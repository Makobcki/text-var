from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TokenCacheMetadata:
    kind: str
    level_vocab_sizes: tuple[int, ...]  # Размер алфавита для каждого уровня
    level_lengths: tuple[int, ...]  # Явная длина последовательности каждого уровня
    codebook_dim: int
    max_token_length: int
    format_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TokenCacheMetadata:
        return cls(
            kind=str(data.get("kind", "vq")),
            level_vocab_sizes=tuple(int(v) for v in data["level_vocab_sizes"]),  # type: ignore[index]
            level_lengths=tuple(int(v) for v in data["level_lengths"]),  # type: ignore[index]
            codebook_dim=int(data.get("codebook_dim", 0)),
            max_token_length=int(data["max_token_length"]),
            format_version=int(data.get("format_version", 1)),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["level_vocab_sizes"] = list(self.level_vocab_sizes)
        data["level_lengths"] = list(self.level_lengths)
        return data

    def __post_init__(self) -> None:
        vocab_sizes = tuple(int(v) for v in self.level_vocab_sizes)
        lengths = tuple(int(l) for l in self.level_lengths)
        object.__setattr__(self, "level_vocab_sizes", vocab_sizes)
        object.__setattr__(self, "level_lengths", lengths)

        if len(vocab_sizes) != len(lengths):
            raise ValueError(
                f"Размерности массивов не совпадают: len(level_lengths)={len(lengths)} "
                f"!= len(level_vocab_sizes)={len(vocab_sizes)}."
            )
        if sum(lengths) > int(self.max_token_length):
            raise ValueError(
                f"Сумма level_lengths ({sum(lengths)}) превышает max_token_length ({self.max_token_length})."
            )


def save_token_cache_metadata(path: str | Path, metadata: TokenCacheMetadata) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_token_cache_metadata(path: str | Path) -> TokenCacheMetadata:
    return TokenCacheMetadata.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_tokenizer_metadata(actual: TokenCacheMetadata, expected: TokenCacheMetadata) -> None:
    if len(actual.level_lengths) != len(actual.level_vocab_sizes):
        raise ValueError("level_lengths и level_vocab_sizes должны иметь одинаковую длину.")

    for field_name in (
        "kind",
        "level_vocab_sizes",
        "level_lengths",
        "codebook_dim",
        "max_token_length",
    ):
        if getattr(actual, field_name) != getattr(expected, field_name):
            raise ValueError(
                "Tokenizer metadata mismatch for "
                f"{field_name}: {getattr(actual, field_name)!r} != "
                f"{getattr(expected, field_name)!r}."
            )
    if sum(actual.level_lengths) > int(actual.max_token_length):
        raise ValueError("Сумма level_lengths превышает max_token_length.")


def build_synthetic_token_entries(
    metadata: TokenCacheMetadata, *, count: int, seed: int = 0
) -> list[dict[str, object]]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    entries: list[dict[str, object]] = []

    for idx in range(int(count)):
        scale_tokens: list[torch.Tensor] = []
        if len(metadata.level_lengths) == 0:
            entries.append({"id": f"synthetic-{idx}", "tokens": scale_tokens})
            continue

        # Генерация токенов первого уровня (Уровень 0)
        lvl_len_0 = int(metadata.level_lengths[0])
        vocab_size_0 = int(metadata.level_vocab_sizes[0])
        root = torch.randint(
            0,
            vocab_size_0,
            (lvl_len_0,),
            generator=generator,
            dtype=torch.long,
        )
        scale_tokens.append(root)
        source = int(root[0].item()) if root.numel() else idx

        # Генерация токенов для последующих уровней с ограничением сверху по индивидуальным размерам словарей
        for scale_idx in range(1, len(metadata.level_lengths)):
            length = int(metadata.level_lengths[scale_idx])
            vocab_size = int(metadata.level_vocab_sizes[scale_idx])
            offset = 997 * scale_idx
            tokens = (source + offset + torch.arange(length, dtype=torch.long)) % vocab_size
            scale_tokens.append(tokens)

        entries.append({"id": f"synthetic-{idx}", "tokens": scale_tokens})
    return entries


def load_token_entries(path: str | Path) -> tuple[list[dict[str, object]], TokenCacheMetadata]:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Token cache payload must be a dictionary.")
    metadata = TokenCacheMetadata.from_dict(dict(payload["metadata"]))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Token cache payload must contain an entries list.")
    return entries, metadata


class MultiscaleTokenDataset(Dataset):
    def __init__(self, entries: list[dict[str, object]], metadata: TokenCacheMetadata) -> None:
        self.entries = list(entries)
        self.metadata = metadata
        if sum(metadata.level_lengths) > int(metadata.max_token_length):
            raise ValueError("Сумма level_lengths превышает max_token_length.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[int(index)]
        tokens = entry.get("tokens")
        if not isinstance(tokens, list) or len(tokens) != len(self.metadata.level_lengths):
            raise ValueError("Token cache entry has invalid scale count.")

        out = [torch.as_tensor(item, dtype=torch.long) for item in tokens]
        for lvl_idx, (expected, vocab_size) in enumerate(
            zip(self.metadata.level_lengths, self.metadata.level_vocab_sizes)
        ):
            item = out[lvl_idx]
            if item.numel() != expected:
                raise ValueError(
                    f"Token level {lvl_idx} expected {expected} tokens, got {item.numel()}."
                )
            if item.numel() and (
                int(item.max().item()) >= vocab_size or int(item.min().item()) < 0
            ):
                raise ValueError("Token value outside tokenizer codebook.")

        return {"id": entry.get("id", str(index)), "tokens": out, "metadata": self.metadata}


def _metadata_from_payload(payload: dict[str, Any]) -> TokenCacheMetadata:
    return TokenCacheMetadata.from_dict(dict(payload["metadata"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate VAR token cache metadata.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--expect", type=Path, default=None)
    args = parser.parse_args(argv)

    actual = load_token_cache_metadata(args.metadata)
    if args.expect is not None:
        expected = load_token_cache_metadata(args.expect)
        validate_tokenizer_metadata(actual, expected)
    else:
        validate_tokenizer_metadata(actual, actual)
    print(json.dumps(actual.to_dict(), sort_keys=True))


__all__ = [
    "MultiscaleTokenDataset",
    "TokenCacheMetadata",
    "build_synthetic_token_entries",
    "load_token_cache_metadata",
    "load_token_entries",
    "save_token_cache_metadata",
    "validate_tokenizer_metadata",
]


if __name__ == "__main__":
    main()
