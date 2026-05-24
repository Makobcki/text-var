from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset, IterableDataset


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
        lengths = tuple(int(length_value) for length_value in self.level_lengths)
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


def load_token_entries_from_directory(
    path: str | Path,
) -> tuple[list[Path], TokenCacheMetadata]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"Token cache path must be a directory: {root}")

    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Token cache metadata file not found: {metadata_path}")

    metadata = load_token_cache_metadata(metadata_path)
    chunk_paths = sorted(root.glob("tokens_chunk_*.pt")) + sorted(root.glob("tokens_chunk_*.safetensors"))
    if not chunk_paths:
        raise ValueError(f"No token chunk files found in {root}")
    return chunk_paths, metadata


def _iter_safetensors_entries(chunk_path: Path, metadata: TokenCacheMetadata):
    payload = load_file(str(chunk_path), device="cpu")
    ids = payload.get("ids")
    if ids is None or ids.dim() != 1:
        raise ValueError(f"Chunk {chunk_path} must contain a 1D ids tensor.")

    level_tensors: list[torch.Tensor] = []
    for level_idx, expected_length in enumerate(metadata.level_lengths):
        level_tensor = payload.get(f"tokens_level_{level_idx}")
        if level_tensor is None or level_tensor.dim() != 2:
            raise ValueError(f"Chunk {chunk_path} missing tokens_level_{level_idx} tensor.")
        if level_tensor.shape[0] != ids.shape[0] or level_tensor.shape[1] != expected_length:
            raise ValueError(f"Chunk {chunk_path} has invalid shape for tokens_level_{level_idx}.")
        level_tensors.append(level_tensor)

    for row_idx in range(ids.shape[0]):
        out = [torch.as_tensor(level_tensor[row_idx], dtype=torch.long) for level_tensor in level_tensors]
        for lvl_idx, vocab_size in enumerate(metadata.level_vocab_sizes):
            item = out[lvl_idx]
            if item.numel() and (int(item.max().item()) >= vocab_size or int(item.min().item()) < 0):
                raise ValueError("Token value outside tokenizer codebook.")
        yield {
            "id": str(int(ids[row_idx].item())),
            "tokens": out,
            "metadata": metadata,
        }


class MultiscaleTokenChunkIterableDataset(IterableDataset):
    def __init__(self, chunk_paths: list[Path], metadata: TokenCacheMetadata) -> None:
        self.chunk_paths = list(chunk_paths)
        self.metadata = metadata

    def __iter__(self):
        for chunk_path in self.chunk_paths:
            if chunk_path.suffix == ".safetensors":
                yield from _iter_safetensors_entries(chunk_path, self.metadata)
                continue
            payload = torch.load(chunk_path, map_location="cpu")
            entries = payload.get("entries") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise ValueError(f"Chunk {chunk_path} must contain an entries list.")

            for index, entry in enumerate(entries):
                tokens = entry.get("tokens") if isinstance(entry, dict) else None
                if not isinstance(tokens, list) or len(tokens) != len(self.metadata.level_lengths):
                    raise ValueError(f"Token cache entry has invalid scale count in {chunk_path}.")

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

                yield {
                    "id": entry.get("id", f"{chunk_path.name}:{index}"),
                    "tokens": out,
                    "metadata": self.metadata,
                }


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
    "MultiscaleTokenChunkIterableDataset",
    "TokenCacheMetadata",
    "build_synthetic_token_entries",
    "load_token_cache_metadata",
    "load_token_entries_from_directory",
    "load_token_entries",
    "save_token_cache_metadata",
    "validate_tokenizer_metadata",
]


if __name__ == "__main__":
    main()
