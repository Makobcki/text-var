from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from token_cache import TokenCacheMetadata


def _read_texts(path: Path, *, field: str | None) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [line for line in lines if line]

    if suffix == ".jsonl":
        texts: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if field is None:
                if "text" not in payload:
                    raise ValueError("JSONL item has no 'text' field. Pass --field to select another key.")
                value = payload["text"]
            else:
                if field not in payload:
                    raise ValueError(f"JSONL item has no '{field}' field.")
                value = payload[field]
            texts.append(str(value).strip())
        return [t for t in texts if t]

    raise ValueError("Unsupported input extension. Use .txt or .jsonl")


def _stable_u64(data: bytes) -> int:
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _encode_scale(tokens: Iterable[str], *, length: int, vocab_size: int, salt: str) -> torch.Tensor:
    result = torch.zeros((length,), dtype=torch.long)
    tok_list = list(tokens)
    for idx in range(length):
        token = tok_list[idx] if idx < len(tok_list) else f"<pad:{idx}>"
        hashed = _stable_u64(f"{salt}::{idx}::{token}".encode("utf-8"))
        result[idx] = int(hashed % vocab_size)
    return result


def _split_words(text: str) -> list[str]:
    return [piece for piece in text.strip().split() if piece]


def build_entries(
    texts: list[str],
    metadata: TokenCacheMetadata,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    if len(metadata.level_lengths) != 3:
        raise ValueError(
            "This preparation script expects exactly 3 token levels (semantic/local/fine)."
        )

    for idx, text in enumerate(texts):
        words = _split_words(text)
        chars = list(text)
        coarse = _encode_scale(
            words,
            length=int(metadata.level_lengths[0]),
            vocab_size=int(metadata.level_vocab_sizes[0]),
            salt="lvl0",
        )
        local = _encode_scale(
            words + chars,
            length=int(metadata.level_lengths[1]),
            vocab_size=int(metadata.level_vocab_sizes[1]),
            salt="lvl1",
        )
        fine = _encode_scale(
            chars,
            length=int(metadata.level_lengths[2]),
            vocab_size=int(metadata.level_vocab_sizes[2]),
            salt="lvl2",
        )
        entries.append({"id": f"sample-{idx}", "text": text, "tokens": [coarse, local, fine]})
    return entries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare token cache dataset for VAR training.")
    parser.add_argument("--input", type=Path, required=True, help="Path to .txt or .jsonl dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt token cache file.")
    parser.add_argument("--field", type=str, default=None, help="Field name for jsonl records.")
    parser.add_argument("--kind", type=str, default="vq")
    parser.add_argument("--level-vocab-sizes", type=int, nargs=3, default=(4096, 2048, 32000))
    parser.add_argument("--level-lengths", type=int, nargs=3, default=(32, 128, 1024))
    parser.add_argument("--codebook-dim", type=int, default=256)

    args = parser.parse_args(argv)

    texts = _read_texts(args.input, field=args.field)
    if not texts:
        raise ValueError("Input dataset is empty after filtering blank lines.")

    metadata = TokenCacheMetadata(
        kind=args.kind,
        level_vocab_sizes=tuple(int(v) for v in args.level_vocab_sizes),
        level_lengths=tuple(int(v) for v in args.level_lengths),
        codebook_dim=int(args.codebook_dim),
        max_token_length=sum(int(v) for v in args.level_lengths),
    )

    entries = build_entries(texts, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata.to_dict(), "entries": entries}, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(entries),
                "metadata": metadata.to_dict(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
