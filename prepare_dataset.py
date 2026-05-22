from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from token_cache import TokenCacheMetadata

_WORKER_TOKENIZERS: dict[tuple[str, bool], object] = {}


def _get_tokenizer(tokenizer_name: str, use_fast: bool):
    key = (tokenizer_name, use_fast)
    tok = _WORKER_TOKENIZERS.get(key)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=use_fast)
        _WORKER_TOKENIZERS[key] = tok
    return tok


def _fit_to_level(token_ids: list[int], *, length: int, vocab_size: int, pad_id: int = 0) -> list[int]:
    """Truncate/pad token sequence and clamp ids into target vocab range."""
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1 for real tokenizer pipeline")

    overflow_id = vocab_size - 1
    clipped = [tok if 0 <= tok < vocab_size else overflow_id for tok in token_ids[:length]]
    if len(clipped) < length:
        clipped.extend([pad_id] * (length - len(clipped)))
    return clipped


def _downsample(token_ids: list[int], stride: int) -> list[int]:
    if stride <= 1:
        return token_ids
    return token_ids[::stride]


def _encode_multiscale(
    token_ids: list[int], *, level_lengths: tuple[int, ...], level_vocab_sizes: tuple[int, ...]
) -> list[list[int]]:
    if len(level_lengths) != 3 or len(level_vocab_sizes) != 3:
        raise ValueError("This pipeline expects exactly 3 token levels.")

    # Coarse -> medium -> fine levels from a single BPE/WordPiece stream.
    lvl0 = _fit_to_level(
        _downsample(token_ids, 4),
        length=level_lengths[0],
        vocab_size=level_vocab_sizes[0],
    )
    lvl1 = _fit_to_level(
        _downsample(token_ids, 2),
        length=level_lengths[1],
        vocab_size=level_vocab_sizes[1],
    )
    lvl2 = _fit_to_level(
        token_ids,
        length=level_lengths[2],
        vocab_size=level_vocab_sizes[2],
    )
    return [lvl0, lvl1, lvl2]


def process_single_line(
    args_tuple: tuple[str, int, tuple[int, ...], tuple[int, ...], str, bool],
) -> dict[str, object]:
    line, index, level_lengths, level_vocab_sizes, tokenizer_name, use_fast = args_tuple

    payload = json.loads(line)
    text = str(payload.get("content", payload.get("text", ""))).strip()

    if not text:
        return {
            "id": str(index),
            "tokens": None,
            "bytes_processed": len(line.encode("utf-8")),
        }

    tokenizer = _get_tokenizer(tokenizer_name, use_fast)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return {
            "id": str(index),
            "tokens": None,
            "bytes_processed": len(line.encode("utf-8")),
        }

    lvl0, lvl1, lvl2 = _encode_multiscale(
        token_ids, level_lengths=level_lengths, level_vocab_sizes=level_vocab_sizes
    )

    return {
        "id": str(index),
        "tokens": [lvl0, lvl1, lvl2],
        "bytes_processed": len(line.encode("utf-8")),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-core Tokenizer for VAR.")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for batched token cache.",
    )
    parser.add_argument("--field", type=str, default="content")
    parser.add_argument("--kind", type=str, default="vq")
    parser.add_argument("--level-vocab-sizes", type=int, nargs=3, default=(4096, 2048, 32000))
    parser.add_argument("--level-lengths", type=int, nargs=3, default=(32, 128, 1024))
    parser.add_argument("--codebook-dim", type=int, default=256)
    parser.add_argument("--tokenizer-name", type=str, default="bert-base-uncased")
    parser.add_argument("--slow-tokenizer", action="store_true", help="Use python tokenizer instead of fast backend.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(8, cpu_count() - 1)),
        help="Worker processes (cap to reduce RAM pressure).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4000,
        help="How many samples per chunk file (lower = less RAM).",
    )

    args = parser.parse_args(argv)

    metadata = TokenCacheMetadata(
        kind=args.kind,
        level_vocab_sizes=tuple(int(v) for v in args.level_vocab_sizes),
        level_lengths=tuple(int(v) for v in args.level_lengths),
        codebook_dim=int(args.codebook_dim),
        max_token_length=sum(int(v) for v in args.level_lengths),
    )

    total_bytes = os.path.getsize(args.input)
    num_workers = int(args.num_workers)
    print(f"Запуск параллельной токенизации на {num_workers} ядрах CPU...")

    # Создаем чистую целевую папку
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем метадату отдельно, чтобы она лежала рядом с батчами
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as fM:
        json.dump(metadata.to_dict(), fM, ensure_ascii=False, indent=2)

    current_batch = []
    batch_index = 0
    total_saved_docs = 0

    with Pool(processes=num_workers) as pool:
        with open(args.input, "r", encoding="utf-8") as f:

            def argument_generator():
                for idx, line in enumerate(f):
                    yield (
                        line,
                        idx,
                        metadata.level_lengths,
                        metadata.level_vocab_sizes,
                        args.tokenizer_name,
                        not args.slow_tokenizer,
                    )

            pbar = tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Токенизация",
            )

            # Использование pool.imap гарантирует сохранение порядка (индексы идут строго вверх)
            # Благодаря этому нам больше не нужна финальная тяжелая сортировка .sort()
            for result in pool.imap(process_single_line, argument_generator(), chunksize=256):
                pbar.update(result["bytes_processed"])

                if result["tokens"] is not None:
                    tensor_tokens = [torch.tensor(t, dtype=torch.int32) for t in result["tokens"]]
                    current_batch.append({"id": result["id"], "tokens": tensor_tokens})
                    total_saved_docs += 1

                # Достигли лимита батча — сбрасываем на диск и полностью чистим память
                if len(current_batch) >= args.batch_size:
                    chunk_path = args.output_dir / f"tokens_chunk_{batch_index:04d}.pt"
                    torch.save({"entries": current_batch}, chunk_path)

                    # Явное освобождение памяти
                    current_batch = []
                    batch_index += 1

            # Сбрасываем остатки данных
            if current_batch:
                chunk_path = args.output_dir / f"tokens_chunk_{batch_index:04d}.pt"
                torch.save({"entries": current_batch}, chunk_path)
                del current_batch

            pbar.close()

    if total_saved_docs == 0:
        raise ValueError("Input dataset is empty or contains no valid records.")

    print(
        f"Успешно завершено! Сохранено {total_saved_docs:,} документов разбитых на {batch_index + 1} чанков в '{args.output_dir}'."
    )


if __name__ == "__main__":
    main()
