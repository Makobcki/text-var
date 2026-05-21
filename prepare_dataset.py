from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
from tqdm import tqdm

from token_cache import TokenCacheMetadata


def _stable_u64(data: bytes) -> int:
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _encode_scale(tokens: list[str], *, length: int, vocab_size: int, prefix: str) -> list[int]:
    # Оптимизация: убираем создание torch.tensor внутри воркера.
    # Возвращаем чистый питоновский list[int]. Это снижает оверхед на сериализацию в multiprocessing.
    out = []
    sub_tokens = tokens[:length]
    for idx, tok in enumerate(sub_tokens):
        salt = f"{prefix}::{idx}::{tok}".encode("utf-8")
        out.append(_stable_u64(salt) % vocab_size)

    while len(out) < length:
        pad_tok = f"<pad:{len(out)}>".encode("utf-8")
        salt = f"{prefix}::{len(out)}::{pad_tok}".encode("utf-8")
        out.append(_stable_u64(salt) % vocab_size)

    return out


def process_single_line(
    args_tuple: tuple[str, int, tuple[int, ...], tuple[int, ...]],
) -> dict[str, object]:
    line, index, level_lengths, level_vocab_sizes = args_tuple

    payload = json.loads(line)
    text = str(payload.get("content", payload.get("text", ""))).strip()

    if not text:
        return {
            "id": str(index),
            "tokens": None,
            "bytes_processed": len(line.encode("utf-8")),
        }

    words = text.split()
    chars = list(text)

    lvl0 = _encode_scale(
        words,
        length=level_lengths[0],
        vocab_size=level_vocab_sizes[0],
        prefix="lvl0",
    )
    lvl1 = _encode_scale(
        words + chars,
        length=level_lengths[1],
        vocab_size=level_vocab_sizes[1],
        prefix="lvl1",
    )
    lvl2 = _encode_scale(
        chars,
        length=level_lengths[2],
        vocab_size=level_vocab_sizes[2],
        prefix="lvl2",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="How many samples per chunk file.",
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
    num_workers = max(1, cpu_count() - 1)
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
            for result in pool.imap(process_single_line, argument_generator(), chunksize=250):
                pbar.update(result["bytes_processed"])

                if result["tokens"] is not None:
                    # Превращаем в тензоры прямо перед упаковкой батча
                    tensor_tokens = [torch.tensor(t, dtype=torch.long) for t in result["tokens"]]
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
