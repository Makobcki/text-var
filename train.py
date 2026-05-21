from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from checkpoint import save_checkpoint
from config import TrainConfig, load_train_config
from loss import multiscale_next_scale_cross_entropy  # Сюда перемещен лосс
from model import VARTransformer
from token_cache import (
    MultiscaleTokenDataset,
    TokenCacheMetadata,
    build_synthetic_token_entries,
    load_token_entries,
    validate_tokenizer_metadata,
)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _collate_tokens(batch: list[dict[str, Any]]) -> list[torch.Tensor]:
    if not batch:
        raise RuntimeError("VAR token batch is empty.")
    scale_count = len(batch[0]["tokens"])
    return [
        torch.stack([item["tokens"][scale_idx] for item in batch], dim=0)
        for scale_idx in range(scale_count)
    ]


def _build_dataset(cfg: TrainConfig) -> MultiscaleTokenDataset:
    if cfg.token_cache_path is not None:
        entries, actual_metadata = load_token_entries(cfg.token_cache_path)
        if cfg.token_metadata is not None:
            validate_tokenizer_metadata(actual_metadata, cfg.token_metadata)
        return MultiscaleTokenDataset(entries, actual_metadata)

    dummy_meta = TokenCacheMetadata(
        kind="synthetic",
        level_vocab_sizes=cfg.model.level_vocab_sizes,
        level_lengths=cfg.model.level_lengths,
        codebook_dim=int(cfg.model.hidden_size),
        max_token_length=sum(cfg.model.level_lengths),
    )
    entries = build_synthetic_token_entries(dummy_meta, count=cfg.synthetic_count, seed=cfg.seed)
    return MultiscaleTokenDataset(entries, dummy_meta)


def _set_trainable(module: torch.nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def apply_phase_freezing(model: VARTransformer, phase_idx: int) -> None:
    for param in model.parameters():
        param.requires_grad = False

    levels = len(model.token_embeddings)
    depth = len(model.blocks)
    segment = max(1, depth // max(1, levels))

    phase = max(0, min(int(phase_idx), levels - 1))
    block_start = min(depth, phase * segment)
    block_end = depth if phase == levels - 1 else min(depth, (phase + 1) * segment)

    # Базовые компоненты всегда обучаемы
    _set_trainable(model.scale_embedding, True)
    _set_trainable(model.local_position_embedding, True)
    model.target_token.requires_grad = True

    # Обучаем только активный уровень (токен-эмбеддинг + головы)
    _set_trainable(model.token_embeddings[phase], True)
    _set_trainable(model.heads[phase], True)

    for key, head in model.early_exit_heads.items():
        scale_tag = f"_scale_{phase}"
        _set_trainable(head, scale_tag in key)

    # Послойная заморозка: активен только сегмент блоков текущей фазы
    for block_idx, block in enumerate(model.blocks):
        _set_trainable(block, block_start <= block_idx < block_end)

    # LayerNorm оставляем обучаемым для адаптации распределений в каждой фазе
    _set_trainable(model.norm, True)

    print(
        "[TRAIN] freeze-policy "
        f"phase={phase + 1}/{levels} "
        f"active_level={phase} "
        f"active_blocks=[{block_start}, {max(block_start, block_end)})"
    )


def run_training(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = _resolve_device(cfg.device)
    model = VARTransformer(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    dataset = _build_dataset(cfg)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate_tokens
    )

    step = 0
    last_loss = 0.0
    accumulated_steps_per_phase = 0
    current_phase = 0
    apply_phase_freezing(model, current_phase)

    model.train()
    print(f"[TRAIN] Инициализация пройдена успешно. Старт на {device}.")

    while step < int(cfg.max_steps):
        saw_batch = False
        for batch in dataloader:
            saw_batch = True

            if (
                current_phase < len(cfg.phase_steps) - 1
                and accumulated_steps_per_phase >= cfg.phase_steps[current_phase]
            ):
                current_phase += 1
                accumulated_steps_per_phase = 0
                apply_phase_freezing(model, current_phase)
                print(f"[TRAIN] Переключение фазы обучения на: {current_phase + 1}")

            moved_tokens = [t.to(device) for t in batch]

            optimizer.zero_grad()
            loss = multiscale_next_scale_cross_entropy(
                model, moved_tokens, level_weights=cfg.level_weights
            )
            loss.backward()

            if float(cfg.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))

            optimizer.step()
            step += 1
            accumulated_steps_per_phase += 1
            last_loss = float(loss.detach().cpu())

            print(
                f"[TRAIN] family=var phase={current_phase + 1} step={step}/{cfg.max_steps} loss={last_loss:.6f}"
            )

            if int(cfg.save_every) > 0 and step % int(cfg.save_every) == 0:
                save_checkpoint(
                    cfg.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    loss=last_loss,
                )
            if step >= int(cfg.max_steps):
                break

        if not saw_batch:
            raise RuntimeError("VAR training dataloader produced no batches.")

    save_checkpoint(
        cfg.checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=step,
        loss=last_loss,
    )
    return cfg.checkpoint_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a standalone VAR model.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint_path = run_training(load_train_config(args.config))
    print(f"[TRAIN] Обучение завершено. Чекпоинт сохранен в {checkpoint_path}")


if __name__ == "__main__":
    main()
