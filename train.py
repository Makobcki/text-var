from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from var_branch.checkpoint import save_checkpoint
from var_branch.config import TrainConfig, load_train_config
from var_branch.loss import multiscale_next_scale_cross_entropy  # Сюда перемещен лосс
from var_branch.model import VARTransformer
from var_branch.token_cache import (
    MultiscaleTokenDataset,
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
        validate_tokenizer_metadata(actual_metadata, cfg.token_metadata)
        return MultiscaleTokenDataset(entries, actual_metadata)
    entries = build_synthetic_token_entries(
        cfg.synthetic_count, cfg.model.level_vocab_sizes, cfg.model.level_lengths
    )
    from var_branch.token_cache import TokenCacheMetadata

    dummy_meta = TokenCacheMetadata(
        kind="synthetic",
        level_vocab_sizes=cfg.model.level_vocab_sizes,
        level_lengths=cfg.model.level_lengths,
        codebook_dim=int(cfg.model.hidden_size),
        max_token_length=sum(cfg.model.level_lengths),
    )
    return MultiscaleTokenDataset(entries, dummy_meta)


def apply_phase_freezing(model: VARTransformer, phase_idx: int) -> None:
    for p in model.parameters():
        p.requires_grad = True

    # Безопасное замораживание без обращения к несуществующим эмбеддингам
    if phase_idx == 0:
        for idx in range(1, len(model.token_embeddings)):
            for p in model.token_embeddings[idx].parameters():
                p.requires_grad = False
    model.local_position_embedding.requires_grad = True


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
