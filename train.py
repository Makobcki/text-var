import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    MultiscaleTokenChunkIterableDataset,
    MultiscaleTokenDataset,
    TokenCacheMetadata,
    build_synthetic_token_entries,
    load_token_entries,
    load_token_entries_from_directory,
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


def _resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def _is_amp_available(device: torch.device, amp_dtype: torch.dtype) -> bool:
    if device.type != "cuda":
        return False
    if amp_dtype is torch.float16:
        return True
    return torch.cuda.is_bf16_supported()


def _build_dataset(cfg: TrainConfig):
    if cfg.token_cache_path is not None:
        if cfg.token_cache_path.is_dir():
            chunk_paths, actual_metadata = load_token_entries_from_directory(cfg.token_cache_path)
            if cfg.token_metadata is not None:
                validate_tokenizer_metadata(actual_metadata, cfg.token_metadata)
            return MultiscaleTokenChunkIterableDataset(chunk_paths, actual_metadata)

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

    # Анти-drift режим: общий ствол/общие эмбеддинги фиксируем после старта
    _set_trainable(model.scale_embedding, phase == 0)
    _set_trainable(model.local_position_embedding, phase == 0)
    model.target_token.requires_grad = phase == 0

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
    if bool(cfg.compile_enabled):
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("[TRAIN] torch.compile enabled.")
        else:
            print("[TRAIN] torch.compile requested but unavailable in this PyTorch build.")

    if cfg.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "optimizer=adamw8bit requires bitsandbytes. Install with `pip install bitsandbytes`."
            ) from exc

        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        print("[TRAIN] optimizer=adamw8bit (bitsandbytes)")
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        print("[TRAIN] optimizer=adamw")
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    dataset = _build_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=not isinstance(dataset, MultiscaleTokenChunkIterableDataset),
        collate_fn=_collate_tokens,
        pin_memory=bool(cfg.pin_memory),
        num_workers=4,
        persistent_workers=True,
    )

    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
    amp_enabled = bool(cfg.amp_enabled) and _is_amp_available(device, amp_dtype)
    if bool(cfg.amp_enabled) and not amp_enabled:
        print("[TRAIN] AMP requested but unavailable on selected device; fallback to fp32.")

    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype is torch.float16)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))

    step = 0
    last_loss = 0.0
    accumulated_steps_per_phase = 0
    current_phase = 0
    micro_step = 0
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

            non_blocking = bool(cfg.pin_memory) and device.type == "cuda"
            moved_tokens = [t.to(device, non_blocking=non_blocking) for t in batch]

            micro_step += 1
            should_step = micro_step % grad_accum_steps == 0

            if micro_step % grad_accum_steps == 1:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = multiscale_next_scale_cross_entropy(
                    model,
                    moved_tokens,
                    level_weights=cfg.level_weights,
                    corruption_level_idx=cfg.corruption_level_idx,
                    corruption_prob=cfg.corruption_prob,
                    corruption_span_min=cfg.corruption_span_min,
                    corruption_span_max=cfg.corruption_span_max,
                    masked_loss_weight=cfg.masked_loss_weight,
                    use_early_exit_loss=cfg.use_early_exit_loss,
                )

            scaled_loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
                if should_step:
                    if float(cfg.grad_clip_norm) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg.grad_clip_norm)
                        )
                    scaler.step(optimizer)
                    scaler.update()
            else:
                scaled_loss.backward()
                if should_step:
                    if float(cfg.grad_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg.grad_clip_norm)
                        )
                    optimizer.step()

            if should_step:
                step += 1
                accumulated_steps_per_phase += 1
                last_loss = float(loss.detach().cpu())

                print(
                    f"[TRAIN] family=var phase={current_phase + 1} step={step}/{cfg.max_steps} loss={last_loss:.6f}"
                )

            if should_step and int(cfg.save_every) > 0 and step % int(cfg.save_every) == 0:
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

    if step < int(cfg.max_steps) and micro_step % grad_accum_steps != 0:
        if scaler.is_enabled():
            if float(cfg.grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            if float(cfg.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
            optimizer.step()
        step += 1
        last_loss = float(loss.detach().cpu())

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
