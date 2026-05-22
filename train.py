import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from config import TrainConfig, load_train_config
from loss import multiscale_next_scale_cross_entropy
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


def _compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        value = float(param.grad.detach().float().norm(2).cpu())
        total += value * value
    return math.sqrt(total)


def _compute_weight_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        value = float(param.detach().float().norm(2).cpu())
        total += value * value
    return math.sqrt(total)


def _generate_validation_sample(step: int, prompt: str, max_new_tokens: int) -> str:
    random_tail = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=8))
    return (
        f"[step={step}] prompt='{prompt}'\n"
        f"sample='{prompt} ... {random_tail}'\n"
        f"max_new_tokens={max_new_tokens}"
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


def _build_dataset(cfg: TrainConfig, *, for_validation: bool = False):
    path = cfg.val_token_cache_path if for_validation else cfg.token_cache_path
    if path is not None:
        if path.is_dir():
            chunk_paths, actual_metadata = load_token_entries_from_directory(path)
            if cfg.token_metadata is not None:
                validate_tokenizer_metadata(actual_metadata, cfg.token_metadata)
            return MultiscaleTokenChunkIterableDataset(chunk_paths, actual_metadata)

        entries, actual_metadata = load_token_entries(path)
        if cfg.token_metadata is not None:
            validate_tokenizer_metadata(actual_metadata, cfg.token_metadata)
        return MultiscaleTokenDataset(entries, actual_metadata)

    if for_validation and cfg.validation_split <= 0:
        return None

    dummy_meta = TokenCacheMetadata(
        kind="synthetic",
        level_vocab_sizes=cfg.model.level_vocab_sizes,
        level_lengths=cfg.model.level_lengths,
        codebook_dim=int(cfg.model.hidden_size),
        max_token_length=sum(cfg.model.level_lengths),
    )
    count = cfg.synthetic_val_count if for_validation else cfg.synthetic_count
    seed = cfg.seed + (10_000 if for_validation else 0)
    entries = build_synthetic_token_entries(dummy_meta, count=count, seed=seed)
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

    _set_trainable(model.scale_embedding, phase == 0)
    _set_trainable(model.local_position_embedding, phase == 0)
    model.target_token.requires_grad = phase == 0

    _set_trainable(model.token_embeddings[phase], True)
    _set_trainable(model.heads[phase], True)

    for key, head in model.early_exit_heads.items():
        scale_tag = f"_scale_{phase}"
        _set_trainable(head, scale_tag in key)

    for block_idx, block in enumerate(model.blocks):
        _set_trainable(block, block_start <= block_idx < block_end)

    _set_trainable(model.norm, True)

    print(
        "[TRAIN] freeze-policy "
        f"phase={phase + 1}/{levels} "
        f"active_level={phase} "
        f"active_blocks=[{block_start}, {max(block_start, block_end)})"
    )


@torch.no_grad()
def evaluate(
    model: VARTransformer,
    dataloader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max(1, int(max_batches)):
            break
        non_blocking = bool(cfg.pin_memory) and device.type == "cuda"
        moved_tokens = [t.to(device, non_blocking=non_blocking) for t in batch]
        with torch.autocast(
            device_type=device.type,
            dtype=_resolve_amp_dtype(cfg.amp_dtype),
            enabled=bool(cfg.amp_enabled),
        ):
            loss = multiscale_next_scale_cross_entropy(
                model,
                moved_tokens,
                level_weights=cfg.level_weights,
                corruption_level_idx=-1,
                corruption_prob=0.0,
                use_early_exit_loss=False,
            )
        total_loss += float(loss.detach().cpu())
        total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)


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

    warmup_steps = int(cfg.max_steps * float(cfg.warmup_ratio))
    warmup_steps = max(1, min(int(cfg.max_steps) - 1, warmup_steps)) if int(cfg.max_steps) > 1 else 0
    min_lr_ratio = min(1.0, max(0.0, float(cfg.min_learning_rate_ratio)))

    def _lr_lambda(current_step: int) -> float:
        if int(cfg.max_steps) <= 1:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress_denominator = max(1, int(cfg.max_steps) - warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(progress_denominator)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    print(
        "[TRAIN] scheduler=cosine_with_warmup "
        f"warmup_steps={warmup_steps} "
        f"min_lr={cfg.learning_rate * min_lr_ratio:.3e}"
    )

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

    val_dataloader = None
    if cfg.validation_every > 0:
        val_dataset = _build_dataset(cfg, for_validation=True)
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=_collate_tokens,
                pin_memory=bool(cfg.pin_memory),
                num_workers=2,
                persistent_workers=True,
            )

    writer = SummaryWriter(log_dir=str(cfg.log_dir)) if cfg.tensorboard_enabled else None
    wandb_run = None
    if cfg.wandb_enabled:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb_enabled=true requires Weights & Biases. Install with `pip install wandb`."
            ) from exc
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "max_steps": cfg.max_steps,
                "amp_dtype": cfg.amp_dtype,
            },
        )
        print(f"[TRAIN] wandb enabled: project={cfg.wandb_project} run={wandb_run.name}")

    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
    amp_enabled = bool(cfg.amp_enabled) and _is_amp_available(device, amp_dtype)
    if bool(cfg.amp_enabled) and not amp_enabled:
        print("[TRAIN] AMP requested but unavailable on selected device; fallback to fp32.")

    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype is torch.float16)
    grad_accum_steps = max(1, int(cfg.grad_accum_steps))

    step = 0
    current_phase = 0
    if cfg.resume_from is not None:
        loaded_model, payload = load_checkpoint(cfg.resume_from, device=device)
        model.load_state_dict(loaded_model.state_dict())
        restore_training_state(payload, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
        step = int(payload.get("step", 0))
        current_phase = min(
            len(cfg.phase_steps) - 1,
            sum(1 for phase_limit in cfg.phase_steps[:-1] if step >= int(phase_limit)),
        )
        print(f"[TRAIN] resumed from checkpoint={cfg.resume_from} step={step}")
    elif cfg.checkpoint_path.exists():
        print(f"[TRAIN] Найден чекпоинт: {cfg.checkpoint_path}. Восстанавливаем состояние...")
        loaded_model, payload = load_checkpoint(cfg.checkpoint_path, device=device)
        model.load_state_dict(loaded_model.state_dict())
        restore_training_state(payload, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
        step = int(payload.get('step', 0))
        phase_boundaries = [0]
        for phase_steps in cfg.phase_steps:
            phase_boundaries.append(phase_boundaries[-1] + int(phase_steps))
        current_phase = 0
        for idx in range(len(phase_boundaries) - 1):
            if phase_boundaries[idx] <= step < phase_boundaries[idx + 1]:
                current_phase = idx
                break
        else:
            current_phase = len(cfg.phase_steps) - 1
        print(f"[TRAIN] Успешно продолжено с шага {step}")
    last_loss = 0.0
    phase_start = sum(int(phase_steps) for phase_steps in cfg.phase_steps[:current_phase])
    accumulated_steps_per_phase = max(0, step - phase_start)
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                scaled_loss.backward()
                if should_step:
                    if float(cfg.grad_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                    optimizer.step()
                    scheduler.step()

            if should_step:
                step += 1
                accumulated_steps_per_phase += 1
                last_loss = float(loss.detach().cpu())
                print(f"[TRAIN] family=var phase={current_phase + 1} step={step}/{cfg.max_steps} loss={last_loss:.6f}")
                if writer:
                    writer.add_scalar("train/loss", last_loss, step)
                    writer.add_scalar("train/lr", float(scheduler.get_last_lr()[0]), step)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": last_loss,
                            "train/lr": float(scheduler.get_last_lr()[0]),
                            "phase": current_phase + 1,
                        },
                        step=step,
                    )

                if int(cfg.log_grad_norm_every) > 0 and step % int(cfg.log_grad_norm_every) == 0:
                    grad_norm = _compute_grad_norm(model)
                    print(f"[MONITOR] step={step} grad_norm={grad_norm:.6f}")
                    if writer:
                        writer.add_scalar("train/grad_norm_l2", grad_norm, step)
                    if wandb_run is not None:
                        wandb_run.log({"train/grad_norm_l2": grad_norm}, step=step)

                if int(cfg.log_weight_norm_every) > 0 and step % int(cfg.log_weight_norm_every) == 0:
                    weight_norm = _compute_weight_norm(model)
                    print(f"[MONITOR] step={step} weight_norm={weight_norm:.6f}")
                    if writer:
                        writer.add_scalar("train/weight_norm_l2", weight_norm, step)
                    if wandb_run is not None:
                        wandb_run.log({"train/weight_norm_l2": weight_norm}, step=step)

                val_every = int(getattr(cfg, "val_every", 0)) or int(cfg.validation_every) or int(cfg.save_every)
                if val_dataloader is not None and val_every > 0 and step % val_every == 0:
                    val_loss = evaluate(
                        model,
                        val_dataloader,
                        cfg,
                        device,
                        max_batches=int(cfg.validation_batches),
                    )
                    val_ppl = math.exp(min(20.0, val_loss))
                    print(f"[VAL] step={step} val_loss={val_loss:.6f} perplexity={val_ppl:.4f}")
                    if writer:
                        writer.add_scalar("val/loss", val_loss, step)
                        writer.add_scalar("val/perplexity", val_ppl, step)
                    if wandb_run is not None:
                        wandb_run.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)

                if int(cfg.sample_every) > 0 and step % int(cfg.sample_every) == 0:
                    generated_text = _generate_validation_sample(
                        step=step,
                        prompt=cfg.sample_prompt,
                        max_new_tokens=cfg.sample_max_new_tokens,
                    )
                    print(f"[SAMPLE]\n{generated_text}")
                    if writer:
                        writer.add_text("samples/validation", generated_text, step)
                    if wandb_run is not None:
                        wandb_run.log({"samples/validation": generated_text}, step=step)

            if should_step and int(cfg.save_every) > 0 and step % int(cfg.save_every) == 0:
                save_checkpoint(
                    cfg.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    loss=last_loss,
                    scaler=scaler,
                    scheduler=scheduler,
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
        scaler=scaler,
        scheduler=scheduler,
    )
    if writer:
        writer.flush()
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    return cfg.checkpoint_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a standalone VAR model.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint_path = run_training(load_train_config(args.config))
    print(f"[TRAIN] Обучение завершено. Чекпоинт сохранен в {checkpoint_path}")


if __name__ == "__main__":
    main()
