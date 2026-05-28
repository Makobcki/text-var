import logging
import signal
import time
from pathlib import Path

import torch
import bitsandbytes as bnb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from torch.utils.data import DataLoader

from src.core.training_logger import StepTiming, TrainingStepLogger
from src.data.token_cache import (
    MultiscaleTokenChunkIterableDataset,
    TokenCacheMetadata,
    load_token_entries_from_directory,
)

from src.vqvae.model import SemanticTextVQVAE
from src.vqvae.training.cli import build_parser
from src.vqvae.training.config import VQVAETrainConfig, load_vqvae_train_config

LOGGER = logging.getLogger(__name__)
CONSOLE = Console()


class TrainingInterruptedError(RuntimeError):
    """Raised when VQ-VAE training is interrupted."""


def _configure_logging(verbose: bool) -> None:
    """Configure logging format and level.

    Args:
        verbose: Whether debug-level verbose logging is enabled.
    """
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="[%(levelname)s] - %(message)s - [%(filename)s:%(lineno)d]"
        )
        return
    logging.basicConfig(level=logging.WARNING, format="%(message)s")


from src.core.checkpoint import CheckpointManager


def _collate_level(level_index: int):
    def collate(batch: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [item["tokens"][level_index] for item in batch]  # type: ignore[index]
        stacked = torch.stack(
            [t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in tokens], dim=0
        )
        stacked = stacked.to(dtype=torch.long)
        padding_mask = stacked.eq(0)
        return stacked, padding_mask

    return collate


def run_training(
    output_path: Path,
    token_cache_dir: Path,
    *,
    steps: int = 500,
    epochs: int = 0,
    save_every: int = 1000,
    max_checkpoints: int = 3,
    batch_size: int = 8,
    vocab_size: int = 32000,
    hidden_size: int = 1024,
    num_semantic_tokens: int = 4096,
    semantic_sequence_length: int = 1,
    pad_token_id: int = 0,
    max_position_embeddings: int = 2048,
    semantic_pad_token_id: int = 0,
    lr: float = 3e-4,
    device: str = "cuda",
    level_index: int = 2,
    gradient_accumulation_steps: int = 1,
    dataloader_num_workers: int = 4,
    dataloader_prefetch_factor: int = 2,
    pin_memory: bool = True,
    use_torch_compile: bool = False,
    compile_mode: str = "default",
    use_turboquant_kv: bool = False,
    turboquant_key_bits: int = 4,
    turboquant_value_bits: int = 4,
    turboquant_qjl_residual_scale: float = 0.5,
    gradient_checkpointing: bool = False,
    use_rotary_embeddings: bool = True,
    use_unpadding: bool = False,
    log_every_steps: int = 10,
    verbose: bool = False,
    weight_decay: float = 0.05,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.1,
    scheduler_type: str = "cosine",
    optimizer_type: str = "adamw",
    max_grad_norm: float = 1.0,
    tensorboard_dir: str = "runs/vqvae",
    resume_from: Path | None = None,
) -> Path:
    _configure_logging(verbose)
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0.")

    if log_every_steps <= 0:
        CONSOLE.warning("log_every_steps must be greater than 0, setting to 1")
        log_every_steps = 1

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    chunk_paths, metadata = load_token_entries_from_directory(token_cache_dir)
    if not (0 <= int(level_index) < len(metadata.level_lengths)):
        raise ValueError(f"level-index must be in [0, {len(metadata.level_lengths) - 1}]")

    level_vocab_size = int(metadata.level_vocab_sizes[level_index])
    if vocab_size <= 0:
        vocab_size = level_vocab_size
    elif vocab_size != level_vocab_size:
        LOGGER.warning(
            "override vocab_size=%s -> metadata level vocab_size=%s",
            vocab_size,
            level_vocab_size,
        )
        vocab_size = level_vocab_size

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    base_model = SemanticTextVQVAE(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_semantic_tokens=num_semantic_tokens,
        semantic_sequence_length=semantic_sequence_length,
        pad_token_id=pad_token_id,
        semantic_pad_token_id=semantic_pad_token_id,
        max_position_embeddings=max_position_embeddings,
        use_turboquant_kv=use_turboquant_kv,
        turboquant_key_bits=turboquant_key_bits,
        turboquant_value_bits=turboquant_value_bits,
        turboquant_qjl_residual_scale=turboquant_qjl_residual_scale,
        gradient_checkpointing=gradient_checkpointing,
        use_rotary_embeddings=use_rotary_embeddings,
        use_unpadding=use_unpadding,
    ).to(dev)
    amp_dtype = torch.bfloat16
    amp_enabled = dev.type == "cuda"

    if use_torch_compile and hasattr(torch, "compile"):
        from src.core.optimization import setup_blackwell_autotune

        setup_blackwell_autotune(compile_mode=compile_mode)
        model = torch.compile(base_model, mode=compile_mode)
    else:
        model = base_model

    from src.core.optimization import build_cosine_warmup_scheduler_lambda, configure_weight_decay

    optim_groups = configure_weight_decay(model, weight_decay=weight_decay)
    if optimizer_type == "adamw8bit":
        optimizer = bnb.optim.AdamW8bit(optim_groups, lr=lr)
    elif optimizer_type == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor(
            optim_groups,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=False,
            relative_step=False,
        )
    else:
        is_cuda = dev.type == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, fused=is_cuda)

    if epochs > 0:
        # Оцениваем количество шагов
        seq_per_chunk = 11000
        total_seq = len(chunk_paths) * seq_per_chunk
        estimated_steps_per_epoch = int(total_seq / (batch_size * gradient_accumulation_steps))
        steps = max(1, estimated_steps_per_epoch * epochs)
        CONSOLE.print(f"[VQVAE] Epochs mode enabled ({epochs} epochs). Estimated ~{steps} total steps.")

    if scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=steps, pct_start=warmup_ratio
        )
    else:
        warmup_steps = int(steps * warmup_ratio)
        _lr_lambda = build_cosine_warmup_scheduler_lambda(
            max_steps=steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    ds = MultiscaleTokenChunkIterableDataset(
        chunk_paths=chunk_paths,
        metadata=metadata,
        validate_ranges=False,
    )
    pin_memory = dev.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=_collate_level(level_index),
        num_workers=max(0, int(dataloader_num_workers)),
        pin_memory=pin_memory,
        persistent_workers=bool(dataloader_num_workers > 0),
        prefetch_factor=int(dataloader_prefetch_factor) if dataloader_num_workers > 0 else None,
    )

    model.train()
    loss = None
    step = 0
    micro_step = 0
    ckpt_manager = CheckpointManager(output_path.parent, max_to_keep=max_checkpoints, prefix=output_path.stem)
    optimizer.zero_grad(set_to_none=True)

    tb_writer = None
    if tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)

    if resume_from is not None and resume_from.exists():
        CONSOLE.print(f"[VQVAE] Resuming training from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location="cpu")
        if "model" in ckpt:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
            base_model.load_state_dict(state_dict)
        step = ckpt.get("steps", 0)
        micro_step = step * gradient_accumulation_steps
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

    step_logger = TrainingStepLogger("vqvae", steps, initial_step=step)

    try:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=CONSOLE
        ) as progress:
            task_id = progress.add_task(f"Training VQ-VAE 0/{steps}", total=None)
            
            completed_epochs = 0
            while step < steps and (epochs == 0 or completed_epochs < epochs):
                if ckpt_manager.stop_requested:
                    CONSOLE.print(f"[VQVAE] Graceful stop requested at step {step}. Saving checkpoint...")
                    payload = {
                        "model": base_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "steps": step,
                    }
                    ckpt_manager.save(payload, step)
                    break
                did_progress = False
                for tokens, padding_mask in loader:
                    if ckpt_manager.stop_requested:
                        break
                    did_progress = True
                    step_start = time.perf_counter()
                    transfer_start = time.perf_counter()
                    tokens = tokens.to(dev, non_blocking=True)
                    padding_mask = padding_mask.to(dev, non_blocking=True)
                    transfer_time = time.perf_counter() - transfer_start
                    forward_start = time.perf_counter()
                    with torch.autocast(
                        device_type=dev.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                            logits, loss, loss_dict = model(tokens, padding_mask=padding_mask)
                    del logits
                    forward_time = time.perf_counter() - forward_start
                    scaled_loss = loss / gradient_accumulation_steps
                    backward_start = time.perf_counter()
                    scaled_loss.backward()
                    backward_time = time.perf_counter() - backward_start
                    micro_step += 1
                    if micro_step % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                        optimizer_start = time.perf_counter()
                        optimizer.step()
                        scheduler.step()
                        optimizer_time = time.perf_counter() - optimizer_start
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        if epochs > 0:
                            progress.update(task_id, description=f"Training VQ-VAE Epoch {completed_epochs+1}/{epochs} (Step {step}/{steps})")
                        else:
                            progress.update(task_id, description=f"Training VQ-VAE {step}/{steps}")
                        if step % log_every_steps == 0 or step == 1 or step == steps:
                            CONSOLE.print(
                                step_logger.build_line(
                                    step=step,
                                    loss=float(loss.detach().item()),
                                    loss_dict={k: float(v.item()) for k, v in loss_dict.items()},
                                    timing=StepTiming(
                                        total=time.perf_counter() - step_start,
                                        stages={
                                            "transfer": transfer_time,
                                            "forward": forward_time,
                                            "backward": backward_time,
                                            "optimizer": optimizer_time,
                                        },
                                    ),
                                )
                            )
                            if tb_writer is not None:
                                tb_writer.add_scalar("train/loss", float(loss.detach().item()), step)
                                for k, v in loss_dict.items():
                                    tb_writer.add_scalar(f"train/{k}", float(v.item()), step)
                                tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                        
                        if save_every > 0 and step % save_every == 0:
                            CONSOLE.print(f"[VQVAE] Auto-saving checkpoint at step {step}...")
                            payload = {
                                "model": base_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "steps": step,
                                "model_config": {
                                    "vocab_size": int(base_model.vocab_size),
                                    "hidden_size": int(base_model.hidden_size),
                                    "num_semantic_tokens": int(base_model.quantizer.codebook_size),
                                    "semantic_sequence_length": int(base_model.semantic_sequence_length),
                                    "pad_token_id": int(base_model.pad_token_id),
                                    "semantic_pad_token_id": int(base_model.semantic_pad_token_id),
                                    "use_turboquant_kv": bool(base_model.use_turboquant_kv),
                                    "turboquant_key_bits": int(base_model.turboquant_key_bits),
                                    "turboquant_value_bits": int(base_model.turboquant_value_bits),
                                    "turboquant_qjl_residual_scale": float(base_model.turboquant_qjl_residual_scale),
                                    "gradient_checkpointing": bool(base_model.gradient_checkpointing),
                                    "use_rotary_embeddings": bool(base_model.use_rotary_embeddings),
                                    "use_unpadding": bool(base_model.use_unpadding),
                                },
                                "metadata": TokenCacheMetadata(
                                    kind=metadata.kind,
                                    level_vocab_sizes=(vocab_size,),
                                    level_lengths=(metadata.level_lengths[level_index],),
                                    codebook_dim=metadata.codebook_dim,
                                    max_token_length=metadata.level_lengths[level_index],
                                ).to_dict(),
                                "source_token_cache": str(token_cache_dir),
                                "level_index": int(level_index),
                            }
                            ckpt_manager.save(payload, step)

                    if step >= steps:
                        break
                
                # 1 эпоха завершена.
                completed_epochs += 1
                CONSOLE.print(f"[VQVAE] Finished {completed_epochs} full epoch(s) over the dataset.")
                if epochs > 0 and completed_epochs >= epochs:
                    break

            if not did_progress:
                raise RuntimeError("No valid token entries were loaded from token cache.")

            if micro_step % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer_start = time.perf_counter()
                optimizer.step()
                scheduler.step()
                optimizer_time = time.perf_counter() - optimizer_start
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % log_every_steps == 0 or step == 1 or step == steps:
                    CONSOLE.print(
                        step_logger.build_line(
                            step=step,
                            loss=float(loss.detach().item()),
                            loss_dict={k: float(v.item()) for k, v in loss_dict.items()},
                            timing=StepTiming(
                                total=time.perf_counter() - step_start,
                                stages={
                                    "transfer": transfer_time,
                                    "forward": forward_time,
                                    "backward": backward_time,
                                    "optimizer": optimizer_time,
                                },
                            ),
                        )
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss", float(loss.detach().item()), step)
                        for k, v in loss_dict.items():
                            tb_writer.add_scalar(f"train/{k}", float(v.item()), step)
                        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
        payload = {
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "steps": step,
            "model_config": {
                "vocab_size": int(base_model.vocab_size),
                "hidden_size": int(base_model.hidden_size),
                "num_semantic_tokens": int(base_model.quantizer.codebook_size),
                "semantic_sequence_length": int(base_model.semantic_sequence_length),
                "pad_token_id": int(base_model.pad_token_id),
                "semantic_pad_token_id": int(base_model.semantic_pad_token_id),
                "use_turboquant_kv": bool(base_model.use_turboquant_kv),
                "turboquant_key_bits": int(base_model.turboquant_key_bits),
                "turboquant_value_bits": int(base_model.turboquant_value_bits),
                "turboquant_qjl_residual_scale": float(
                    base_model.turboquant_qjl_residual_scale
                ),
                "gradient_checkpointing": bool(base_model.gradient_checkpointing),
                "use_rotary_embeddings": bool(base_model.use_rotary_embeddings),
                "use_unpadding": bool(base_model.use_unpadding),
            },
            "metadata": TokenCacheMetadata(
                kind=metadata.kind,
                level_vocab_sizes=(vocab_size,),
                level_lengths=(metadata.level_lengths[level_index],),
                codebook_dim=metadata.codebook_dim,
                max_token_length=metadata.level_lengths[level_index],
            ).to_dict(),
            "source_token_cache": str(token_cache_dir),
            "level_index": int(level_index),
        }
        
        ckpt_manager.save(payload, step)
        CONSOLE.print(f"[VQVAE] Final checkpoint saved: {output_path}")
        if tb_writer is not None:
            tb_writer.close()
        return output_path
    finally:
        ckpt_manager.restore_handlers()


def main() -> None:
    """Run VQ-VAE training from CLI options."""
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        cfg = load_vqvae_train_config(Path(args.config))
    else:
        if not args.output or not args.token_cache_dir:
            parser.error(
                "--output and --token-cache-dir are required when --config is not provided"
            )
        cfg = VQVAETrainConfig(
            output=Path(args.output),
            token_cache_dir=Path(args.token_cache_dir),
            steps=args.steps,
            epochs=args.epochs,
            save_every=args.save_every,
            max_checkpoints=args.max_checkpoints,
            batch_size=args.batch_size,
            device=args.device,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_semantic_tokens=args.num_semantic_tokens,
            semantic_sequence_length=args.semantic_sequence_length,
            pad_token_id=args.pad_token_id,
            semantic_pad_token_id=args.semantic_pad_token_id,
            max_position_embeddings=args.max_position_embeddings,
            lr=args.lr,
            level_index=args.level_index,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            pin_memory=True,
            use_torch_compile=args.use_torch_compile,
            compile_mode="default",
            use_turboquant_kv=args.use_turboquant_kv,
            turboquant_key_bits=args.turboquant_key_bits,
            turboquant_value_bits=args.turboquant_value_bits,
            turboquant_qjl_residual_scale=args.turboquant_qjl_residual_scale,
            gradient_checkpointing=args.gradient_checkpointing,
            use_rotary_embeddings=not args.disable_rotary_embeddings,
            use_unpadding=args.use_unpadding,
            log_every_steps=args.log_every_steps,
            verbose=args.verbose,
            optimizer_type=args.optimizer,
            tensorboard_dir=args.tensorboard_dir,
            resume_from=Path(args.resume_from) if args.resume_from else None,
        )
    try:
        run_training(
            output_path=cfg.output,
            token_cache_dir=cfg.token_cache_dir,
            steps=cfg.steps,
            epochs=cfg.epochs,
            save_every=cfg.save_every,
            max_checkpoints=cfg.max_checkpoints,
            batch_size=cfg.batch_size,
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_semantic_tokens=cfg.num_semantic_tokens,
            semantic_sequence_length=cfg.semantic_sequence_length,
            pad_token_id=cfg.pad_token_id,
            semantic_pad_token_id=cfg.semantic_pad_token_id,
            max_position_embeddings=cfg.max_position_embeddings,
            lr=cfg.lr,
            device=cfg.device,
            level_index=cfg.level_index,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            dataloader_num_workers=cfg.dataloader_num_workers,
            dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,
            pin_memory=cfg.pin_memory,
            use_torch_compile=cfg.use_torch_compile,
            compile_mode=cfg.compile_mode,
            use_turboquant_kv=cfg.use_turboquant_kv,
            turboquant_key_bits=cfg.turboquant_key_bits,
            turboquant_value_bits=cfg.turboquant_value_bits,
            turboquant_qjl_residual_scale=cfg.turboquant_qjl_residual_scale,
            gradient_checkpointing=cfg.gradient_checkpointing,
            use_rotary_embeddings=cfg.use_rotary_embeddings,
            use_unpadding=cfg.use_unpadding,
            log_every_steps=cfg.log_every_steps,
            verbose=cfg.verbose,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            min_lr_ratio=cfg.min_lr_ratio,
            scheduler_type=cfg.scheduler_type,
            optimizer_type=cfg.optimizer_type,
            max_grad_norm=cfg.max_grad_norm,
            tensorboard_dir=cfg.tensorboard_dir,
            resume_from=cfg.resume_from,
        )
    except TrainingInterruptedError:
        CONSOLE.print("[yellow]Training interrupted gracefully.[/yellow]")


if __name__ == "__main__":
    main()
