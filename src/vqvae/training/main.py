import logging
import signal
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from torch.utils.data import DataLoader

from src.core.training_logger import StepTiming, TrainingStepLogger
from src.data.token_cache import (
    MultiscaleTokenChunkIterableDataset,
    TokenCacheMetadata,
    load_token_entries_from_directory,
)
from src.vqvae.checkpoint import save_vqvae_checkpoint
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


def _install_signal_handlers(stop_state: dict[str, bool]) -> dict[int, signal.Handlers]:
    """Install SIGINT/SIGTERM handlers that request graceful stop.

    Args:
        stop_state: Mutable stop flag container.

    Returns:
        Previous signal handlers.
    """

    def _handle_signal(signum: int, _frame: object) -> None:
        LOGGER.warning("Training stop requested by signal %s", signum)
        stop_state["requested"] = True

    previous = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    return previous


def _restore_signal_handlers(previous: dict[int, signal.Handlers]) -> None:
    """Restore previous signal handlers.

    Args:
        previous: Previous signal handlers.
    """
    for signum, handler in previous.items():
        signal.signal(signum, handler)


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
    amp_dtype_name: str = "bf16",
    use_torch_compile: bool = False,
    use_triton_ema: bool = False,
    use_turboquant_kv: bool = False,
    turboquant_key_bits: int = 4,
    turboquant_value_bits: int = 4,
    turboquant_qjl_residual_scale: float = 0.5,
    gradient_checkpointing: bool = False,
    use_rotary_embeddings: bool = True,
    log_every_steps: int = 10,
    verbose: bool = False,
) -> Path:
    _configure_logging(verbose)
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0.")

    if log_every_steps <= 0:
        CONSOLE.warning("log_every_steps must be greater than 0, setting to 1")
        log_every_steps = 1

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
        use_triton_ema=use_triton_ema,
        use_turboquant_kv=use_turboquant_kv,
        turboquant_key_bits=turboquant_key_bits,
        turboquant_value_bits=turboquant_value_bits,
        turboquant_qjl_residual_scale=turboquant_qjl_residual_scale,
        gradient_checkpointing=gradient_checkpointing,
        use_rotary_embeddings=use_rotary_embeddings,
    ).to(dev)
    amp_dtype_lookup: dict[str, torch.dtype | None] = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "none": None,
    }
    if amp_dtype_name not in amp_dtype_lookup:
        raise ValueError("amp_dtype_name must be one of: bf16, fp16, none.")
    amp_dtype = amp_dtype_lookup[amp_dtype_name]
    amp_enabled = dev.type == "cuda" and amp_dtype is not None

    if use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(base_model)
    else:
        model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype is torch.float16)

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
    step_logger = TrainingStepLogger("vqvae", steps)
    loss = None
    step = 0
    micro_step = 0
    stop_state: dict[str, bool] = {"requested": False}
    previous_handlers = _install_signal_handlers(stop_state)
    optimizer.zero_grad(set_to_none=True)
    try:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=CONSOLE
        ) as progress:
            task_id = progress.add_task(f"Training VQ-VAE 0/{steps}", total=None)
            while step < steps:
                if stop_state["requested"]:
                    raise TrainingInterruptedError("Training interrupted by signal.")
                did_progress = False
                for tokens, padding_mask in loader:
                    if stop_state["requested"]:
                        raise TrainingInterruptedError("Training interrupted by signal.")
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
                        _, loss = model(tokens, padding_mask=padding_mask)
                    forward_time = time.perf_counter() - forward_start
                    scaled_loss = loss / gradient_accumulation_steps
                    backward_start = time.perf_counter()
                    scaler.scale(scaled_loss).backward()
                    backward_time = time.perf_counter() - backward_start
                    micro_step += 1
                    if micro_step % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer_start = time.perf_counter()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_time = time.perf_counter() - optimizer_start
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        progress.update(task_id, description=f"Training VQ-VAE {step}/{steps}")
                        if step % log_every_steps == 0 or step == 1 or step == steps:
                            CONSOLE.print(
                                step_logger.build_line(
                                    step=step,
                                    loss=float(loss.detach().item()),
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
                    if step >= steps:
                        break

            if not did_progress:
                raise RuntimeError("No valid token entries were loaded from token cache.")

            if micro_step % gradient_accumulation_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_start = time.perf_counter()
                scaler.step(optimizer)
                scaler.update()
                optimizer_time = time.perf_counter() - optimizer_start
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % log_every_steps == 0 or step == 1 or step == steps:
                    CONSOLE.print(
                        step_logger.build_line(
                            step=step,
                            loss=float(loss.detach().item()),
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
        save_vqvae_checkpoint(
            {
                "model": model.state_dict(),
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
            },
            output_path,
        )
        CONSOLE.print(f"[VQVAE] checkpoint saved: {output_path}")
        return output_path
    finally:
        _restore_signal_handlers(previous_handlers)


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
            amp_dtype=args.amp_dtype,
            use_torch_compile=args.use_torch_compile,
            use_triton_ema=args.use_triton_ema,
            use_turboquant_kv=args.use_turboquant_kv,
            turboquant_key_bits=args.turboquant_key_bits,
            turboquant_value_bits=args.turboquant_value_bits,
            turboquant_qjl_residual_scale=args.turboquant_qjl_residual_scale,
            gradient_checkpointing=args.gradient_checkpointing,
            use_rotary_embeddings=not args.disable_rotary_embeddings,
            log_every_steps=args.log_every_steps,
            verbose=args.verbose,
        )
    try:
        run_training(
            cfg.output,
            cfg.token_cache_dir,
            steps=cfg.steps,
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
            amp_dtype_name=cfg.amp_dtype,
            use_torch_compile=cfg.use_torch_compile,
            use_triton_ema=cfg.use_triton_ema,
            use_turboquant_kv=cfg.use_turboquant_kv,
            turboquant_key_bits=cfg.turboquant_key_bits,
            turboquant_value_bits=cfg.turboquant_value_bits,
            turboquant_qjl_residual_scale=cfg.turboquant_qjl_residual_scale,
            gradient_checkpointing=cfg.gradient_checkpointing,
            use_rotary_embeddings=cfg.use_rotary_embeddings,
            log_every_steps=cfg.log_every_steps,
            verbose=cfg.verbose,
        )
    except TrainingInterruptedError:
        CONSOLE.print("[yellow]Training interrupted gracefully.[/yellow]")


if __name__ == "__main__":
    main()
