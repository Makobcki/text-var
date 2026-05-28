"""Configuration model and loading for VQ-VAE training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VQVAEConfigError(ValueError):
    """Raised when VQ-VAE config payload is invalid."""


@dataclass(frozen=True)
class VQVAETrainConfig:
    """Typed VQ-VAE training configuration.

    Args:
        output: Output checkpoint path.
        token_cache_dir: Directory with token cache chunks.
        steps: Number of training steps.
        batch_size: Batch size.
        device: Device string.
        vocab_size: Vocabulary size (0 means infer from metadata).
        hidden_size: Hidden size.
        num_semantic_tokens: Number of semantic tokens.
        semantic_sequence_length: Number of semantic positions produced by encoder pooling.
        pad_token_id: Padding token id used for masking.
        max_position_embeddings: Maximum positional embedding length.
        lr: Learning rate.
        level_index: Multiscale level index.
        gradient_accumulation_steps: Number of micro-steps to accumulate before optimizer step.
    """

    output: Path
    token_cache_dir: Path
    steps: int = 500
    max_checkpoints: int = 3
    batch_size: int = 8
    device: str = "cuda"
    vocab_size: int = 0
    hidden_size: int = 1024
    num_semantic_tokens: int = 4096
    semantic_sequence_length: int = 1
    pad_token_id: int = 0
    max_position_embeddings: int = 2048
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    scheduler_type: str = "cosine"
    optimizer_type: str = "adamw"
    max_grad_norm: float = 1.0
    level_index: int = 2
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 2
    pin_memory: bool = True
    use_torch_compile: bool = False
    compile_mode: str = "default"
    log_every_steps: int = 10
    verbose: bool = False
    semantic_pad_token_id: int = 0
    use_turboquant_kv: bool = False
    turboquant_key_bits: int = 4
    turboquant_value_bits: int = 4
    turboquant_qjl_residual_scale: float = 0.5
    gradient_checkpointing: bool = False
    use_unpadding: bool = False
    use_rotary_embeddings: bool = True
    tensorboard_dir: str = "runs/vqvae"
    resume_from: Path | None = None


def _require_str(data: dict[str, Any], key: str) -> str:
    """Return required string field from payload.

    Args:
        data: Parsed JSON payload.
        key: Key to fetch.

    Returns:
        String value.

    Raises:
        VQVAEConfigError: If key is missing or not a string.
    """

    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VQVAEConfigError(f"Config field '{key}' must be a non-empty string.")
    return value


def load_vqvae_train_config(path: Path) -> VQVAETrainConfig:
    """Load VQ-VAE training config from JSON file.

    Args:
        path: Path to JSON config file.

    Returns:
        Parsed train config.

    Raises:
        VQVAEConfigError: If config is malformed.
        OSError: If file cannot be read.
        json.JSONDecodeError: If JSON is invalid.
    """

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise VQVAEConfigError("Top-level VQ-VAE config payload must be an object.")

    return VQVAETrainConfig(
        output=Path(_require_str(data, "output")),
        token_cache_dir=Path(_require_str(data, "token_cache_dir")),
        steps=int(data.get("steps", 500)),
        max_checkpoints=int(data.get("max_checkpoints", 3)),
        batch_size=int(data.get("batch_size", 8)),
        device=str(data.get("device", "cuda")),
        vocab_size=int(data.get("vocab_size", 0)),
        hidden_size=int(data.get("hidden_size", 1024)),
        num_semantic_tokens=int(data.get("num_semantic_tokens", data.get("semantic_tokens", 4096))),
        semantic_sequence_length=int(data.get("semantic_sequence_length", 1)),
        pad_token_id=int(data.get("pad_token_id", 0)),
        max_position_embeddings=int(data.get("max_position_embeddings", 2048)),
        lr=float(data.get("lr", 3e-4)),
        weight_decay=float(data.get("weight_decay", 0.05)),
        warmup_ratio=float(data.get("warmup_ratio", 0.05)),
        min_lr_ratio=float(data.get("min_lr_ratio", 0.1)),
        scheduler_type=str(data.get("scheduler_type", "cosine")),
        optimizer_type=str(data.get("optimizer", data.get("optimizer_type", "adamw"))),
        max_grad_norm=float(data.get("max_grad_norm", 1.0)),
        level_index=int(data.get("level_index", 2)),
        gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 1)),
        dataloader_num_workers=int(data.get("dataloader_num_workers", 4)),
        dataloader_prefetch_factor=int(data.get("dataloader_prefetch_factor", 2)),
        pin_memory=bool(data.get("pin_memory", True)),
        use_torch_compile=bool(data.get("use_torch_compile", False)),
        compile_mode=str(data.get("compile_mode", "default")),
        log_every_steps=int(data.get("log_every_steps", 10)),
        verbose=bool(data.get("verbose", False)),
        semantic_pad_token_id=int(data.get("semantic_pad_token_id", 0)),
        use_turboquant_kv=bool(data.get("use_turboquant_kv", False)),
        turboquant_key_bits=int(data.get("turboquant_key_bits", 4)),
        turboquant_value_bits=int(data.get("turboquant_value_bits", 4)),
        turboquant_qjl_residual_scale=float(data.get("turboquant_qjl_residual_scale", 0.5)),
        gradient_checkpointing=bool(data.get("gradient_checkpointing", False)),
        use_unpadding=bool(data.get("use_unpadding", False)),
        use_rotary_embeddings=bool(data.get("use_rotary_embeddings", True)),
        tensorboard_dir=str(data.get("tensorboard_dir", "runs/vqvae")),
        resume_from=Path(data["resume_from"]) if data.get("resume_from") else None,
    )
