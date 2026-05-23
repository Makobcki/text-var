"""Configuration model for VQ-VAE training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
        semantic_tokens: Number of semantic tokens.
        lr: Learning rate.
        level_index: Multiscale level index.
    """

    output: Path
    token_cache_dir: Path
    steps: int = 500
    batch_size: int = 8
    device: str = "cuda"
    vocab_size: int = 0
    hidden_size: int = 1024
    semantic_tokens: int = 4096
    lr: float = 3e-4
    level_index: int = 2
