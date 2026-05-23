"""Checkpoint utilities for VQ-VAE training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_vqvae_checkpoint(payload: dict[str, Any], output_path: Path) -> None:
    """Save VQ-VAE checkpoint payload.

    Args:
        payload: Serialized checkpoint data.
        output_path: Destination checkpoint path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
