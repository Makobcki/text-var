"""VQ-VAE loss helpers."""

from __future__ import annotations

import torch


def total_vqvae_loss(reconstruction_loss: torch.Tensor, commitment_loss: torch.Tensor) -> torch.Tensor:
    """Compute total VQ-VAE loss.

    Args:
        reconstruction_loss: Reconstruction term.
        commitment_loss: Commitment/quantization term.

    Returns:
        Combined scalar loss tensor.
    """

    return reconstruction_loss + commitment_loss
