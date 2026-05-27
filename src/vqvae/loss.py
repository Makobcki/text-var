"""VQ-VAE loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def total_vqvae_loss(reconstruction_loss: torch.Tensor, commitment_loss: torch.Tensor) -> torch.Tensor:
    """Compute total VQ-VAE loss.

    Args:
        reconstruction_loss: Reconstruction term.
        commitment_loss: Commitment/quantization term.

    Returns:
        Combined scalar loss tensor.
    """
    return reconstruction_loss + commitment_loss


def feature_matching_loss(pre_quant: torch.Tensor, post_quant: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute MSE loss between continuous representations before and after quantization/projection.
    
    This enforces that the deep continuous features just before quantization closely 
    resemble the decoded features just after the upsampling block.
    """
    loss = F.mse_loss(pre_quant, post_quant, reduction="none")
    if mask is not None:
        valid_mask = (~mask).unsqueeze(-1).float()
        loss = loss * valid_mask
        return loss.sum() / (valid_mask.sum() + 1e-8)
    return loss.mean()


def contrastive_latent_loss(latents: torch.Tensor, mask: torch.Tensor | None = None, temperature: float = 0.1) -> torch.Tensor:
    """Compute InfoNCE contrastive loss using dropout-based augmentation on pooled sequence latents.
    
    Ensures diverse and well-separated representations across the batch.
    """
    # Pool latents over time (mean pooling)
    if mask is not None:
        valid = (~mask).unsqueeze(-1).float()
        pooled = (latents * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)
    else:
        pooled = latents.mean(dim=1)
    
    # Create two views via dropout
    z1 = F.dropout(pooled, p=0.1, training=True)
    z2 = F.dropout(pooled, p=0.1, training=True)
    
    # Normalize
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    
    # Targets are the diagonal elements (positive pairs)
    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z1.device)
    
    # Cross entropy over the similarity matrix
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
