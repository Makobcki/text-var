"""VQ-VAE loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def total_vqvae_loss(
    reconstruction_loss: torch.Tensor, commitment_loss: torch.Tensor
) -> torch.Tensor:
    """Compute total VQ-VAE loss."""
    return reconstruction_loss + commitment_loss


def feature_matching_loss(
    pre_quant: torch.Tensor, post_quant: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute MSE loss between continuous representations before and after quantization/projection."""  # noqa: E501
    loss = F.mse_loss(pre_quant, post_quant, reduction="none")
    if mask is not None:
        valid_mask = (~mask).unsqueeze(-1).float()
        loss = loss * valid_mask
        return loss.sum() / (valid_mask.sum() * loss.size(-1) + 1e-8)
    return loss.mean()


def contrastive_latent_loss(
    latents: torch.Tensor, mask: torch.Tensor | None = None, temperature: float = 0.1
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss using dropout-based augmentation on pooled sequence latents."""  # noqa: E501
    if mask is not None:
        valid = (~mask).unsqueeze(-1).float()
        pooled = (latents * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-8)
    else:
        pooled = latents.mean(dim=1)

    z1 = F.dropout(pooled, p=0.1, training=True)
    z2 = F.dropout(pooled, p=0.1, training=True)

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    sim_matrix = torch.matmul(z1, z2.T) / temperature

    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z1.device)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def token_level_contrastive_loss(
    latents: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss at the individual token level."""
    batch_size, seq_len, dim = latents.shape

    # Flatten latents
    flat_latents = latents.view(-1, dim)
    if mask is not None:
        flat_mask = mask.view(-1)
        valid_indices = torch.where(~flat_mask)[0]
        valid_latents = flat_latents[valid_indices]
    else:
        valid_latents = flat_latents

    num_valid = valid_latents.size(0)
    if num_valid == 0:
        return torch.tensor(0.0, device=latents.device)

    # Subsample if too many valid tokens to prevent memory explosion
    if num_valid > max_tokens:
        perm = torch.randperm(num_valid, device=latents.device)[:max_tokens]
        valid_latents = valid_latents[perm]
        num_valid = max_tokens

    # Dropout augmentation for positive pairs
    z1 = F.dropout(valid_latents, p=0.1, training=True)
    z2 = F.dropout(valid_latents, p=0.1, training=True)

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(num_valid, device=latents.device)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def kl_divergence_loss(latents: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute KL-divergence penalty against a standard normal prior N(0, I)."""
    loss = 0.5 * latents.pow(2).mean(dim=-1)
    if mask is not None:
        valid_mask = (~mask).float()
        loss = loss * valid_mask
        return loss.sum() / (valid_mask.sum() + 1e-8)
    return loss.mean()
