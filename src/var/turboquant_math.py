"""Core TurboQuant math helpers (PolarQuant + QJL simulation)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TurboQuantCompressed:
    """Compressed representation for one KV tensor block.

    Args:
        x_q: Base quantized tensor.
        scale: Per-token scale.
        res_sign: 1-bit residual sign map.
        res_scale: Residual scale.
    """

    x_q: torch.Tensor
    scale: torch.Tensor
    res_sign: torch.Tensor
    res_scale: torch.Tensor


def generate_orthogonal_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Generate Haar-like random orthogonal matrix.

    Args:
        dim: Square matrix dimension.
        device: Target device.

    Returns:
        Orthogonal matrix [dim, dim].
    """
    random_matrix = torch.randn((dim, dim), device=device)
    q, r = torch.linalg.qr(random_matrix)
    phase = torch.sign(torch.diag(r))
    return q * phase


def turboquant_compress(x: torch.Tensor, rotation: torch.Tensor, bits: int = 4) -> TurboQuantCompressed:
    """Compress tensor with PolarQuant rotation and 1-bit QJL residual.

    Args:
        x: Input tensor [B, L, H, D].
        rotation: Orthogonal matrix [D, D].
        bits: Quantization bits.

    Returns:
        Compressed tensor bundle.
    """
    x_rot = torch.einsum("blhd,df->blhf", x, rotation.transpose(0, 1))
    abs_max = x_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    q_max = float((2 ** (bits - 1)) - 1)
    scale = abs_max / q_max
    x_q = torch.round(x_rot / scale).clamp(-q_max, q_max).to(torch.int8)
    x_deq = x_q.to(x.dtype) * scale
    residual = x_rot - x_deq
    res_scale = residual.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
    res_sign = residual >= 0
    return TurboQuantCompressed(x_q=x_q, scale=scale, res_sign=res_sign, res_scale=res_scale)


def turboquant_decompress(compressed: TurboQuantCompressed, dtype: torch.dtype) -> torch.Tensor:
    """Decompress TurboQuant representation in rotated space.

    Args:
        compressed: Compressed bundle.
        dtype: Output dtype.

    Returns:
        Rotated-space approximation [B, L, H, D].
    """
    base = compressed.x_q.to(dtype) * compressed.scale.to(dtype)
    signs = torch.where(compressed.res_sign, 1.0, -1.0).to(dtype)
    return base + signs * compressed.res_scale.to(dtype)

