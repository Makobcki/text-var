from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _row_l2_kernel(  # pragma: no cover
        x_ptr,
        out_ptr,
        stride_x0,
        stride_x1,
        rows,
        cols,
        BLOCK: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        row = pid
        if row >= rows:
            return
        offs = tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        running = tl.zeros((1,), dtype=tl.float32)
        col = 0
        while col < cols:
            idx = col + offs
            mask = idx < cols
            x = tl.load(x_ptr + row * stride_x0 + idx * stride_x1, mask=mask, other=0.0)
            acc = x * x
            running += tl.sum(acc, axis=0)
            col += BLOCK
        tl.store(out_ptr + row, running)

    @triton.jit
    def _bincount_kernel(  # pragma: no cover
        index_ptr,
        out_ptr,
        n_items,
        n_bins,
        BLOCK: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        start = pid * BLOCK
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_items
        idx = tl.load(index_ptr + offsets, mask=mask, other=0).to(tl.int32)
        in_range = idx < n_bins
        add_mask = mask & in_range
        tl.atomic_add(out_ptr + idx, 1.0, mask=add_mask)

    @triton.jit
    def _sum_by_index_kernel(  # pragma: no cover
        index_ptr,
        value_ptr,
        out_ptr,
        n_items,
        n_bins,
        n_cols,
        stride_v0,
        stride_v1,
        stride_o0,
        stride_o1,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ) -> None:
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        n_mask = offs_n < n_items
        d_mask = offs_d < n_cols

        idx = tl.load(index_ptr + offs_n, mask=n_mask, other=0).to(tl.int32)
        valid_idx = idx < n_bins
        values = tl.load(
            value_ptr + offs_n[:, None] * stride_v0 + offs_d[None, :] * stride_v1,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        for i in range(BLOCK_N):
            idx_i = idx[i]
            if idx_i >= 0:
                row_ptr = out_ptr + idx_i * stride_o0 + offs_d * stride_o1
                tl.atomic_add(row_ptr, values[i, :], mask=valid_idx[i] & d_mask)


def _compute_row_l2_triton(x: torch.Tensor) -> torch.Tensor:
    """Compute per-row squared L2 norms with Triton when available.

    Args:
        x: 2D tensor of shape ``(rows, cols)``.

    Returns:
        1D tensor with shape ``(rows,)`` containing row-wise squared norms.
    """
    if triton is None or tl is None or (not x.is_cuda):
        return x.pow(2).sum(dim=1)
    rows, cols = x.shape
    out = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block = 128
    _row_l2_kernel[(rows,)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        rows,
        cols,
        BLOCK=block,
    )
    return out.to(dtype=x.dtype)


def _bincount_triton(indices: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Compute bincount with Triton on CUDA.

    Args:
        indices: Integer vector of shape ``(N,)``.
        n_bins: Number of bins.

    Returns:
        Float32 bincount vector of shape ``(n_bins,)``.
    """
    if triton is None or tl is None or (not indices.is_cuda):
        return torch.bincount(indices, minlength=n_bins).to(dtype=torch.float32)
    out = torch.zeros((n_bins,), device=indices.device, dtype=torch.float32)
    block = 256
    n_items = int(indices.numel())
    grid = (triton.cdiv(n_items, block),)
    _bincount_kernel[grid](indices, out, n_items, int(n_bins), BLOCK=block)
    return out


def _sum_by_index_triton(indices: torch.Tensor, values: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Sum rows in ``values`` grouped by ``indices`` with Triton on CUDA.

    Args:
        indices: Integer vector of shape ``(N,)``.
        values: Value matrix of shape ``(N, D)``.
        n_bins: Number of output rows.

    Returns:
        Reduced tensor of shape ``(n_bins, D)``.
    """
    if triton is None or tl is None or (not values.is_cuda):
        out = torch.zeros((n_bins, values.shape[1]), device=values.device, dtype=values.dtype)
        out.index_add_(0, indices, values)
        return out
    n_items, n_cols = values.shape
    out = torch.zeros((n_bins, n_cols), device=values.device, dtype=values.dtype)
    block_n = 64
    block_d = 32
    grid = (triton.cdiv(n_items, block_n), triton.cdiv(n_cols, block_d))
    _sum_by_index_kernel[grid](
        indices,
        values,
        out,
        int(n_items),
        int(n_bins),
        int(n_cols),
        values.stride(0),
        values.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return out


def ema_update_torch(
    *,
    encoding_indices: torch.Tensor,
    flat_inputs: torch.Tensor,
    ema_cluster_size: torch.Tensor,
    ema_w: torch.Tensor,
    decay: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update EMA cluster statistics for vector quantization.

    Args:
        encoding_indices: Flattened assignment ids of shape ``(N,)``.
        flat_inputs: Flattened inputs of shape ``(N, D)``.
        ema_cluster_size: Running cluster counts of shape ``(K,)``.
        ema_w: Running weighted sums of shape ``(K, D)``.
        decay: EMA decay factor.
        epsilon: Numerical stability epsilon.

    Returns:
        Tuple with updated ``(ema_cluster_size, ema_w)`` tensors.
    """
    cluster_size = torch.bincount(encoding_indices, minlength=ema_cluster_size.shape[0]).to(
        dtype=flat_inputs.dtype
    )
    updated_cluster_size = ema_cluster_size * decay + cluster_size * (1.0 - decay)

    dw = torch.zeros_like(ema_w)
    dw.index_add_(0, encoding_indices, flat_inputs)
    updated_ema_w = ema_w * decay + dw * (1.0 - decay)

    n = updated_cluster_size.sum()
    normalized_cluster_size = (
        (updated_cluster_size + epsilon) / (n + updated_cluster_size.numel() * epsilon)
    ) * n
    return normalized_cluster_size, updated_ema_w


def ema_update_triton(
    *,
    encoding_indices: torch.Tensor,
    flat_inputs: torch.Tensor,
    ema_cluster_size: torch.Tensor,
    ema_w: torch.Tensor,
    decay: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hybrid Triton-backed EMA update path.

    Uses Triton for row-wise norm precompute on CUDA as a first migration step, while
    keeping numerically stable scatter/reduction logic in torch.
    """
    num_embeddings = int(ema_cluster_size.shape[0])
    cluster_size = _bincount_triton(encoding_indices, n_bins=num_embeddings).to(dtype=flat_inputs.dtype)
    updated_cluster_size = ema_cluster_size * decay + cluster_size * (1.0 - decay)
    dw = _sum_by_index_triton(encoding_indices, flat_inputs, n_bins=num_embeddings)
    updated_ema_w = ema_w * decay + dw * (1.0 - decay)
    n = updated_cluster_size.sum()
    normalized_cluster_size = (
        (updated_cluster_size + epsilon) / (n + num_embeddings * epsilon)
    ) * n
    return normalized_cluster_size, updated_ema_w
