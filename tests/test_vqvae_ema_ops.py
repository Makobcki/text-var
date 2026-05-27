import torch
from src.vqvae.ema_ops import _compute_row_l2_triton, ema_update_torch, ema_update_triton


def test_ema_update_triton_matches_torch_placeholder() -> None:
    encoding_indices = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    flat_inputs = torch.randn(4, 3)
    ema_cluster_size = torch.zeros(3)
    ema_w = torch.zeros(3, 3)

    torch_cluster, torch_w = ema_update_torch(
        encoding_indices=encoding_indices,
        flat_inputs=flat_inputs,
        ema_cluster_size=ema_cluster_size,
        ema_w=ema_w,
        decay=0.99,
        epsilon=1e-5,
    )
    triton_cluster, triton_w = ema_update_triton(
        encoding_indices=encoding_indices,
        flat_inputs=flat_inputs,
        ema_cluster_size=ema_cluster_size,
        ema_w=ema_w,
        decay=0.99,
        epsilon=1e-5,
    )

    assert torch.allclose(torch_cluster, triton_cluster)
    assert torch.allclose(torch_w, triton_w)


def test_compute_row_l2_triton_matches_torch_fallback() -> None:
    x = torch.randn(5, 7)
    triton_norm = _compute_row_l2_triton(x)
    torch_norm = x.pow(2).sum(dim=1)
    assert torch.allclose(triton_norm, torch_norm)
