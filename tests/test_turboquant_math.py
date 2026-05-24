"""Tests for TurboQuant math helpers."""

import pytest
from src.var.turboquant_math import generate_orthogonal_matrix, turboquant_compress, turboquant_decompress


torch = pytest.importorskip("torch")


def test_generate_orthogonal_matrix_shape() -> None:
    """Orthogonal helper returns square matrix of requested dimension."""
    mat = generate_orthogonal_matrix(8, torch.device("cpu"))
    assert tuple(mat.shape) == (8, 8)


def test_turboquant_roundtrip_shapes() -> None:
    """Compress/decompress preserves tensor shape."""
    x = torch.randn((2, 3, 1, 8), dtype=torch.float32)
    rotation = generate_orthogonal_matrix(8, torch.device("cpu"))
    compressed = turboquant_compress(x, rotation, bits=4)
    out = turboquant_decompress(compressed, dtype=torch.float32)
    assert tuple(out.shape) == tuple(x.shape)

