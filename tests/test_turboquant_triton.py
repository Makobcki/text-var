"""Tests for TurboQuant Triton wrapper fallback behavior."""

import pytest

torch = pytest.importorskip("torch")

from src.var.turboquant_triton import (
    TurboQuantKernelError,
    TurboQuantTritonInputs,
    _reshape_scales,
    turboquant_attention_causal,
    turboquant_attention,
    _unpack_4bit_tensor,
    _validate_inputs_for_kernel,
)


def test_reshape_scales_accepts_rank_three() -> None:
    """Scale tensors with rank-3 are expanded for broadcast."""
    scales = torch.ones((2, 3, 4), dtype=torch.float32)
    q = torch.ones((2, 3, 4, 8), dtype=torch.int8)
    reshaped = _reshape_scales(scales, q)
    assert tuple(reshaped.shape) == (2, 3, 4, 1)


def test_reshape_scales_rejects_invalid_rank() -> None:
    """Invalid scale rank raises explicit kernel error."""
    scales = torch.ones((2, 3), dtype=torch.float32)
    q = torch.ones((2, 3, 4, 8), dtype=torch.int8)
    with pytest.raises(TurboQuantKernelError):
        _reshape_scales(scales, q)


def test_triton_inputs_dataclass_stores_tensors() -> None:
    """Input dataclass keeps all tensor references."""
    q = torch.ones((1, 1, 1, 8), dtype=torch.float16)
    bundle = TurboQuantTritonInputs(q=q, k_quant=q.to(torch.int8), v_quant=q.to(torch.int8), k_scales=torch.ones((1, 1, 1)), v_scales=torch.ones((1, 1, 1)))
    assert bundle.q.shape[-1] == 8


def test_turboquant_attention_causal_fallback_shape() -> None:
    """Causal wrapper returns SDPA-compatible output shape on CPU fallback."""
    q = torch.randn((1, 1, 2, 8), dtype=torch.float16)
    k = torch.randn((1, 1, 2, 8), dtype=torch.float16)
    v = torch.randn((1, 1, 2, 8), dtype=torch.float16)
    bundle = TurboQuantTritonInputs(q=q, k_quant=k.to(torch.int8), v_quant=v.to(torch.int8), k_scales=torch.empty(0), v_scales=torch.empty(0))
    out = turboquant_attention_causal(bundle, fallback_k=k, fallback_v=v)
    assert tuple(out.shape) == (1, 1, 2, 8)


def test_turboquant_attention_noncausal_fallback_shape() -> None:
    """Non-causal wrapper returns SDPA-compatible output shape on CPU fallback."""
    q = torch.randn((1, 1, 2, 8), dtype=torch.float16)
    k = torch.randn((1, 1, 2, 8), dtype=torch.float16)
    v = torch.randn((1, 1, 2, 8), dtype=torch.float16)
    bundle = TurboQuantTritonInputs(
        q=q,
        k_quant=k.to(torch.int8),
        v_quant=v.to(torch.int8),
        k_scales=torch.empty(0),
        v_scales=torch.empty(0),
    )
    out = turboquant_attention(bundle, is_causal=False, fallback_k=k, fallback_v=v)
    assert tuple(out.shape) == (1, 1, 2, 8)


def test_unpack_4bit_tensor_expands_last_dim() -> None:
    """Packed nibble tensor is expanded into full dimension."""
    packed = torch.tensor([[[[0x21, 0x43]]]], dtype=torch.uint8)
    unpacked = _unpack_4bit_tensor(packed, output_dim=4)
    assert unpacked.tolist() == [[[[1, 2, 3, 4]]]]


def test_triton_inputs_include_qjl_and_bits_fields() -> None:
    """Input dataclass stores QJL metadata for kernel path."""
    q = torch.ones((1, 1, 1, 8), dtype=torch.float16)
    signs = torch.ones((1, 1, 1, 8), dtype=torch.bool)
    bundle = TurboQuantTritonInputs(
        q=q,
        k_quant=q.to(torch.uint8),
        v_quant=q.to(torch.uint8),
        k_scales=torch.ones((1, 1, 1, 1), dtype=torch.float16),
        v_scales=torch.ones((1, 1, 1, 1), dtype=torch.float16),
        k_qjl_signs=signs,
        v_qjl_signs=signs,
        key_bits=4,
        value_bits=4,
    )
    assert bundle.key_bits == 4
    assert bundle.value_bits == 4
    assert bundle.k_qjl_signs is not None


def test_validate_inputs_rejects_bad_packed_width() -> None:
    """Packed 4-bit input width must match ceil(head_dim/2)."""
    q = torch.ones((1, 1, 2, 8), dtype=torch.float16)
    bad_packed = torch.ones((1, 1, 2, 3), dtype=torch.uint8)
    bundle = TurboQuantTritonInputs(
        q=q,
        k_quant=bad_packed,
        v_quant=bad_packed,
        k_scales=torch.ones((1, 1, 2, 1), dtype=torch.float16),
        v_scales=torch.ones((1, 1, 2, 1), dtype=torch.float16),
        key_bits=4,
        value_bits=4,
    )
    with pytest.raises(TurboQuantKernelError):
        _validate_inputs_for_kernel(inputs=bundle, fallback_k=q, fallback_v=q)


def test_validate_inputs_rejects_unsupported_bits() -> None:
    """Unsupported packed bits are rejected early."""
    q = torch.ones((1, 1, 2, 8), dtype=torch.float16)
    bundle = TurboQuantTritonInputs(
        q=q,
        k_quant=q.to(torch.uint8),
        v_quant=q.to(torch.uint8),
        k_scales=torch.ones((1, 1, 2, 1), dtype=torch.float16),
        v_scales=torch.ones((1, 1, 2, 1), dtype=torch.float16),
        key_bits=3,
        value_bits=3,
    )
    with pytest.raises(TurboQuantKernelError):
        _validate_inputs_for_kernel(inputs=bundle, fallback_k=q, fallback_v=q)


def test_validate_inputs_rejects_asymmetric_bits() -> None:
    """Asymmetric K/V bit widths are rejected for fused Triton kernel path."""
    q = torch.ones((1, 1, 2, 8), dtype=torch.float16)
    packed_4bit = torch.ones((1, 1, 2, 4), dtype=torch.uint8)
    bundle = TurboQuantTritonInputs(
        q=q,
        k_quant=packed_4bit,
        v_quant=q.to(torch.uint8),
        k_scales=torch.ones((1, 1, 2, 1), dtype=torch.float16),
        v_scales=torch.ones((1, 1, 2, 1), dtype=torch.float16),
        key_bits=4,
        value_bits=8,
    )
    with pytest.raises(TurboQuantKernelError):
        _validate_inputs_for_kernel(inputs=bundle, fallback_k=q, fallback_v=q)
