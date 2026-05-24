"""TurboQuant attention wrappers with optional Triton acceleration."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Final

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


@dataclass(frozen=True)
class TurboQuantTritonInputs:
    """Container for quantized KV tensors passed to attention backend.

    Args:
        q: Query tensor in shape [B, H, T, D].
        k_quant: Quantized key tensor [B, H, S, D] in int8/uint8-compatible format.
        v_quant: Quantized value tensor [B, H, S, D] in int8/uint8-compatible format.
        k_scales: Per-token/per-head key scales.
        v_scales: Per-token/per-head value scales.
    """

    q: torch.Tensor
    k_quant: torch.Tensor
    v_quant: torch.Tensor
    k_scales: torch.Tensor
    v_scales: torch.Tensor
    k_qjl_signs: torch.Tensor | None = None
    v_qjl_signs: torch.Tensor | None = None
    bits: int = 8
    qjl_residual_scale: float = 0.5


class TurboQuantKernelError(RuntimeError):
    """Raised when TurboQuant Triton kernel cannot be executed safely."""


_SUPPORTED_HEAD_DIMS: Final[set[int]] = {16, 32, 64, 128}
_SUPPORTED_PACKED_BITS: Final[set[int]] = {4, 8}


if triton is not None:

    @triton.jit
    def _unpack_4bit(packed, idx):  # pragma: no cover
        byte = tl.load(packed + (idx // 2))
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        return tl.where((idx % 2) == 0, lo, hi)

    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=5),
    ]

    @triton.autotune(
        configs=_AUTOTUNE_CONFIGS,
        key=["seqlen_q", "seqlen_k", "causal", "PACKED_BITS"],
        prune_configs_by={
            "early_config_prune": lambda configs, named_args, **kwargs: [
                cfg
                for cfg in configs
                if cfg.kwargs["BLOCK_M"] <= max(16, int(named_args["seqlen_q"]))
            ],
        },
    )
    @triton.jit
    def _turboquant_fused_attention_kernel(  # pragma: no cover
        Q,
        K,
        V,
        K_SCALES,
        V_SCALES,
        K_QJL,
        V_QJL,
        O,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ksz,
        stride_ksh,
        stride_ksn,
        stride_ksd,
        stride_vsz,
        stride_vsh,
        stride_vsn,
        stride_vsd,
        stride_kqz,
        stride_kqh,
        stride_kqn,
        stride_kqd,
        stride_vqz,
        stride_vqh,
        stride_vqn,
        stride_vqd,
        stride_oz,
        stride_oh,
        stride_om,
        stride_od,
        seqlen_q,
        seqlen_k,
        nheads,
        causal,
        qjl_residual_scale,
        sm_scale,
        PACKED_BITS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_d = tl.arange(0, BLOCK_D)
        b = pid_bh // nheads
        h = pid_bh % nheads
        q_ptr = Q + b * stride_qz + h * stride_qh + off_m[:, None] * stride_qm + off_d[None, :] * stride_qd
        q = tl.load(q_ptr, mask=off_m[:, None] < seqlen_q, other=0.0)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start_n in range(0, tl.cdiv(seqlen_k, BLOCK_N)):
            token_idx = start_n * BLOCK_N + off_n
            if PACKED_BITS == 8:
                k_ptr = K + b * stride_kz + h * stride_kh + token_idx[:, None] * stride_kn + off_d[None, :] * stride_kd
                v_ptr = V + b * stride_vz + h * stride_vh + token_idx[:, None] * stride_vn + off_d[None, :] * stride_vd
                k_q = tl.load(k_ptr, mask=token_idx[:, None] < seqlen_k, other=0).to(tl.float16)
                v_q = tl.load(v_ptr, mask=token_idx[:, None] < seqlen_k, other=0).to(tl.float16)
            elif PACKED_BITS == 4:
                byte_idx = off_d[None, :] // 2
                nibble_sel = off_d[None, :] % 2
                k_ptr = K + b * stride_kz + h * stride_kh + token_idx[:, None] * stride_kn + byte_idx * stride_kd
                v_ptr = V + b * stride_vz + h * stride_vh + token_idx[:, None] * stride_vn + byte_idx * stride_vd
                k_byte = tl.load(k_ptr, mask=token_idx[:, None] < seqlen_k, other=0)
                v_byte = tl.load(v_ptr, mask=token_idx[:, None] < seqlen_k, other=0)
                k_lo = (k_byte & 0x0F).to(tl.float16)
                k_hi = ((k_byte >> 4) & 0x0F).to(tl.float16)
                v_lo = (v_byte & 0x0F).to(tl.float16)
                v_hi = ((v_byte >> 4) & 0x0F).to(tl.float16)
                k_q = tl.where(nibble_sel == 0, k_lo, k_hi)
                v_q = tl.where(nibble_sel == 0, v_lo, v_hi)
            else:
                k_q = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float16)
                v_q = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float16)

            ks_ptr = K_SCALES + b * stride_ksz + h * stride_ksh + token_idx[:, None] * stride_ksn + tl.zeros((1, BLOCK_D), dtype=tl.int32) * stride_ksd
            vs_ptr = V_SCALES + b * stride_vsz + h * stride_vsh + token_idx[:, None] * stride_vsn + tl.zeros((1, BLOCK_D), dtype=tl.int32) * stride_vsd
            ks = tl.load(ks_ptr, mask=token_idx[:, None] < seqlen_k, other=0.0)
            vs = tl.load(vs_ptr, mask=token_idx[:, None] < seqlen_k, other=0.0)

            k = k_q * ks
            v = v_q * vs

            kqjl_ptr = K_QJL + b * stride_kqz + h * stride_kqh + token_idx[:, None] * stride_kqn + off_d[None, :] * stride_kqd
            vqjl_ptr = V_QJL + b * stride_vqz + h * stride_vqh + token_idx[:, None] * stride_vqn + off_d[None, :] * stride_vqd
            kqjl = tl.load(kqjl_ptr, mask=token_idx[:, None] < seqlen_k, other=0)
            vqjl = tl.load(vqjl_ptr, mask=token_idx[:, None] < seqlen_k, other=0)
            ksign = tl.where(kqjl > 0, 1.0, -1.0)
            vsign = tl.where(vqjl > 0, 1.0, -1.0)
            k = k + ksign * ks * qjl_residual_scale
            v = v + vsign * vs * qjl_residual_scale
            logits = tl.dot(q, tl.trans(k)) * sm_scale
            if causal:
                q_pos = off_m[:, None]
                k_pos = start_n * BLOCK_N + off_n[None, :]
                logits = tl.where(q_pos >= k_pos, logits, float("-inf"))
            block_mask = (start_n * BLOCK_N + off_n[None, :]) < seqlen_k
            logits = tl.where(block_mask, logits, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(logits, axis=1))
            p = tl.exp(logits - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            acc = acc + tl.dot(p.to(v.dtype), v)
            m_i = m_ij
        l_i = tl.where(l_i > 0.0, l_i, 1.0)
        acc = acc / l_i[:, None]
        o_ptr = O + b * stride_oz + h * stride_oh + off_m[:, None] * stride_om + off_d[None, :] * stride_od
        tl.store(o_ptr, acc.to(tl.float16), mask=off_m[:, None] < seqlen_q)


def turboquant_attention(
    inputs: TurboQuantTritonInputs,
    *,
    is_causal: bool,
    fallback_k: torch.Tensor,
    fallback_v: torch.Tensor,
) -> torch.Tensor:
    """Compute attention with Triton when available, otherwise PyTorch fallback.

    Args:
        inputs: Quantized input bundle.
        is_causal: Whether to apply causal masking.
        fallback_k: Dequantized key fallback [B,H,S,D].
        fallback_v: Dequantized value fallback [B,H,S,D].

    Returns:
        Attention output tensor [B,H,T,D].
    """
    q = inputs.q.contiguous()
    _validate_inputs_for_kernel(inputs=inputs, fallback_k=fallback_k, fallback_v=fallback_v)
    k, v = _dequantize_kv(inputs=inputs, fallback_k=fallback_k, fallback_v=fallback_v)
    if triton is None or tl is None or (not q.is_cuda):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
    return _launch_turboquant_kernel(q=q, k=k, v=v, causal=is_causal, raw_inputs=inputs)


def _dequantize_kv(
    *,
    inputs: TurboQuantTritonInputs,
    fallback_k: torch.Tensor,
    fallback_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize K/V tensors if scales are provided.

    Args:
        inputs: TurboQuant inputs.
        fallback_k: Dense fallback key tensor.
        fallback_v: Dense fallback value tensor.

    Returns:
        Dequantized or fallback K/V tensors.
    """
    if inputs.k_scales.numel() == 0 or inputs.v_scales.numel() == 0:
        return fallback_k.contiguous(), fallback_v.contiguous()
    if inputs.bits not in _SUPPORTED_PACKED_BITS:
        raise TurboQuantKernelError(f"Unsupported quantization bits: {inputs.bits}.")
    if inputs.bits == 4 and (inputs.k_quant.shape[-1] * 2) == fallback_k.shape[-1]:
        kq = _unpack_4bit_tensor(inputs.k_quant, fallback_k.shape[-1])
        vq = _unpack_4bit_tensor(inputs.v_quant, fallback_v.shape[-1])
    else:
        kq = inputs.k_quant
        vq = inputs.v_quant
    k_scale = _reshape_scales(inputs.k_scales, inputs.k_quant)
    v_scale = _reshape_scales(inputs.v_scales, inputs.v_quant)
    k = kq.to(torch.float16) * k_scale
    v = vq.to(torch.float16) * v_scale
    if inputs.k_qjl_signs is not None:
        k = k + _qjl_residual(inputs.k_qjl_signs, k_scale)
    if inputs.v_qjl_signs is not None:
        v = v + _qjl_residual(inputs.v_qjl_signs, v_scale)
    return k.contiguous(), v.contiguous()


def _unpack_4bit_tensor(packed: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Unpack 4-bit tensor from trailing packed-byte dimension.

    Args:
        packed: Packed uint8 tensor with trailing byte dimension.
        output_dim: Target unpacked dimension.

    Returns:
        Unpacked uint8 tensor.
    """
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    stacked = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return stacked[..., :output_dim]


def _validate_inputs_for_kernel(
    *,
    inputs: TurboQuantTritonInputs,
    fallback_k: torch.Tensor,
    fallback_v: torch.Tensor,
) -> None:
    """Validate shape/bit assumptions for TurboQuant paths.

    Args:
        inputs: Quantized attention inputs.
        fallback_k: Dense fallback key tensor.
        fallback_v: Dense fallback value tensor.

    Raises:
        TurboQuantKernelError: If shapes/bitness are inconsistent.
    """
    if inputs.bits not in _SUPPORTED_PACKED_BITS:
        raise TurboQuantKernelError(f"Unsupported quantization bits: {inputs.bits}.")
    if inputs.q.ndim != 4 or fallback_k.ndim != 4 or fallback_v.ndim != 4:
        raise TurboQuantKernelError("Expected rank-4 q/fallback_k/fallback_v tensors.")
    if fallback_k.shape != fallback_v.shape:
        raise TurboQuantKernelError("fallback_k and fallback_v must have identical shapes.")
    if inputs.bits == 4 and inputs.k_scales.numel() > 0:
        expected_packed = (fallback_k.shape[-1] + 1) // 2
        if inputs.k_quant.shape[-1] != expected_packed or inputs.v_quant.shape[-1] != expected_packed:
            raise TurboQuantKernelError(
                "Packed 4-bit tensors must have trailing dim ceil(head_dim / 2)."
            )


def _qjl_residual(signs: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode QJL residual signs to additive residual correction.

    Args:
        signs: Boolean sign tensor.
        scales: Dequantization scale tensor.

    Returns:
        Residual correction tensor.
    """
    signed = torch.where(signs, 1.0, -1.0).to(scales.dtype)
    return signed * scales * 0.5


def _reshape_scales(scales: torch.Tensor, q_tensor: torch.Tensor) -> torch.Tensor:
    """Reshape scales into [B, H, S, 1] for broadcast multiply."""
    if scales.ndim == 4:
        return scales.to(torch.float16)
    if scales.ndim != 3:
        raise TurboQuantKernelError(f"Expected scales ndim 3 or 4, got {scales.ndim}.")
    return scales.unsqueeze(-1).to(torch.float16)


def _launch_turboquant_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    raw_inputs: TurboQuantTritonInputs,
) -> torch.Tensor:
    """Launch Triton attention kernel for causal or non-causal path.

    Args:
        q: Query tensor [B,H,T,D].
        k: Key tensor [B,H,S,D].
        v: Value tensor [B,H,S,D].
        causal: Whether to apply causal masking in kernel.

    Returns:
        Attention output [B,H,T,D].

    Raises:
        TurboQuantKernelError: If unsupported tensor layout is provided.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise TurboQuantKernelError("Expected q/k/v to be rank-4 tensors.")
    bsz, heads, seqlen_q, head_dim = q.shape
    if raw_inputs.k_scales.numel() == 0 or raw_inputs.v_scales.numel() == 0:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
    if raw_inputs.bits not in _SUPPORTED_PACKED_BITS:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
    out = torch.empty_like(q)
    block_m = 16 if seqlen_q <= 1 else 64
    block_n = 64
    grid = (triton.cdiv(seqlen_q, block_m), bsz * heads)
    k_scales = _reshape_scales(raw_inputs.k_scales, raw_inputs.k_quant) if raw_inputs.k_scales.numel() > 0 else torch.ones((*k.shape[:-1], 1), device=k.device, dtype=k.dtype)
    v_scales = _reshape_scales(raw_inputs.v_scales, raw_inputs.v_quant) if raw_inputs.v_scales.numel() > 0 else torch.ones((*v.shape[:-1], 1), device=v.device, dtype=v.dtype)
    k_qjl = raw_inputs.k_qjl_signs.to(torch.int8) if raw_inputs.k_qjl_signs is not None else torch.ones_like(k, dtype=torch.int8)
    v_qjl = raw_inputs.v_qjl_signs.to(torch.int8) if raw_inputs.v_qjl_signs is not None else torch.ones_like(v, dtype=torch.int8)

    _turboquant_fused_attention_kernel[grid](
        q,
        raw_inputs.k_quant,
        raw_inputs.v_quant,
        k_scales,
        v_scales,
        k_qjl,
        v_qjl,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        k_scales.stride(0),
        k_scales.stride(1),
        k_scales.stride(2),
        k_scales.stride(3),
        v_scales.stride(0),
        v_scales.stride(1),
        v_scales.stride(2),
        v_scales.stride(3),
        k_qjl.stride(0),
        k_qjl.stride(1),
        k_qjl.stride(2),
        k_qjl.stride(3),
        v_qjl.stride(0),
        v_qjl.stride(1),
        v_qjl.stride(2),
        v_qjl.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        seqlen_q,
        k.shape[2],
        heads,
        causal,
        raw_inputs.qjl_residual_scale,
        1.0 / (head_dim**0.5),
        PACKED_BITS=raw_inputs.bits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=head_dim,
    )
    return out


def turboquant_attention_causal(inputs: TurboQuantTritonInputs, *, fallback_k: torch.Tensor, fallback_v: torch.Tensor) -> torch.Tensor:
    """Explicit causal attention entrypoint with fallback logic.

    Args:
        inputs: Quantized inputs bundle.
        fallback_k: Dequantized key fallback.
        fallback_v: Dequantized value fallback.

    Returns:
        Causal attention output.
    """
    q = inputs.q.contiguous()
    k, v = _dequantize_kv(inputs=inputs, fallback_k=fallback_k, fallback_v=fallback_v)
    if triton is None or tl is None or (not q.is_cuda):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
    return _launch_turboquant_kernel(q=q, k=k, v=v, causal=True, raw_inputs=inputs)
