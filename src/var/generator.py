from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F

from src.var.model import RingKVCacheView, VARTransformer


class RollbackEvent(RuntimeError):
    def __init__(self, block_start: int, block_end: int) -> None:
        super().__init__(f"Rollback requested for block [{block_start}, {block_end})")
        self.block_start = block_start
        self.block_end = block_end


@dataclass(frozen=True)
class TurboQuantConfig:
    """Configuration for TurboQuant-inspired KV compression."""

    key_bits: int = 4
    value_bits: int = 4
    qjl_residual_scale: float = 0.5


class TurboQuantCodec:
    """Quantize/dequantize KV tensors with random orthogonal rotation."""

    def __init__(self, cfg: TurboQuantConfig, head_dim: int, device: torch.device) -> None:
        self.cfg = cfg
        self.head_dim = head_dim
        self.rotation = self._make_rotation(head_dim=head_dim, device=device)
        self.inv_rotation = self.rotation.transpose(0, 1)
        self._key_scales: list[torch.Tensor] = []
        self._key_zeros: list[torch.Tensor] = []
        self._value_scales: list[torch.Tensor] = []
        self._value_zeros: list[torch.Tensor] = []
        self._key_residual_signs: list[torch.Tensor] = []
        self._value_residual_signs: list[torch.Tensor] = []

    def init_layer_state(self) -> None:
        self._key_scales.append(torch.empty(0))
        self._key_zeros.append(torch.empty(0))
        self._value_scales.append(torch.empty(0))
        self._value_zeros.append(torch.empty(0))
        self._key_residual_signs.append(torch.empty(0, dtype=torch.bool))
        self._value_residual_signs.append(torch.empty(0, dtype=torch.bool))

    def allocate_layer(self, layer_idx: int, b: int, w: int, h: int, device: torch.device) -> None:
        base_shape = (b, w, h, 1)
        resid_shape = (b, w, h, self.head_dim)
        self._key_scales[layer_idx] = torch.empty(base_shape, dtype=torch.float32, device=device)
        self._key_zeros[layer_idx] = torch.empty(base_shape, dtype=torch.float32, device=device)
        self._value_scales[layer_idx] = torch.empty(base_shape, dtype=torch.float32, device=device)
        self._value_zeros[layer_idx] = torch.empty(base_shape, dtype=torch.float32, device=device)
        self._key_residual_signs[layer_idx] = torch.empty(resid_shape, dtype=torch.bool, device=device)
        self._value_residual_signs[layer_idx] = torch.empty(resid_shape, dtype=torch.bool, device=device)

    def packed_dim(self, bits: int) -> int:
        """Return packed byte width for one head vector."""
        return (self.head_dim * bits + 7) // 8

    def quantize_step(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, write_ptr: int) -> tuple[torch.Tensor, torch.Tensor]:
        qk, ks, kz, kr = self._quantize_tensor(key, bits=self.cfg.key_bits)
        qv, vs, vz, vr = self._quantize_tensor(value, bits=self.cfg.value_bits)
        sl = slice(write_ptr, write_ptr + 1)
        self._key_scales[layer_idx][:, sl] = ks
        self._key_zeros[layer_idx][:, sl] = kz
        self._value_scales[layer_idx][:, sl] = vs
        self._value_zeros[layer_idx][:, sl] = vz
        self._key_residual_signs[layer_idx][:, sl] = kr
        self._value_residual_signs[layer_idx][:, sl] = vr
        return self._pack_bits(qk, self.cfg.key_bits), self._pack_bits(qv, self.cfg.value_bits)

    def dequantize_view(self, view: RingKVCacheView) -> tuple[torch.Tensor, torch.Tensor]:
        if view.layer_idx < 0 or view.layer_idx >= len(self._key_scales):
            raise ValueError(f"Invalid layer_idx for RingKVCacheView: {view.layer_idx}")
        layer_idx = view.layer_idx
        positions = view.positions
        key = self._dequantize_tensor(
            self._unpack_bits(view.keys.index_select(1, positions), self.cfg.key_bits).to(torch.float32),
            self._key_scales[layer_idx].index_select(1, positions),
            self._key_zeros[layer_idx].index_select(1, positions),
            self._key_residual_signs[layer_idx].index_select(1, positions),
        )
        value = self._dequantize_tensor(
            self._unpack_bits(view.values.index_select(1, positions), self.cfg.value_bits).to(torch.float32),
            self._value_scales[layer_idx].index_select(1, positions),
            self._value_zeros[layer_idx].index_select(1, positions),
            self._value_residual_signs[layer_idx].index_select(1, positions),
        )
        return key.to(dtype=torch.float16), value.to(dtype=torch.float16)

    def quantized_view_payload(
        self,
        view: RingKVCacheView,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ordered packed KV and metadata without dense materialization."""
        if view.layer_idx < 0 or view.layer_idx >= len(self._key_scales):
            raise ValueError(f"Invalid layer_idx for RingKVCacheView: {view.layer_idx}")
        layer_idx = view.layer_idx
        positions = view.positions
        if view.current_length > 0 and view.current_length < view.keys.shape[1]:
            sl = slice(0, int(view.current_length))
            return (
                view.keys[:, sl, :, :],
                view.values[:, sl, :, :],
                self._key_scales[layer_idx][:, sl, :, :],
                self._value_scales[layer_idx][:, sl, :, :],
                self._key_residual_signs[layer_idx][:, sl, :, :],
                self._value_residual_signs[layer_idx][:, sl, :, :],
            )
        return (
            view.keys.index_select(1, positions),
            view.values.index_select(1, positions),
            self._key_scales[layer_idx].index_select(1, positions),
            self._value_scales[layer_idx].index_select(1, positions),
            self._key_residual_signs[layer_idx].index_select(1, positions),
            self._value_residual_signs[layer_idx].index_select(1, positions),
        )

    def _quantize_tensor(self, x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xr = torch.einsum("bthd,df->bthf", x.to(torch.float32), self.rotation)
        xmin = xr.amin(dim=-1, keepdim=True)
        xmax = xr.amax(dim=-1, keepdim=True)
        levels = float((1 << bits) - 1)
        scale = torch.clamp((xmax - xmin) / levels, min=1e-6)
        zero = xmin
        q = torch.clamp(torch.round((xr - zero) / scale), min=0, max=levels).to(torch.uint8)
        recon = q.to(torch.float32) * scale + zero
        residual_sign = (xr - recon) >= 0
        return q, scale, zero, residual_sign

    def _dequantize_tensor(self, q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, residual_sign: torch.Tensor) -> torch.Tensor:
        residual = torch.where(residual_sign, 1.0, -1.0) * scale * self.cfg.qjl_residual_scale
        xr = (q * scale) + zero + residual
        return torch.einsum("bthd,df->bthf", xr, self.inv_rotation)

    def _make_rotation(self, head_dim: int, device: torch.device) -> torch.Tensor:
        gaussian = torch.randn((head_dim, head_dim), dtype=torch.float32, device=device)
        q, _ = torch.linalg.qr(gaussian)
        return q

    def _pack_bits(self, values: torch.Tensor, bits: int) -> torch.Tensor:
        flat = values.to(torch.int32).reshape(-1, self.head_dim)
        packed_cols = self.packed_dim(bits)
        out = torch.zeros((flat.shape[0], packed_cols), dtype=torch.uint8, device=values.device)
        mask = (1 << bits) - 1
        for i in range(self.head_dim):
            v = flat[:, i] & mask
            bit_offset = i * bits
            byte_idx = bit_offset // 8
            shift = bit_offset % 8
            out[:, byte_idx] |= (v << shift).to(torch.uint8)
            spill = shift + bits - 8
            if spill > 0 and byte_idx + 1 < packed_cols:
                out[:, byte_idx + 1] |= (v >> (bits - spill)).to(torch.uint8)
        return out.view(*values.shape[:-1], packed_cols)

    def _unpack_bits(self, packed: torch.Tensor, bits: int) -> torch.Tensor:
        packed_flat = packed.to(torch.int32).reshape(-1, packed.shape[-1])
        out = torch.zeros((packed_flat.shape[0], self.head_dim), dtype=torch.uint8, device=packed.device)
        mask = (1 << bits) - 1
        for i in range(self.head_dim):
            bit_offset = i * bits
            byte_idx = bit_offset // 8
            shift = bit_offset % 8
            v = (packed_flat[:, byte_idx] >> shift) & mask
            spill = shift + bits - 8
            if spill > 0 and byte_idx + 1 < packed_flat.shape[1]:
                v |= (packed_flat[:, byte_idx + 1] << (bits - spill)) & mask
            out[:, i] = v.to(torch.uint8)
        return out.view(*packed.shape[:-1], self.head_dim)


class KVCacheRingBuffer:
    """Ring buffer for per-layer KV tensors used during autoregressive decoding."""

    def __init__(self, max_window: int, turboquant_config: TurboQuantConfig | None = None) -> None:
        """Initialize the ring buffer.

        Args:
            max_window: Maximum number of KV tokens to keep.
        """
        self.max_window = max(1, int(max_window))
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._lengths: list[int] = []
        self._write_ptrs: list[int] = []
        self._codec: TurboQuantCodec | None = None
        self._turboquant_config = turboquant_config

    def update(
        self,
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        skip_write: bool = False,
    ) -> list[RingKVCacheView]:
        """Append a single decoding step and expose ordered cache tensors.

        Args:
            present_key_values: Per-layer present KV tensors from the model.

        Returns:
            Ordered per-layer KV tensors for the next decoding step.
        """
        if not self._keys:
            self._allocate(present_key_values)
        if skip_write:
            self._advance_all_layers()
            return self.materialize()
        for layer_idx, (key, value) in enumerate(present_key_values):
            self._append_layer(layer_idx, key, value)
        return self.materialize()

    def _advance_all_layers(self) -> None:
        """Advance ring metadata when the model has already written the current step."""
        for layer_idx in range(len(self._keys)):
            write_ptr = self._write_ptrs[layer_idx]
            self._write_ptrs[layer_idx] = (write_ptr + 1) % self.max_window
            self._lengths[layer_idx] = min(self.max_window, self._lengths[layer_idx] + 1)

    def materialize(self) -> list[RingKVCacheView]:
        """Return static ring buffers and logical token positions.

        Returns:
            A per-layer immutable view for downstream attention.
        """
        ordered: list[RingKVCacheView] = []
        for layer_idx, key_buffer in enumerate(self._keys):
            cur_len = self._lengths[layer_idx]
            if cur_len == 0:
                positions = torch.empty((0,), dtype=torch.long, device=key_buffer.device)
                ordered.append(
                    RingKVCacheView(
                        keys=key_buffer,
                        values=self._values[layer_idx],
                        positions=positions,
                        codec=self._codec,
                        layer_idx=layer_idx,
                        current_idx=self._write_ptrs[layer_idx],
                        current_length=cur_len,
                    )
                )
                continue
            if cur_len < self.max_window:
                positions = torch.arange(cur_len, device=key_buffer.device, dtype=torch.long)
                ordered.append(
                    RingKVCacheView(
                        keys=key_buffer,
                        values=self._values[layer_idx],
                        positions=positions,
                        codec=self._codec,
                        layer_idx=layer_idx,
                        current_idx=self._write_ptrs[layer_idx],
                        current_length=cur_len,
                    )
                )
                continue
            ptr = self._write_ptrs[layer_idx]
            positions = (torch.arange(self.max_window, device=key_buffer.device, dtype=torch.long) + ptr) % self.max_window
            ordered.append(
                RingKVCacheView(
                    keys=key_buffer,
                    values=self._values[layer_idx],
                    positions=positions,
                    codec=self._codec,
                    layer_idx=layer_idx,
                    current_idx=self._write_ptrs[layer_idx],
                    current_length=cur_len,
                )
            )
        return ordered

    def _allocate(self, present_key_values: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Allocate fixed-size KV storage for each layer."""
        for key, value in present_key_values:
            b, _, h, d = key.shape
            if self._turboquant_config is not None:
                if self._codec is None:
                    self._codec = TurboQuantCodec(self._turboquant_config, head_dim=d, device=key.device)
                self._codec.init_layer_state()
                key_pack_dim = self._codec.packed_dim(self._turboquant_config.key_bits)
                value_pack_dim = self._codec.packed_dim(self._turboquant_config.value_bits)
                self._keys.append(torch.empty((b, self.max_window, h, key_pack_dim), dtype=torch.uint8, device=key.device))
                self._values.append(torch.empty((b, self.max_window, h, value_pack_dim), dtype=torch.uint8, device=value.device))
                self._codec.allocate_layer(len(self._keys) - 1, b, self.max_window, h, key.device)
            else:
                self._keys.append(torch.empty((b, self.max_window, h, d), dtype=key.dtype, device=key.device))
                self._values.append(torch.empty((b, self.max_window, h, d), dtype=value.dtype, device=value.device))
            self._lengths.append(0)
            self._write_ptrs.append(0)

    def _append_layer(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write the newest token KV for a layer into the ring."""
        step_tokens = key[:, -1:, :, :]
        step_values = value[:, -1:, :, :]
        write_ptr = self._write_ptrs[layer_idx]
        if self._codec is None:
            self._keys[layer_idx][:, write_ptr : write_ptr + 1, :, :] = step_tokens
            self._values[layer_idx][:, write_ptr : write_ptr + 1, :, :] = step_values
        else:
            qk, qv = self._codec.quantize_step(layer_idx, step_tokens, step_values, write_ptr)
            self._keys[layer_idx][:, write_ptr : write_ptr + 1, :, :] = qk
            self._values[layer_idx][:, write_ptr : write_ptr + 1, :, :] = qv
        self._write_ptrs[layer_idx] = (write_ptr + 1) % self.max_window
        self._lengths[layer_idx] = min(self.max_window, self._lengths[layer_idx] + 1)


def _to_batch_vector(
    value: Union[float, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert scalar or tensor sampling argument to a batch-aligned vector."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.to(device=device, dtype=dtype).repeat(batch_size)
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError("Expected a [batch_size] tensor for per-sample sampling arguments.")
        return value.to(device=device, dtype=dtype)
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


def thermodynamic_sampling_with_stats(
    logits: torch.Tensor,
    alpha: float = 1.0,
    temperature: Union[float, torch.Tensor] = 1.0,
    top_p: Union[float, torch.Tensor] = 1.0,
    t_min: float = 0.1,
    t_max: float = 2.0,
    healthy_entropy_limit: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    topk_vals, _ = torch.topk(logits, k=2, dim=-1)
    delta_top2 = torch.clamp(topk_vals[..., 0] - topk_vals[..., 1], min=1e-5)

    t_base = t_min + (t_max - t_min) * torch.exp(-alpha * delta_top2)

    chaos_diff = torch.clamp(entropy - healthy_entropy_limit, min=0.0)
    chaos_penalty = torch.exp(-chaos_diff)

    t_dynamic = t_min + (t_base - t_min) * chaos_penalty
    t_dynamic = torch.clamp(t_dynamic, min=t_min, max=t_max).unsqueeze(-1)

    batch_size = logits.shape[0]
    temperatures = torch.clamp(
        _to_batch_vector(temperature, batch_size=batch_size, device=logits.device, dtype=logits.dtype),
        min=1e-5,
    ).unsqueeze(-1)
    top_ps = torch.clamp(
        _to_batch_vector(top_p, batch_size=batch_size, device=logits.device, dtype=logits.dtype),
        min=0.0,
        max=1.0,
    )

    scaled_logits = (logits / temperatures) / t_dynamic
    if bool(torch.any((top_ps > 0.0) & (top_ps < 1.0)).item()):
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > top_ps.unsqueeze(-1)
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        scaled_logits = torch.full_like(scaled_logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    next_tokens = torch.distributions.Categorical(logits=scaled_logits).sample()

    return next_tokens, entropy, chaos_diff


def _decode_next_ar_token(
    model: VARTransformer,
    *,
    prefix_inputs: list[torch.Tensor],
    target_level: int,
    sequence: torch.Tensor,
    cfg_scale: float,
    alpha: float,
    healthy_entropy_limit: float,
    temperature: Union[float, torch.Tensor],
    top_p: Union[float, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = sequence.shape[0]
    bos = torch.full((batch, 1), model.cfg.mask_token_id, dtype=torch.long, device=sequence.device)
    ar_input = torch.cat([bos, sequence], dim=1)
    logits = model(
        prefix_inputs=prefix_inputs,
        target_level=target_level,
        current_level_input=ar_input,
        batch_size=batch,
        cfg_scale=cfg_scale,
    )
    pos = sequence.shape[1]
    return thermodynamic_sampling_with_stats(
        logits[:, pos, :],
        alpha=alpha,
        temperature=temperature,
        top_p=top_p,
        healthy_entropy_limit=healthy_entropy_limit,
    )


def _decode_next_ar_token_with_cache(
    model: VARTransformer,
    *,
    prefix_inputs: list[torch.Tensor],
    target_level: int,
    token_input: torch.Tensor,
    cfg_scale: float,
    alpha: float,
    healthy_entropy_limit: float,
    temperature: Union[float, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    past_key_values: list[tuple[torch.Tensor, torch.Tensor] | RingKVCacheView] | None,
    cache_ring_buffer: KVCacheRingBuffer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[RingKVCacheView]]:
    logits, present_key_values = model(
        prefix_inputs=prefix_inputs,
        target_level=target_level,
        current_level_input=token_input,
        cfg_scale=cfg_scale,
        past_key_values=past_key_values,
        use_cache=True,
    )
    next_token, entropy, chaos = thermodynamic_sampling_with_stats(
        logits[:, -1, :],
        alpha=alpha,
        temperature=temperature,
        top_p=top_p,
        healthy_entropy_limit=healthy_entropy_limit,
    )
    return (
        next_token,
        entropy,
        chaos,
        cache_ring_buffer.update(
            present_key_values,
            skip_write=bool(
                past_key_values
                and isinstance(past_key_values[0], RingKVCacheView)
                and past_key_values[0].codec is not None
            ),
        ),
    )


def _parallel_block_draft(
    model: VARTransformer,
    *,
    prefix_inputs: list[torch.Tensor],
    len_lvl_2: int,
    block_count: int,
    block_size: int,
    batch_size: int,
    device: torch.device,
    cfg_scale: float,
    alpha: float,
    healthy_entropy_limit: float,
    temperature: Union[float, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    rollback_chaos_threshold: float = 0.75,
) -> torch.Tensor:
    lvl_2_tokens = torch.full((batch_size, len_lvl_2), model.cfg.pad_token_id, dtype=torch.long, device=device)

    for block_idx in range(block_count):
        start_idx = block_idx * block_size
        if start_idx >= len_lvl_2:
            break
        end_idx = len_lvl_2 if block_idx == block_count - 1 else min(len_lvl_2, start_idx + block_size)
        block_len = end_idx - start_idx

        chunk_inputs = torch.full(
            (batch_size, block_len),
            model.cfg.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        logits = model(
            prefix_inputs=prefix_inputs,
            target_level=2,
            current_level_input=chunk_inputs,
            cfg_scale=cfg_scale,
            compact_memory_for_final_level=True,
        )
        sampled, _, chaos_diff = thermodynamic_sampling_with_stats(
            logits.view(-1, logits.shape[-1]),
            alpha=alpha,
            healthy_entropy_limit=healthy_entropy_limit,
            temperature=temperature,
            top_p=top_p,
        )
        if float(chaos_diff.mean().item()) > float(rollback_chaos_threshold):
            raise RollbackEvent(start_idx, end_idx)
        lvl_2_tokens[:, start_idx:end_idx] = sampled.view(batch_size, block_len)
    return lvl_2_tokens


def _inpaint_block_seams(
    model: VARTransformer,
    *,
    prefix_inputs: list[torch.Tensor],
    lvl_2_tokens: torch.Tensor,
    block_count: int,
    block_size: int,
    cfg_scale: float,
    alpha: float,
    healthy_entropy_limit: float,
    temperature: Union[float, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    seam_tokens: int = 3,
    max_seams_per_pass: int = 10,
) -> torch.Tensor:
    batch_size, seq_len = lvl_2_tokens.shape
    seam_spans: list[tuple[int, int]] = []
    for block_idx in range(block_count - 1):
        seam = (block_idx + 1) * block_size
        if seam <= 0 or seam >= seq_len:
            continue
        left = max(0, seam - seam_tokens)
        right = min(seq_len, seam + seam_tokens)
        if left < right:
            seam_spans.append((left, right))

    if not seam_spans:
        return lvl_2_tokens

    stitched = lvl_2_tokens.clone()
    seams_per_pass = max(1, int(max_seams_per_pass))
    for seam_offset in range(0, len(seam_spans), seams_per_pass):
        seam_chunk = seam_spans[seam_offset : seam_offset + seams_per_pass]
        expanded = stitched.repeat_interleave(len(seam_chunk), dim=0)
        mask_positions: list[tuple[int, int]] = []
        row = 0
        for b in range(batch_size):
            for left, right in seam_chunk:
                expanded[row, left:right] = model.cfg.mask_token_id
                for pos in range(left, right):
                    mask_positions.append((row, pos))
                row += 1

        logits = model(
            prefix_inputs=[x.repeat_interleave(len(seam_chunk), dim=0) for x in prefix_inputs],
            target_level=2,
            current_level_input=expanded,
            cfg_scale=cfg_scale,
            compact_memory_for_final_level=True,
        )

        pos_rows = torch.tensor([r for r, _ in mask_positions], device=lvl_2_tokens.device)
        pos_cols = torch.tensor([c for _, c in mask_positions], device=lvl_2_tokens.device)
        masked_logits = logits[pos_rows, pos_cols, :]
        sampled, _, _ = thermodynamic_sampling_with_stats(
            masked_logits,
            alpha=alpha,
            healthy_entropy_limit=healthy_entropy_limit,
            temperature=temperature,
            top_p=top_p,
        )
        expanded[pos_rows, pos_cols] = sampled

        row = 0
        for b in range(batch_size):
            for left, right in seam_chunk:
                stitched[b, left:right] = expanded[row, left:right]
                row += 1
    return stitched


def _resolve_phase3_level2_length(*, full_lvl_2_len: int, nominal_lvl_1_len: int, actual_lvl_1_len: int) -> int:
    """Resolve target level-2 length for phase 3 based on configured level capacities.

    Args:
        full_lvl_2_len: Configured max token length for level 2.
        nominal_lvl_1_len: Configured token length for level 1.
        actual_lvl_1_len: Generated token length for level 1 before EOS.

    Returns:
        Non-zero target length for level 2 bounded by configuration.

    Raises:
        ValueError: If configured or generated lengths are invalid.
    """
    if full_lvl_2_len <= 0:
        raise ValueError("level_lengths[2] must be positive.")
    if nominal_lvl_1_len <= 0:
        raise ValueError("level_lengths[1] must be positive.")
    if actual_lvl_1_len < 0:
        raise ValueError("actual level-1 length must be non-negative.")
    if actual_lvl_1_len == 0:
        return 1

    scaled_len = (actual_lvl_1_len * full_lvl_2_len + nominal_lvl_1_len - 1) // nominal_lvl_1_len
    return max(1, min(full_lvl_2_len, scaled_len))


@torch.no_grad()
def hybrid_cascade_decode(
    model: VARTransformer,
    *,
    batch_size: int,
    device: torch.device,
    prefix_inputs: list[torch.Tensor] | None = None,
    nar_steps: int = 4,
    cfg_scale: float = 1.0,
    alpha: float = 1.0,
    healthy_entropy_limit: float = 1.5,
    temperature: Union[float, torch.Tensor] = 1.0,
    top_p: Union[float, torch.Tensor] = 1.0,
    min_block_size_lvl2: int = 16,
    max_seams_per_inpaint_pass: int = 10,
    bpe_chunk_length: int = 128,
    turboquant_kv: bool = False,
) -> list[torch.Tensor]:
    max_local_window = 1024
    batch = int(batch_size)
    out: list[torch.Tensor] = []
    conditioned_prefixes = prefix_inputs or []

    for level_idx, level_tokens in enumerate(conditioned_prefixes):
        if level_idx >= len(model.cfg.level_lengths):
            break
        expected_len = int(model.cfg.level_lengths[level_idx])
        if level_tokens.dim() != 2:
            raise ValueError(f"prefix_inputs[{level_idx}] must be rank-2 tensor with shape (B, L).")
        if level_tokens.shape[0] != batch:
            raise ValueError(
                f"prefix_inputs[{level_idx}] batch mismatch: expected {batch}, got {level_tokens.shape[0]}."
            )
        if level_tokens.shape[1] > expected_len:
            raise ValueError(
                f"prefix_inputs[{level_idx}] length exceeds level capacity "
                f"({level_tokens.shape[1]} > {expected_len})."
            )
        out.append(level_tokens.to(device))

    len_lvl_0 = model.cfg.level_lengths[0]
    if len(out) < 1:
        print(f"[HYBRID] Фаза 1: AR Генерация макро-плана ({len_lvl_0} шагов)...")
        lvl_0_sequence = torch.empty((batch, 0), dtype=torch.long, device=device)
    else:
        lvl_0_sequence = out[0]
        print(
            "[HYBRID] Фаза 1: AR продолжение макро-плана "
            f"(prefix={lvl_0_sequence.shape[1]}, target={len_lvl_0})..."
        )
    for _ in range(lvl_0_sequence.shape[1], len_lvl_0):
        next_token, _, _ = _decode_next_ar_token(
            model,
            prefix_inputs=[],
            target_level=0,
            sequence=lvl_0_sequence,
            cfg_scale=cfg_scale,
            alpha=alpha,
            healthy_entropy_limit=healthy_entropy_limit,
            temperature=temperature,
            top_p=top_p,
        )
        lvl_0_sequence = torch.cat([lvl_0_sequence, next_token.unsqueeze(1)], dim=1)
    if len(out) < 1:
        out.append(lvl_0_sequence)
    else:
        out[0] = lvl_0_sequence

    len_lvl_1 = model.cfg.level_lengths[1]
    if len(out) < 2:
        print(f"[HYBRID] Фаза 2: AR Генерация структурного каркаса ({len_lvl_1} шагов)...")
        lvl_1_sequence = torch.empty((batch, 0), dtype=torch.long, device=device)
    else:
        lvl_1_sequence = out[1]
        print(
            "[HYBRID] Фаза 2: AR продолжение структурного каркаса "
            f"(prefix={lvl_1_sequence.shape[1]}, target={len_lvl_1})..."
        )
    past_key_values: list[tuple[torch.Tensor, torch.Tensor] | RingKVCacheView] | None = None
    turbo_cfg = TurboQuantConfig() if turboquant_kv else None
    cache_ring_buffer = KVCacheRingBuffer(max_window=max_local_window, turboquant_config=turbo_cfg)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    actual_lvl_1_len = len_lvl_1
    chunk_size = max(1, int(bpe_chunk_length))
    tokens_remaining = max(0, len_lvl_1 - lvl_1_sequence.shape[1])
    num_chunks = (tokens_remaining + chunk_size - 1) // chunk_size
    global_level_0_memory = out[0]
    for chunk_idx in range(num_chunks):
        chunk_start = lvl_1_sequence.shape[1]
        chunk_end = min(len_lvl_1, chunk_start + chunk_size)
        for _ in range(chunk_start, chunk_end):
            is_active = ~finished
            token_input = (
                torch.full((batch, 1), model.cfg.mask_token_id, dtype=torch.long, device=device)
                if lvl_1_sequence.shape[1] == 0
                else lvl_1_sequence[:, -1:].contiguous()
            )
            if bool(torch.any(~is_active).item()):
                token_input = token_input.clone()
                token_input[~is_active] = int(model.cfg.pad_token_id)
            next_token, _, _, past_key_values = _decode_next_ar_token_with_cache(
                model,
                prefix_inputs=[global_level_0_memory],
                target_level=1,
                token_input=token_input,
                cfg_scale=cfg_scale,
                alpha=alpha,
                healthy_entropy_limit=healthy_entropy_limit,
                temperature=temperature,
                top_p=top_p,
                past_key_values=past_key_values,
                cache_ring_buffer=cache_ring_buffer,
            )
            if bool(torch.any(~is_active).item()):
                next_token = torch.where(
                    ~is_active,
                    torch.full_like(next_token, int(model.cfg.pad_token_id)),
                    next_token,
                )
            is_eos = next_token == int(model.cfg.eos_token_id)
            finished |= is_eos
            lvl_1_sequence = torch.cat([lvl_1_sequence, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                actual_lvl_1_len = lvl_1_sequence.shape[1]
                break
        if finished.all():
            break
    if len(out) < 2:
        out.append(lvl_1_sequence)
    else:
        out[1] = lvl_1_sequence

    full_lvl_2_len = model.cfg.level_lengths[2]
    len_lvl_2 = _resolve_phase3_level2_length(
        full_lvl_2_len=full_lvl_2_len,
        nominal_lvl_1_len=len_lvl_1,
        actual_lvl_1_len=actual_lvl_1_len,
    )
    min_block = max(1, int(min_block_size_lvl2))
    block_count = max(1, min(actual_lvl_1_len, len_lvl_2 // min_block))
    block_size = max(min_block, len_lvl_2 // block_count)

    print(f"[HYBRID] Фаза 3.1: Параллельный драфт ({block_count} блоков, block_size={block_size})...")
    try:
        lvl_2_draft = _parallel_block_draft(
            model,
            prefix_inputs=out,
            len_lvl_2=len_lvl_2,
            block_count=block_count,
            block_size=block_size,
            batch_size=batch,
            device=device,
            cfg_scale=cfg_scale,
            alpha=alpha,
            healthy_entropy_limit=healthy_entropy_limit,
            temperature=temperature,
            top_p=top_p,
        )
    except RollbackEvent as event:
        print(f"[HYBRID] rollback block=[{event.block_start}, {event.block_end}) -> conservative resample")
        lvl_2_draft = _parallel_block_draft(
            model,
            prefix_inputs=out,
            len_lvl_2=len_lvl_2,
            block_count=block_count,
            block_size=block_size,
            batch_size=batch,
            device=device,
            cfg_scale=cfg_scale,
            alpha=max(0.25, alpha * 0.5),
            healthy_entropy_limit=healthy_entropy_limit,
            temperature=temperature,
            top_p=top_p,
            rollback_chaos_threshold=1.0,
        )

    print("[HYBRID] Фаза 3.2: Шовная склейка (latent inpainting)...")
    lvl_2_tokens = _inpaint_block_seams(
        model,
        prefix_inputs=out,
        lvl_2_tokens=lvl_2_draft,
        block_count=block_count,
        block_size=block_size,
        cfg_scale=cfg_scale,
        alpha=alpha,
        healthy_entropy_limit=healthy_entropy_limit,
        temperature=temperature,
        top_p=top_p,
        max_seams_per_pass=max_seams_per_inpaint_pass,
    )

    out.append(lvl_2_tokens)
    print("[HYBRID] Генерация завершена.")
    return out
