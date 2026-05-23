from typing import Union

import torch
import torch.nn.functional as F

from src.var.model import RingKVCacheView, VARTransformer


class RollbackEvent(RuntimeError):
    def __init__(self, block_start: int, block_end: int) -> None:
        super().__init__(f"Rollback requested for block [{block_start}, {block_end})")
        self.block_start = block_start
        self.block_end = block_end


class KVCacheRingBuffer:
    """Ring buffer for per-layer KV tensors used during autoregressive decoding."""

    def __init__(self, max_window: int) -> None:
        """Initialize the ring buffer.

        Args:
            max_window: Maximum number of KV tokens to keep.
        """
        self.max_window = max(1, int(max_window))
        self._keys: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._lengths: list[int] = []
        self._write_ptrs: list[int] = []

    def update(
        self,
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[RingKVCacheView]:
        """Append a single decoding step and expose ordered cache tensors.

        Args:
            present_key_values: Per-layer present KV tensors from the model.

        Returns:
            Ordered per-layer KV tensors for the next decoding step.
        """
        if not self._keys:
            self._allocate(present_key_values)
        for layer_idx, (key, value) in enumerate(present_key_values):
            self._append_layer(layer_idx, key, value)
        return self.materialize()

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
                ordered.append(RingKVCacheView(keys=key_buffer, values=self._values[layer_idx], positions=positions))
                continue
            if cur_len < self.max_window:
                positions = torch.arange(cur_len, device=key_buffer.device, dtype=torch.long)
                ordered.append(RingKVCacheView(keys=key_buffer, values=self._values[layer_idx], positions=positions))
                continue
            ptr = self._write_ptrs[layer_idx]
            positions = (torch.arange(self.max_window, device=key_buffer.device, dtype=torch.long) + ptr) % self.max_window
            ordered.append(RingKVCacheView(keys=key_buffer, values=self._values[layer_idx], positions=positions))
        return ordered

    def _allocate(self, present_key_values: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Allocate fixed-size KV storage for each layer."""
        for key, value in present_key_values:
            b, _, h, d = key.shape
            self._keys.append(torch.empty((b, self.max_window, h, d), dtype=key.dtype, device=key.device))
            self._values.append(torch.empty((b, self.max_window, h, d), dtype=value.dtype, device=value.device))
            self._lengths.append(0)
            self._write_ptrs.append(0)

    def _append_layer(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write the newest token KV for a layer into the ring."""
        step_tokens = key[:, -1:, :, :]
        step_values = value[:, -1:, :, :]
        write_ptr = self._write_ptrs[layer_idx]
        self._keys[layer_idx][:, write_ptr : write_ptr + 1, :, :] = step_tokens
        self._values[layer_idx][:, write_ptr : write_ptr + 1, :, :] = step_values
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
        cache_ring_buffer.update(present_key_values),
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
    cache_ring_buffer = KVCacheRingBuffer(max_window=max_local_window)
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
            token_input = (
                torch.full((batch, 1), model.cfg.mask_token_id, dtype=torch.long, device=device)
                if lvl_1_sequence.shape[1] == 0
                else lvl_1_sequence[:, -1:].contiguous()
            )
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
    scale_factor = max(1, full_lvl_2_len // len_lvl_1)
    len_lvl_2 = min(full_lvl_2_len, actual_lvl_1_len * scale_factor)
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
