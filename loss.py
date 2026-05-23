import math

import torch
import torch.nn.functional as F

from model import VARTransformer

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _flash_cross_entropy_loss
except Exception:  # flash-attn is optional
    _flash_cross_entropy_loss = None


def _cross_entropy_per_token(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    use_flash: bool,
    ignore_index: int | None = None,
) -> torch.Tensor:
    if use_flash and _flash_cross_entropy_loss is not None and logits.is_cuda:
        losses, _ = _flash_cross_entropy_loss(logits, target)
        if ignore_index is not None:
            valid_tokens = target != ignore_index
            losses = losses[valid_tokens]
        return losses
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    ce_ignore_index = ignore_index if ignore_index is not None else -100
    losses = F.cross_entropy(flat_logits, flat_target, reduction="none", ignore_index=ce_ignore_index)
    if ignore_index is not None:
        valid_tokens = flat_target != ignore_index
        losses = losses[valid_tokens]
    return losses


def multiscale_next_scale_cross_entropy(
    model: VARTransformer,
    moved_tokens: list[torch.Tensor],
    *,
    level_weights: list[float] | None = None,
    corruption_level_idx: int = -1,
    corruption_prob: float = 0.35,
    corruption_span_min: int = 8,
    corruption_span_max: int = 64,
    masked_loss_weight: float = 0.85,
    use_early_exit_loss: bool = False,
) -> torch.Tensor:
    def _build_span_mask(
        batch: int,
        seq_len: int,
        prob: float,
        min_span: int,
        max_span: int,
        device: torch.device,
    ) -> torch.Tensor:
        if prob <= 0.0 or seq_len <= 0:
            return torch.zeros((batch, seq_len), dtype=torch.bool, device=device)

        span_lo = max(1, min(int(min_span), seq_len))
        span_hi = max(span_lo, min(int(max_span), seq_len))
        mean_span = max(1.0, float(span_lo + span_hi) / 2.0)
        clipped_prob = min(max(float(prob), 1e-5), 1.0 - 1e-5)
        num_spans = int(math.ceil(-seq_len / mean_span * math.log(1.0 - clipped_prob)))
        num_spans = max(1, num_spans)

        starts = torch.randint(0, seq_len, (batch, num_spans, 1), device=device)
        lengths = torch.randint(span_lo, span_hi + 1, (batch, num_spans, 1), device=device)
        positions = torch.arange(seq_len, device=device).view(1, 1, -1)
        return ((positions >= starts) & (positions < starts + lengths)).any(dim=1)

    scale_losses = []
    batch_size = moved_tokens[0].size(0)
    use_flash = bool(getattr(model.cfg, "flash_cross_entropy", True))
    ignore_index = getattr(model.cfg, "pad_token_id", None)

    for target_idx in range(len(moved_tokens)):
        prefix_inputs = moved_tokens[:target_idx]
        target = moved_tokens[target_idx]
        model_input = target
        mask_positions = None
        effective_corruption_level = corruption_level_idx if corruption_level_idx >= 0 else (len(moved_tokens) - 1)
        if target_idx < len(moved_tokens) - 1:
            bos = torch.full(
                (batch_size, 1),
                int(model.cfg.mask_token_id),
                dtype=torch.long,
                device=target.device,
            )
            model_input = torch.cat([bos, target[:, :-1]], dim=1)
        elif target_idx == effective_corruption_level:
            mask_positions = _build_span_mask(
                batch=batch_size,
                seq_len=target.size(1),
                prob=corruption_prob,
                min_span=corruption_span_min,
                max_span=corruption_span_max,
                device=target.device,
            )
            if mask_positions.any():
                model_input = target.clone()
                model_input[mask_positions] = int(model.cfg.mask_token_id)

        if use_early_exit_loss:
            final_pred, early_outputs = model(
                prefix_inputs,
                target_level=target_idx,
                current_level_input=model_input,
                batch_size=batch_size,
                return_early_outputs=True,
            )
            all_predictions = early_outputs + [final_pred]
        else:
            final_pred = model(
                prefix_inputs,
                target_level=target_idx,
                current_level_input=model_input,
                batch_size=batch_size,
                return_early_outputs=False,
            )
            all_predictions = [final_pred]

        level_weight = (
            level_weights[target_idx] if level_weights and target_idx < len(level_weights) else 1.0
        )
        current_scale_loss = 0.0

        flat_target = target.reshape(-1)
        flat_mask = mask_positions.reshape(-1) if mask_positions is not None else None

        for i, pred in enumerate(all_predictions):
            is_early = i < (len(all_predictions) - 1)
            per_token = _cross_entropy_per_token(
                pred,
                flat_target,
                use_flash=use_flash,
                ignore_index=ignore_index,
            )

            if flat_mask is not None and flat_mask.any():
                masked_part = per_token[flat_mask]
                unmasked_part = per_token[~flat_mask]
                if masked_part.numel() > 0 and unmasked_part.numel() > 0:
                    loss = masked_loss_weight * masked_part.mean() + (1.0 - masked_loss_weight) * unmasked_part.mean()
                elif masked_part.numel() > 0:
                    loss = masked_part.mean()
                else:
                    loss = unmasked_part.mean()
            else:
                loss = per_token.mean()

            if is_early:
                loss = loss * 0.25

            current_scale_loss += loss

        scale_losses.append(current_scale_loss * level_weight)

    return sum(scale_losses)
