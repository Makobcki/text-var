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
) -> torch.Tensor:
    if use_flash and _flash_cross_entropy_loss is not None and logits.is_cuda:
        losses, _ = _flash_cross_entropy_loss(logits, target)
        return losses
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_target = target.reshape(-1)
    return F.cross_entropy(flat_logits, flat_target, reduction="none")


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
    def _build_span_mask(batch: int, seq_len: int, prob: float, min_span: int, max_span: int, device: torch.device) -> torch.Tensor:
        if prob <= 0.0 or seq_len <= 0:
            return torch.zeros((batch, seq_len), dtype=torch.bool, device=device)
        mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=device)
        span = max(1, min(int(min_span), seq_len))
        span_cap = max(span, min(int(max_span), seq_len))
        starts = torch.rand((batch, seq_len), device=device) < float(prob)
        for b_idx in range(batch):
            start_ids = starts[b_idx].nonzero(as_tuple=False).flatten()
            for s in start_ids.tolist():
                cur_span = int(torch.randint(span, span_cap + 1, (1,), device=device).item())
                e = min(seq_len, s + cur_span)
                mask[b_idx, s:e] = True
        return mask

    scale_losses = []
    batch_size = moved_tokens[0].size(0)
    use_flash = bool(getattr(model.cfg, "flash_cross_entropy", True))

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
            per_token = _cross_entropy_per_token(pred, flat_target, use_flash=use_flash)

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
