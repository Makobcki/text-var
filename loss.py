import torch
import torch.nn.functional as F

from model import VARTransformer


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

        outputs = model(
            prefix_inputs,
            target_level=target_idx,
            current_level_input=model_input,
            batch_size=batch_size,
            return_early_outputs=True,
        )

        final_pred, early_outputs = outputs
        all_predictions = early_outputs + [final_pred]

        level_weight = (
            level_weights[target_idx] if level_weights and target_idx < len(level_weights) else 1.0
        )
        current_scale_loss = 0.0

        for i, pred in enumerate(all_predictions):
            is_early = i < (len(all_predictions) - 1)
            flat_pred = pred.reshape(-1, pred.size(-1))
            flat_target = target.reshape(-1)
            per_token = F.cross_entropy(flat_pred, flat_target, reduction="none")
            if mask_positions is not None and mask_positions.any():
                flat_mask = mask_positions.reshape(-1)
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
