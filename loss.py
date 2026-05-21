import torch
import torch.nn.functional as F

from model import VARTransformer


def multiscale_next_scale_cross_entropy(
    model: VARTransformer,
    moved_tokens: list[torch.Tensor],
    *,
    level_weights: list[float] | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:

    scale_losses = []
    batch_size = moved_tokens[0].size(0)

    for target_idx in range(len(moved_tokens)):
        prefix_inputs = moved_tokens[:target_idx]
        target = moved_tokens[target_idx]

        outputs = model(
            prefix_inputs,
            target_level=target_idx,
            current_level_input=target,
            batch_size=batch_size,
            return_early_outputs=True,
        )

        final_pred, early_outputs = outputs
        all_predictions = early_outputs + [final_pred]

        is_continuous = target.dtype.is_floating_point
        level_weight = (
            level_weights[target_idx] if level_weights and target_idx < len(level_weights) else 1.0
        )
        current_scale_loss = 0.0

        for i, pred in enumerate(all_predictions):
            is_early = i < (len(all_predictions) - 1)

            if is_continuous:
                H = pred.size(-1)
                pred_flat = F.normalize(pred.reshape(-1, H), dim=-1)
                target_flat = F.normalize(target.reshape(-1, H), dim=-1)

                logits_p2t = torch.matmul(pred_flat, target_flat.T) / temperature
                logits_t2p = torch.matmul(target_flat, pred_flat.T) / temperature

                labels = torch.arange(logits_p2t.size(0), device=pred.device)
                loss_p2t = F.cross_entropy(logits_p2t, labels)
                loss_t2p = F.cross_entropy(logits_t2p, labels)
                loss = (loss_p2t + loss_t2p) / 2.0
            else:
                loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))

            if is_early:
                loss = loss * 0.25

            current_scale_loss += loss

        scale_losses.append(current_scale_loss * level_weight)

    return sum(scale_losses)
