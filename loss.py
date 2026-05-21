import torch
import torch.nn.functional as F

from model import VARTransformer


def multiscale_next_scale_cross_entropy(
    model: VARTransformer,
    moved_tokens: list[torch.Tensor],
    *,
    level_weights: list[float] | None = None,
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

        level_weight = (
            level_weights[target_idx] if level_weights and target_idx < len(level_weights) else 1.0
        )
        current_scale_loss = 0.0

        for i, pred in enumerate(all_predictions):
            is_early = i < (len(all_predictions) - 1)
            loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))

            if is_early:
                loss = loss * 0.25

            current_scale_loss += loss

        scale_losses.append(current_scale_loss * level_weight)

    return sum(scale_losses)
