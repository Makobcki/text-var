import torch
from src.var.loss import multiscale_next_scale_cross_entropy


class _DummyCfg:
    flash_cross_entropy = False
    pad_token_id = None
    mask_token_id = 1


class _DummyModel:
    def __init__(self):
        self.cfg = _DummyCfg()
        self.recorded_prefix_lengths = []

    def __call__(self, prefix_inputs, *, target_level, current_level_input, batch_size, return_early_outputs):  # noqa: E501
        if prefix_inputs:
            self.recorded_prefix_lengths.append(prefix_inputs[0].shape[1])
        vocab = 16
        logits = torch.zeros(
            current_level_input.shape[0],
            current_level_input.shape[1],
            vocab,
            dtype=torch.float32,
        )
        return logits


def test_stateful_level0_context_is_appended_to_prefix() -> None:
    model = _DummyModel()
    moved_tokens = [
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[4, 5, 6, 7]], dtype=torch.long),
    ]
    historical = torch.tensor([[9, 9]], dtype=torch.long)

    _ = multiscale_next_scale_cross_entropy(
        model,
        moved_tokens,
        historical_level0_tokens=historical,
    )

    assert model.recorded_prefix_lengths == [5]
