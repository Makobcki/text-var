import torch

from src.var.loss import _cross_entropy_per_token


def test_cross_entropy_per_token_ignores_padding_tokens() -> None:
    logits = torch.tensor([[[3.0, 0.5], [0.1, 3.2], [2.0, 0.2]]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 0], dtype=torch.long)
    ignore_index = 0

    masked_losses = _cross_entropy_per_token(
        logits,
        targets,
        use_flash=False,
        ignore_index=ignore_index,
    )
    expected_losses = _cross_entropy_per_token(
        logits,
        targets,
        use_flash=False,
        ignore_index=None,
    )[targets != ignore_index]

    assert masked_losses.numel() == 1
    assert torch.allclose(masked_losses, expected_losses)
