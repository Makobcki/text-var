import torch

from src.var.training.main import _apply_unconditional_prefix_dropout


def test_unconditional_prefix_dropout_zeros_only_prefix_levels() -> None:
    level0 = torch.tensor([[5, 6], [7, 8]], dtype=torch.long)
    level1 = torch.tensor([[9, 10], [11, 12]], dtype=torch.long)
    target = torch.tensor([[13, 14], [15, 16]], dtype=torch.long)

    dropped = _apply_unconditional_prefix_dropout([level0, level1, target], drop_prob=1.0)

    assert torch.equal(dropped[0], torch.zeros_like(level0))
    assert torch.equal(dropped[1], torch.zeros_like(level1))
    assert torch.equal(dropped[2], target)


def test_unconditional_prefix_dropout_keeps_tokens_when_disabled() -> None:
    levels = [torch.tensor([[1, 2], [3, 4]], dtype=torch.long) for _ in range(3)]
    dropped = _apply_unconditional_prefix_dropout(levels, drop_prob=0.0)

    for original, current in zip(levels, dropped):
        assert torch.equal(original, current)
