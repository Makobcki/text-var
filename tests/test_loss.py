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

    assert masked_losses[targets != ignore_index].numel() == 1
    assert torch.allclose(masked_losses[targets != ignore_index], expected_losses)
    assert torch.all(masked_losses[targets == ignore_index] == 0.0)


def test_cross_entropy_per_token_returns_finite_when_all_tokens_ignored() -> None:
    logits = torch.tensor([[[3.0, 0.5], [0.1, 3.2]]], dtype=torch.float32)
    targets = torch.tensor([0, 0], dtype=torch.long)
    masked_losses = _cross_entropy_per_token(
        logits, targets, use_flash=False, ignore_index=0
    )
    assert torch.isfinite(masked_losses).all()
    assert float(masked_losses.sum()) == 0.0


class _DummyConfig:
    def __init__(self):
        self.pad_token_id = 0
        self.mask_token_id = 1


class _DummyModel:
    def __init__(self):
        self.cfg = _DummyConfig()

    def __call__(
        self,
        prefix_inputs,
        target_level: int,
        current_level_input: torch.Tensor,
        batch_size: int | None = None,
        return_early_outputs: bool = False,
        precomputed_final_memory: torch.Tensor | None = None,
    ):
        del prefix_inputs, target_level, batch_size, return_early_outputs, precomputed_final_memory
        seq_len = current_level_input.shape[1]
        return torch.zeros((1, seq_len, 4), dtype=torch.float32)


def test_multiscale_loss_is_finite_when_level_is_fully_padding() -> None:
    from src.var.loss import multiscale_next_scale_cross_entropy

    model = _DummyModel()
    moved_tokens = [torch.zeros((1, 3), dtype=torch.long)]

    loss = multiscale_next_scale_cross_entropy(model, moved_tokens)

    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_multiscale_loss_is_invariant_to_fully_padding_level() -> None:
    from src.var.loss import multiscale_next_scale_cross_entropy

    model = _DummyModel()
    informative = torch.tensor([[1, 2, 3]], dtype=torch.long)
    fully_padded = torch.zeros((1, 3), dtype=torch.long)

    loss_without_padding_level = multiscale_next_scale_cross_entropy(model, [informative])
    loss_with_padding_level = multiscale_next_scale_cross_entropy(model, [informative, fully_padded])  # noqa: E501

    assert torch.isfinite(loss_without_padding_level)
    assert torch.isfinite(loss_with_padding_level)
    assert torch.allclose(loss_without_padding_level, loss_with_padding_level, atol=1e-6)


def test_masked_weighting_is_stable_for_sparse_mask() -> None:
    from src.var.loss import multiscale_next_scale_cross_entropy

    model = _DummyModel()
    tokens = torch.tensor([[1, 2, 3, 1, 2, 3]], dtype=torch.long)
    loss_a = multiscale_next_scale_cross_entropy(
        model,
        [tokens],
        corruption_level_idx=0,
        corruption_prob=0.01,
        corruption_span_min=1,
        corruption_span_max=1,
        masked_loss_weight=0.85,
    )
    loss_b = multiscale_next_scale_cross_entropy(
        model,
        [tokens],
        corruption_level_idx=0,
        corruption_prob=0.90,
        corruption_span_min=1,
        corruption_span_max=1,
        masked_loss_weight=0.85,
    )
    assert torch.isfinite(loss_a)
    assert torch.isfinite(loss_b)
    assert float(loss_a) > 0.0
    assert float(loss_b) > 0.0
