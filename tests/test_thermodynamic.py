import torch

from src.var.generator import thermodynamic_sampling_with_stats


def test_thermodynamic_sampling_accepts_per_item_temperature_and_top_p() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[8.0, 7.9, 1.0], [8.0, 7.9, 1.0]], dtype=torch.float32)

    sampled, entropy, chaos = thermodynamic_sampling_with_stats(
        logits,
        temperature=torch.tensor([0.1, 3.0]),
        top_p=torch.tensor([0.5, 1.0]),
    )

    assert sampled.shape == (2,)
    assert entropy.shape == (2,)
    assert chaos.shape == (2,)
    # Low top-p should always keep only the top token.
    assert sampled[0].item() == 0


def test_thermodynamic_sampling_accepts_column_vector_sampling_arguments() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[8.0, 7.9, 1.0], [8.0, 7.9, 1.0]], dtype=torch.float32)

    sampled, entropy, chaos = thermodynamic_sampling_with_stats(
        logits,
        temperature=torch.tensor([[0.1], [3.0]], dtype=torch.float32),
        top_p=torch.tensor([[0.5], [1.0]], dtype=torch.float32),
    )

    assert sampled.shape == (2,)
    assert entropy.shape == (2,)
    assert chaos.shape == (2,)


def test_thermodynamic_sampling_accepts_singleton_tensor_sampling_arguments() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[8.0, 7.9, 1.0], [8.0, 7.9, 1.0]], dtype=torch.float32)

    sampled, entropy, chaos = thermodynamic_sampling_with_stats(
        logits,
        temperature=torch.tensor([0.4], dtype=torch.float32),
        top_p=torch.tensor([1.0], dtype=torch.float32),
    )

    assert sampled.shape == (2,)
    assert entropy.shape == (2,)
    assert chaos.shape == (2,)
