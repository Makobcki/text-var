from pathlib import Path

import pytest
import torch
from src.vqvae.training import main as training_main


class DummyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.vocab_size = 8
        self.hidden_size = 4
        self.semantic_sequence_length = 2
        self.pad_token_id = 0
        self.quantizer = type("Quantizer", (), {"num_embeddings": 16})()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.config = type("Config", (), {"to_dict": lambda self: {}})()

    def to(self, _dev: torch.device) -> "DummyModel":
        return self

    def parameters(self):
        return [torch.nn.Parameter(torch.tensor(1.0))]

    def train(self) -> None:
        return None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor([1.0])}

    def __call__(self, _tokens: torch.Tensor, padding_mask: torch.Tensor):
        loss = (padding_mask.float().sum() * 0.0) + torch.tensor(1.0, requires_grad=True)
        return torch.zeros_like(_tokens), loss, {"loss": loss}


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr) -> None:
        self.step_calls = 0
        self.zero_calls = 0
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def zero_grad(self, set_to_none: bool = True) -> None:
        _ = set_to_none
        self.zero_calls += 1

    def step(self) -> None:
        self.step_calls += 1


def test_run_training_uses_gradient_accumulation(monkeypatch, tmp_path: Path) -> None:
    metadata = training_main.TokenCacheMetadata(
        kind="multiscale-token-cache",
        level_vocab_sizes=(32,),
        level_lengths=(3,),
        codebook_dim=4,
        max_token_length=3,
    )
    batches = [
        (
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[False, False, False]]),
        )
        for _ in range(4)
    ]

    optimizer_holder: dict[str, DummyOptimizer] = {}

    monkeypatch.setattr(
        training_main,
        "load_token_entries_from_directory",
        lambda *_: ([Path("chunk.pt")], metadata),
    )
    monkeypatch.setattr(training_main, "MultiscaleTokenChunkIterableDataset", lambda **_: object())
    monkeypatch.setattr(training_main, "DataLoader", lambda *_, **__: batches)
    monkeypatch.setattr(training_main, "SemanticTextVQVAE", lambda *_, **__: DummyModel())

    def _make_optimizer(params, lr: float, **kwargs) -> DummyOptimizer:
        optimizer = DummyOptimizer(params, lr)
        optimizer_holder["opt"] = optimizer
        return optimizer

    monkeypatch.setattr(training_main.torch.optim, "AdamW", _make_optimizer)

    output = tmp_path / "vqvae.ckpt"
    training_main.run_training(
        output,
        tmp_path,
        steps=2,
        batch_size=1,
        vocab_size=32,
        hidden_size=4,
        num_semantic_tokens=16,
        lr=1e-3,
        device="cpu",
        level_index=0,
        gradient_accumulation_steps=2,
    )

    assert optimizer_holder["opt"].step_calls == 2


def test_run_training_rejects_non_positive_gradient_accumulation_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        training_main.run_training(
            tmp_path / "vqvae.ckpt",
            tmp_path,
            gradient_accumulation_steps=0,
        )
