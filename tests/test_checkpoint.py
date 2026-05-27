import random
from pathlib import Path

import torch
from src.var.checkpoint import restore_training_state, save_checkpoint
from src.var.model import VARTransformer
from src.var.training.config import VARConfig


def test_save_checkpoint_persists_scaler_and_rng_state(tmp_path: Path) -> None:
    cfg = VARConfig(
        level_vocab_sizes=(16, 16),
        level_lengths=(2, 2),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        exit_layers=(),
    )
    model = VARTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=False)

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=7,
        loss=1.23,
        scaler=scaler,
        scheduler=None,
    )
    payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    assert "scaler" in payload
    assert "rng_state" in payload
    assert "torch" in payload["rng_state"]
    assert "python" in payload["rng_state"]
    assert "numpy" in payload["rng_state"]


def test_restore_training_state_loads_scaler_and_optimizer() -> None:
    cfg = VARConfig(
        level_vocab_sizes=(16, 16),
        level_lengths=(2, 2),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        exit_layers=(),
    )
    model = VARTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=False)

    payload = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": {"torch": torch.random.get_rng_state(), "python": random.getstate()},
    }
    restore_training_state(payload, optimizer=optimizer, scaler=scaler, scheduler=None)


def test_load_checkpoint_uses_weights_only_false(monkeypatch, tmp_path: Path) -> None:
    from src.var.checkpoint import load_checkpoint

    captured: dict[str, object] = {}

    cfg = VARConfig(
        level_vocab_sizes=(16, 16),
        level_lengths=(2, 2),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        exit_layers=(),
    )
    model = VARTransformer(cfg)
    checkpoint_path = tmp_path / "load-model.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        step=1,
        loss=0.0,
        scaler=None,
        scheduler=None,
    )

    original_load = torch.load

    def _spy_load(*args, **kwargs):
        captured.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_load)
    loaded_model, payload = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    assert isinstance(loaded_model, VARTransformer)
    assert isinstance(payload, dict)
    assert captured.get("weights_only") is False
