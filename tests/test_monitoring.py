import json
from pathlib import Path

import torch

from src.var.training.config import load_train_config
from train import _compute_grad_norm, _compute_weight_norm, _generate_validation_sample


def test_config_loads_monitoring_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "train.json"
    payload = {
        "checkpoint_path": "checkpoints/latest.pt",
        "log_grad_norm_every": 10,
        "log_weight_norm_every": 20,
        "sample_every": 50,
        "sample_prompt": "Validate me",
        "sample_max_new_tokens": 64,
        "wandb_enabled": True,
        "wandb_project": "demo",
        "wandb_run_name": "run-1",
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_train_config(cfg_path)

    assert cfg.log_grad_norm_every == 10
    assert cfg.log_weight_norm_every == 20
    assert cfg.sample_every == 50
    assert cfg.sample_prompt == "Validate me"
    assert cfg.sample_max_new_tokens == 64
    assert cfg.wandb_enabled is True
    assert cfg.wandb_project == "demo"
    assert cfg.wandb_run_name == "run-1"


def test_norm_helpers_return_positive_values() -> None:
    layer = torch.nn.Linear(4, 2)
    output = layer(torch.ones(1, 4))
    output.sum().backward()

    grad_norm = _compute_grad_norm(layer)
    weight_norm = _compute_weight_norm(layer)

    assert grad_norm > 0.0
    assert weight_norm > 0.0


def test_validation_sample_contains_prompt() -> None:
    text = _generate_validation_sample(5, "hello", 32)
    assert "hello" in text
    assert "step=5" in text
