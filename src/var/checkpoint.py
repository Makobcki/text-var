from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.var.model import VARTransformer
from src.var.training.config import VARConfig


def _collect_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.random.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: VARTransformer,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    loss: float,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "md-var-checkpoint-v2",
        "model_family": "var",
        "model_config": model.cfg.to_dict(),
        "model": model.state_dict(),
        "step": int(step),
        "loss": float(loss),
        "rng_state": _collect_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, ckpt_path)


def load_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[VARTransformer, dict[str, Any]]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if payload.get("model_family") != "var":
        raise ValueError("Checkpoint is not a VAR checkpoint.")
    cfg = VARConfig.from_dict(dict(payload["model_config"]))
    model = VARTransformer(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def restore_training_state(
    payload: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    rng_state = payload.get("rng_state")
    if isinstance(rng_state, dict):
        _restore_rng_state(rng_state)
