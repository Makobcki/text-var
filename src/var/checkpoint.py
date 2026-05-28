from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.var.model import VARTransformer
from src.var.training.config import VARConfig


from src.core.checkpoint import restore_rng_state


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
        restore_rng_state(rng_state)
