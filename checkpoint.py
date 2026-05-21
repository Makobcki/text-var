from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from var_branch.config import VARConfig
from var_branch.model import VARTransformer


def save_checkpoint(
    path: str | Path,
    *,
    model: VARTransformer,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    loss: float,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "md-var-checkpoint-v1",
        "model_family": "var",
        "model_config": model.cfg.to_dict(),
        "model": model.state_dict(),
        "step": int(step),
        "loss": float(loss),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, ckpt_path)


def load_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[VARTransformer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if payload.get("model_family") != "var":
        raise ValueError("Checkpoint is not a VAR checkpoint.")
    cfg = VARConfig.from_dict(dict(payload["model_config"]))
    model = VARTransformer(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
