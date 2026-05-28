import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch
import random

def collect_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.random.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state: dict[str, Any]) -> None:
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

class CheckpointManager:
    def __init__(self, output_dir: Path | str, max_to_keep: int = 3, prefix: str = "checkpoint"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.prefix = prefix
        self.stop_requested = False
        self._install_signal_handlers()

    def _install_signal_handlers(self):
        def _handle_signal(signum: int, _frame: object) -> None:
            print(f"\n[CheckpointManager] Training stop requested by signal {signum}")
            self.stop_requested = True

        self.previous_handlers = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def restore_handlers(self):
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)

    def save(self, payload: dict[str, Any], step: int) -> Path:
        payload["rng_state"] = collect_rng_state()
        
        ckpt_name = f"{self.prefix}-{step}.pt"
        ckpt_path = self.output_dir / ckpt_name
        
        # Atomic write
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        tmp_path.rename(ckpt_path)
        
        self._cleanup()
        return ckpt_path

    def _cleanup(self):
        checkpoints = []
        for p in self.output_dir.glob(f"{self.prefix}-*.pt"):
            try:
                step_str = p.stem.split("-")[-1]
                step = int(step_str)
                checkpoints.append((step, p))
            except ValueError:
                continue
        
        checkpoints.sort(key=lambda x: x[0])
        
        if len(checkpoints) > self.max_to_keep:
            for _, p in checkpoints[:-self.max_to_keep]:
                p.unlink(missing_ok=True)
