from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepTiming:
    """Structured timing data for a single optimization step.

    Args:
        total: Full step duration in seconds.
        stages: Mapping of stage name to duration in seconds.
    """

    total: float
    stages: dict[str, float] = field(default_factory=dict)


class TrainingStepLogger:
    """Unified training logger with step-level stats for model training.

    Args:
        architecture: Short architecture tag (for example ``"var"`` or ``"vqvae"``).
        total_steps: Planned total count of optimizer steps.
    """

    def __init__(self, architecture: str, total_steps: int) -> None:
        self._architecture = architecture
        self._total_steps = max(1, int(total_steps))
        self._time_origin = time.perf_counter()

    def build_line(
        self,
        *,
        step: int,
        loss: float,
        timing: StepTiming,
    ) -> str:
        """Build a unified console line for a training step.

        Args:
            step: Current optimizer step (1-indexed).
            loss: Step loss value.
            timing: Timing stats for this step.

        Returns:
            A fully formatted log line.
        """
        safe_loss = float(loss)
        perplexity = math.exp(min(20.0, safe_loss))
        elapsed = max(1e-9, time.perf_counter() - self._time_origin)
        eta_minutes = self._estimate_eta_minutes(step=step, elapsed=elapsed)
        stage_stats = " ".join(f"{name}={value:.3f}s" for name, value in timing.stages.items())

        return (
            f"[{self._architecture}] step={step}/{self._total_steps} "
            f"loss={safe_loss:.6f} "
            f"perplexity={perplexity:.4f} "
            f"time={timing.total:.3f}s"
            f" {stage_stats} "
            f"eta={eta_minutes:.2f}m"
        )

    def _estimate_eta_minutes(self, *, step: int, elapsed: float) -> float:
        """Estimate time to completion in minutes.

        Args:
            step: Current completed step.
            elapsed: Elapsed wall-clock seconds since logger creation.

        Returns:
            Estimated remaining minutes.
        """
        safe_step = max(1, min(int(step), self._total_steps))
        average_step_seconds = elapsed / float(safe_step)
        remaining_steps = max(0, self._total_steps - safe_step)
        return (remaining_steps * average_step_seconds) / 60.0
