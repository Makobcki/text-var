import re
import time

from src.core.training_logger import StepTiming, TrainingStepLogger


def test_training_logger_line_contains_required_fields() -> None:
    logger = TrainingStepLogger("var", 100)
    line = logger.build_line(
        step=10,
        loss=2.0,
        timing=StepTiming(
            total=0.321,
            stages={"transfer": 0.010, "forward": 0.200, "backward": 0.090, "optimizer": 0.021},
        ),
    )
    assert "[var] step=10/100" in line
    assert "loss=2.000000" in line
    assert "perplexity=7.3891" in line
    assert "time=0.321s" in line
    assert "stages=(transfer=0.010s forward=0.200s backward=0.090s optimizer=0.021s)" in line
    assert re.search(r"eta=\d+\.\d{2}m", line)


def test_training_logger_eta_decreases_with_progress() -> None:
    logger = TrainingStepLogger("vqvae", 20)
    time.sleep(0.01)
    eta_step_1 = logger._estimate_eta_minutes(step=1, elapsed=0.5)
    eta_step_10 = logger._estimate_eta_minutes(step=10, elapsed=0.5)
    assert eta_step_10 < eta_step_1
