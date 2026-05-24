"""Tests for TextVAR API engine batching behavior."""

from src.api.engine import GenerationParams, TextVAREngine


class DummyPipeline:
    """Pipeline stub returning deterministic batched outputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, float, float, bool]] = []

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        per_item_temperatures: list[float] | None = None,
        per_item_top_ps: list[float] | None = None,
        turboquant_kv: bool = False,
    ) -> list[str]:
        self.calls.append((prompts, max_new_tokens, temperature, top_p, turboquant_kv))
        temperatures = per_item_temperatures or [temperature] * len(prompts)
        top_ps = per_item_top_ps or [top_p] * len(prompts)
        return [f"{prompt}|{max_new_tokens}|{temp}|{tp}" for prompt, temp, tp in zip(prompts, temperatures, top_ps)]


def test_generate_batch_supports_mixed_sampling_settings() -> None:
    """Ensure engine can process mixed sampling parameters in one input batch."""
    pipeline = DummyPipeline()
    engine = TextVAREngine(pipeline)  # type: ignore[arg-type]

    outputs = engine.generate_batch(
        [
            GenerationParams("a", 8, 0.7, 0.9),
            GenerationParams("b", 16, 0.9, 0.95),
            GenerationParams("c", 8, 0.7, 0.9),
        ]
    )

    assert outputs == ["a|8|0.7|0.9", "b|16|0.9|0.95", "c|8|0.7|0.9"]
    assert len(pipeline.calls) == 2


def test_generate_batch_keeps_batching_for_mixed_temperatures() -> None:
    """Ensure requests with same max tokens remain in one pipeline call."""
    pipeline = DummyPipeline()
    engine = TextVAREngine(pipeline)  # type: ignore[arg-type]
    outputs = engine.generate_batch(
        [
            GenerationParams("a", 8, 0.2, 0.7),
            GenerationParams("b", 8, 0.9, 0.95),
            GenerationParams("c", 8, 1.4, 1.0),
        ]
    )
    assert outputs == ["a|8|0.2|0.7", "b|8|0.9|0.95", "c|8|1.4|1.0"]
    assert len(pipeline.calls) == 1


def test_generate_batch_splits_by_turboquant_flag() -> None:
    """Ensure batching key separates turboquant-enabled and disabled requests."""
    pipeline = DummyPipeline()
    engine = TextVAREngine(pipeline)  # type: ignore[arg-type]
    _ = engine.generate_batch(
        [
            GenerationParams("a", 8, 0.7, 0.9, turboquant_kv=True),
            GenerationParams("b", 8, 0.7, 0.9, turboquant_kv=False),
        ]
    )
    assert len(pipeline.calls) == 2
