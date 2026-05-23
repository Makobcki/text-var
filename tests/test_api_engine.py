"""Tests for TextVAR API engine batching behavior."""

from src.api.engine import GenerationParams, TextVAREngine


class DummyPipeline:
    """Pipeline stub returning deterministic batched outputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int, float, float]] = []

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        self.calls.append((prompts, max_new_tokens, temperature, top_p))
        return [f"{prompt}|{max_new_tokens}|{temperature}|{top_p}" for prompt in prompts]


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

