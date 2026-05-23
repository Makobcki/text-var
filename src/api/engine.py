"""API service layer for TextVAR inference."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.pipeline import TextVARPipeline


@dataclass(frozen=True)
class GenerationParams:
    """Generation settings for TextVAR completion.

    Args:
        prompt: Input prompt text.
        max_tokens: Maximum number of output tokens.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
    """

    prompt: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0


class TextVAREngine:
    """Business-logic service for API generation operations.

    Args:
        pipeline: Ready-to-use inference pipeline.
    """

    def __init__(self, pipeline: TextVARPipeline) -> None:
        self._pipeline = pipeline

    def generate(self, params: GenerationParams) -> str:
        """Generate output text.

        Args:
            params: Generation parameters.

        Returns:
            Generated text.
        """

        return self._pipeline.generate(
            params.prompt,
            max_new_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
        )

    def generate_batch(self, params_list: list[GenerationParams]) -> list[str]:
        """Generate outputs for a batch of prompts.

        Args:
            params_list: Ordered generation parameters.

        Returns:
            Generated text strings preserving input order.

        Raises:
            ValueError: If params_list is empty.
        """
        if not params_list:
            raise ValueError("Prompt list cannot be empty.")
        prompts = [params.prompt for params in params_list]
        max_tokens = params_list[0].max_tokens
        if any(params.max_tokens != max_tokens for params in params_list):
            raise ValueError("All batch items must use the same max_tokens.")
        temperature = params_list[0].temperature
        top_p = params_list[0].top_p
        if any(params.temperature != temperature for params in params_list):
            raise ValueError("All batch items must use the same temperature.")
        if any(params.top_p != top_p for params in params_list):
            raise ValueError("All batch items must use the same top_p.")
        return self._pipeline.generate_batch(
            prompts,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
