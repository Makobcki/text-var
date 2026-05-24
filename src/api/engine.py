"""API service layer for TextVAR inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    turboquant_kv: bool = False


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
            turboquant_kv=params.turboquant_kv,
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
        return self._generate_with_grouped_sampling(params_list)

    def _generate_with_grouped_sampling(self, params_list: list[GenerationParams]) -> list[str]:
        """Generate a mixed-parameter batch by grouping requests.

        Args:
            params_list: Ordered generation parameters.

        Returns:
            Generated text preserving the original request ordering.
        """
        grouped: Dict[int, List[Tuple[int, GenerationParams]]] = {}
        for index, params in enumerate(params_list):
            key = (params.max_tokens, params.turboquant_kv)
            grouped.setdefault(key, []).append((index, params))

        results: list[str] = [""] * len(params_list)
        for group_key, grouped_items in grouped.items():
            max_tokens, turboquant_kv = group_key
            prompts = [item.prompt for _, item in grouped_items]
            temperatures = [item.temperature for _, item in grouped_items]
            top_ps = [item.top_p for _, item in grouped_items]
            outputs = self._pipeline.generate_batch(
                prompts,
                max_new_tokens=max_tokens,
                per_item_temperatures=temperatures,
                per_item_top_ps=top_ps,
                turboquant_kv=turboquant_kv,
            )
            for output_idx, (original_idx, _) in enumerate(grouped_items):
                results[original_idx] = outputs[output_idx]
        return results
