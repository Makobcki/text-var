from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from src.var.training.config import VARConfig
from src.var.generator import hybrid_cascade_decode
from src.var.model import VARTransformer
from src.vqvae.model import SemanticTextVQVAE


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the text generation pipeline.

    Args:
        vqvae_path: Path to pretrained VQ-VAE checkpoint.
        var_path: Path to pretrained VAR checkpoint.
        bpe_tokenizer_path: Path to tokenizer.json for fast tokenizer.
        device: Target torch device.
        max_bpe_len: Maximum length for prompt encoding and decoding.
    """

    vqvae_path: Path
    var_path: Path
    bpe_tokenizer_path: Path
    device: str = "cpu"
    max_bpe_len: int = 128


class TextVARPipeline:
    """Inference-only integration pipeline for BPE + VQ-VAE + VAR.

    Args:
        config: PipelineConfig with model/tokenizer artifacts.

    Raises:
        FileNotFoundError: If required artifacts are missing.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._tokenizer = self._load_tokenizer(config.bpe_tokenizer_path)
        self._vqvae = self._load_vqvae(config.vqvae_path).to(self._device).eval()
        self._var_model = self._load_var(config.var_path).to(self._device).eval()

    @property
    def device(self) -> torch.device:
        """Return pipeline torch device.

        Returns:
            Device where all modules are placed.
        """

        return self._device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        turboquant_kv: bool = False,
    ) -> str:
        """Generate text continuation from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum output token length after latent decoding.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Decoded text string.
        """

        generated_texts = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            turboquant_kv=turboquant_kv,
        )
        return generated_texts[0]

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        per_item_temperatures: list[float] | None = None,
        per_item_top_ps: list[float] | None = None,
        turboquant_kv: bool = False,
    ) -> list[str]:
        """Generate text continuations for a batch of prompts.

        Args:
            prompts: Ordered prompt strings for batched generation.
            max_new_tokens: Maximum output token length after latent decoding.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            per_item_temperatures: Optional per-request temperatures.
            per_item_top_ps: Optional per-request top-p values.

        Returns:
            Generated text strings preserving input order.

        Raises:
            ValueError: If prompts is empty.
        """
        if not prompts:
            raise ValueError("Prompt list cannot be empty.")
        if per_item_temperatures is not None and len(per_item_temperatures) != len(prompts):
            raise ValueError("per_item_temperatures length must match prompts length.")
        if per_item_top_ps is not None and len(per_item_top_ps) != len(prompts):
            raise ValueError("per_item_top_ps length must match prompts length.")

        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._config.max_bpe_len,
        )
        bpe_tokens = encoded["input_ids"].to(self._device)
        padding_mask = ~encoded["attention_mask"].bool().to(self._device)

        semantic_indices, _ = self._vqvae.encode_sentence(
            bpe_tokens,
            padding_mask=padding_mask,
        )
        semantic_prefix = semantic_indices.view(semantic_indices.shape[0], -1).long()
        sampling_temperatures = (
            torch.tensor(per_item_temperatures, dtype=torch.float32, device=self._device)
            if per_item_temperatures is not None
            else temperature
        )
        sampling_top_ps = (
            torch.tensor(per_item_top_ps, dtype=torch.float32, device=self._device)
            if per_item_top_ps is not None
            else top_p
        )
        generated_levels = hybrid_cascade_decode(
            self._var_model,
            batch_size=bpe_tokens.shape[0],
            device=self._device,
            prefix_inputs=[semantic_prefix],
            temperature=sampling_temperatures,
            top_p=sampling_top_ps,
            turboquant_kv=turboquant_kv,
        )

        decoded_bpe = self._vqvae.decode_from_semantic_indices(
            generated_levels[0],
            max_length=max_new_tokens,
            bos_token_id=self._tokenizer.bos_token_id or self._tokenizer.eos_token_id or 0,
            eos_token_id=self._tokenizer.eos_token_id,
            temperature=float(temperature),
            top_p=float(top_p),
        )
        return self._tokenizer.batch_decode(decoded_bpe.tolist(), skip_special_tokens=True)

    @staticmethod
    def _load_tokenizer(path: Path) -> PreTrainedTokenizerFast:
        """Load a fast tokenizer from local tokenizer file.

        Args:
            path: Path to tokenizer JSON artifact.

        Returns:
            Initialized tokenizer.

        Raises:
            FileNotFoundError: If tokenizer file does not exist.
        """

        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path))
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def _load_vqvae(self, path: Path) -> SemanticTextVQVAE:
        """Load VQ-VAE checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Loaded SemanticTextVQVAE model in eval mode.

        Raises:
            FileNotFoundError: If checkpoint is missing.
        """

        if not path.exists():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict: dict[str, Any] = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

        if not isinstance(payload, dict):
            raise ValueError("VQ-VAE checkpoint payload must be a dictionary.")
        model_config = payload.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("VQ-VAE checkpoint model_config is required for loading.")
        required_fields = (
            "vocab_size",
            "hidden_size",
            "num_semantic_tokens",
            "semantic_sequence_length",
            "pad_token_id",
        )
        for field_name in required_fields:
            if field_name not in model_config:
                raise ValueError(f"VQ-VAE checkpoint model_config.{field_name} is required for loading.")
        model = SemanticTextVQVAE(
            vocab_size=int(model_config["vocab_size"]),
            hidden_size=int(model_config["hidden_size"]),
            num_semantic_tokens=int(model_config["num_semantic_tokens"]),
            semantic_sequence_length=int(model_config["semantic_sequence_length"]),
            pad_token_id=int(model_config["pad_token_id"]),
        )
        model.load_state_dict(state_dict, strict=False)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _load_var(self, path: Path) -> VARTransformer:
        """Load VAR transformer checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Loaded VAR transformer.

        Raises:
            FileNotFoundError: If checkpoint is missing.
        """

        if not path.exists():
            raise FileNotFoundError(f"VAR checkpoint not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict: dict[str, Any] = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

        if not isinstance(payload, dict):
            raise ValueError("VAR checkpoint payload must be a dictionary.")
        model_config = payload.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError("VAR checkpoint model_config is required for loading.")
        cfg = VARConfig.from_dict(dict(model_config))
        model = VARTransformer(cfg)
        model.load_state_dict(state_dict, strict=False)
        for param in model.parameters():
            param.requires_grad = False
        return model
