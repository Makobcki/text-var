from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from config import VARConfig
from generator import hybrid_cascade_decode
from model import VARTransformer
from vqvae import SemanticTextVQVAE


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
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text continuation from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum output token length after latent decoding.

        Returns:
            Decoded text string.
        """

        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_bpe_len,
        )
        bpe_tokens = encoded["input_ids"].to(self._device)
        padding_mask = ~encoded["attention_mask"].bool().to(self._device)

        latent_context, _ = self._vqvae.encode_sentence(bpe_tokens, padding_mask=padding_mask)
        semantic_prefix = latent_context.view(latent_context.shape[0], -1).long()
        generated_levels = hybrid_cascade_decode(
            self._var_model,
            batch_size=bpe_tokens.shape[0],
            device=self._device,
            prefix_inputs=[semantic_prefix],
        )

        decoded_bpe = self._vqvae.decode_from_semantic_indices(
            generated_levels[0],
            max_length=max_new_tokens,
            bos_token_id=self._tokenizer.bos_token_id or self._tokenizer.eos_token_id or 0,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        return self._tokenizer.decode(decoded_bpe[0].tolist(), skip_special_tokens=True)

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
        return PreTrainedTokenizerFast(tokenizer_file=str(path))

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

        payload = torch.load(path, map_location="cpu")
        state_dict: dict[str, Any] = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

        vocab_size = int(payload.get("metadata", {}).get("level_vocab_sizes", [32000])[0]) if isinstance(payload, dict) else 32000
        model = SemanticTextVQVAE(vocab_size=vocab_size)
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

        payload = torch.load(path, map_location="cpu")
        state_dict: dict[str, Any] = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

        cfg = VARConfig()
        if isinstance(payload, dict) and "model_config" in payload:
            cfg = VARConfig.from_dict(dict(payload["model_config"]))
        model = VARTransformer(cfg)
        model.load_state_dict(state_dict, strict=False)
        for param in model.parameters():
            param.requires_grad = False
        return model
