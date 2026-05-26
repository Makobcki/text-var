"""CLI utility to validate VQ-VAE encode/quantize/decode cycle."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.vqvae.model import SemanticTextVQVAE

LOGGER = logging.getLogger(__name__)
CONSOLE = Console()


class VQVAEInferenceError(RuntimeError):
    """Raised when VQ-VAE inference cannot be completed."""


@dataclass(frozen=True)
class InferConfig:
    """Configuration for VQ-VAE inference CLI.

    Args:
        checkpoint: Path to VQ-VAE checkpoint.
        input_text: Input text for roundtrip inference.
        tokenizer: Optional path to tokenizer JSON.
        device: Torch device string.
        max_length: Maximum tokenized sequence length.
    """

    checkpoint: Path
    input_text: str
    tokenizer: str | None
    device: str
    max_length: int


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser.

    Returns:
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(description="VQ-VAE roundtrip inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input text")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path (tokenizer.json) or pretrained tokenizer name (e.g., gpt2)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--max-length", type=int, default=128, help="Max input token length")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure logging format and level.

    Args:
        verbose: Enables DEBUG log level when True.
    """

    level = logging.DEBUG if verbose else logging.WARNING
    message_format = (
        "[%(levelname)s] - %(message)s - [%(filename)s:%(lineno)d]" if verbose else "%(message)s"
    )
    logging.basicConfig(level=level, format=message_format)


def _load_tokenizer(tokenizer_ref: str | None) -> PreTrainedTokenizerFast:
    """Load tokenizer for text encoding and decoding.

    Args:
        tokenizer_ref: Tokenizer JSON file path or pretrained tokenizer name.

    Returns:
        Loaded fast tokenizer.

    Raises:
        FileNotFoundError: If tokenizer path string points to a missing file.
        VQVAEInferenceError: If tokenizer cannot be initialized.
    """

    if tokenizer_ref is None:
        raise VQVAEInferenceError("--tokenizer is required for roundtrip text decoding.")
    path_candidate = Path(tokenizer_ref)
    if path_candidate.exists():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path_candidate))
    elif path_candidate.suffix:
        raise FileNotFoundError(f"Tokenizer file not found: {path_candidate}")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, use_fast=True)
        except OSError as error:
            raise VQVAEInferenceError(
                f"Failed to load tokenizer '{tokenizer_ref}' as a pretrained tokenizer."
            ) from error
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise VQVAEInferenceError(
                f"Tokenizer '{tokenizer_ref}' does not provide a fast tokenizer implementation."
            )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def _load_model(checkpoint_path: Path, device: torch.device) -> SemanticTextVQVAE:
    """Load VQ-VAE model from checkpoint.

    Args:
        checkpoint_path: Path to serialized checkpoint.
        device: Target torch device.

    Returns:
        Initialized VQ-VAE model.

    Raises:
        FileNotFoundError: If checkpoint path does not exist.
        VQVAEInferenceError: If checkpoint payload is invalid.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise VQVAEInferenceError("Checkpoint payload must be a dictionary.")

    state_dict = payload.get("model", payload)
    model_config = payload.get("model_config", {})

    if not isinstance(state_dict, dict):
        raise VQVAEInferenceError("Checkpoint model state must be a dictionary.")
    if not isinstance(model_config, dict):
        raise VQVAEInferenceError("Checkpoint model_config must be a dictionary.")

    # --- ИСПРАВЛЕНИЕ: Очищаем ключи от префикса torch.compile ---
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            clean_key = key[len("_orig_mod.") :]
        else:
            clean_key = key
        clean_state_dict[clean_key] = value
    # -----------------------------------------------------------

    model = SemanticTextVQVAE(**model_config)

    # Используем очищенный словарь весов
    model.load_state_dict(clean_state_dict)
    return model.to(device).eval()


def run_roundtrip(config: InferConfig) -> str:
    """Run encode-quantize-decode cycle and return decoded text.

    Args:
        config: Immutable inference settings.

    Returns:
        Reconstructed text string.

    Raises:
        VQVAEInferenceError: On invalid runtime assumptions.
    """

    device = torch.device(config.device)
    tokenizer = _load_tokenizer(config.tokenizer)
    model = _load_model(config.checkpoint, device)

    encoded = tokenizer(
        [config.input_text],
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
        padding=True,
    )
    input_ids = encoded["input_ids"].to(device)
    padding_mask = ~encoded["attention_mask"].bool().to(device)

    with torch.no_grad():
        semantic_indices, _ = model.encode_sentence(input_ids, padding_mask=padding_mask)
        decoded_ids = model.decode_from_semantic_indices(
            semantic_indices,
            max_length=input_ids.shape[1],
            bos_token_id=int(tokenizer.bos_token_id or input_ids[0, 0].item()),
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0,
        )

    decoded_text = tokenizer.batch_decode(decoded_ids.tolist(), skip_special_tokens=True)[0]
    return decoded_text


def main() -> None:
    """Execute CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    config = InferConfig(
        checkpoint=args.checkpoint,
        input_text=args.input,
        tokenizer=args.tokenizer,
        device=args.device,
        max_length=args.max_length,
    )

    try:
        with CONSOLE.status("Running VQ-VAE encode/quant/decode..."):
            result_text = run_roundtrip(config)
    except KeyboardInterrupt:
        CONSOLE.print("[yellow]Interrupted by user.[/yellow]")
        return
    CONSOLE.print(result_text)


if __name__ == "__main__":
    main()
