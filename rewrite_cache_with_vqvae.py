from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from token_cache import TokenCacheMetadata, load_token_cache_metadata
from vqvae import SemanticTextVQVAE

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure logging for CLI execution.

    Args:
        verbose: Enables DEBUG-level metadata-rich logging when True.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    fmt = "[%(levelname)s] - %(message)s - [%(filename)s:%(lineno)d]" if verbose else "%(message)s"
    logging.basicConfig(level=level, format=fmt)


def _load_vqvae_checkpoint(
    checkpoint_path: Path,
    *,
    vocab_size: int,
    hidden_size: int,
    semantic_tokens: int,
    device: torch.device,
) -> SemanticTextVQVAE:
    """Load trained VQ-VAE model from checkpoint.

    Args:
        checkpoint_path: Path to `.pt` checkpoint.
        vocab_size: Input token vocabulary size for lvl2.
        hidden_size: Hidden size used during training.
        semantic_tokens: Number of semantic codebook entries.
        device: Destination device.

    Returns:
        Loaded `SemanticTextVQVAE` in eval mode.

    Raises:
        ValueError: If checkpoint does not contain a model state dictionary.
    """
    payload = torch.load(checkpoint_path, map_location=device)
    model_state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} has no 'model' state dict.")

    model = SemanticTextVQVAE(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_semantic_tokens=semantic_tokens,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model


def rewrite_cache_with_vqvae(
    token_cache_dir: Path,
    checkpoint_path: Path,
    *,
    level_from: int = 2,
    level_to: int = 0,
    hidden_size: int = 1024,
    semantic_tokens: int = 4096,
    device: str = "cuda",
) -> int:
    """Rewrite cache level with semantic ids produced from another level.

    Args:
        token_cache_dir: Directory containing metadata.json and chunk files.
        checkpoint_path: Trained VQ-VAE checkpoint path.
        level_from: Source level index (default lvl2).
        level_to: Destination level index to overwrite (default lvl0).
        hidden_size: Model hidden size.
        semantic_tokens: Number of VQ codebook entries.
        device: Device selection, `cuda` or `cpu`.

    Returns:
        Number of processed entries.

    Raises:
        ValueError: If metadata/chunk content is invalid.
    """
    metadata_path = token_cache_dir / "metadata.json"
    metadata = load_token_cache_metadata(metadata_path)

    if not (0 <= level_from < len(metadata.level_lengths)):
        raise ValueError(f"Invalid source level index: {level_from}.")
    if not (0 <= level_to < len(metadata.level_lengths)):
        raise ValueError(f"Invalid destination level index: {level_to}.")

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _load_vqvae_checkpoint(
        checkpoint_path,
        vocab_size=int(metadata.level_vocab_sizes[level_from]),
        hidden_size=hidden_size,
        semantic_tokens=semantic_tokens,
        device=dev,
    )

    chunk_paths = sorted(token_cache_dir.glob("tokens_chunk_*.pt"))
    if not chunk_paths:
        raise ValueError(f"No chunk files found under: {token_cache_dir}")

    processed_entries = 0
    with torch.inference_mode():
        for chunk_path in chunk_paths:
            payload = torch.load(chunk_path, map_location="cpu")
            entries = payload.get("entries") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise ValueError(f"Chunk {chunk_path} has invalid 'entries' payload.")

            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(f"Chunk {chunk_path} contains non-dict entry.")
                tokens = entry.get("tokens")
                if not isinstance(tokens, list) or len(tokens) != len(metadata.level_lengths):
                    raise ValueError(f"Chunk {chunk_path} entry contains invalid token levels.")

                source_tokens = torch.as_tensor(tokens[level_from], dtype=torch.long).unsqueeze(0).to(dev)
                semantic_idx, _ = model.encode_sentence(source_tokens, padding_mask=source_tokens.eq(0))

                semantic_ids = semantic_idx.view(-1).to(dtype=torch.long, device="cpu")
                expected_len = int(metadata.level_lengths[level_to])
                if semantic_ids.numel() == 1 and expected_len > 1:
                    semantic_ids = semantic_ids.repeat(expected_len)
                if semantic_ids.numel() != expected_len:
                    raise ValueError(
                        f"Produced semantic token length ({semantic_ids.numel()}) does not match "
                        f"target level length ({expected_len})."
                    )

                tokens[level_to] = semantic_ids
                processed_entries += 1

            torch.save(payload, chunk_path)
            LOGGER.debug("Rewritten chunk: %s", chunk_path)

    return processed_entries


def main() -> None:
    """Parse CLI arguments and execute cache rewrite flow."""
    parser = argparse.ArgumentParser(description="Rewrite lvl0 cache tokens using VQ-VAE semantic ids from lvl2")
    parser.add_argument("--token-cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--level-from", type=int, default=2)
    parser.add_argument("--level-to", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--semantic-tokens", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    rewritten = rewrite_cache_with_vqvae(
        token_cache_dir=args.token_cache_dir,
        checkpoint_path=args.checkpoint,
        level_from=args.level_from,
        level_to=args.level_to,
        hidden_size=args.hidden_size,
        semantic_tokens=args.semantic_tokens,
        device=args.device,
    )
    print(f"Rewritten entries: {rewritten}")


if __name__ == "__main__":
    main()
