"""CLI helpers for VQ-VAE training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build VQ-VAE training CLI parser.

    Returns:
        Parser for VQ-VAE training command.
    """

    parser = argparse.ArgumentParser(description="Pretrain SemanticTextVQVAE")
    parser.add_argument('--config', type=str, default='configs/vqvae_train.json')
    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('--token-cache-dir', type=str, required=False)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--vocab-size', type=int, default=0)
    parser.add_argument('--hidden-size', type=int, default=1024)
    parser.add_argument('--num-semantic-tokens', type=int, default=4096)
    parser.add_argument('--semantic-sequence-length', type=int, default=1)
    parser.add_argument('--pad-token-id', type=int, default=0)
    parser.add_argument('--max-position-embeddings', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--level-index', type=int, default=2)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    return parser
