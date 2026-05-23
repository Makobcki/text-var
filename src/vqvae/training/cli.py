"""CLI helpers for VQ-VAE training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build VQ-VAE training CLI parser.

    Returns:
        Parser for VQ-VAE training command.
    """

    parser = argparse.ArgumentParser(description="Pretrain SemanticTextVQVAE")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--token-cache-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--semantic-tokens", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--level-index", type=int, default=2)
    return parser
