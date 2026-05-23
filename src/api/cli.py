"""CLI parsing helpers for API server."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build API server CLI parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="VAR OpenAI-Compatible API Server")
    parser.add_argument("--vqvae-path", type=Path, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--var-path", type=Path, required=True, help="Path to VAR checkpoint")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to BPE tokenizer JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--max-bpe-len", type=int, default=128)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser
