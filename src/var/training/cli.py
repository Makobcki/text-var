"""CLI helpers for VAR training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build training CLI parser.

    Returns:
        Parser for VAR training command.
    """

    parser = argparse.ArgumentParser(description="Train VAR model")
    parser.add_argument("--config", type=str, required=True, help="Path to train config JSON")
    return parser
