"""CLI helpers for VQ-VAE training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build VQ-VAE training CLI parser.

    Returns:
        Parser for VQ-VAE training command.
    """

    parser = argparse.ArgumentParser(description="Pretrain SemanticTextVQVAE")
    parser.add_argument("--config", type=str, default="configs/vqvae_train.json")
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--token-cache-dir", type=str, required=False)
    parser.add_argument("--tensorboard-dir", type=str, default="runs/vqvae")
    parser.add_argument("--resume-from", type=str, required=False, help="Path to checkpoint to resume training from")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=0, help="Train for N epochs (0 means use steps)")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-semantic-tokens", type=int, default=4096)
    parser.add_argument("--semantic-sequence-length", type=int, default=1)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--semantic-pad-token-id", type=int, default=0)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--level-index", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=("bf16", "fp16", "none"))
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adamw8bit", "adafactor"])
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "onecycle"])
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--use-turboquant-kv", action="store_true")
    parser.add_argument("--turboquant-key-bits", type=int, default=4)
    parser.add_argument("--turboquant-value-bits", type=int, default=4)
    parser.add_argument("--turboquant-qjl-residual-scale", type=float, default=0.5)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-unpadding", action="store_true")
    parser.add_argument("--disable-rotary-embeddings", action="store_true")
    parser.add_argument("--word-dropout-prob", type=float, default=0.1)
    parser.add_argument("--encoder-num-heads", type=int, default=8)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--encoder-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--compression-rate", type=int, default=4)
    parser.add_argument("--downsample-num-blocks", type=int, default=2)
    parser.add_argument("--fsq-levels", type=int, nargs="+", default=[8, 8, 8, 8, 8, 8, 8, 8])
    parser.add_argument("--decoder-num-heads", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser
