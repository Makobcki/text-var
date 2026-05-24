from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.token_cache import (
    MultiscaleTokenChunkIterableDataset,
    TokenCacheMetadata,
    load_token_entries_from_directory,
)
from src.vqvae.checkpoint import save_vqvae_checkpoint
from src.vqvae.model import SemanticTextVQVAE
from src.vqvae.training.cli import build_parser
from src.vqvae.training.config import VQVAETrainConfig


def _collate_level(level_index: int):
    def collate(batch: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [item["tokens"][level_index] for item in batch]  # type: ignore[index]
        stacked = torch.stack([torch.as_tensor(t, dtype=torch.long) for t in tokens], dim=0)
        padding_mask = stacked.eq(0)
        return stacked, padding_mask

    return collate


def run_training(
    output_path: Path,
    token_cache_dir: Path,
    *,
    steps: int = 500,
    batch_size: int = 8,
    vocab_size: int = 32000,
    hidden_size: int = 1024,
    semantic_tokens: int = 4096,
    lr: float = 3e-4,
    device: str = "cuda",
    level_index: int = 2,
) -> Path:
    chunk_paths, metadata = load_token_entries_from_directory(token_cache_dir)
    if not (0 <= int(level_index) < len(metadata.level_lengths)):
        raise ValueError(f"level-index must be in [0, {len(metadata.level_lengths) - 1}]")

    level_vocab_size = int(metadata.level_vocab_sizes[level_index])
    if vocab_size <= 0:
        vocab_size = level_vocab_size
    elif vocab_size != level_vocab_size:
        print(
            f"[VQVAE] override vocab_size={vocab_size} -> metadata level vocab_size={level_vocab_size}"
        )
        vocab_size = level_vocab_size

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = SemanticTextVQVAE(vocab_size=vocab_size, hidden_size=hidden_size, num_semantic_tokens=semantic_tokens).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    ds = MultiscaleTokenChunkIterableDataset(chunk_paths=chunk_paths, metadata=metadata)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=_collate_level(level_index))

    model.train()
    step = 0
    while step < steps:
        did_progress = False
        for tokens, padding_mask in loader:
            did_progress = True
            tokens = tokens.to(dev, non_blocking=True)
            padding_mask = padding_mask.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(tokens, padding_mask=padding_mask)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f"[VQVAE] step={step}/{steps} loss={float(loss.detach().cpu()):.6f}")
            if step >= steps:
                break

        if not did_progress:
            raise RuntimeError("No valid token entries were loaded from token cache.")

    save_vqvae_checkpoint({
            "model": model.state_dict(),
            "steps": step,
            "model_config": {
                "vocab_size": int(model.vocab_size),
                "hidden_size": int(model.hidden_size),
                "num_semantic_tokens": int(model.quantizer.num_embeddings),
                "semantic_sequence_length": int(model.semantic_sequence_length),
                "pad_token_id": int(model.pad_token_id),
            },
            "metadata": TokenCacheMetadata(
                kind=metadata.kind,
                level_vocab_sizes=(vocab_size,),
                level_lengths=(metadata.level_lengths[level_index],),
                codebook_dim=metadata.codebook_dim,
                max_token_length=metadata.level_lengths[level_index],
            ).to_dict(),
            "source_token_cache": str(token_cache_dir),
            "level_index": int(level_index),
        }, output_path)
    print(f"[VQVAE] checkpoint saved: {output_path}")
    return output_path


def main() -> None:
    """Run VQ-VAE training from CLI options."""
    parser = build_parser()
    args = parser.parse_args()
    cfg = VQVAETrainConfig(
        output=Path(args.output),
        token_cache_dir=Path(args.token_cache_dir),
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        semantic_tokens=args.semantic_tokens,
        lr=args.lr,
        level_index=args.level_index,
    )
    run_training(
        cfg.output,
        cfg.token_cache_dir,
        steps=cfg.steps,
        batch_size=cfg.batch_size,
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        semantic_tokens=cfg.semantic_tokens,
        lr=cfg.lr,
        device=cfg.device,
        level_index=cfg.level_index,
    )


if __name__ == "__main__":
    main()
