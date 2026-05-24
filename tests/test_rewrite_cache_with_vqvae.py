from __future__ import annotations

from pathlib import Path

import torch
from src.data.token_cache import TokenCacheMetadata, save_token_cache_metadata
from src.data.utils.rewrite_cache_with_vqvae import rewrite_cache_with_vqvae
from src.vqvae.model import SemanticTextVQVAE


def test_rewrite_lvl0_from_lvl2(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)

    metadata = TokenCacheMetadata(
        kind="vq",
        level_vocab_sizes=(8, 16, 32),
        level_lengths=(1, 2, 4),
        codebook_dim=64,
        max_token_length=7,
    )
    save_token_cache_metadata(cache_dir / "metadata.json", metadata)

    chunk_path = cache_dir / "tokens_chunk_00000.pt"
    original_lvl0 = torch.tensor([7], dtype=torch.long)
    lvl2 = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    torch.save(
        {
            "metadata": metadata.to_dict(),
            "entries": [
                {
                    "id": "sample-1",
                    "tokens": [original_lvl0.clone(), torch.tensor([0, 1]), lvl2.clone()],
                }
            ],
        },
        chunk_path,
    )

    model = SemanticTextVQVAE(vocab_size=32, hidden_size=32, num_semantic_tokens=16)
    checkpoint_path = tmp_path / "vqvae.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    rewritten = rewrite_cache_with_vqvae(
        token_cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        hidden_size=32,
        semantic_tokens=16,
        device="cpu",
    )
    assert rewritten == 1

    payload = torch.load(chunk_path, map_location="cpu")
    entry = payload["entries"][0]
    new_lvl0 = torch.as_tensor(entry["tokens"][0], dtype=torch.long)
    assert new_lvl0.shape == original_lvl0.shape
    assert 0 <= int(new_lvl0.item()) < 16

    expected_idx, _ = model.encode_sentence(lvl2.unsqueeze(0), padding_mask=lvl2.unsqueeze(0).eq(0))
    assert int(new_lvl0.item()) == int(expected_idx.view(-1).item())


def test_rewrite_multitoken_semantic_target(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_multi"
    cache_dir.mkdir(parents=True)

    metadata = TokenCacheMetadata(
        kind="vq",
        level_vocab_sizes=(16, 16, 32),
        level_lengths=(2, 2, 4),
        codebook_dim=64,
        max_token_length=7,
    )
    save_token_cache_metadata(cache_dir / "metadata.json", metadata)

    chunk_path = cache_dir / "tokens_chunk_00000.pt"
    lvl0 = torch.tensor([7, 7], dtype=torch.long)
    lvl2 = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    torch.save({"metadata": metadata.to_dict(), "entries": [{"id": "sample-1", "tokens": [lvl0, torch.tensor([0, 1]), lvl2]}]}, chunk_path)

    model = SemanticTextVQVAE(vocab_size=32, hidden_size=32, num_semantic_tokens=16, semantic_sequence_length=2)
    checkpoint_path = tmp_path / "vqvae_multi.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    rewritten = rewrite_cache_with_vqvae(
        token_cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        hidden_size=32,
        semantic_tokens=16,
        device="cpu",
    )
    assert rewritten == 1
    payload = torch.load(chunk_path, map_location="cpu")
    new_lvl0 = torch.as_tensor(payload["entries"][0]["tokens"][0], dtype=torch.long)
    assert tuple(new_lvl0.shape) == (2,)
