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
        level_lengths=(2, 2, 8),
        codebook_dim=64,
        max_token_length=12,
    )
    save_token_cache_metadata(cache_dir / "metadata.json", metadata)

    chunk_path = cache_dir / "tokens_chunk_00000.safetensors"
    original_lvl0 = torch.tensor([7, 7], dtype=torch.long)
    lvl2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    from safetensors.torch import save_file
    import json
    save_file(
        {
            "tokens_level_0": torch.stack([original_lvl0.clone()]),
            "tokens_level_1": torch.stack([torch.tensor([0, 1])]),
            "tokens_level_2": torch.stack([lvl2.clone()]),
        },
        chunk_path,
    )
    with open(chunk_path.with_suffix(".json"), "w") as f:
        json.dump([{"id": "sample-1"}], f)

    model = SemanticTextVQVAE(vocab_size=32, hidden_size=32, num_semantic_tokens=16)
    checkpoint_path = tmp_path / "vqvae.safetensors"
    from safetensors.torch import save_model
    save_model(model, checkpoint_path)

    rewritten = rewrite_cache_with_vqvae(
        token_cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        hidden_size=32,
        semantic_tokens=16,
        device="cpu",
    )
    assert rewritten == 1

    from safetensors.torch import load_file
    payload = load_file(chunk_path)
    new_lvl0 = torch.as_tensor(payload["tokens_level_0"][0], dtype=torch.long)
    assert new_lvl0.shape == original_lvl0.shape

    expected_idx, _ = model.encode_sentence(lvl2.unsqueeze(0), padding_mask=lvl2.unsqueeze(0).eq(0))
    assert torch.equal(new_lvl0, expected_idx.view(-1).to(torch.long))


def test_rewrite_multitoken_semantic_target(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_multi"
    cache_dir.mkdir(parents=True)

    metadata = TokenCacheMetadata(
        kind="vq",
        level_vocab_sizes=(16, 16, 32),
        level_lengths=(2, 2, 8),
        codebook_dim=64,
        max_token_length=12,
    )
    save_token_cache_metadata(cache_dir / "metadata.json", metadata)

    chunk_path = cache_dir / "tokens_chunk_00000.safetensors"
    lvl0 = torch.tensor([7, 7], dtype=torch.long)
    lvl2 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    
    from safetensors.torch import save_file, load_file
    import json
    save_file(
        {
            "tokens_level_0": torch.stack([lvl0]),
            "tokens_level_1": torch.stack([torch.tensor([0, 1])]),
            "tokens_level_2": torch.stack([lvl2]),
        },
        chunk_path,
    )
    with open(chunk_path.with_suffix(".json"), "w") as f:
        json.dump([{"id": "sample-1"}], f)

    model = SemanticTextVQVAE(vocab_size=32, hidden_size=32, num_semantic_tokens=16, semantic_sequence_length=2)  # noqa: E501
    checkpoint_path = tmp_path / "vqvae_multi.safetensors"
    from safetensors.torch import save_model
    save_model(model, checkpoint_path)

    rewritten = rewrite_cache_with_vqvae(
        token_cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        hidden_size=32,
        semantic_tokens=16,
        device="cpu",
    )
    assert rewritten == 1
    payload = load_file(chunk_path)
    new_lvl0 = payload["tokens_level_0"][0]
    assert tuple(new_lvl0.shape) == (2,)
