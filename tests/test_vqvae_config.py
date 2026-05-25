from pathlib import Path

import pytest

from src.vqvae.training.config import VQVAEConfigError, load_vqvae_train_config


def test_load_vqvae_train_config_success(tmp_path: Path) -> None:
    config_path = tmp_path / 'vqvae.json'
    config_path.write_text(
        '{"output":"out.pt","token_cache_dir":"cache","steps":7,"num_semantic_tokens":123,"semantic_sequence_length":4,"pad_token_id":9,"max_position_embeddings":777,"gradient_accumulation_steps":2}',
        encoding='utf-8',
    )

    cfg = load_vqvae_train_config(config_path)

    assert cfg.output == Path('out.pt')
    assert cfg.token_cache_dir == Path('cache')
    assert cfg.steps == 7
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.num_semantic_tokens == 123
    assert cfg.semantic_sequence_length == 4
    assert cfg.pad_token_id == 9
    assert cfg.max_position_embeddings == 777


def test_load_vqvae_train_config_missing_required_fields(tmp_path: Path) -> None:
    config_path = tmp_path / 'vqvae.json'
    config_path.write_text('{"output":"x.pt"}', encoding='utf-8')

    with pytest.raises(VQVAEConfigError):
        load_vqvae_train_config(config_path)


def test_load_vqvae_train_config_supports_legacy_semantic_tokens_key(tmp_path: Path) -> None:
    config_path = tmp_path / "vqvae_legacy.json"
    config_path.write_text(
        "{\"output\":\"out.pt\",\"token_cache_dir\":\"cache\",\"semantic_tokens\":321}",
        encoding="utf-8",
    )

    cfg = load_vqvae_train_config(config_path)

    assert cfg.num_semantic_tokens == 321
