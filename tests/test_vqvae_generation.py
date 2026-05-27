import torch
from src.vqvae.model import SemanticTextVQVAE


def test_decode_from_semantic_indices_returns_expected_shape() -> None:
    model = SemanticTextVQVAE(
        vocab_size=128,
        hidden_size=32,
        num_semantic_tokens=64,
        semantic_sequence_length=2,
        max_position_embeddings=32,
    ).eval()
    semantic_indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

    generated = model.decode_from_semantic_indices(
        semantic_indices,
        max_length=6,
        bos_token_id=1,
        eos_token_id=None,
        temperature=0.9,
        top_p=0.95,
    )

    assert tuple(generated.shape) == (2, 6)
    assert torch.all(generated[:, 0] == 1)


def test_decoder_cache_is_built_for_incremental_step() -> None:
    model = SemanticTextVQVAE(
        vocab_size=64,
        hidden_size=32,
        num_semantic_tokens=32,
        semantic_sequence_length=2,
        max_position_embeddings=16,
    ).eval()
    memory = torch.randn(1, 2, 32)
    first_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    first_pos = model._position_ids(first_tokens.size(1), first_tokens.device)
    first_emb = model.embedding(first_tokens) + model.pos_embedding(first_pos)
    _, cache = model._run_decoder(tgt_emb=first_emb, memory=memory, incremental=False)

    next_tokens = torch.tensor([[3]], dtype=torch.long)
    next_pos = model._position_ids(3, next_tokens.device)[:, -1:]
    next_emb = model.embedding(next_tokens) + model.pos_embedding(next_pos)
    decoded_step, updated_cache = model._run_decoder(
        tgt_emb=next_emb,
        memory=memory,
        past_key_values=cache,
        incremental=True,
    )

    assert decoded_step.shape == (1, 1, 32)
    assert len(updated_cache) == len(model.decoder_layers)


def test_decode_uses_kv_cache_ring_buffer(monkeypatch) -> None:
    model = SemanticTextVQVAE(
        vocab_size=64,
        hidden_size=32,
        num_semantic_tokens=32,
        semantic_sequence_length=2,
        max_position_embeddings=16,
    ).eval()
    semantic_indices = torch.tensor([[1, 2]], dtype=torch.long)
    update_calls: list[int] = []

    original_update = model.decode_from_semantic_indices.__globals__["KVCacheRingBuffer"].update

    def _tracked_update(self, present_key_values, *, skip_write=False):  # noqa: ANN001, ANN003
        _ = skip_write
        update_calls.append(len(present_key_values))
        return original_update(self, present_key_values)

    monkeypatch.setattr(
        model.decode_from_semantic_indices.__globals__["KVCacheRingBuffer"],
        "update",
        _tracked_update,
    )

    _ = model.decode_from_semantic_indices(
        semantic_indices,
        max_length=4,
        bos_token_id=1,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
    )
    assert len(update_calls) == 3
    assert all(call_size == len(model.decoder_layers) for call_size in update_calls)


def test_apply_semantic_padding_mask_zeroes_padded_positions() -> None:
    model = SemanticTextVQVAE(
        vocab_size=64,
        hidden_size=32,
        num_semantic_tokens=32,
        semantic_sequence_length=2,
        max_position_embeddings=16,
    ).eval()
    semantic_states = torch.ones((1, 2, 32), dtype=torch.float32)
    semantic_padding_mask = torch.tensor([[False, True]])

    masked = model._apply_semantic_padding_mask(semantic_states, semantic_padding_mask)
    assert torch.allclose(masked[:, 0], torch.ones((1, 32)))
    assert torch.allclose(masked[:, 1], torch.zeros((1, 32)))


def test_decode_applies_semantic_padding_mask(monkeypatch) -> None:
    model = SemanticTextVQVAE(
        vocab_size=64,
        hidden_size=32,
        num_semantic_tokens=32,
        semantic_sequence_length=2,
        semantic_pad_token_id=31,
        max_position_embeddings=16,
    ).eval()
    semantic_indices = torch.tensor([[1, 31]], dtype=torch.long)
    observed_masks: list[torch.Tensor] = []

    original_apply = model._apply_semantic_padding_mask

    def _capture_apply(
        semantic_states: torch.Tensor,
        semantic_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if semantic_padding_mask is not None:
            observed_masks.append(semantic_padding_mask.detach().clone())
        return original_apply(semantic_states, semantic_padding_mask)

    monkeypatch.setattr(model, "_apply_semantic_padding_mask", _capture_apply)
    _ = model.decode_from_semantic_indices(
        semantic_indices,
        max_length=3,
        bos_token_id=1,
        temperature=1.0,
        top_p=1.0,
    )

    assert len(observed_masks) >= 1
    assert torch.equal(observed_masks[0], torch.tensor([[False, True]]))


def test_decode_passes_turboquant_config_to_ring_buffer(monkeypatch) -> None:
    model = SemanticTextVQVAE(
        vocab_size=64,
        hidden_size=32,
        num_semantic_tokens=32,
        semantic_sequence_length=2,
        max_position_embeddings=16,
        use_turboquant_kv=True,
        turboquant_key_bits=3,
        turboquant_value_bits=4,
        turboquant_qjl_residual_scale=0.25,
    ).eval()
    observed: dict[str, object] = {}
    original_init = model.decode_from_semantic_indices.__globals__["KVCacheRingBuffer"].__init__

    def _capture_init(self, max_window, turboquant_config=None):  # noqa: ANN001, ANN003
        observed["max_window"] = max_window
        observed["turbo_cfg"] = turboquant_config
        original_init(self, max_window=max_window, turboquant_config=turboquant_config)

    monkeypatch.setattr(
        model.decode_from_semantic_indices.__globals__["KVCacheRingBuffer"],
        "__init__",
        _capture_init,
    )
    _ = model.decode_from_semantic_indices(
        torch.tensor([[1, 2]], dtype=torch.long),
        max_length=4,
        bos_token_id=1,
        top_p=1.0,
        temperature=1.0,
    )

    turbo_cfg = observed["turbo_cfg"]
    assert observed["max_window"] == 4
    assert turbo_cfg is not None
    assert turbo_cfg.key_bits == 3
    assert turbo_cfg.value_bits == 4
    assert turbo_cfg.qjl_residual_scale == 0.25


def test_model_exposes_rotary_and_checkpointing_flags() -> None:
    model = SemanticTextVQVAE(
        vocab_size=64,
        hidden_size=32,
        num_semantic_tokens=32,
        semantic_sequence_length=2,
        max_position_embeddings=16,
        gradient_checkpointing=True,
        use_rotary_embeddings=True,
    )
    assert model.gradient_checkpointing is True
    assert model.use_rotary_embeddings is True
