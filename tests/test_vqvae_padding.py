import torch
from src.vqvae.model import SemanticTextVQVAE, VectorQuantizer


def test_resolve_padding_mask_infers_from_pad_token_id() -> None:
    model = SemanticTextVQVAE(vocab_size=32, hidden_size=8, num_semantic_tokens=8, pad_token_id=99)
    tokens = torch.tensor([[5, 99, 7, 99]], dtype=torch.long)

    mask = model._resolve_padding_mask(tokens, padding_mask=None)

    assert torch.equal(mask, torch.tensor([[False, True, False, True]]))


def test_resolve_padding_mask_merges_provided_and_inferred_masks() -> None:
    model = SemanticTextVQVAE(vocab_size=32, hidden_size=8, num_semantic_tokens=8, pad_token_id=0)
    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    provided = torch.tensor([[False, False, True, False]], dtype=torch.bool)

    mask = model._resolve_padding_mask(tokens, padding_mask=provided)

    assert torch.equal(mask, torch.tensor([[True, False, True, False]]))


def test_encode_sentence_preserves_semantic_sequence_length() -> None:
    model = SemanticTextVQVAE(
        vocab_size=32,
        hidden_size=8,
        num_semantic_tokens=16,
        semantic_sequence_length=4,
        pad_token_id=0,
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    semantic_idx, _ = model.encode_sentence(tokens, padding_mask=tokens.eq(0))

    assert semantic_idx.shape == (1, 4)


def test_position_ids_raise_for_sequences_longer_than_limit() -> None:
    model = SemanticTextVQVAE(
        vocab_size=32,
        hidden_size=8,
        num_semantic_tokens=16,
        max_position_embeddings=4,
    )

    try:
        _ = model._position_ids(seq_len=5, device=torch.device("cpu"))
    except ValueError as exc:
        assert "max_position_embeddings" in str(exc)
        return

    raise AssertionError("Expected ValueError for sequence length overflow.")


def test_decode_from_semantic_indices_respects_position_limit() -> None:
    model = SemanticTextVQVAE(
        vocab_size=64,
        hidden_size=16,
        num_semantic_tokens=32,
        max_position_embeddings=3,
    ).eval()
    semantic_indices = torch.tensor([[1]], dtype=torch.long)

    try:
        _ = model.decode_from_semantic_indices(
            semantic_indices,
            max_length=5,
            bos_token_id=1,
            top_p=1.0,
        )
    except ValueError as exc:
        assert "max_position_embeddings" in str(exc)
        return

    raise AssertionError("Expected ValueError for generation beyond position limit.")


def test_pool_semantic_tokens_ignores_padded_positions() -> None:
    model = SemanticTextVQVAE(
        vocab_size=32,
        hidden_size=2,
        num_semantic_tokens=8,
        semantic_sequence_length=2,
        pad_token_id=0,
    )
    encoded = torch.tensor(
        [[[2.0, 4.0], [8.0, 10.0], [100.0, 100.0], [100.0, 100.0]]],
        dtype=torch.float32,
    )
    padding_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)

    pooled = model._pool_semantic_tokens(encoded, padding_mask=padding_mask)

    expected = torch.tensor([[[5.0, 7.0], [0.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose(pooled, expected, atol=1e-5)


def test_pool_semantic_padding_mask_marks_all_padded_windows() -> None:
    model = SemanticTextVQVAE(
        vocab_size=32,
        hidden_size=2,
        num_semantic_tokens=8,
        semantic_sequence_length=2,
        pad_token_id=0,
    )
    padding_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)

    semantic_mask = model._pool_semantic_padding_mask(padding_mask)

    assert torch.equal(semantic_mask, torch.tensor([[False, True]]))


def test_forward_ignores_padding_tokens_in_reconstruction_loss() -> None:
    model = SemanticTextVQVAE(
        vocab_size=16,
        hidden_size=8,
        num_semantic_tokens=8,
        semantic_sequence_length=2,
        pad_token_id=0,
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    padding_mask = tokens.eq(0)

    logits, total_loss = model(tokens, padding_mask=padding_mask)
    recon_targets = tokens[:, 1:]
    loss_mask = padding_mask[:, 1:]
    masked_targets = recon_targets.masked_fill(loss_mask, 0)
    expected_recon = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        masked_targets.reshape(-1),
        ignore_index=0,
        reduction="mean",
    )

    expected_total = expected_recon + model.quantizer.last_commitment_loss
    assert torch.isfinite(total_loss)
    assert torch.allclose(total_loss, expected_total, atol=1e-5)


def test_vector_quantizer_chunked_distances_match_direct_cdist() -> None:
    quantizer = VectorQuantizer(num_embeddings=7, embedding_dim=3)
    flat_inputs = torch.tensor(
        [[0.1, 0.2, 0.3], [1.0, -0.1, 0.5], [-0.2, 0.4, -0.7]],
        dtype=torch.float32,
    )

    expected = torch.cdist(flat_inputs, quantizer.codebook.weight, p=2.0)
    actual = quantizer._compute_distances_chunked(flat_inputs, chunk_size=2)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_vector_quantizer_chunked_distances_reject_invalid_chunk_size() -> None:
    quantizer = VectorQuantizer(num_embeddings=5, embedding_dim=2)
    flat_inputs = torch.randn(4, 2)

    try:
        _ = quantizer._compute_distances_chunked(flat_inputs, chunk_size=0)
    except ValueError as exc:
        assert "chunk_size must be >= 1." in str(exc)
        return

    raise AssertionError("Expected ValueError for invalid chunk_size.")


def test_pool_to_semantic_length_handles_non_contiguous_inputs() -> None:
    model = SemanticTextVQVAE(
        vocab_size=32,
        hidden_size=4,
        num_semantic_tokens=8,
        semantic_sequence_length=2,
        pad_token_id=0,
    )
    base = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    non_contiguous = base.transpose(1, 2)
    assert not non_contiguous.is_contiguous()

    pooled = model._pool_to_semantic_length(non_contiguous)

    assert pooled.shape == (2, 2, 4)


def test_vector_quantizer_forward_accepts_non_contiguous_inputs() -> None:
    quantizer = VectorQuantizer(num_embeddings=11, embedding_dim=4)
    contiguous = torch.randn(2, 4, 4, dtype=torch.float32)
    non_contiguous = contiguous.transpose(1, 2)
    assert not non_contiguous.is_contiguous()

    quantized, vq_loss, indices = quantizer(non_contiguous)

    assert quantized.shape == non_contiguous.shape
    assert indices.shape == non_contiguous.shape[:-1]
    assert torch.isfinite(vq_loss)


def test_build_causal_mask_shape_and_finite_entries() -> None:
    model = SemanticTextVQVAE(vocab_size=32, hidden_size=8, num_semantic_tokens=8, pad_token_id=0)

    mask = model._build_causal_mask(seq_len=4, device=torch.device("cpu"))

    assert mask.shape == (4, 4)
    assert torch.isfinite(mask.diag()).all()
    assert torch.isinf(mask[0, 1])
