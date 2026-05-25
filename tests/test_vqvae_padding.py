import torch

from src.vqvae.model import SemanticTextVQVAE


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
