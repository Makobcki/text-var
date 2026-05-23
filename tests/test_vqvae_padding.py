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
