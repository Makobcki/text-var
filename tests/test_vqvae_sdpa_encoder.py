import torch
from src.vqvae.sdpa_blocks import SDPAEncoder, SDPAEncoderLayer


def test_sdpa_encoder_layer_rejects_invalid_head_config() -> None:
    try:
        _ = SDPAEncoderLayer(hidden=10, num_heads=3, mlp_ratio=2.0)
    except ValueError as exc:
        assert "divisible" in str(exc)
        return
    raise AssertionError("Expected ValueError for incompatible head configuration.")


def test_sdpa_encoder_preserves_shape() -> None:
    encoder = SDPAEncoder(hidden=16, num_heads=4, depth=2, mlp_ratio=2.0, dropout=0.0).eval()
    x = torch.randn(2, 5, 16)
    padding_mask = torch.tensor([[False, False, True, True, True], [False, False, False, False, True]])  # noqa: E501

    output = encoder(x, key_padding_mask=padding_mask)
    assert output.shape == x.shape


def test_sdpa_encoder_accepts_rotary_frequencies() -> None:
    encoder = SDPAEncoder(hidden=16, num_heads=4, depth=1, mlp_ratio=2.0, dropout=0.0).eval()
    x = torch.randn(1, 4, 16)
    rotary_freqs = torch.randn(4, 4)
    output = encoder(x, key_padding_mask=None, rotary_freqs=rotary_freqs)
    assert output.shape == x.shape


def test_sdpa_encoder_padding_mask_blocks_padded_keys() -> None:
    layer = SDPAEncoderLayer(hidden=8, num_heads=2, mlp_ratio=1.0, dropout=0.0).eval()

    mask = torch.tensor([[False, False, True]])

    sdpa_mask = layer._build_padding_mask(mask)
    assert sdpa_mask is not None
    assert bool(sdpa_mask[0, 0, 0, 0].item()) is True
    assert bool(sdpa_mask[0, 0, 0, 2].item()) is False
