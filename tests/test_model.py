import torch

from src.var.training.config import VARConfig
from src.var.model import SDPADecoderLayer, VARTransformer


class _CaptureDecoderLayer(SDPADecoderLayer):
    def __init__(self, hidden: int, num_heads: int) -> None:
        super().__init__(hidden=hidden, num_heads=num_heads, mlp_ratio=1.0, dropout=0.0)
        self.masks: list[torch.Tensor | None] = []
        self.causal_flags: list[bool] = []

    def forward(
        self,
        *,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_is_causal: bool = False,
        rotary_freqs_tgt: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        cross_kv_memory: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        del memory, rotary_freqs_tgt, cross_kv_memory
        self.masks.append(self_attn_mask)
        self.causal_flags.append(self_is_causal)
        if past_key_value is None:
            b, l, _ = tgt.shape
            empty = torch.empty((b, l, self.num_heads, self.head_dim), dtype=tgt.dtype, device=tgt.device)
            return tgt, (empty, empty)
        return tgt, past_key_value


def test_prefix_uses_sliding_window_causal_attention_when_radius_enabled() -> None:
    cfg = VARConfig(
        level_vocab_sizes=(32, 64),
        level_lengths=(4, 4),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        exit_layers=(),
        local_attention_radius=1,
    )
    model = VARTransformer(cfg).eval()
    capture_layer = _CaptureDecoderLayer(hidden=cfg.hidden_size, num_heads=cfg.num_heads)
    model.blocks = torch.nn.ModuleList([capture_layer])

    prefix_tokens = [torch.tensor([[1, 2, 3, 4]], dtype=torch.long)]
    current_tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    _ = model(prefix_tokens, target_level=1, current_level_input=current_tokens)

    assert len(capture_layer.masks) == 2
    assert capture_layer.masks[0] is not None
    assert capture_layer.causal_flags[0] is False


def test_prefix_mask_disabled_when_radius_zero() -> None:
    cfg = VARConfig(
        level_vocab_sizes=(32, 64),
        level_lengths=(3, 3),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        exit_layers=(),
        local_attention_radius=0,
    )
    model = VARTransformer(cfg).eval()
    capture_layer = _CaptureDecoderLayer(hidden=cfg.hidden_size, num_heads=cfg.num_heads)
    model.blocks = torch.nn.ModuleList([capture_layer])

    prefix_tokens = [torch.tensor([[1, 2, 3]], dtype=torch.long)]
    current_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    _ = model(prefix_tokens, target_level=1, current_level_input=current_tokens)

    assert len(capture_layer.masks) == 2
    assert capture_layer.masks[0] is None
    assert capture_layer.causal_flags[0] is False


def test_decoder_layer_returns_concatenated_kv_cache() -> None:
    layer = SDPADecoderLayer(hidden=8, num_heads=2, mlp_ratio=1.0, dropout=0.0).eval()
    tgt = torch.randn(1, 2, 8)
    memory = torch.randn(1, 3, 8)
    past_k = torch.randn(1, 4, 2, 4)
    past_v = torch.randn(1, 4, 2, 4)

    _, present = layer(tgt=tgt, memory=memory, past_key_value=(past_k, past_v))
    present_k, present_v = present

    assert tuple(present_k.shape) == (1, 6, 2, 4)
    assert tuple(present_v.shape) == (1, 6, 2, 4)


def test_rotary_embedding_uses_cache_position_offset() -> None:
    cfg = VARConfig(
        level_vocab_sizes=(32, 64),
        level_lengths=(4, 4),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        exit_layers=(),
    )
    model = VARTransformer(cfg).eval()
    prefix_tokens = [torch.tensor([[1, 2, 3, 4]], dtype=torch.long)]
    current_tokens = torch.tensor([[1]], dtype=torch.long)

    past_k = torch.zeros((1, 5, cfg.num_heads, cfg.hidden_size // cfg.num_heads))
    past_v = torch.zeros((1, 5, cfg.num_heads, cfg.hidden_size // cfg.num_heads))
    _, cache_with_offset = model(
        prefix_tokens,
        target_level=1,
        current_level_input=current_tokens,
        use_cache=True,
        past_key_values=[(past_k, past_v)],
    )

    _, cache_without_offset = model(
        prefix_tokens,
        target_level=1,
        current_level_input=current_tokens,
        use_cache=True,
        past_key_values=None,
    )

    next_k_with_offset = cache_with_offset[0][0][:, -1]
    next_k_without_offset = cache_without_offset[0][0][:, -1]
    assert not torch.allclose(next_k_with_offset, next_k_without_offset)


def test_decoder_layer_stores_configured_attention_dropout() -> None:
    layer = SDPADecoderLayer(hidden=8, num_heads=2, mlp_ratio=1.0, dropout=0.25).train()
    assert layer.attention_dropout == 0.25


def test_local_sliding_window_mask_uses_banded_causal_pattern() -> None:
    cfg = VARConfig(
        level_vocab_sizes=(16, 16),
        level_lengths=(4, 4),
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        exit_layers=(),
    )
    model = VARTransformer(cfg).eval()
    mask = model._build_local_causal_mask(seq_len=5, radius=2, device=torch.device("cpu"))

    expected = torch.full((5, 5), float("-inf"))
    expected[0, 0] = 0.0
    expected[1, 0:2] = 0.0
    expected[2, 1:3] = 0.0
    expected[3, 2:4] = 0.0
    expected[4, 3:5] = 0.0
    assert torch.equal(mask, expected)
