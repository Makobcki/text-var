import torch
from src.var.generator import KVCacheRingBuffer, TurboQuantConfig
from src.var.model import RingKVCacheView


def _make_layer_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, len(values), 1, 1)


def test_kv_ring_buffer_keeps_latest_window_ordered() -> None:
    ring = KVCacheRingBuffer(max_window=3)

    for step in [1, 2, 3, 4, 5]:
        present = [(_make_layer_tensor([step]), _make_layer_tensor([step + 100]))]
        cached = ring.update(present)

    layer_view = cached[0]
    assert isinstance(layer_view, RingKVCacheView)
    key = layer_view.keys.index_select(1, layer_view.positions)
    value = layer_view.values.index_select(1, layer_view.positions)
    assert key.view(-1).tolist() == [3.0, 4.0, 5.0]
    assert value.view(-1).tolist() == [103.0, 104.0, 105.0]


def test_kv_ring_buffer_returns_partial_window_before_full() -> None:
    ring = KVCacheRingBuffer(max_window=4)
    present = [(_make_layer_tensor([7]), _make_layer_tensor([9]))]

    cached = ring.update(present)
    layer_view = cached[0]
    key = layer_view.keys.index_select(1, layer_view.positions)
    value = layer_view.values.index_select(1, layer_view.positions)

    assert tuple(key.shape) == (1, 1, 1, 1)
    assert tuple(value.shape) == (1, 1, 1, 1)
    assert key.item() == 7.0
    assert value.item() == 9.0


def test_kv_ring_buffer_exposes_wrapped_positions_without_reallocation() -> None:
    ring = KVCacheRingBuffer(max_window=3)
    for step in [1, 2, 3, 4]:
        present = [(_make_layer_tensor([step]), _make_layer_tensor([step + 10]))]
        cached = ring.update(present)
    layer_view = cached[0]
    assert layer_view.positions.tolist() == [1, 2, 0]


def test_kv_ring_buffer_turboquant_compresses_to_uint8_and_materializes() -> None:
    ring = KVCacheRingBuffer(max_window=3, turboquant_config=TurboQuantConfig())
    for step in [1, 2, 3]:
        present = [(_make_layer_tensor([step]), _make_layer_tensor([step + 100]))]
        cached = ring.update(present)
    layer_view = cached[0]
    assert layer_view.keys.dtype == torch.uint8
    assert layer_view.values.dtype == torch.uint8
    assert layer_view.codec is not None
    assert layer_view.layer_idx == 0

    key, value = layer_view.codec.dequantize_view(layer_view)
    assert tuple(key.shape) == (1, 3, 1, 1)
    assert tuple(value.shape) == (1, 3, 1, 1)


def test_turboquant_stores_bitpacked_width() -> None:
    ring = KVCacheRingBuffer(max_window=2, turboquant_config=TurboQuantConfig(key_bits=3, value_bits=4))  # noqa: E501
    present = [(
        torch.ones((1, 1, 1, 8), dtype=torch.float32),
        torch.ones((1, 1, 1, 8), dtype=torch.float32),
    )]
    cached = ring.update(present)
    layer_view = cached[0]
    assert layer_view.keys.shape[-1] == 3
    assert layer_view.values.shape[-1] == 4
