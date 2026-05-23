import torch

from src.var.generator import KVCacheRingBuffer
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
