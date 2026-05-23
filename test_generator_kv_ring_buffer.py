import torch

from generator import KVCacheRingBuffer


def _make_layer_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, len(values), 1, 1)


def test_kv_ring_buffer_keeps_latest_window_ordered() -> None:
    ring = KVCacheRingBuffer(max_window=3)

    for step in [1, 2, 3, 4, 5]:
        present = [(_make_layer_tensor([step]), _make_layer_tensor([step + 100]))]
        cached = ring.update(present)

    key, value = cached[0]
    assert key.view(-1).tolist() == [3.0, 4.0, 5.0]
    assert value.view(-1).tolist() == [103.0, 104.0, 105.0]


def test_kv_ring_buffer_returns_partial_window_before_full() -> None:
    ring = KVCacheRingBuffer(max_window=4)
    present = [(_make_layer_tensor([7]), _make_layer_tensor([9]))]

    cached = ring.update(present)
    key, value = cached[0]

    assert tuple(key.shape) == (1, 1, 1, 1)
    assert tuple(value.shape) == (1, 1, 1, 1)
    assert key.item() == 7.0
    assert value.item() == 9.0
