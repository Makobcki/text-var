import torch

from config import VARConfig
from generator import hybrid_cascade_decode


class _ToyModel:
    def __init__(self, eos_token_id: int) -> None:
        self.cfg = VARConfig(
            level_vocab_sizes=(8, 8, 8),
            level_lengths=(2, 8, 32),
            hidden_size=8,
            depth=1,
            num_heads=1,
            mlp_ratio=1.0,
            exit_layers=(),
            eos_token_id=eos_token_id,
        )

    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("The model forward is monkeypatched in tests")


def test_hierarchical_eos_truncates_level_2(monkeypatch) -> None:
    model = _ToyModel(eos_token_id=2)
    sampled_tokens = [torch.tensor([5]), torch.tensor([2])]
    draft_lengths: list[int] = []

    def _fake_decode_with_cache(**kwargs):
        del kwargs
        token = sampled_tokens.pop(0)
        return (
            token,
            torch.zeros((1,), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.float32),
            [],
        )

    def _fake_decode_no_cache(**kwargs):
        del kwargs
        return (
            torch.tensor([1]),
            torch.zeros((1,), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.float32),
        )

    def _fake_parallel_block_draft(**kwargs):
        draft_lengths.append(int(kwargs["len_lvl_2"]))
        return torch.zeros((1, kwargs["len_lvl_2"]), dtype=torch.long)

    monkeypatch.setattr("generator._decode_next_ar_token", _fake_decode_no_cache)
    monkeypatch.setattr("generator._decode_next_ar_token_with_cache", _fake_decode_with_cache)
    monkeypatch.setattr("generator._parallel_block_draft", _fake_parallel_block_draft)
    monkeypatch.setattr("generator._inpaint_block_seams", lambda **kwargs: kwargs["lvl_2_tokens"])

    output = hybrid_cascade_decode(model, batch_size=1, device=torch.device("cpu"))

    assert draft_lengths == [8]
    assert output[1].shape[1] == 2
    assert output[2].shape[1] == 8


def test_encode_multiscale_appends_eos() -> None:
    from prepare_dataset import _encode_multiscale

    levels = _encode_multiscale(
        [7, 8, 9],
        level_lengths=(2, 3, 4),
        level_vocab_sizes=(32, 32, 32),
        eos_token_id=2,
    )

    assert levels[2] == [7, 8, 9, 2]
