import torch
from src.var.generator import hybrid_cascade_decode
from src.var.generator import _resolve_phase3_level2_length
from src.var.training.config import VARConfig


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

    monkeypatch.setattr("src.var.generator._decode_next_ar_token", _fake_decode_no_cache)
    monkeypatch.setattr(
        "src.var.generator._decode_next_ar_token_with_cache", _fake_decode_with_cache
    )
    monkeypatch.setattr("src.var.generator._parallel_block_draft", _fake_parallel_block_draft)
    monkeypatch.setattr(
        "src.var.generator._inpaint_block_seams", lambda **kwargs: kwargs["lvl_2_tokens"]
    )

    output = hybrid_cascade_decode(model, batch_size=1, device=torch.device("cpu"))

    assert draft_lengths == [8]
    assert output[1].shape[1] == 2
    assert output[2].shape[1] == 8


def test_phase2_finished_rows_are_padded(monkeypatch) -> None:
    model = _ToyModel(eos_token_id=2)
    model.cfg.pad_token_id = 0
    sampled_tokens = [torch.tensor([2, 5]), torch.tensor([7, 6])]

    def _fake_decode_with_cache(**kwargs):
        del kwargs
        token = sampled_tokens.pop(0)
        return token, torch.zeros((2,), dtype=torch.float32), torch.zeros((2,), dtype=torch.float32), []

    def _fake_decode_no_cache(**kwargs):
        del kwargs
        return torch.tensor([1, 1]), torch.zeros((2,), dtype=torch.float32), torch.zeros((2,), dtype=torch.float32)

    monkeypatch.setattr("src.var.generator._decode_next_ar_token", _fake_decode_no_cache)
    monkeypatch.setattr("src.var.generator._decode_next_ar_token_with_cache", _fake_decode_with_cache)
    monkeypatch.setattr("src.var.generator._parallel_block_draft", lambda **kwargs: torch.zeros((2, kwargs["len_lvl_2"]), dtype=torch.long))
    monkeypatch.setattr("src.var.generator._inpaint_block_seams", lambda **kwargs: kwargs["lvl_2_tokens"])

    output = hybrid_cascade_decode(model, batch_size=2, device=torch.device("cpu"))
    lvl1 = output[1]
    assert int(lvl1[0, 1].item()) == int(model.cfg.pad_token_id)


def test_hybrid_cascade_decode_retries_twice_after_rollback(monkeypatch) -> None:
    model = _ToyModel(eos_token_id=2)
    attempt_count = {"count": 0}

    def _fake_decode_with_cache(**kwargs):
        del kwargs
        return (
            torch.tensor([2]),
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
        attempt_count["count"] += 1
        threshold = kwargs.get("rollback_chaos_threshold", 0.5)
        if threshold != float("inf"):
            from src.var.generator import RollbackEvent

            raise RollbackEvent(0, 16)
        return torch.zeros((1, kwargs["len_lvl_2"]), dtype=torch.long)

    monkeypatch.setattr("src.var.generator._decode_next_ar_token", _fake_decode_no_cache)
    monkeypatch.setattr("src.var.generator._decode_next_ar_token_with_cache", _fake_decode_with_cache)
    monkeypatch.setattr("src.var.generator._parallel_block_draft", _fake_parallel_block_draft)
    monkeypatch.setattr(
        "src.var.generator._inpaint_block_seams", lambda **kwargs: kwargs["lvl_2_tokens"]
    )

    output = hybrid_cascade_decode(model, batch_size=1, device=torch.device("cpu"))
    assert attempt_count["count"] == 3
    assert output[2].shape[0] == 1


def test_encode_multiscale_appends_eos() -> None:
    from src.data.utils.prepare_dataset import _encode_multiscale

    levels = _encode_multiscale(
        [7, 8, 9],
        level_lengths=(2, 3, 4),
        level_vocab_sizes=(32, 32, 32),
        eos_token_id=2,
    )

    assert levels[2] == [7, 8, 9, 2]


def test_encode_multiscale_supports_dynamic_levels() -> None:
    from src.data.utils.prepare_dataset import _encode_multiscale

    levels = _encode_multiscale(
        [1, 2, 3, 4, 5, 6, 7],
        level_lengths=(2, 3, 4, 8),
        level_vocab_sizes=(32, 32, 32, 32),
        eos_token_id=2,
    )

    assert len(levels) == 4
    assert levels[0][:2] == [1, 2]
    assert levels[1][:3] == [1, 3, 5]
    assert levels[2][:4] == [1, 2, 3, 4]
    assert levels[3][:8] == [1, 2, 3, 4, 5, 6, 7, 2]


def test_resolve_phase3_level2_length_scales_by_config_proportion() -> None:
    resolved = _resolve_phase3_level2_length(
        full_lvl_2_len=1024,
        nominal_lvl_1_len=128,
        actual_lvl_1_len=64,
    )
    assert resolved == 512
