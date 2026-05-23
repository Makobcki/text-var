from pathlib import Path

import torch
from src.core.pipeline import PipelineConfig, TextVARPipeline
from src.var.generator import RollbackEvent, _parallel_block_draft
from src.vqvae.model import SemanticTextVQVAE


class _DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str,
        truncation: bool,
        max_length: int,
        padding: bool = False,
    ):
        del return_tensors, truncation, max_length, padding
        batch_size = len(text) if isinstance(text, list) else 1
        return {
            "input_ids": torch.tensor([[1, 7, 2]] * batch_size, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]] * batch_size, dtype=torch.long),
        }

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "|".join(str(i) for i in ids)

    def batch_decode(self, ids: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        del skip_special_tokens
        return ["|".join(str(i) for i in row) for row in ids]


class _DummyVQVAE(torch.nn.Module):
    def encode_sentence(self, bpe_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None):
        del padding_mask
        batch = bpe_tokens.shape[0]
        return torch.zeros((batch, 1), dtype=torch.long), torch.tensor(0.0)

    def decode_from_semantic_indices(
        self,
        semantic_indices: torch.Tensor,
        *,
        max_length: int,
        bos_token_id: int,
        eos_token_id: int | None = None,
    ):
        del semantic_indices, eos_token_id
        return torch.full((1, max_length), bos_token_id, dtype=torch.long)


class _DummyVAR(torch.nn.Module):
    pass


def test_generate_flow(monkeypatch, tmp_path: Path) -> None:
    cfg = PipelineConfig(
        vqvae_path=tmp_path / "vqvae.pt",
        var_path=tmp_path / "var.pt",
        bpe_tokenizer_path=tmp_path / "tokenizer.json",
    )
    cfg.vqvae_path.write_text("x", encoding="utf-8")
    cfg.var_path.write_text("x", encoding="utf-8")
    cfg.bpe_tokenizer_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        TextVARPipeline, "_load_tokenizer", staticmethod(lambda _: _DummyTokenizer())
    )
    monkeypatch.setattr(TextVARPipeline, "_load_vqvae", lambda self, _: _DummyVQVAE())
    monkeypatch.setattr(TextVARPipeline, "_load_var", lambda self, _: _DummyVAR())
    capture: dict[str, torch.Tensor] = {}

    def _fake_decode(model, batch_size, device, prefix_inputs=None):
        del model
        assert prefix_inputs is not None
        capture["prefix"] = prefix_inputs[0]
        return [torch.zeros((batch_size, 1), dtype=torch.long, device=device)]

    monkeypatch.setattr("src.core.pipeline.hybrid_cascade_decode", _fake_decode)

    pipeline = TextVARPipeline(cfg)
    result = pipeline.generate("hello", max_new_tokens=4)

    assert result == "1|1|1|1"
    assert tuple(capture["prefix"].shape) == (1, 1)


def test_missing_tokenizer_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    try:
        TextVARPipeline._load_tokenizer(missing)
    except FileNotFoundError as exc:
        assert "Tokenizer file not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_load_var_uses_model_config(monkeypatch, tmp_path: Path) -> None:
    cfg = PipelineConfig(
        vqvae_path=tmp_path / "vqvae.pt",
        var_path=tmp_path / "var.pt",
        bpe_tokenizer_path=tmp_path / "tokenizer.json",
    )
    cfg.vqvae_path.write_text("x", encoding="utf-8")
    cfg.var_path.write_text("x", encoding="utf-8")
    cfg.bpe_tokenizer_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        TextVARPipeline, "_load_tokenizer", staticmethod(lambda _: _DummyTokenizer())
    )
    monkeypatch.setattr(TextVARPipeline, "_load_vqvae", lambda self, _: _DummyVQVAE())

    captured: dict[str, object] = {}

    class _FakeVAR(torch.nn.Module):
        def __init__(self, cfg_obj):
            super().__init__()
            captured["cfg"] = cfg_obj

        def load_state_dict(self, state_dict, strict=False):
            del strict
            captured["state_dict"] = state_dict
            return self

    monkeypatch.setattr("src.core.pipeline.VARTransformer", _FakeVAR)
    monkeypatch.setattr(
        "src.core.pipeline.torch.load",
        lambda path, map_location: {
            "model": {"w": torch.tensor(1)},
            "model_config": {
                "level_vocab_sizes": [101, 202, 303],
                "level_lengths": [3, 4, 5],
                "hidden_size": 32,
                "depth": 2,
                "num_heads": 2,
                "mlp_ratio": 2.0,
            },
        },
    )

    _ = TextVARPipeline(cfg)
    loaded_cfg = captured["cfg"]
    assert getattr(loaded_cfg, "hidden_size") == 32
    assert tuple(getattr(loaded_cfg, "level_lengths")) == (3, 4, 5)


def test_vqvae_decode_from_semantic_indices_shape_and_bos() -> None:
    model = SemanticTextVQVAE(vocab_size=16, hidden_size=8, num_semantic_tokens=8).eval()
    semantic_indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    generated = model.decode_from_semantic_indices(
        semantic_indices,
        max_length=5,
        bos_token_id=7,
        eos_token_id=None,
    )
    assert tuple(generated.shape) == (2, 5)
    assert bool(torch.all(generated[:, 0] == 7))


def test_vqvae_decode_rejects_invalid_max_length() -> None:
    model = SemanticTextVQVAE(vocab_size=16, hidden_size=8, num_semantic_tokens=8).eval()
    try:
        model.decode_from_semantic_indices(
            torch.tensor([[1]], dtype=torch.long),
            max_length=0,
            bos_token_id=1,
        )
    except ValueError as exc:
        assert "max_length must be >= 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for max_length=0")


def test_parallel_block_draft_raises_rollback_on_high_chaos(monkeypatch) -> None:
    class _Cfg:
        pad_token_id = 0
        mask_token_id = 1

    class _Model:
        cfg = _Cfg()

        def __call__(
            self,
            prefix_inputs,
            target_level,
            current_level_input,
            cfg_scale,
            compact_memory_for_final_level,
        ):
            del prefix_inputs, target_level, cfg_scale, compact_memory_for_final_level
            batch, block_len = current_level_input.shape
            return torch.zeros((batch, block_len, 4), dtype=torch.float32)

    def _fake_sampling(logits, alpha, healthy_entropy_limit):
        del logits, alpha, healthy_entropy_limit
        return (
            torch.zeros((2,), dtype=torch.long),
            torch.zeros((2,), dtype=torch.float32),
            torch.ones((2,), dtype=torch.float32),
        )

    monkeypatch.setattr("generator.thermodynamic_sampling_with_stats", _fake_sampling)

    try:
        _parallel_block_draft(
            _Model(),
            prefix_inputs=[],
            len_lvl_2=2,
            block_count=1,
            block_size=2,
            batch_size=1,
            device=torch.device("cpu"),
            cfg_scale=1.0,
            alpha=1.0,
            healthy_entropy_limit=1.5,
            rollback_chaos_threshold=0.5,
        )
    except RollbackEvent as exc:
        assert exc.block_start == 0
        assert exc.block_end == 2
    else:
        raise AssertionError("Expected RollbackEvent when chaos_diff exceeds threshold")
