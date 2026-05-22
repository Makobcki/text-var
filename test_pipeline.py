from pathlib import Path

import torch

from pipeline import PipelineConfig, TextVARPipeline


class _DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text: str, return_tensors: str, truncation: bool, max_length: int):
        del text, return_tensors, truncation, max_length
        return {
            "input_ids": torch.tensor([[1, 7, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "|".join(str(i) for i in ids)


class _DummyVQVAE(torch.nn.Module):
    def encode_sentence(self, bpe_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None):
        del padding_mask
        batch = bpe_tokens.shape[0]
        return torch.zeros((batch, 1), dtype=torch.long), torch.tensor(0.0)

    def decode_from_semantic_indices(self, semantic_indices: torch.Tensor, *, max_length: int, bos_token_id: int, eos_token_id: int | None = None):
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

    monkeypatch.setattr(TextVARPipeline, "_load_tokenizer", staticmethod(lambda _: _DummyTokenizer()))
    monkeypatch.setattr(TextVARPipeline, "_load_vqvae", lambda self, _: _DummyVQVAE())
    monkeypatch.setattr(TextVARPipeline, "_load_var", lambda self, _: _DummyVAR())
    monkeypatch.setattr("pipeline.hybrid_cascade_decode", lambda model, batch_size, device: [torch.zeros((batch_size, 1), dtype=torch.long, device=device)])

    pipeline = TextVARPipeline(cfg)
    result = pipeline.generate("hello", max_new_tokens=4)

    assert result == "1|1|1|1"


def test_missing_tokenizer_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    try:
        TextVARPipeline._load_tokenizer(missing)
    except FileNotFoundError as exc:
        assert "Tokenizer file not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
