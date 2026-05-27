from pathlib import Path

import torch
from src.vqvae.infer_cli import InferConfig, _load_tokenizer, build_parser, run_roundtrip


class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 4, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def batch_decode(self, values, skip_special_tokens=True):  # noqa: ARG002
        assert values
        return ["decoded text"]


class _FakeModel:
    def encode_sentence(self, input_ids: torch.Tensor, padding_mask: torch.Tensor):
        _ = padding_mask
        return torch.zeros((input_ids.shape[0], 1), dtype=torch.long), torch.tensor(0.0)

    def decode_from_semantic_indices(self, *_args, **_kwargs):
        return torch.tensor([[1, 5, 2]], dtype=torch.long)


def test_build_parser_accepts_required_args() -> None:
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "model.pt", "--input", "hello", "--tokenizer", "gpt2"])  # noqa: E501

    assert args.checkpoint == Path("model.pt")
    assert args.input == "hello"
    assert args.tokenizer == "gpt2"


def test_load_tokenizer_supports_pretrained_name(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.vqvae.infer_cli.AutoTokenizer.from_pretrained",
        lambda _name, use_fast: _FakeTokenizer(),
    )
    tokenizer = _load_tokenizer("gpt2")
    assert isinstance(tokenizer, _FakeTokenizer)


def test_run_roundtrip_returns_decoded_text(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"placeholder")

    monkeypatch.setattr("src.vqvae.infer_cli._load_tokenizer", lambda _path: _FakeTokenizer())
    monkeypatch.setattr("src.vqvae.infer_cli._load_model", lambda _path, _device: _FakeModel())

    config = InferConfig(
        checkpoint=checkpoint_path,
        input_text="hello",
        tokenizer=tmp_path / "tok.json",
        device="cpu",
        max_length=32,
    )

    assert run_roundtrip(config) == "decoded text"
