from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import VARConfig
from generator import hybrid_cascade_decode
from model import VARTransformer
from token_cache import TokenCacheMetadata
from vqvae import SemanticTextVQVAE


@dataclass
class PipelineArtifacts:
    bpe_tokens: torch.Tensor
    semantic_tokens: torch.Tensor
    token_cache: dict[str, object]
    var_tokens: list[torch.Tensor]
    decoded_bpe: torch.Tensor
    decoded_text: list[str]


def build_token_cache(semantic_tokens: torch.Tensor, var_cfg: VARConfig) -> dict[str, object]:
    metadata = TokenCacheMetadata(
        kind="vq-var",
        level_vocab_sizes=tuple(int(v) for v in var_cfg.level_vocab_sizes),
        level_lengths=tuple(int(v) for v in var_cfg.level_lengths),
        codebook_dim=0,
        max_token_length=sum(var_cfg.level_lengths),
    )

    entries = []
    for i in range(semantic_tokens.shape[0]):
        level0_len = int(var_cfg.level_lengths[0])
        seed_tok = int(semantic_tokens[i].view(-1)[0].item())
        lvl0 = (seed_tok + torch.arange(level0_len)) % int(var_cfg.level_vocab_sizes[0])
        lvl1 = torch.zeros(var_cfg.level_lengths[1], dtype=torch.long)
        lvl2 = torch.zeros(var_cfg.level_lengths[2], dtype=torch.long)
        entries.append({"id": f"sample-{i}", "tokens": [lvl0, lvl1, lvl2]})

    return {"metadata": metadata.to_dict(), "entries": entries}


@torch.no_grad()
def run_full_cycle(
    text: str,
    *,
    tokenizer_name: str = "gpt2",
    max_bpe_len: int = 128,
    var_cfg: VARConfig | None = None,
    device: str = "cpu",
) -> PipelineArtifacts:
    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_bpe_len)
    bpe_tokens = encoded["input_ids"].to(dev)
    padding_mask = ~encoded["attention_mask"].bool().to(dev)

    vqvae = SemanticTextVQVAE(vocab_size=tokenizer.vocab_size).to(dev).eval()
    semantic_tokens, _ = vqvae.encode_sentence(bpe_tokens, padding_mask=padding_mask)

    cfg = var_cfg or VARConfig()
    var_model = VARTransformer(cfg).to(dev).eval()
    token_cache = build_token_cache(semantic_tokens, cfg)

    var_tokens = hybrid_cascade_decode(var_model, batch_size=bpe_tokens.shape[0], device=dev)

    decoded_bpe = vqvae.decode_from_semantic_indices(
        var_tokens[0],
        max_length=max_bpe_len,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id or 0,
        eos_token_id=tokenizer.eos_token_id,
    )
    decoded_text = tokenizer.batch_decode(decoded_bpe, skip_special_tokens=True)

    return PipelineArtifacts(
        bpe_tokens=bpe_tokens,
        semantic_tokens=semantic_tokens,
        token_cache=token_cache,
        var_tokens=var_tokens,
        decoded_bpe=decoded_bpe,
        decoded_text=decoded_text,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end VQ-VAE + VAR inference pipeline.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-bpe-len", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-cache", type=Path, default=None)
    args = parser.parse_args()

    artifacts = run_full_cycle(
        args.text,
        tokenizer_name=args.tokenizer,
        max_bpe_len=args.max_bpe_len,
        device=args.device,
    )

    if args.save_cache is not None:
        args.save_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts.token_cache, args.save_cache)

    print("=== INPUT TEXT ===")
    print(args.text)
    print("=== RECONSTRUCTED TEXT ===")
    print("\n".join(artifacts.decoded_text))


if __name__ == "__main__":
    main()
