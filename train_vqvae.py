import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from vqvae import SemanticTextVQVAE


def run_training(
    output_path: Path,
    *,
    steps: int = 500,
    batch_size: int = 8,
    seq_len: int = 128,
    vocab_size: int = 32000,
    hidden_size: int = 1024,
    semantic_tokens: int = 4096,
    lr: float = 3e-4,
    device: str = "cuda",
) -> Path:
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = SemanticTextVQVAE(vocab_size=vocab_size, hidden_size=hidden_size, num_semantic_tokens=semantic_tokens).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    synthetic = torch.randint(low=2, high=vocab_size, size=(max(steps * batch_size, 512), seq_len))
    ds = TensorDataset(synthetic)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    step = 0
    while step < steps:
        for (tokens,) in loader:
            tokens = tokens.to(dev)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(tokens)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f"[VQVAE] step={step}/{steps} loss={float(loss.detach().cpu()):.6f}")
            if step >= steps:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": step}, output_path)
    print(f"[VQVAE] checkpoint saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain SemanticTextVQVAE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_training(args.output, steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len, device=args.device)


if __name__ == "__main__":
    main()
