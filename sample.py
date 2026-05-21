from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from checkpoint import load_checkpoint
from config import SampleConfig, load_sample_config
from generator import hybrid_cascade_decode  # Замена несуществующего генератора


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def run_sampling(cfg: SampleConfig) -> Path:
    device = _resolve_device(cfg.device)
    model, checkpoint = load_checkpoint(cfg.checkpoint_path, device=device)

    print(
        f"[SAMPLE] Запуск гибридного каскадного декодирования. "
        f"Entropy Limit: {cfg.entropy_threshold}, Alpha: {cfg.thermodynamic_alpha}"
    )

    # Применение существующего в вашей кодовой базе алгоритма генерации
    tokens = hybrid_cascade_decode(
        model,
        batch_size=int(cfg.batch_size),
        device=device,
        cfg_scale=cfg.cfg_scale,
        alpha=cfg.thermodynamic_alpha,
        healthy_entropy_limit=cfg.entropy_threshold,
        nar_steps=4,
    )

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokens": [item.detach().cpu() for item in tokens],
        "metadata": {
            "family": "var",
            "checkpoint_step": int(checkpoint.get("step", 0)),
            "model_config": model.cfg.to_dict(),
        },
    }
    torch.save(payload, cfg.output_path)
    cfg.output_path.with_suffix(".json").write_text(
        json.dumps(payload["metadata"], indent=2) + "\n",
        encoding="utf-8",
    )
    return cfg.output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Автономная генерация VAR.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    out = run_sampling(load_sample_config(args.config))
    print(f"[SAMPLE] Выходные токены успешно сохранены в: {out}")


if __name__ == "__main__":
    main()
