from pathlib import Path

from src.vqvae.training.cli import build_parser
from src.vqvae.training.config import load_vqvae_train_config


def test_cli_parses_runtime_optimization_flags() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--output", "out.ckpt",
        "--token-cache-dir", "cache",
        "--dataloader-num-workers", "6",
        "--dataloader-prefetch-factor", "3",
        "--amp-dtype", "fp16",
        "--use-torch-compile",
        "--log-every-steps", "25",
    ])

    assert args.dataloader_num_workers == 6
    assert args.dataloader_prefetch_factor == 3
    assert args.amp_dtype == "fp16"
    assert args.use_torch_compile is True
    assert args.log_every_steps == 25
    assert args.verbose is False


def test_cli_parses_verbose_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--output", "out.ckpt", "--token-cache-dir", "cache", "--verbose"])
    assert args.verbose is True


def test_config_loads_runtime_optimization_options(tmp_path: Path) -> None:
    cfg_path = tmp_path / "vqvae_cfg.json"
    cfg_path.write_text(
        """
{
  "output": "checkpoint.pt",
  "token_cache_dir": "cache-dir",
  "dataloader_num_workers": 2,
  "dataloader_prefetch_factor": 4,
  "amp_dtype": "none",
  "use_torch_compile": true,
  "log_every_steps": 7
}
""".strip(),
        encoding="utf-8",
    )

    cfg = load_vqvae_train_config(cfg_path)

    assert cfg.dataloader_num_workers == 2
    assert cfg.dataloader_prefetch_factor == 4
    assert cfg.amp_dtype == "none"
    assert cfg.use_torch_compile is True
    assert cfg.log_every_steps == 7
    assert cfg.verbose is False
