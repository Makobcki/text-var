import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class VARConfig:
    # Уровень 0: Семантические токены (VQ-VAE словарь сюжетов)
    # Уровень 1: Локальные токены (BPE словарь текста)
    level_vocab_sizes: tuple[int, ...] = (4096, 32000)

    # 32 "сюжетных" шага, 1024 текстовых BPE-шага
    level_lengths: tuple[int, ...] = (32, 1024)

    hidden_size: int = 1024
    depth: int = 16
    num_heads: int = 16
    mlp_ratio: float = 4.0

    exit_layers: tuple[int, ...] = (4, 8, 12)  # Слои Early Exit
    pad_token_id: int = 0
    mask_token_id: int = 1  # Используется для NAR генерации

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VARConfig":
        return cls(
            level_vocab_sizes=tuple(int(v) for v in data.get("level_vocab_sizes", (4096, 32000))),
            level_lengths=tuple(int(v) for v in data.get("level_lengths", (32, 1024))),
            hidden_size=int(data.get("hidden_size", 1024)),
            depth=int(data.get("depth", 16)),
            num_heads=int(data.get("num_heads", 16)),
            mlp_ratio=float(data.get("mlp_ratio", 4.0)),
            exit_layers=tuple(int(v) for v in data.get("exit_layers", (4, 8, 12))),
            pad_token_id=int(data.get("pad_token_id", 0)),
            mask_token_id=int(data.get("mask_token_id", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["level_vocab_sizes"] = list(self.level_vocab_sizes)
        data["level_lengths"] = list(self.level_lengths)
        data["exit_layers"] = list(self.exit_layers)
        return data


@dataclass(frozen=True)
class TrainConfig:
    model: VARConfig
    checkpoint_path: Path
    device: str = "cuda"
    seed: int = 42
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    grad_clip_norm: float = 1.0
    save_every: int = 1000
    phase_steps: tuple[int, ...] = (5000, 5000)
    level_weights: Optional[list[float]] = None
    token_cache_path: Optional[Path] = None
    synthetic_count: int = 1000
    token_metadata: Optional[Any] = None


@dataclass(frozen=True)
class SampleConfig:
    checkpoint_path: Path
    output_path: Path
    device: str = "cuda"
    batch_size: int = 1
    entropy_threshold: float = 1.5
    early_exit_threshold: float = 0.8
    max_retries: int = 3
    thermodynamic_alpha: float = 1.0
    t_min: float = 0.1
    t_max: float = 2.0
    cfg_scale: float = 1.0


def load_train_config(path: Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_cfg = VARConfig.from_dict(data.get("model", {}))

    # Ленивый импорт для предотвращения циклической зависимости при сборке метаданных
    token_metadata = None
    if data.get("token_metadata"):
        from var_branch.token_cache import TokenCacheMetadata

        token_metadata = TokenCacheMetadata.from_dict(data["token_metadata"])

    return TrainConfig(
        model=model_cfg,
        checkpoint_path=Path(data["checkpoint_path"]),
        device=data.get("device", "cuda"),
        seed=int(data.get("seed", 42)),
        batch_size=int(data.get("batch_size", 4)),
        learning_rate=float(data.get("learning_rate", 1e-4)),
        weight_decay=float(data.get("weight_decay", 0.01)),
        max_steps=int(data.get("max_steps", 10000)),
        grad_clip_norm=float(data.get("grad_clip_norm", 1.0)),
        save_every=int(data.get("save_every", 1000)),
        phase_steps=tuple(int(v) for v in data.get("phase_steps", (5000, 5000))),
        level_weights=data.get("level_weights"),
        token_cache_path=Path(data["token_cache_path"]) if data.get("token_cache_path") else None,
        synthetic_count=int(data.get("synthetic_count", 1000)),
        token_metadata=token_metadata,
    )


def load_sample_config(path: Path) -> SampleConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return SampleConfig(
        checkpoint_path=Path(data["checkpoint_path"]),
        output_path=Path(data["output_path"]),
        device=data.get("device", "cuda"),
        batch_size=int(data.get("batch_size", 1)),
        entropy_threshold=float(data.get("entropy_threshold", 1.5)),
        early_exit_threshold=float(data.get("early_exit_threshold", 0.8)),
        max_retries=int(data.get("max_retries", 3)),
        thermodynamic_alpha=float(data.get("thermodynamic_alpha", 1.0)),
        t_min=float(data.get("t_min", 0.1)),
        t_max=float(data.get("t_max", 2.0)),
        cfg_scale=float(data.get("cfg_scale", 1.0)),
    )
