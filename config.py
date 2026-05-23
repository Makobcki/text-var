import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class VARConfig:
    # Уровень 0: Семантические токены (VQ-VAE словарь сюжетов)
    # Уровень 1: Локальные токены (BPE словарь текста)
    level_vocab_sizes: tuple[int, ...] = (4096, 2048, 32000)

    # 32 "сюжетных" шага, 1024 текстовых BPE-шага
    level_lengths: tuple[int, ...] = (32, 128, 1024)

    hidden_size: int = 1024
    depth: int = 16
    num_heads: int = 16
    mlp_ratio: float = 4.0

    exit_layers: tuple[int, ...] = (4, 8, 12)  # Слои Early Exit
    pad_token_id: int = 0
    mask_token_id: int = 1  # Используется для NAR генерации
    eos_token_id: int = 2
    gradient_checkpointing: bool = False
    local_attention_radius: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VARConfig":
        return cls(
            level_vocab_sizes=tuple(int(v) for v in data.get("level_vocab_sizes", (4096, 2048, 32000))),
            level_lengths=tuple(int(v) for v in data.get("level_lengths", (32, 128, 1024))),
            hidden_size=int(data.get("hidden_size", 1024)),
            depth=int(data.get("depth", 16)),
            num_heads=int(data.get("num_heads", 16)),
            mlp_ratio=float(data.get("mlp_ratio", 4.0)),
            exit_layers=tuple(int(v) for v in data.get("exit_layers", (4, 8, 12))),
            pad_token_id=int(data.get("pad_token_id", 0)),
            mask_token_id=int(data.get("mask_token_id", 1)),
            eos_token_id=int(data.get("eos_token_id", 2)),
            gradient_checkpointing=bool(data.get("gradient_checkpointing", False)),
            local_attention_radius=int(data.get("local_attention_radius", 0)),
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
    min_learning_rate_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_steps: int = 10000
    grad_clip_norm: float = 1.0
    save_every: int = 1000
    phase_steps: tuple[int, ...] = (5000, 5000)
    level_weights: Optional[list[float]] = None
    token_cache_path: Optional[Path] = None
    synthetic_count: int = 1000
    token_metadata: Optional[Any] = None
    corruption_level_idx: int = -1
    corruption_prob: float = 0.35
    corruption_span_min: int = 8
    corruption_span_max: int = 64
    masked_loss_weight: float = 0.85
    amp_enabled: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    compile_enabled: bool = False
    pin_memory: bool = True
    grad_accum_steps: int = 1
    flash_cross_entropy: bool = True
    use_early_exit_loss: bool = False
    optimizer: Literal["adamw", "adamw8bit"] = "adamw"
    resume_from: Optional[Path] = None
    validation_every: int = 0
    val_every: int = 0
    validation_batches: int = 8
    validation_split: float = 0.0
    val_token_cache_path: Optional[Path] = None
    synthetic_val_count: int = 256
    tensorboard_enabled: bool = False
    log_dir: Path = Path("runs/var")
    log_grad_norm_every: int = 0
    log_weight_norm_every: int = 0
    sample_every: int = 0
    sample_prompt: str = "The story begins"
    sample_max_new_tokens: int = 96
    wandb_enabled: bool = False
    wandb_project: str = "text-var"
    wandb_run_name: Optional[str] = None
    unconditional_drop_prob: float = 0.1
    stateful_context_enabled: bool = False
    stateful_context_max_tokens: int = 0


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
    max_backtracks_per_block: int = 2
    min_block_size_lvl2: int = 16
    max_seams_per_inpaint_pass: int = 10


def load_train_config(path: Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_cfg = VARConfig.from_dict(data.get("model", {}))

    # Ленивый импорт для предотвращения циклической зависимости при сборке метаданных
    token_metadata = None
    if data.get("token_metadata"):
        from token_cache import TokenCacheMetadata

        token_metadata = TokenCacheMetadata.from_dict(data["token_metadata"])

    return TrainConfig(
        model=model_cfg,
        checkpoint_path=Path(data["checkpoint_path"]),
        device=data.get("device", "cuda"),
        seed=int(data.get("seed", 42)),
        batch_size=int(data.get("batch_size", 4)),
        learning_rate=float(data.get("learning_rate", 1e-4)),
        min_learning_rate_ratio=float(data.get("min_learning_rate_ratio", 0.1)),
        warmup_ratio=float(data.get("warmup_ratio", 0.03)),
        weight_decay=float(data.get("weight_decay", 0.01)),
        max_steps=int(data.get("max_steps", 10000)),
        grad_clip_norm=float(data.get("grad_clip_norm", 1.0)),
        save_every=int(data.get("save_every", 1000)),
        phase_steps=tuple(int(v) for v in data.get("phase_steps", (5000, 5000))),
        level_weights=data.get("level_weights"),
        token_cache_path=Path(data["token_cache_path"]) if data.get("token_cache_path") else None,
        synthetic_count=int(data.get("synthetic_count", 1000)),
        token_metadata=token_metadata,
        corruption_level_idx=int(data.get("corruption_level_idx", -1)),
        corruption_prob=float(data.get("corruption_prob", 0.35)),
        corruption_span_min=int(data.get("corruption_span_min", 8)),
        corruption_span_max=int(data.get("corruption_span_max", 64)),
        masked_loss_weight=float(data.get("masked_loss_weight", 0.85)),
        amp_enabled=bool(data.get("amp_enabled", True)),
        amp_dtype=str(data.get("amp_dtype", "bf16")).lower(),
        compile_enabled=bool(data.get("compile_enabled", False)),
        pin_memory=bool(data.get("pin_memory", True)),
        grad_accum_steps=max(1, int(data.get("grad_accum_steps", 1))),
        flash_cross_entropy=bool(data.get("flash_cross_entropy", True)),
        use_early_exit_loss=bool(data.get("use_early_exit_loss", False)),
        optimizer=str(data.get("optimizer", "adamw")).lower(),
        resume_from=Path(data["resume_from"]) if data.get("resume_from") else None,
        validation_every=int(data.get("validation_every", 0)),
        val_every=int(data.get("val_every", data.get("validation_every", 0))),
        validation_batches=int(data.get("validation_batches", 8)),
        validation_split=float(data.get("validation_split", 0.0)),
        val_token_cache_path=Path(data["val_token_cache_path"]) if data.get("val_token_cache_path") else None,
        synthetic_val_count=int(data.get("synthetic_val_count", 256)),
        tensorboard_enabled=bool(data.get("tensorboard_enabled", False)),
        log_dir=Path(data.get("log_dir", "runs/var")),
        log_grad_norm_every=max(0, int(data.get("log_grad_norm_every", 0))),
        log_weight_norm_every=max(0, int(data.get("log_weight_norm_every", 0))),
        sample_every=max(0, int(data.get("sample_every", 0))),
        sample_prompt=str(data.get("sample_prompt", "The story begins")),
        sample_max_new_tokens=max(1, int(data.get("sample_max_new_tokens", 96))),
        wandb_enabled=bool(data.get("wandb_enabled", False)),
        wandb_project=str(data.get("wandb_project", "text-var")),
        wandb_run_name=str(data["wandb_run_name"]) if data.get("wandb_run_name") else None,
        unconditional_drop_prob=min(1.0, max(0.0, float(data.get("unconditional_drop_prob", 0.1)))),
        stateful_context_enabled=bool(data.get("stateful_context_enabled", False)),
        stateful_context_max_tokens=max(0, int(data.get("stateful_context_max_tokens", 0))),
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
        max_backtracks_per_block=int(data.get("max_backtracks_per_block", 2)),
        min_block_size_lvl2=int(data.get("min_block_size_lvl2", 16)),
        max_seams_per_inpaint_pass=int(data.get("max_seams_per_inpaint_pass", 10)),
    )
