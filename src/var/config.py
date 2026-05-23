from dataclasses import asdict, dataclass
from typing import Any


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
            level_vocab_sizes=tuple(
                int(v) for v in data.get("level_vocab_sizes", (4096, 2048, 32000))
            ),
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
