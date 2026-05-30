from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VQVAEConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_semantic_tokens: int = 4096
    semantic_sequence_length: int = 1
    pad_token_id: int = 0
    semantic_pad_token_id: int = 0
    max_position_embeddings: int = 2048
    
    use_turboquant_kv: bool = False
    turboquant_key_bits: int = 4
    turboquant_value_bits: int = 4
    turboquant_qjl_residual_scale: float = 0.5
    gradient_checkpointing: bool = False
    use_rotary_embeddings: bool = True
    word_dropout_prob: float = 0.1
    use_unpadding: bool = False
    use_fp8: bool = False
    
    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    router_aux_loss_coef: float = 0.01

    encoder_num_heads: int = 8
    encoder_depth: int = 4
    encoder_mlp_ratio: float = 4.0
    encoder_dropout: float = 0.1
    
    compression_rate: int = 4
    downsample_num_blocks: int = 2
    
    fsq_levels: list[int] = field(default_factory=lambda: [8, 8, 8, 8, 8, 8, 8, 8])
    
    decoder_num_heads: int = 8
    decoder_depth: int = 4
    decoder_mlp_ratio: float = 4.0
    decoder_dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VQVAEConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
