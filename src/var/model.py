from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.var.training.config import VARConfig
from src.var.turboquant_math import generate_orthogonal_matrix
from src.var.turboquant_triton import TurboQuantTritonInputs, turboquant_attention
from src.var.fp8 import get_linear_layer

if TYPE_CHECKING:
    from src.var.generator import TurboQuantCodec


@dataclass(frozen=True)
class RingKVCacheView:
    """Read-only view of static KV ring buffers for one decoder layer.

    Attributes:
        keys: Preallocated ring buffer for keys with shape [B, W, H, D].
        values: Preallocated ring buffer for values with shape [B, W, H, D].
        positions: Logical order of valid KV entries in `keys/values`.
    """

    keys: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor
    codec: "TurboQuantCodec | None" = None
    layer_idx: int = -1
    current_idx: int = 0
    current_length: int = 0


class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> torch.Tensor:
        t = torch.arange(start_pos, start_pos + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    cos = freqs.cos()
    sin = freqs.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.0, use_fp8: bool = False):
        super().__init__()
        self.w1 = get_linear_layer(in_features, hidden_features, bias=False, use_fp8=use_fp8)
        self.w2 = get_linear_layer(in_features, hidden_features, bias=False, use_fp8=use_fp8)
        self.w3 = get_linear_layer(hidden_features, out_features, bias=False, use_fp8=use_fp8)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

try:
    from megablocks.layers.arguments import Arguments as MoEArgs
    from megablocks.layers.dmoe import dMoE
    HAS_MEGABLOCKS = True
except ImportError:
    HAS_MEGABLOCKS = False

class MegablocksMoEWrapper(nn.Module):
    def __init__(self, hidden: int, ff_hidden: int, num_experts: int, top_k: int):
        super().__init__()
        args = MoEArgs(
            hidden_size=hidden,
            ffn_hidden_size=ff_hidden,
            moe_num_experts=num_experts,
            moe_top_k=top_k,
            moe_capacity_factor=1.0,
            moe_loss_weight=1.0,
            moe_expert_model_parallelism=False,
            device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"),
            bf16=True, 
            mlp_type='swiglu',
            mlp_impl='grouped'
        )
        self.moe = dMoE(args)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        out_flat, _ = self.moe(x_flat)
        out = out_flat.view(batch_size, seq_len, hidden)
        # megablocks router typically stores l_aux
        aux_loss = getattr(self.moe.router, "l_aux", torch.tensor(0.0, device=x.device, dtype=x.dtype))
        return out, aux_loss

class SparseMoE(nn.Module):
    def __init__(self, hidden: int, ff_hidden: int, num_experts: int, top_k: int, dropout: float = 0.0, use_fp8: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = get_linear_layer(hidden, num_experts, bias=False, use_fp8=False)
        self.experts = nn.ModuleList([
            SwiGLU(hidden, ff_hidden, hidden, dropout, use_fp8=use_fp8)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
        router_probs = routing_weights.mean(dim=0)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx)
            token_indices, k_indices = torch.where(expert_mask)
            tokens_per_expert[expert_idx] = token_indices.numel() / (x_flat.shape[0] * self.top_k)
            
            if token_indices.numel() > 0:
                expert_inputs = x_flat[token_indices]
                expert_outputs = self.experts[expert_idx](expert_inputs)
                weights = routing_weights_topk[token_indices, k_indices].unsqueeze(-1)
                final_output[token_indices] += expert_outputs * weights
                
        aux_loss = self.num_experts * torch.sum(router_probs * tokens_per_expert)
        return final_output.view(batch_size, seq_len, hidden), aux_loss


class SDPADecoderLayer(nn.Module):
    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1, use_fp8: bool = False, use_moe: bool = False, num_experts: int = 8, moe_top_k: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        if hidden % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.self_qkv = get_linear_layer(hidden, hidden * 3, use_fp8=use_fp8)
        self.cross_q = get_linear_layer(hidden, hidden, use_fp8=use_fp8)
        self.cross_kv = get_linear_layer(hidden, hidden * 2, use_fp8=use_fp8)
        self.self_out = get_linear_layer(hidden, hidden, use_fp8=use_fp8)
        self.cross_out = get_linear_layer(hidden, hidden, use_fp8=use_fp8)

        ff_hidden = max(hidden, int(hidden * float(mlp_ratio) * 2 / 3))
        # Round up to multiple of 128 for FP8 compatibility and optimal CUDA tiling
        if use_fp8:
            ff_hidden = ((ff_hidden + 127) // 128) * 128
            
        self.use_moe = use_moe
        if self.use_moe:
            if HAS_MEGABLOCKS:
                self.ffn = MegablocksMoEWrapper(hidden, ff_hidden, num_experts, moe_top_k)
            else:
                self.ffn = SparseMoE(hidden, ff_hidden, num_experts, moe_top_k, dropout, use_fp8=use_fp8)
        else:
            self.ffn = SwiGLU(hidden, ff_hidden, hidden, dropout, use_fp8=use_fp8)

        # Per-sublayer Pre-Norm (sequential residual connections)
        self.norm_self_attn = RMSNorm(hidden)
        self.norm_cross_attn = RMSNorm(hidden)
        self.norm_ffn = RMSNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = float(dropout)
        self.register_buffer(
            "R_k", generate_orthogonal_matrix(self.head_dim, torch.device("cpu")), persistent=True
        )
        self.register_buffer(
            "R_v", generate_orthogonal_matrix(self.head_dim, torch.device("cpu")), persistent=True
        )

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

    def _compress_memory(self, x: torch.Tensor, target_tokens: int) -> torch.Tensor:
        seq_len = x.shape[1]
        keep = max(1, min(int(target_tokens), seq_len))
        if keep >= seq_len:
            return x
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), keep).transpose(1, 2)
        return pooled

    def _materialize_past(
        self,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | RingKVCacheView,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize ordered past K/V tensors from cache representation.

        Args:
            past_key_value: Either dense ordered past tensors or ring-buffer view.

        Returns:
            Ordered past key and value tensors.
        """
        if isinstance(past_key_value, RingKVCacheView):
            positions = past_key_value.positions
            if past_key_value.codec is not None:
                return past_key_value.codec.dequantize_view(past_key_value)
            return (
                past_key_value.keys.index_select(1, positions),
                past_key_value.values.index_select(1, positions),
            )
        return past_key_value

    def _append_turboquant_step(
        self,
        past_key_value: RingKVCacheView,
        step_k: torch.Tensor,
        step_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append current step to preallocated quantized cache and return ordered payload."""
        if past_key_value.codec is None:
            raise ValueError("TurboQuant append requires codec.")
        layer_idx = past_key_value.layer_idx
        write_ptr = past_key_value.current_idx
        qk, qv = past_key_value.codec.quantize_step(layer_idx, step_k, step_v, write_ptr)
        past_key_value.keys[:, write_ptr : write_ptr + 1, :, :] = qk
        past_key_value.values[:, write_ptr : write_ptr + 1, :, :] = qv
        new_length = min(past_key_value.keys.shape[1], past_key_value.current_length + 1)
        if new_length < past_key_value.keys.shape[1]:
            ordered_positions = torch.arange(
                new_length, device=past_key_value.keys.device, dtype=torch.long
            )
        else:
            ordered_positions = (
                torch.arange(
                    past_key_value.keys.shape[1],
                    device=past_key_value.keys.device,
                    dtype=torch.long,
                )
                + ((write_ptr + 1) % past_key_value.keys.shape[1])
            ) % past_key_value.keys.shape[1]
        staged = RingKVCacheView(
            keys=past_key_value.keys,
            values=past_key_value.values,
            positions=ordered_positions,
            codec=past_key_value.codec,
            layer_idx=layer_idx,
            current_idx=(write_ptr + 1) % past_key_value.keys.shape[1],
            current_length=new_length,
        )
        return past_key_value.codec.quantized_view_payload(staged)

    def forward(
        self,
        *,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        cross_kv_memory: tuple[torch.Tensor, torch.Tensor] | None = None,
        self_attn_mask: torch.Tensor | None = None,
        self_is_causal: bool = False,
        rotary_freqs_tgt: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | RingKVCacheView | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = tgt

        # 1. Self-Attention с собственной нормализацией
        x1 = self.norm_self_attn(x)
        qkv = self.self_qkv(x1)
        q, k, v = qkv.chunk(3, dim=-1)
        batch_size, seq_len, _ = x.shape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        if rotary_freqs_tgt is not None:
            q, k = apply_rotary_pos_emb(q, k, rotary_freqs_tgt)
        if past_key_value is not None and not (
            bool(getattr(self, "use_turboquant", False))
            and isinstance(past_key_value, RingKVCacheView)
            and past_key_value.codec is not None
        ):
            past_k, past_v = self._materialize_past(past_key_value)
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        present_key_value = (k, v)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        can_use_turboquant = (
            isinstance(past_key_value, RingKVCacheView) and past_key_value.codec is not None
        )
        if (
            bool(getattr(self, "use_turboquant", False))
            and can_use_turboquant
            and self_attn_mask is None
        ):
            assert past_key_value.codec is not None
            step_k = k[:, -1:, :, :]
            step_v = v[:, -1:, :, :]
            pk, pv, ks, vs, ksign, vsign = self._append_turboquant_step(
                past_key_value, step_k, step_v
            )
            qh = torch.einsum("blhd,df->blhf", q, self.R_k.to(q.device).transpose(0, 1)).transpose(
                1, 2
            )
            tq_inputs = TurboQuantTritonInputs(
                q=qh,
                k_quant=pk.transpose(1, 2).contiguous(),
                v_quant=pv.transpose(1, 2).contiguous(),
                k_scales=ks.transpose(1, 2).contiguous(),
                v_scales=vs.transpose(1, 2).contiguous(),
                k_qjl_signs=ksign.transpose(1, 2).contiguous(),
                v_qjl_signs=vsign.transpose(1, 2).contiguous(),
                key_bits=int(past_key_value.codec.cfg.key_bits),
                value_bits=int(past_key_value.codec.cfg.value_bits),
                qjl_residual_scale=float(past_key_value.codec.cfg.qjl_residual_scale),
            )
            present_key_value = (k[:, -1:, :, :], v[:, -1:, :, :])
            self_attn = turboquant_attention(
                tq_inputs,
                is_causal=self_is_causal,
                fallback_k=kh[:, :, -1:, :],
                fallback_v=vh[:, :, -1:, :],
            )
            self_attn = torch.einsum(
                "blhd,df->blhf",
                self_attn.transpose(1, 2),
                self.R_v.to(self_attn.device),
            ).transpose(1, 2)
        else:
            self_attn = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=self_attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self_is_causal and self_attn_mask is None,
            )
        self_attn_out = self.dropout(self.self_out(self._merge_heads(self_attn)))
        x = x + self_attn_out

        # 2. Cross-Attention с собственной нормализацией
        x2 = self.norm_cross_attn(x)
        cq = self.cross_q(x2)
        if cross_kv_memory is None:
            ckv = self.cross_kv(memory)
            ck, cv = ckv.chunk(2, dim=-1)
        else:
            ck, cv = cross_kv_memory
        cqh, ckh, cvh = self._shape_heads(cq), self._shape_heads(ck), self._shape_heads(cv)
        cross_attn = F.scaled_dot_product_attention(
            cqh,
            ckh,
            cvh,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        cross_attn_out = self.dropout(self.cross_out(self._merge_heads(cross_attn)))
        x = x + cross_attn_out

        # 3. FFN с собственной нормализацией
        x3 = self.norm_ffn(x)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(x3)
            ffn_out = self.dropout(ffn_out)
        else:
            ffn_out = self.dropout(self.ffn(x3))
        x = x + ffn_out
        return x, present_key_value, aux_loss


class VARTransformer(nn.Module):
    def __init__(self, cfg: VARConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.hidden_size)

        self.token_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden) for vocab_size in cfg.level_vocab_sizes]
        )
        self.scale_embedding = nn.Embedding(len(cfg.level_vocab_sizes), hidden)
        self.rotary_emb = RotaryEmbedding(dim=hidden // int(cfg.num_heads))

        self.target_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_size))

        self.null_token_embedding = nn.Embedding(1, hidden)
        self.null_token_embedding.weight.data.fill_(0.0)

        self.blocks = nn.ModuleList(
            [
                SDPADecoderLayer(
                    hidden=hidden,
                    num_heads=int(cfg.num_heads),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                    use_fp8=bool(getattr(cfg, "use_fp8", False)),
                    use_moe=bool(getattr(cfg, "use_moe", False)),
                    num_experts=int(getattr(cfg, "num_experts", 8)),
                    moe_top_k=int(getattr(cfg, "moe_top_k", 2)),
                )
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = RMSNorm(hidden)

        self.heads = nn.ModuleList(
            [get_linear_layer(hidden, vocab_size, use_fp8=bool(getattr(cfg, "use_fp8", False))) for vocab_size in cfg.level_vocab_sizes]
        )

        self.early_exit_heads = nn.ModuleDict()
        for layer_idx in cfg.exit_layers:
            for scale_idx, vocab_size in enumerate(cfg.level_vocab_sizes):
                head_key = f"layer_{layer_idx}_scale_{scale_idx}"
                self.early_exit_heads[head_key] = get_linear_layer(hidden, vocab_size, use_fp8=bool(getattr(cfg, "use_fp8", False)))

    def _validate_token_ids(self, token_ids: torch.Tensor, *, level_idx: int, source: str) -> None:
        """Validate that token ids are in-range for the given hierarchical level.

        Args:
            token_ids: Integer token tensor used as embedding indices.
            level_idx: Hierarchical level index that selects embedding vocabulary.
            source: Human-readable token source for error messages.

        Raises:
            ValueError: If token ids contain values outside `[0, vocab_size)`.
        """
        vocab_size = int(self.cfg.level_vocab_sizes[level_idx])
        if token_ids.numel() == 0:
            return

        min_id = int(token_ids.min().item())
        max_id = int(token_ids.max().item())
        if min_id < 0 or max_id >= vocab_size:
            raise ValueError(
                f"{source} contains out-of-range token ids for level {level_idx}: "
                f"valid range is [0, {vocab_size - 1}], observed min={min_id}, max={max_id}."
            )

    def _compress_memory(self, x: torch.Tensor, target_tokens: int) -> torch.Tensor:
        seq_len = x.shape[1]
        keep = max(1, min(int(target_tokens), seq_len))
        if keep >= seq_len:
            return x
        return F.adaptive_avg_pool1d(x.transpose(1, 2), keep).transpose(1, 2)

    def _maybe_turboquant_memory(self, x: torch.Tensor) -> torch.Tensor:
        """Compress/decompress cross-attention memory for memory reduction simulation."""
        turbo_cfg = getattr(self.cfg, "turboquant_memory", None)
        if turbo_cfg is None or not isinstance(turbo_cfg, dict):
            return x
        enabled = bool(turbo_cfg.get("enabled", False))
        if not enabled:
            return x
        from src.var.generator import TurboQuantCodec, TurboQuantConfig

        num_heads = int(self.cfg.num_heads)
        head_dim = x.shape[-1] // num_heads
        if head_dim * num_heads != x.shape[-1]:
            return x
        shaped = x.view(x.shape[0], x.shape[1], num_heads, head_dim)
        cfg = TurboQuantConfig(
            key_bits=int(turbo_cfg.get("key_bits", 4)),
            value_bits=int(turbo_cfg.get("value_bits", 4)),
            qjl_residual_scale=float(turbo_cfg.get("qjl_residual_scale", 0.5)),
        )
        codec = TurboQuantCodec(cfg=cfg, head_dim=head_dim, device=x.device)
        quant, scale, zero, signs = codec._quantize_tensor(shaped, bits=cfg.value_bits)
        packed = codec._pack_bits(quant, cfg.value_bits)
        unpacked = codec._unpack_bits(packed, cfg.value_bits).to(torch.float32)
        deq = codec._dequantize_tensor(unpacked, scale, zero, signs)
        return deq.view_as(x).to(dtype=x.dtype)

    def _run_prefix_block_causal(
        self,
        block: SDPADecoderLayer,
        x_scale: torch.Tensor,
        current_context: torch.Tensor,
        rotary_freqs_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Run one prefix block with standard causal attention.

        Args:
            block: Decoder block to execute.
            x_scale: Prefix embeddings for one scale.
            current_context: Cross-attention memory.
            rotary_freqs_tgt: Rotary frequencies for prefix sequence.

        Returns:
            Tuple of (Updated prefix features, aux_loss).
        """
        out, _, l_aux = block(
            tgt=x_scale,
            memory=current_context,
            self_attn_mask=None,
            self_is_causal=True,
            rotary_freqs_tgt=rotary_freqs_tgt,
            past_key_value=None,
        )
        return out, l_aux

    def _encode_prefix_memories(
        self,
        prefix_inputs: list[torch.Tensor],
        *,
        batch_size: int,
        compact_memory_for_final_level: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Encode all prefix levels once and build per-target memories.

        Args:
            prefix_inputs: Hierarchical prefix tokens for each level.
            batch_size: Batch size used when there are no prefix levels.
            compact_memory_for_final_level: Whether to compress memory to the prior level length.

        Returns:
            Tuple of null memory tensor, per-target final memories, and accumulated aux_loss.
        """
        b = prefix_inputs[0].shape[0] if prefix_inputs else batch_size
        null_mem = self.null_token_embedding.weight.unsqueeze(0).expand(b, 1, -1)
        current_context = null_mem
        last_scale_memory = null_mem
        memories_by_target: list[torch.Tensor] = [null_mem]
        prefix_aux_loss = torch.tensor(0.0, device=null_mem.device, dtype=null_mem.dtype)

        for s_idx, scale_input in enumerate(prefix_inputs):
            self._validate_token_ids(scale_input, level_idx=s_idx, source=f"prefix_inputs[{s_idx}]")
            emb = self.token_embeddings[s_idx](scale_input)
            emb = emb + self.scale_embedding.weight[s_idx].view(1, 1, -1)
            seq_len = scale_input.shape[1]
            rotary_freqs_tgt = self.rotary_emb(emb, seq_len)

            x_scale = emb
            use_ckpt = (
                bool(self.cfg.gradient_checkpointing) and self.training and torch.is_grad_enabled()
            )
            local_radius = int(getattr(self.cfg, "local_attention_radius", 0))
            for block in self.blocks:
                if local_radius > 0 and seq_len > 1:
                    x_scale, l_aux = self._run_prefix_block_causal(
                        block=block,
                        x_scale=x_scale,
                        current_context=current_context,
                        rotary_freqs_tgt=rotary_freqs_tgt,
                    )
                elif use_ckpt:
                    x_scale, _, l_aux = checkpoint(
                        block,
                        use_reentrant=False,
                        tgt=x_scale,
                        memory=current_context,
                        self_attn_mask=None,
                        self_is_causal=False,
                        rotary_freqs_tgt=rotary_freqs_tgt,
                        past_key_value=None,
                    )
                else:
                    x_scale, _, l_aux = block(
                        tgt=x_scale,
                        memory=current_context,
                        self_attn_mask=None,
                        self_is_causal=False,
                        rotary_freqs_tgt=rotary_freqs_tgt,
                        past_key_value=None,
                    )
                prefix_aux_loss = prefix_aux_loss + l_aux

            encoded = self.norm(x_scale)
            compress_tokens = (
                self.cfg.level_lengths[max(0, s_idx - 1)]
                if compact_memory_for_final_level
                else encoded.shape[1]
            )
            compressed = self._compress_memory(encoded, target_tokens=compress_tokens)
            last_scale_memory = compressed
            current_context = self._maybe_turboquant_memory(
                torch.cat([null_mem, compressed], dim=1)
            )
            memories_by_target.append(
                self._maybe_turboquant_memory(torch.cat([null_mem, last_scale_memory], dim=1))
            )
        return null_mem, memories_by_target, prefix_aux_loss

    def forward(
        self,
        prefix_inputs: list[torch.Tensor],
        *,
        target_level: int | None = None,
        current_level_input: torch.Tensor | None = None,
        batch_size: int | None = None,
        cfg_scale: float = 1.0,
        return_early_outputs: bool = False,
        compact_memory_for_final_level: bool = True,
        precomputed_final_memory: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | RingKVCacheView] | None = None,
        use_cache: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[tuple[torch.Tensor, list[torch.Tensor]], torch.Tensor]
    ):
        if cfg_scale != 1.0 and prefix_inputs:
            out_cond = self.forward(
                prefix_inputs,
                target_level=target_level,
                current_level_input=current_level_input,
                batch_size=batch_size,
                cfg_scale=1.0,
                return_early_outputs=return_early_outputs,
                compact_memory_for_final_level=compact_memory_for_final_level,
            )
            uncond_prefixes = [torch.zeros_like(p) for p in prefix_inputs]
            out_uncond = self.forward(
                uncond_prefixes,
                target_level=target_level,
                current_level_input=current_level_input,
                batch_size=batch_size,
                cfg_scale=1.0,
                return_early_outputs=return_early_outputs,
                compact_memory_for_final_level=compact_memory_for_final_level,
            )
            if return_early_outputs:
                (logits_cond, early_cond), aux_loss_cond = out_cond
                (logits_uncond, early_uncond), aux_loss_uncond = out_uncond
                final_logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                final_early = [u + cfg_scale * (c - u) for c, u in zip(early_cond, early_uncond, strict=True)]  # noqa: E501
                return (final_logits, final_early), aux_loss_cond + aux_loss_uncond
            out_cond_tensor, aux_loss_cond = out_cond
            out_uncond_tensor, aux_loss_uncond = out_uncond
            return out_uncond_tensor + cfg_scale * (out_cond_tensor - out_uncond_tensor), aux_loss_cond + aux_loss_uncond

        if prefix_inputs:
            b = prefix_inputs[0].shape[0]
        elif current_level_input is not None:
            b = current_level_input.shape[0]
        else:
            b = batch_size if batch_size else 1

        target_idx = target_level if target_level is not None else len(prefix_inputs)
        total_aux_loss = torch.tensor(0.0, device=x.device if 'x' in locals() else prefix_inputs[0].device if prefix_inputs else self.target_token.device, dtype=self.target_token.dtype)
        if precomputed_final_memory is None:
            _, memories_by_target, prefix_aux = self._encode_prefix_memories(
                prefix_inputs[:target_idx],
                batch_size=b,
                compact_memory_for_final_level=compact_memory_for_final_level,
            )
            final_memory = memories_by_target[target_idx]
            total_aux_loss = total_aux_loss + prefix_aux
        else:
            final_memory = precomputed_final_memory
        projected_final_memory = [
            tuple(block.cross_kv(final_memory).chunk(2, dim=-1)) for block in self.blocks
        ]

        if current_level_input is not None:
            self._validate_token_ids(
                current_level_input, level_idx=target_idx, source="current_level_input"
            )
            x = self.token_embeddings[target_idx](current_level_input)
            target_len = current_level_input.shape[1]
        else:
            target_len = self.cfg.level_lengths[target_idx]
            x = self.target_token.expand(b, target_len, -1)

        x = x + self.scale_embedding.weight[target_idx].view(1, 1, -1)
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_past = past_key_values[0]
            if isinstance(first_past, RingKVCacheView):
                past_len = int(first_past.current_length)
            else:
                past_len = int(first_past[0].shape[1])
        rotary_freqs_tgt = self.rotary_emb(x, target_len, start_pos=past_len)

        # Keep attn_mask=None so SDPA can stay on Flash Attention kernels.
        # Dense custom masks often force fallback to math backend with O(N^2) memory.
        self_mask = None
        self_is_causal = True

        early_outputs = []
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        use_ckpt = (
            bool(self.cfg.gradient_checkpointing) and self.training and torch.is_grad_enabled()
        )
        if use_cache and use_ckpt:
            raise ValueError("KV-cache is not supported with gradient checkpointing enabled.")
        for layer_idx, block in enumerate(self.blocks):
            layer_past = None
            if past_key_values is not None and layer_idx < len(past_key_values):
                layer_past = past_key_values[layer_idx]
            if use_ckpt:
                x, present, l_aux = checkpoint(
                    block,
                    use_reentrant=False,
                    tgt=x,
                    memory=final_memory,
                    self_attn_mask=self_mask,
                    self_is_causal=self_is_causal,
                    rotary_freqs_tgt=rotary_freqs_tgt,
                    past_key_value=layer_past,
                    cross_kv_memory=projected_final_memory[layer_idx],
                )
            else:
                x, present, l_aux = block(
                    tgt=x,
                    memory=final_memory,
                    self_attn_mask=self_mask,
                    self_is_causal=self_is_causal,
                    rotary_freqs_tgt=rotary_freqs_tgt,
                    past_key_value=layer_past,
                    cross_kv_memory=projected_final_memory[layer_idx],
                )
            total_aux_loss = total_aux_loss + l_aux
            if use_cache:
                present_key_values.append(present)
            if return_early_outputs and layer_idx in self.cfg.exit_layers:
                target_features = self.norm(x)
                head_key = f"layer_{layer_idx}_scale_{target_idx}"
                logits = self.early_exit_heads[head_key](target_features)
                early_outputs.append(logits)

        out_features = self.heads[target_idx](self.norm(x))
        if use_cache:
            return out_features, present_key_values
        if return_early_outputs:
            return (out_features, early_outputs), total_aux_loss
        return out_features, total_aux_loss
