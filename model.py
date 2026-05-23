import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import VARConfig


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
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


class SDPADecoderLayer(nn.Module):
    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        if hidden % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.self_qkv = nn.Linear(hidden, hidden * 3)
        self.cross_q = nn.Linear(hidden, hidden)
        self.cross_kv = nn.Linear(hidden, hidden * 2)
        self.self_out = nn.Linear(hidden, hidden)
        self.cross_out = nn.Linear(hidden, hidden)

        ff_hidden = max(hidden, int(hidden * float(mlp_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, hidden),
        )

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, l, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, h * d)

    def _compress_memory(self, x: torch.Tensor, target_tokens: int) -> torch.Tensor:
        seq_len = x.shape[1]
        keep = max(1, min(int(target_tokens), seq_len))
        if keep >= seq_len:
            return x
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), keep).transpose(1, 2)
        return pooled

    def forward(
        self,
        *,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_is_causal: bool = False,
        rotary_freqs_tgt: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = tgt

        x1 = self.norm1(x)
        qkv = self.self_qkv(x1)
        q, k, v = qkv.chunk(3, dim=-1)
        b, l, _ = x.shape
        q = q.view(b, l, self.num_heads, self.head_dim)
        k = k.view(b, l, self.num_heads, self.head_dim)
        v = v.view(b, l, self.num_heads, self.head_dim)
        if rotary_freqs_tgt is not None:
            q, k = apply_rotary_pos_emb(q, k, rotary_freqs_tgt)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        present_key_value = (k, v)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        self_attn = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=self_attn_mask,
            dropout_p=0.0,
            is_causal=self_is_causal and self_attn_mask is None,
        )
        x = x + self.dropout(self.self_out(self._merge_heads(self_attn)))

        x2 = self.norm2(x)
        cq = self.cross_q(x2)
        ckv = self.cross_kv(memory)
        ck, cv = ckv.chunk(2, dim=-1)
        cqh, ckh, cvh = self._shape_heads(cq), self._shape_heads(ck), self._shape_heads(cv)
        cross_attn = F.scaled_dot_product_attention(
            cqh,
            ckh,
            cvh,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        x = x + self.dropout(self.cross_out(self._merge_heads(cross_attn)))

        x3 = self.norm3(x)
        x = x + self.dropout(self.ffn(x3))
        return x, present_key_value


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
                    dropout=0.1,
                )
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(hidden)

        self.heads = nn.ModuleList([nn.Linear(hidden, vocab_size) for vocab_size in cfg.level_vocab_sizes])

        self.early_exit_heads = nn.ModuleDict()
        for layer_idx in cfg.exit_layers:
            for scale_idx, vocab_size in enumerate(cfg.level_vocab_sizes):
                head_key = f"layer_{layer_idx}_scale_{scale_idx}"
                self.early_exit_heads[head_key] = nn.Linear(hidden, vocab_size)


    def _compress_memory(self, x: torch.Tensor, target_tokens: int) -> torch.Tensor:
        seq_len = x.shape[1]
        keep = max(1, min(int(target_tokens), seq_len))
        if keep >= seq_len:
            return x
        return F.adaptive_avg_pool1d(x.transpose(1, 2), keep).transpose(1, 2)

    def _make_local_bidirectional_mask(self, seq_len: int, radius: int, device: torch.device) -> torch.Tensor:
        """Return a local bidirectional attention mask for SDPA.

        Tokens can attend to neighbors on both sides inside ``radius`` positions.
        Anything outside the local window is masked out.
        """
        idx = torch.arange(seq_len, device=device)
        row = idx.unsqueeze(1)
        col = idx.unsqueeze(0)
        invalid = (col - row).abs() > radius
        return invalid.view(1, 1, seq_len, seq_len)

    def _build_prefix_self_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build a local self-attention mask for prefix encoding.

        Args:
            seq_len: Prefix token sequence length.
            device: Device where the mask should be allocated.

        Returns:
            Local bidirectional mask for SDPA when configured and meaningful,
            otherwise ``None`` to allow dense attention kernels.
        """
        local_radius = int(getattr(self.cfg, "local_attention_radius", 0))
        if local_radius <= 0 or seq_len <= 1:
            return None
        return self._make_local_bidirectional_mask(seq_len, local_radius, device)

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
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if cfg_scale != 1.0 and prefix_inputs:
            out_cond = self.forward(prefix_inputs, target_level=target_level, current_level_input=current_level_input, batch_size=batch_size, cfg_scale=1.0, return_early_outputs=return_early_outputs, compact_memory_for_final_level=compact_memory_for_final_level)
            uncond_prefixes = [torch.zeros_like(p) for p in prefix_inputs]
            out_uncond = self.forward(uncond_prefixes, target_level=target_level, current_level_input=current_level_input, batch_size=batch_size, cfg_scale=1.0, return_early_outputs=return_early_outputs, compact_memory_for_final_level=compact_memory_for_final_level)
            if return_early_outputs:
                logits_cond, early_cond = out_cond
                logits_uncond, early_uncond = out_uncond
                final_logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                final_early = [u + cfg_scale * (c - u) for c, u in zip(early_cond, early_uncond)]
                return final_logits, final_early
            return out_uncond + cfg_scale * (out_cond - out_uncond)

        if prefix_inputs:
            b = prefix_inputs[0].shape[0]
            device = prefix_inputs[0].device
        elif current_level_input is not None:
            b = current_level_input.shape[0]
            device = current_level_input.device
        else:
            b = batch_size if batch_size else 1
            device = self.target_token.device

        target_idx = target_level if target_level is not None else len(prefix_inputs)

        null_mem = self.null_token_embedding.weight.unsqueeze(0).expand(b, 1, -1)
        current_context = null_mem
        last_scale_memory = null_mem

        for s_idx, scale_input in enumerate(prefix_inputs):
            if s_idx >= target_idx:
                break
            emb = self.token_embeddings[s_idx](scale_input)
            emb = emb + self.scale_embedding.weight[s_idx].view(1, 1, -1)
            l = scale_input.shape[1]
            rotary_freqs_tgt = self.rotary_emb(emb, l)

            x_scale = emb
            use_ckpt = bool(self.cfg.gradient_checkpointing) and self.training and torch.is_grad_enabled()
            prefix_mask = self._build_prefix_self_attention_mask(l, device)
            for block in self.blocks:
                if use_ckpt:
                    x_scale, _ = checkpoint(
                        block,
                        use_reentrant=False,
                        tgt=x_scale,
                        memory=current_context,
                        self_attn_mask=prefix_mask,
                        self_is_causal=False,
                        rotary_freqs_tgt=rotary_freqs_tgt,
                        past_key_value=None,
                    )
                else:
                    x_scale, _ = block(
                        tgt=x_scale,
                        memory=current_context,
                        self_attn_mask=prefix_mask,
                        self_is_causal=False,
                        rotary_freqs_tgt=rotary_freqs_tgt,
                        past_key_value=None,
                    )

            encoded = self.norm(x_scale)
            compress_tokens = self.cfg.level_lengths[max(0, s_idx - 1)] if compact_memory_for_final_level else encoded.shape[1]
            compressed = self._compress_memory(encoded, target_tokens=compress_tokens)
            last_scale_memory = compressed
            current_context = torch.cat([null_mem, compressed], dim=1)

        final_memory = torch.cat([null_mem, last_scale_memory], dim=1) if target_idx > 0 else null_mem

        if current_level_input is not None:
            x = self.token_embeddings[target_idx](current_level_input)
            target_len = current_level_input.shape[1]
        else:
            target_len = self.cfg.level_lengths[target_idx]
            x = self.target_token.expand(b, target_len, -1)

        x = x + self.scale_embedding.weight[target_idx].view(1, 1, -1)
        rotary_freqs_tgt = self.rotary_emb(x, target_len)

        # Keep attn_mask=None so SDPA can stay on Flash Attention kernels.
        # Dense custom masks often force fallback to math backend with O(N^2) memory.
        self_mask = None
        self_is_causal = True

        early_outputs = []
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        use_ckpt = bool(self.cfg.gradient_checkpointing) and self.training and torch.is_grad_enabled()
        if use_cache and use_ckpt:
            raise ValueError("KV-cache is not supported with gradient checkpointing enabled.")
        for layer_idx, block in enumerate(self.blocks):
            layer_past = None
            if past_key_values is not None and layer_idx < len(past_key_values):
                layer_past = past_key_values[layer_idx]
            if use_ckpt:
                x, present = checkpoint(
                    block,
                    use_reentrant=False,
                    tgt=x,
                    memory=final_memory,
                    self_attn_mask=self_mask,
                    self_is_causal=self_is_causal,
                    rotary_freqs_tgt=rotary_freqs_tgt,
                    past_key_value=layer_past,
                )
            else:
                x, present = block(
                    tgt=x,
                    memory=final_memory,
                    self_attn_mask=self_mask,
                    self_is_causal=self_is_causal,
                    rotary_freqs_tgt=rotary_freqs_tgt,
                    past_key_value=layer_past,
                )
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
            return out_features, early_outputs
        return out_features
