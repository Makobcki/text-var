import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from src.var.model import apply_rotary_pos_emb, RMSNorm, SwiGLU


class SDPAEncoderLayer(nn.Module):
    """Transformer encoder layer implemented with SDPA primitives."""

    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float, dropout: float = 0.1) -> None:
        """Initialize encoder layer.

        Args:
            hidden: Hidden size.
            num_heads: Number of attention heads.
            mlp_ratio: Feed-forward expansion ratio.
            dropout: Dropout probability.

        Raises:
            ValueError: If hidden size is not divisible by num_heads.
        """
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden // self.num_heads
        self.attn_dropout = float(dropout)

        self.qkv = nn.Linear(self.hidden, self.hidden * 3)
        self.out_proj = nn.Linear(self.hidden, self.hidden)
        ff_hidden = max(self.hidden, int(self.hidden * float(mlp_ratio) * 2 / 3))
        self.ffn = SwiGLU(self.hidden, ff_hidden, self.hidden, dropout)
        self.norm1 = RMSNorm(self.hidden)
        self.dropout = nn.Dropout(dropout)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

    def _build_padding_mask(
        self,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Convert key-padding mask to SDPA boolean attention mask.

        Args:
            key_padding_mask: Optional ``(B, T)`` mask where ``True`` means padding.

        Returns:
            Optional SDPA-compatible mask with shape ``(B, 1, 1, T)`` where
            ``True`` indicates positions that are allowed for attention.
        """
        if key_padding_mask is None:
            return None
        return (~key_padding_mask).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        rotary_freqs: torch.Tensor | None = None,
        unpad_info: tuple | None = None,
    ) -> torch.Tensor:
        """Run encoder layer forward pass.

        Args:
            x: Input tensor of shape ``(B, T, H)``.
            key_padding_mask: Optional ``(B, T)`` mask where ``True`` means padding.
            rotary_freqs: Optional RoPE frequencies for sequence length.

        Returns:
            Output tensor of shape ``(B, T, H)``.
        """
        x1 = self.norm1(x)
        qkv = self.qkv(x1)
        q, k, v = qkv.chunk(3, dim=-1)

        if unpad_info is not None:
            indices, cu_seqlens, max_seqlen_in_batch, total_tokens, _ = unpad_info
            q = q.view(1, total_tokens, self.num_heads, self.head_dim)
            k = k.view(1, total_tokens, self.num_heads, self.head_dim)
            v = v.view(1, total_tokens, self.num_heads, self.head_dim)
            if rotary_freqs is not None:
                q, k = apply_rotary_pos_emb(q, k, rotary_freqs)
            
            from flash_attn import flash_attn_varlen_func
            q_var = q.squeeze(0)
            k_var = k.squeeze(0)
            v_var = v.squeeze(0)
            
            attn = flash_attn_varlen_func(
                q_var, k_var, v_var,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen_in_batch,
                max_seqlen_k=max_seqlen_in_batch,
                dropout_p=self.attn_dropout if self.training else 0.0,
                causal=False,
            )
            attn = attn.view(1, total_tokens, self.hidden)
            attn_out = self.dropout(self.out_proj(attn))
        else:
            q = q.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            k = k.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            if rotary_freqs is not None:
                q, k = apply_rotary_pos_emb(q, k, rotary_freqs)
            q = q.reshape(x.shape[0], x.shape[1], self.hidden)
            k = k.reshape(x.shape[0], x.shape[1], self.hidden)
            qh = self._shape_heads(q)
            kh = self._shape_heads(k)
            vh = self._shape_heads(v)
            attn_mask = self._build_padding_mask(key_padding_mask)
            attn = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_out = self.dropout(self.out_proj(self._merge_heads(attn)))
        
        ffn_out = self.dropout(self.ffn(x1))
        return x + attn_out + ffn_out


class SDPAEncoder(nn.Module):
    """Encoder stack built from SDPAEncoderLayer blocks."""

    def __init__(
        self, hidden: int, num_heads: int, depth: int, mlp_ratio: float, dropout: float = 0.1, gradient_checkpointing: bool = False, use_unpadding: bool = False
    ) -> None:
        """Initialize encoder stack.

        Args:
            hidden: Hidden size.
            num_heads: Number of attention heads.
            depth: Number of layers.
            mlp_ratio: Feed-forward expansion ratio.
            dropout: Dropout probability.
        """
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.use_unpadding = use_unpadding
        self.layers = nn.ModuleList(
            [
                SDPAEncoderLayer(
                    hidden=hidden, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = RMSNorm(hidden)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        rotary_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run encoder stack.

        Args:
            x: Input tensor of shape ``(B, T, H)``.
            key_padding_mask: Optional ``(B, T)`` padding mask.
            rotary_freqs: Optional RoPE frequencies for sequence length.

        Returns:
            Encoded tensor of shape ``(B, T, H)``.
        """
        unpad_info = None
        if self.use_unpadding and key_padding_mask is not None:
            try:
                from flash_attn.bert_padding import unpad_input, pad_input
                attention_mask = (~key_padding_mask).long()
                x_unpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(x, attention_mask)
                x = x_unpad.unsqueeze(0)
                unpad_info = (indices, cu_seqlens, max_seqlen_in_batch, x.shape[1], attention_mask)

                if rotary_freqs is not None:
                    freqs_expanded = rotary_freqs.unsqueeze(0).expand(key_padding_mask.shape[0], -1, -1)
                    freqs_unpad, _, _, _ = unpad_input(freqs_expanded, attention_mask)
                    rotary_freqs = freqs_unpad
            except ImportError:
                pass

        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        hidden = x
        for layer in self.layers:
            if use_ckpt:
                hidden = checkpoint(
                    layer,
                    hidden,
                    use_reentrant=False,
                    key_padding_mask=key_padding_mask,
                    rotary_freqs=rotary_freqs,
                    unpad_info=unpad_info,
                )
            else:
                hidden = layer(hidden, key_padding_mask=key_padding_mask, rotary_freqs=rotary_freqs, unpad_info=unpad_info)
        
        hidden = self.norm(hidden)

        if unpad_info is not None:
            from flash_attn.bert_padding import pad_input
            indices, cu_seqlens, max_seqlen_in_batch, total_tokens, attention_mask = unpad_info
            hidden_squeeze = hidden.squeeze(0)
            hidden = pad_input(hidden_squeeze, indices, key_padding_mask.shape[0], key_padding_mask.shape[1])

        return hidden
