import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from vector_quantize_pytorch import FSQ

from src.var.generator import KVCacheRingBuffer, TurboQuantConfig, thermodynamic_sampling_with_stats
from src.var.model import RingKVCacheView, RotaryEmbedding, SDPADecoderLayer
from src.vqvae.cnn_blocks import HierarchicalDownsample1D, HierarchicalUpsample1D
from src.vqvae.loss import (
    contrastive_latent_loss,
    feature_matching_loss,
    kl_divergence_loss,
    token_level_contrastive_loss,
)
from src.vqvae.sdpa_blocks import SDPAEncoder


class SemanticTextVQVAE(nn.Module):
    """Semantic VQ-VAE model for token reconstruction and semantic quantization."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_semantic_tokens: int = 4096,
        semantic_sequence_length: int = 1,
        pad_token_id: int = 0,
        semantic_pad_token_id: int = 0,
        max_position_embeddings: int = 2048,
        use_turboquant_kv: bool = False,
        turboquant_key_bits: int = 4,
        turboquant_value_bits: int = 4,
        turboquant_qjl_residual_scale: float = 0.5,
        gradient_checkpointing: bool = False,
        use_rotary_embeddings: bool = True,
        use_triton_ema: bool = False,
        semantic_mask_prob: float = 0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = int(pad_token_id)
        self.semantic_pad_token_id = int(semantic_pad_token_id)
        self.semantic_sequence_length = max(1, int(semantic_sequence_length))
        self.max_position_embeddings = int(max_position_embeddings)
        self.use_turboquant_kv = bool(use_turboquant_kv)
        self.turboquant_key_bits = int(turboquant_key_bits)
        self.turboquant_value_bits = int(turboquant_value_bits)
        self.turboquant_qjl_residual_scale = float(turboquant_qjl_residual_scale)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.use_rotary_embeddings = bool(use_rotary_embeddings)
        self.use_triton_ema = bool(use_triton_ema)
        self.semantic_mask_prob = float(semantic_mask_prob)

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=self.pad_token_id)
        self.pos_embedding = nn.Embedding(self.max_position_embeddings, hidden_size)

        self.encoder = SDPAEncoder(
            hidden=hidden_size,
            num_heads=8,
            depth=4,
            mlp_ratio=4.0,
            dropout=0.1,
        )

        self.compression_rate = 4

        self.downsample = HierarchicalDownsample1D(
            dim=hidden_size, compression_rate=self.compression_rate, num_blocks=2
        )

        self.upsample = HierarchicalUpsample1D(
            dim=hidden_size, compression_rate=self.compression_rate, num_blocks=2
        )

        # =====================================================================
        # 3. Finite Scalar Quantization (FSQ)
        # =====================================================================
        self.levels = [8, 8, 8, 8]  # 8^4 = 4096 levels
        self.fsq_dim = len(self.levels)
        self.pre_quant_proj = nn.Linear(hidden_size, self.fsq_dim)
        self.quantizer = FSQ(levels=self.levels)
        self.post_quant_proj = nn.Linear(self.fsq_dim, hidden_size)

        # Компоненты декодера
        self.decoder_layers = nn.ModuleList(
            [
                SDPADecoderLayer(hidden=hidden_size, num_heads=8, mlp_ratio=4.0, dropout=0.1)
                for _ in range(4)
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // 8, max_position_embeddings=self.max_position_embeddings
        )
        for layer in self.decoder_layers:
            layer.use_turboquant = self.use_turboquant_kv

    def _pool_to_semantic_length(self, sequence: torch.Tensor) -> torch.Tensor:
        """Сжимает последовательность в 4 раза с сохранением пространственной информации."""
        channel_first = sequence.transpose(1, 2).contiguous()
        compressed = self.downsample(channel_first)
        return compressed.transpose(1, 2).contiguous()

    def _expand_to_full_length(self, latents: torch.Tensor) -> torch.Tensor:
        """Разжимает последовательность обратно в исходную длину."""
        channel_first = latents.transpose(1, 2).contiguous()
        expanded = self.upsample(channel_first)
        return expanded.transpose(1, 2).contiguous()

    def _pool_semantic_tokens(
        self, encoded: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if padding_mask is None:
            return self._pool_to_semantic_length(encoded)

        valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=encoded.dtype)
        masked_encoded = encoded * valid_mask
        # Свертка сама выполнит пространственный пулинг нужных фичей
        return self._pool_to_semantic_length(masked_encoded)

    def _pool_semantic_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """Pool token padding mask to semantic-token resolution using Max Pooling."""
        pm_float = padding_mask.float().unsqueeze(1)
        pooled = F.max_pool1d(
            pm_float, kernel_size=self.compression_rate, stride=self.compression_rate
        )
        return pooled.squeeze(1) > 0.5

    def _apply_semantic_padding_mask(
        self,
        semantic_states: torch.Tensor,
        semantic_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if semantic_padding_mask is None:
            return semantic_states
        valid = (~semantic_padding_mask).unsqueeze(-1).to(dtype=semantic_states.dtype)
        return semantic_states * valid

    def _resolve_padding_mask(
        self,
        bpe_tokens: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        inferred_padding_mask = bpe_tokens.eq(int(self.pad_token_id))
        if padding_mask is None:
            return inferred_padding_mask
        return padding_mask.bool() | inferred_padding_mask

    def _build_turboquant_config(self) -> TurboQuantConfig | None:
        if not self.use_turboquant_kv:
            return None
        return TurboQuantConfig(
            key_bits=self.turboquant_key_bits,
            value_bits=self.turboquant_value_bits,
            qjl_residual_scale=self.turboquant_qjl_residual_scale,
        )

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(int(seq_len), device=device)

    def _position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if int(seq_len) > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings="
                f"{self.max_position_embeddings}."
            )
        return torch.arange(seq_len, device=device).unsqueeze(0)

    def _run_decoder(
        self,
        *,
        tgt_emb: torch.Tensor,
        memory: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | RingKVCacheView] | None = None,
        incremental: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if past_key_values is not None and len(past_key_values) != len(self.decoder_layers):
            raise ValueError("past_key_values length must match decoder layers.")
        hidden = tgt_emb
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        self_attn_mask = (
            None if incremental else self._build_causal_mask(tgt_emb.size(1), tgt_emb.device)
        )
        for layer_idx, layer in enumerate(self.decoder_layers):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            rotary_freqs_tgt = self.rotary_emb(hidden, seq_len=hidden.shape[1])
            if self.gradient_checkpointing and self.training and layer_past is None:
                hidden, present = checkpoint(
                    lambda a, b, c, layer_=layer: layer_(
                        tgt=a,
                        memory=b,
                        self_attn_mask=self_attn_mask,
                        self_is_causal=not incremental,
                        rotary_freqs_tgt=c,
                        past_key_value=None,
                    ),
                    hidden,
                    memory,
                    rotary_freqs_tgt,
                    use_reentrant=False,
                )
            else:
                hidden, present = layer(
                    tgt=hidden,
                    memory=memory,
                    self_attn_mask=self_attn_mask,
                    self_is_causal=not incremental,
                    rotary_freqs_tgt=rotary_freqs_tgt,
                    past_key_value=layer_past,
                )
            present_key_values.append(present)
        return self.decoder_norm(hidden), present_key_values

    def encode_sentence(
        self, bpe_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = self._resolve_padding_mask(bpe_tokens, padding_mask)
        positions = self._position_ids(bpe_tokens.size(1), bpe_tokens.device)
        x = self.embedding(bpe_tokens) + self.pos_embedding(positions)
        rotary_freqs = (
            self.rotary_emb(x, seq_len=x.shape[1]) if self.use_rotary_embeddings else None
        )
        encoded = self.encoder(x, key_padding_mask=padding_mask, rotary_freqs=rotary_freqs)

        semantic_inputs = self._pool_semantic_tokens(encoded, padding_mask=padding_mask)
        projected_inputs = self.pre_quant_proj(semantic_inputs)
        _, indices = self.quantizer(projected_inputs)
        return indices, torch.tensor(0.0, device=indices.device)

    def decode_from_semantic_indices(
        self,
        semantic_indices: torch.Tensor,
        *,
        max_length: int,
        bos_token_id: int,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        if int(max_length) < 1:
            raise ValueError("max_length must be >= 1.")
        if float(temperature) <= 0:
            raise ValueError("temperature must be > 0.")
        if not 0 < float(top_p) <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if semantic_indices.dim() == 1:
            semantic_indices = semantic_indices.unsqueeze(1)
        if semantic_indices.dim() != 2:
            raise ValueError("semantic_indices must have shape (B,) or (B, S).")

        semantic_padding_mask = semantic_indices.eq(int(self.semantic_pad_token_id))

        # Получаем векторы из индексов
        if hasattr(self.quantizer, "indices_to_codes"):
            fsq_features = self.quantizer.indices_to_codes(semantic_indices)
        elif hasattr(self.quantizer, "get_codes_from_indices"):
            fsq_features = self.quantizer.get_codes_from_indices(semantic_indices)
        else:
            raise AttributeError(
                "Quantizer does not support indices_to_codes or get_codes_from_indices"
            )

        semantic_features = self.post_quant_proj(fsq_features)

        memory = self._apply_semantic_padding_mask(semantic_features, semantic_padding_mask)

        batch_size = semantic_indices.shape[0]
        generated = torch.full(
            (batch_size, 1),
            int(bos_token_id),
            dtype=torch.long,
            device=semantic_indices.device,
        )

        turbo_cfg = self._build_turboquant_config()
        cache_ring_buffer = KVCacheRingBuffer(
            max_window=int(max_length), turboquant_config=turbo_cfg
        )
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | RingKVCacheView] | None = None
        for step_idx in range(int(max_length) - 1):
            if step_idx == 0:
                step_tokens = generated
                step_positions = self._position_ids(step_tokens.size(1), step_tokens.device)
            else:
                step_tokens = generated[:, -1:]
                step_positions = self._position_ids(generated.size(1), generated.device)[:, -1:]
            step_emb = self.embedding(step_tokens) + self.pos_embedding(step_positions)
            decoded, past_key_values = self._run_decoder(
                tgt_emb=step_emb,
                memory=memory,
                past_key_values=past_key_values,
                incremental=step_idx > 0,
            )
            past_key_values = cache_ring_buffer.update(past_key_values)
            logits = self.lm_head(decoded)[:, -1, :]
            next_token, _, _ = thermodynamic_sampling_with_stats(
                logits=logits,
                temperature=float(temperature),
                top_p=float(top_p),
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            if eos_token_id is not None and bool((next_token == int(eos_token_id)).all()):
                break

        return generated

    def forward(
        self, bpe_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = self._resolve_padding_mask(bpe_tokens, padding_mask)

        positions = self._position_ids(bpe_tokens.size(1), bpe_tokens.device)
        x = self.embedding(bpe_tokens) + self.pos_embedding(positions)
        rotary_freqs = (
            self.rotary_emb(x, seq_len=x.shape[1]) if self.use_rotary_embeddings else None
        )
        encoded = self.encoder(x, key_padding_mask=padding_mask, rotary_freqs=rotary_freqs)

        semantic_inputs = self._pool_semantic_tokens(encoded, padding_mask=padding_mask)
        semantic_padding_mask = self._pool_semantic_padding_mask(padding_mask)

        projected_inputs = self.pre_quant_proj(semantic_inputs)
        quantized_fsq, indices = self.quantizer(projected_inputs)
        quantized = self.post_quant_proj(quantized_fsq)
        vq_loss = torch.tensor(0.0, device=quantized.device)

        tgt_tokens = bpe_tokens[:, :-1]
        tgt_positions = positions[:, :-1]
        tgt_emb = self.embedding(tgt_tokens) + self.pos_embedding(tgt_positions)

        memory = self._apply_semantic_padding_mask(quantized, semantic_padding_mask)

        # Masked Latent Modeling (MLM)
        if self.training and self.semantic_mask_prob > 0.0:
            rand_tensor = torch.rand(memory.shape[:2], device=memory.device)
            mlm_mask = (rand_tensor < self.semantic_mask_prob) & (~semantic_padding_mask)
            memory[mlm_mask] = 0.0

        decoded, _ = self._run_decoder(
            tgt_emb=tgt_emb,
            memory=memory,
            past_key_values=None,
            incremental=False,
        )

        valid_mask = ~padding_mask[:, :-1]

        active_decoded = decoded[valid_mask]
        active_logits = self.lm_head(active_decoded)

        active_targets = bpe_tokens[:, 1:][valid_mask]

        # 1. Label Smoothing Reconstruction Loss
        recon_loss = F.cross_entropy(
            active_logits, active_targets, reduction="mean", label_smoothing=0.1
        )

        # 2. Latent Feature Matching Loss
        fm_loss = feature_matching_loss(semantic_inputs, quantized, mask=semantic_padding_mask)

        # 3. Sequence-Level Contrastive Latent Loss
        cl_loss = contrastive_latent_loss(semantic_inputs, mask=semantic_padding_mask)

        # 4. Token-Level Contrastive Loss
        tl_cl_loss = token_level_contrastive_loss(semantic_inputs, mask=semantic_padding_mask)

        # 5. KL-Divergence Prior Loss
        kl_loss = kl_divergence_loss(semantic_inputs, mask=semantic_padding_mask)

        total_loss = (
            recon_loss + vq_loss + 0.5 * fm_loss + 0.1 * cl_loss + 0.1 * tl_cl_loss + 0.01 * kl_loss
        )

        logits = torch.zeros(
            bpe_tokens.size(0),
            bpe_tokens.size(1) - 1,
            self.vocab_size,
            device=active_logits.device,
            dtype=active_logits.dtype,
        )
        logits[valid_mask] = active_logits

        return logits, total_loss
