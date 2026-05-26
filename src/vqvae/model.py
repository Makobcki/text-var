from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.var.generator import KVCacheRingBuffer, TurboQuantConfig, thermodynamic_sampling_with_stats
from src.var.model import RingKVCacheView, RotaryEmbedding, SDPADecoderLayer
from src.vqvae.ema_ops import ema_update_torch, ema_update_triton
from src.vqvae.sdpa_blocks import SDPAEncoder


class VectorQuantizer(nn.Module):
    """Vector quantizer with EMA codebook updates."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_triton_ema: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.use_triton_ema = bool(use_triton_ema)

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())
        self.last_commitment_loss: torch.Tensor | None = None

    def _compute_distances_chunked(
        self,
        flat_inputs: torch.Tensor,
        *,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Compute squared L2 distances to the codebook with bounded peak memory.

        Uses ``||x-w||^2 = ||x||^2 + ||w||^2 - 2xw^T`` to map the operation to
        matrix multiplication kernels, which are typically faster than ``torch.cdist``.

        Args:
            flat_inputs: Flattened input tensor with shape ``(N, D)``.
            chunk_size: Number of rows processed per chunk.

        Returns:
            Squared distance tensor with shape ``(N, num_embeddings)``.

        Raises:
            ValueError: If ``chunk_size`` is smaller than 1.
        """
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1.")
        codebook_weight = self.codebook.weight
        codebook_norm = codebook_weight.pow(2).sum(dim=1).unsqueeze(0)
        distances: list[torch.Tensor] = []
        total_rows = int(flat_inputs.size(0))
        for start_idx in range(0, total_rows, int(chunk_size)):
            end_idx = min(start_idx + int(chunk_size), total_rows)
            current_chunk = flat_inputs[start_idx:end_idx]
            chunk_norm = current_chunk.pow(2).sum(dim=1, keepdim=True)
            cross_term = current_chunk.matmul(codebook_weight.transpose(0, 1))
            chunk_distances = torch.clamp(chunk_norm + codebook_norm - (2.0 * cross_term), min=0.0)
            distances.append(chunk_distances)
        return torch.cat(distances, dim=0)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize semantic inputs against the codebook.

        Args:
            inputs: Tensor with shape ``(B, S, D)``.

        Returns:
            Tuple ``(quantized, vq_loss, indices)``.
        """
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        distances = self._compute_distances_chunked(flat_inputs)

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).reshape(inputs.shape)

        if self.training:
            with torch.no_grad():
                ema_update_fn = ema_update_triton if self.use_triton_ema else ema_update_torch
                cluster_size, updated_ema_w = ema_update_fn(
                    encoding_indices=encoding_indices,
                    flat_inputs=flat_inputs,
                    ema_cluster_size=self.ema_cluster_size,
                    ema_w=self.ema_w,
                    decay=self.decay,
                    epsilon=self.epsilon,
                )
                self.ema_cluster_size.copy_(cluster_size)
                self.ema_w.copy_(updated_ema_w)
                self.codebook.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * e_latent_loss
        self.last_commitment_loss = vq_loss.detach()

        quantized = inputs + (quantized - inputs).detach()
        return quantized, vq_loss, encoding_indices.reshape(inputs.shape[:-1])


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
    ):
        """Initialize semantic VQ-VAE components.

        Args:
            vocab_size: Size of text vocabulary.
            hidden_size: Hidden dimension for encoder/decoder blocks.
            num_semantic_tokens: Size of VQ codebook.
            semantic_sequence_length: Number of semantic tokens produced per sample.
            pad_token_id: Padding token id for text input.
            semantic_pad_token_id: Padding token id in semantic-token space.
            max_position_embeddings: Maximum supported sequence length.
            use_turboquant_kv: Enables turboquant KV cache for decoding.
            turboquant_key_bits: Bitwidth for turboquant key cache.
            turboquant_value_bits: Bitwidth for turboquant value cache.
            turboquant_qjl_residual_scale: Residual scale for turboquant cache.
            gradient_checkpointing: Enables gradient checkpointing in decoder layers.
            use_rotary_embeddings: Enables rotary embeddings in decoder attention.
            use_triton_ema: Enables Triton-accelerated EMA codebook updates.
        """
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

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=self.pad_token_id)
        self.pos_embedding = nn.Embedding(self.max_position_embeddings, hidden_size)

        self.encoder = SDPAEncoder(
            hidden=hidden_size,
            num_heads=8,
            depth=4,
            mlp_ratio=4.0,
            dropout=0.1,
        )

        # Квантизатор (Codebook)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_semantic_tokens,
            embedding_dim=hidden_size,
            use_triton_ema=self.use_triton_ema,
        )

        # Компоненты декодера для деквантования и автоэнкодинга
        self.decoder_layers = nn.ModuleList(
            [
                SDPADecoderLayer(hidden=hidden_size, num_heads=8, mlp_ratio=4.0, dropout=0.1)
                for _ in range(4)
            ]
        )
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.rotary_emb = RotaryEmbedding(dim=hidden_size // 8, max_position_embeddings=self.max_position_embeddings)
        for layer in self.decoder_layers:
            layer.use_turboquant = self.use_turboquant_kv

    def _build_turboquant_config(self) -> TurboQuantConfig | None:
        """Build TurboQuant config for decode-time KV ring buffer.

        Returns:
            TurboQuant configuration object if enabled, otherwise ``None``.
        """
        if not self.use_turboquant_kv:
            return None
        return TurboQuantConfig(
            key_bits=self.turboquant_key_bits,
            value_bits=self.turboquant_value_bits,
            qjl_residual_scale=self.turboquant_qjl_residual_scale,
        )

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build an upper-triangular causal mask for decoder self-attention.

        Args:
            seq_len: Target sequence length.
            device: Target device for mask allocation.

        Returns:
            Float mask tensor compatible with ``nn.TransformerDecoder``.
        """
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
        """Run decoder stack with optional incremental KV caching.

        Args:
            tgt_emb: Decoder input embeddings with shape ``(B, T, H)``.
            memory: Cross-attention memory with shape ``(B, S, H)``.
            past_key_values: Optional per-layer cached KV tensors.
            incremental: Whether this is an incremental decoding step.

        Returns:
            Tuple of decoded states and present per-layer KV cache tensors.

        Raises:
            ValueError: If cache depth does not match decoder depth.
        """
        if past_key_values is not None and len(past_key_values) != len(self.decoder_layers):
            raise ValueError("past_key_values length must match decoder layers.")
        hidden = tgt_emb
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        self_attn_mask = None if incremental else self._build_causal_mask(tgt_emb.size(1), tgt_emb.device)
        for layer_idx, layer in enumerate(self.decoder_layers):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            rotary_freqs_tgt = self.rotary_emb(hidden, seq_len=hidden.shape[1])
            if self.gradient_checkpointing and self.training and layer_past is None:
                hidden, present = checkpoint(
                    lambda a, b, c: layer(
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

    def _pool_to_semantic_length(self, sequence: torch.Tensor) -> torch.Tensor:
        """Pool a ``(B, T, C)`` sequence to semantic length.

        Args:
            sequence: Input tensor shaped ``(batch, time, channels)``.

        Returns:
            Tensor shaped ``(batch, semantic_time, channels)``.

        Raises:
            ValueError: If the source sequence length is not positive.
        """
        source_len = int(sequence.shape[1])
        target_len = int(self.semantic_sequence_length)
        if source_len <= 0:
            raise ValueError("Encoded sequence length must be positive.")
        stride = max(1, ceil(source_len / target_len))
        channel_first = sequence.transpose(1, 2).contiguous()
        pooled = F.avg_pool1d(
            channel_first,
            kernel_size=stride,
            stride=stride,
            ceil_mode=True,
        ).transpose(1, 2)
        if int(pooled.shape[1]) == target_len:
            return pooled
        return F.adaptive_avg_pool1d(pooled.transpose(1, 2), target_len).transpose(1, 2)

    def _pool_semantic_tokens(
        self, encoded: torch.Tensor, padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Pool encoded features into a fixed-length semantic representation.

        If a padding mask is provided, padded tokens are masked out (zeroed)
        before optional adaptive resampling to the configured semantic length.

        Args:
            encoded: Encoder output tensor with shape ``(B, T, H)``.
            padding_mask: Optional token-level padding mask with shape ``(B, T)``
                where ``True`` marks padded positions.

        Returns:
            Semantic feature tensor with shape ``(B, S, H)`` where
            ``S == self.semantic_sequence_length``.
        """
        if padding_mask is None:
            return self._pool_to_semantic_length(encoded)

        valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=encoded.dtype)
        masked_encoded = encoded * valid_mask
        pooled_sum = self._pool_to_semantic_length(masked_encoded)
        pooled_valid = self._pool_to_semantic_length(valid_mask).clamp_min(1e-6)
        return pooled_sum / pooled_valid

    def _pool_semantic_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """Pool token padding mask to semantic-token resolution."""
        pooled = self._pool_to_semantic_length(padding_mask.unsqueeze(-1).float()).squeeze(-1)
        return pooled > 0

    def _apply_semantic_padding_mask(
        self,
        semantic_states: torch.Tensor,
        semantic_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Zero semantic states in padded positions.

        Args:
            semantic_states: Semantic tensor with shape ``(B, S, H)``.
            semantic_padding_mask: Optional semantic padding mask ``(B, S)``.

        Returns:
            Masked semantic tensor with unchanged shape ``(B, S, H)``.
        """
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

    def encode_sentence(
        self, bpe_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = self._resolve_padding_mask(bpe_tokens, padding_mask)
        positions = self._position_ids(bpe_tokens.size(1), bpe_tokens.device)
        x = self.embedding(bpe_tokens) + self.pos_embedding(positions)
        rotary_freqs = self.rotary_emb(x, seq_len=x.shape[1]) if self.use_rotary_embeddings else None
        encoded = self.encoder(x, key_padding_mask=padding_mask, rotary_freqs=rotary_freqs)

        semantic_inputs = self._pool_semantic_tokens(encoded, padding_mask=padding_mask)
        _, vq_loss, semantic_idx = self.quantizer(semantic_inputs)
        return semantic_idx, vq_loss

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
        """Decode semantic codebook indices into autoregressive BPE ids.

        Args:
            semantic_indices: Semantic token indices of shape ``(B,)`` or ``(B, S)``.
            max_length: Maximum number of BPE tokens to generate.
            bos_token_id: Token id used as first autoregressive token.
            eos_token_id: Optional early-stop token id.
            temperature: Sampling temperature (>0).
            top_p: Nucleus sampling threshold in (0, 1].

        Returns:
            Tensor of generated BPE token ids with shape ``(B, L)``.

        Raises:
            ValueError: If arguments are invalid.
        """
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
        semantic_features = self.quantizer.codebook(semantic_indices.long())
        memory = self._apply_semantic_padding_mask(semantic_features, semantic_padding_mask)

        batch_size = semantic_indices.shape[0]
        generated = torch.full(
            (batch_size, 1),
            int(bos_token_id),
            dtype=torch.long,
            device=semantic_indices.device,
        )

        turbo_cfg = self._build_turboquant_config()
        cache_ring_buffer = KVCacheRingBuffer(max_window=int(max_length), turboquant_config=turbo_cfg)
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

        # --- 1. ЭТАП ЭНКОДИНГА ---
        positions = self._position_ids(bpe_tokens.size(1), bpe_tokens.device)
        x = self.embedding(bpe_tokens) + self.pos_embedding(positions)
        rotary_freqs = self.rotary_emb(x, seq_len=x.shape[1]) if self.use_rotary_embeddings else None
        encoded = self.encoder(x, key_padding_mask=padding_mask, rotary_freqs=rotary_freqs)

        semantic_inputs = self._pool_semantic_tokens(encoded, padding_mask=padding_mask)
        semantic_padding_mask = self._pool_semantic_padding_mask(padding_mask)

        # --- 2. ЭТАП КВАНТОВАНИЯ ---
        quantized, vq_loss, _ = self.quantizer(semantic_inputs)

        # --- 3. ЭТАП ДЕКОДИРОВАНИЯ (Causal AR Reconstruction) ---
        tgt_tokens = bpe_tokens[:, :-1]
        tgt_positions = positions[:, :-1]
        tgt_emb = self.embedding(tgt_tokens) + self.pos_embedding(tgt_positions)

        memory = self._apply_semantic_padding_mask(quantized, semantic_padding_mask)

        decoded, _ = self._run_decoder(
            tgt_emb=tgt_emb,
            memory=memory,
            past_key_values=None,
            incremental=False,
        )

        # --- 4. РАСЧЕТ РЕКОНСТРУКЦИИ (Active Token Slicing) ---
        # Находим полезные (не padding) токены
        valid_mask = ~padding_mask[:, :-1]

        # ВЫРЕЗАЕМ мусор ДО проекции! (Экономия до 60% VRAM и TFLOPS)
        active_decoded = decoded[valid_mask]
        active_logits = self.lm_head(active_decoded)

        # Берем только правильные таргеты
        active_targets = bpe_tokens[:, 1:][valid_mask]

        # Лосс считается только по активным токенам
        recon_loss = F.cross_entropy(active_logits, active_targets, reduction="mean")
        total_loss = recon_loss + vq_loss

        # Рассеиваем логиты обратно в полную форму (нулями)
        logits = torch.zeros(
            bpe_tokens.size(0),
            bpe_tokens.size(1) - 1,
            self.vocab_size,
            device=active_logits.device,
            dtype=active_logits.dtype,
        )
        logits[valid_mask] = active_logits

        return logits, total_loss
