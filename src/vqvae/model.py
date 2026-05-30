import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False

if HAS_LIGER:
    @torch.compiler.disable
    def _compute_liger_loss(weight, active_decoded, active_targets, bias):
        lce = LigerFusedLinearCrossEntropyLoss(label_smoothing=0.1)
        return lce(weight, active_decoded, active_targets, bias=bias)
from vector_quantize_pytorch import FSQ

from src.var.generator import KVCacheRingBuffer, TurboQuantConfig, thermodynamic_sampling_with_stats
from src.var.model import RingKVCacheView, RMSNorm, RotaryEmbedding, SDPADecoderLayer
from src.var.fp8 import get_linear_layer
from src.vqvae.cnn_blocks import HierarchicalDownsample1D
from src.vqvae.loss import (
    contrastive_latent_loss,
    feature_matching_loss,
    kl_divergence_loss,
    token_level_contrastive_loss,
)
from src.vqvae.sdpa_blocks import SDPAEncoder
from src.vqvae.config import VQVAEConfig


class SemanticTextVQVAE(nn.Module):
    """Semantic VQ-VAE model for token reconstruction and semantic quantization."""

    def __init__(self, config: VQVAEConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = VQVAEConfig(**kwargs)
        else:
            self.config = config
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
        self.pad_token_id = self.config.pad_token_id
        self.semantic_pad_token_id = self.config.semantic_pad_token_id
        self.semantic_sequence_length = max(1, self.config.semantic_sequence_length)
        self.max_position_embeddings = self.config.max_position_embeddings
        self.use_turboquant_kv = self.config.use_turboquant_kv
        self.turboquant_key_bits = self.config.turboquant_key_bits
        self.turboquant_value_bits = self.config.turboquant_value_bits
        self.turboquant_qjl_residual_scale = self.config.turboquant_qjl_residual_scale
        self.gradient_checkpointing = self.config.gradient_checkpointing
        self.use_rotary_embeddings = self.config.use_rotary_embeddings
        self.word_dropout_prob = self.config.word_dropout_prob
        self.use_unpadding = self.config.use_unpadding

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_token_id)
        # Learned mask token для word dropout (вместо заполнения нулями)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)

        self.encoder = SDPAEncoder(
            hidden=self.hidden_size,
            num_heads=self.config.encoder_num_heads,
            depth=self.config.encoder_depth,
            mlp_ratio=self.config.encoder_mlp_ratio,
            dropout=self.config.encoder_dropout,
            gradient_checkpointing=self.gradient_checkpointing,
            use_unpadding=self.use_unpadding,
            use_fp8=self.config.use_fp8,
        )

        self.compression_rate = self.config.compression_rate

        self.downsample = HierarchicalDownsample1D(
            dim=self.hidden_size,
            compression_rate=self.compression_rate,
            num_blocks=self.config.downsample_num_blocks,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        # =====================================================================
        # 3. Finite Scalar Quantization (FSQ)
        # =====================================================================
        self.levels = self.config.fsq_levels
        self.fsq_dim = len(self.levels)
        self.num_fsq_codes = 1
        for lvl in self.levels:
            self.num_fsq_codes *= lvl
        self.pre_quant_proj = nn.Linear(self.hidden_size, self.fsq_dim)
        self.quantizer = FSQ(levels=self.levels)
        self.post_quant_proj = nn.Linear(self.fsq_dim, self.hidden_size)

        # Компоненты декодера
        self.decoder_layers = nn.ModuleList(
            [
                SDPADecoderLayer(
                    hidden=self.hidden_size,
                    num_heads=self.config.decoder_num_heads,
                    mlp_ratio=self.config.decoder_mlp_ratio,
                    dropout=self.config.decoder_dropout,
                    use_fp8=self.config.use_fp8,
                    use_moe=getattr(self.config, "use_moe", False),
                    num_experts=getattr(self.config, "num_experts", 8),
                    moe_top_k=getattr(self.config, "moe_top_k", 2),
                )
                for _ in range(self.config.decoder_depth)
            ]
        )
        self.decoder_norm = RMSNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        # Weight Tying: Share weights between embedding and lm_head
        self.lm_head.weight = self.embedding.weight
        self.encoder_rotary_emb = RotaryEmbedding(
            dim=self.hidden_size // self.config.encoder_num_heads, max_position_embeddings=self.max_position_embeddings
        )
        self.decoder_rotary_emb = RotaryEmbedding(
            dim=self.hidden_size // self.config.decoder_num_heads, max_position_embeddings=self.max_position_embeddings
        )
        for layer in self.decoder_layers:
            layer.use_turboquant = self.use_turboquant_kv

        # Initialize weights properly
        self.apply(self._init_weights)
        # Re-apply weight tying just in case
        self.lm_head.weight = self.embedding.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _pool_to_semantic_length(self, sequence: torch.Tensor) -> torch.Tensor:
        """Сжимает последовательность в 4 раза с сохранением пространственной информации."""
        channel_first = sequence.transpose(1, 2)
        compressed = self.downsample(channel_first)
        return compressed.transpose(1, 2)

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
        """Pool token padding mask to semantic-token resolution using Logical OR."""
        seq_len = padding_mask.shape[1]
        valid_len = (seq_len // self.compression_rate) * self.compression_rate
        return (
            padding_mask[:, :valid_len]
            .view(padding_mask.shape[0], -1, self.compression_rate)
            .any(dim=-1)
        )

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
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        if past_key_values is not None and len(past_key_values) != len(self.decoder_layers):
            raise ValueError("past_key_values length must match decoder layers.")
        hidden = tgt_emb
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        self_attn_mask = None
        
        start_pos = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_past = past_key_values[0]
            if isinstance(first_past, RingKVCacheView):
                start_pos = int(first_past.current_length)
            else:
                start_pos = int(first_past[0].shape[1])
                
        rotary_freqs_tgt = self.decoder_rotary_emb(tgt_emb, seq_len=tgt_emb.shape[1], start_pos=start_pos)
        total_aux_loss = torch.tensor(0.0, device=tgt_emb.device, dtype=tgt_emb.dtype)
        for layer_idx, layer in enumerate(self.decoder_layers):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            if self.gradient_checkpointing and self.training and layer_past is None:
                hidden, present, l_aux = checkpoint(
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
                hidden, present, l_aux = layer(
                    tgt=hidden,
                    memory=memory,
                    self_attn_mask=self_attn_mask,
                    self_is_causal=not incremental,
                    rotary_freqs_tgt=rotary_freqs_tgt,
                    past_key_value=layer_past,
                )
            total_aux_loss = total_aux_loss + l_aux
            present_key_values.append(present)
        return self.decoder_norm(hidden), present_key_values, total_aux_loss

    def encode_sentence(
        self, bpe_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = self._resolve_padding_mask(bpe_tokens, padding_mask)
        x = self.embedding(bpe_tokens)

        # Encoder на полной последовательности (до pooling)
        rotary_freqs = (
            self.encoder_rotary_emb(x, seq_len=x.shape[1])
            if self.use_rotary_embeddings
            else None
        )
        encoded = self.encoder(x, key_padding_mask=padding_mask, rotary_freqs=rotary_freqs)

        # Pooling обогащённых представлений
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
            else:
                step_tokens = generated[:, -1:]
            step_emb = self.embedding(step_tokens)
            decoded, past_key_values, _ = self._run_decoder(
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        padding_mask = self._resolve_padding_mask(bpe_tokens, padding_mask)

        x = self.embedding(bpe_tokens)

        # Encoder на полной последовательности (до pooling)
        rotary_freqs = (
            self.encoder_rotary_emb(x, seq_len=x.shape[1])
            if self.use_rotary_embeddings
            else None
        )
        encoded = self.encoder(x, key_padding_mask=padding_mask, rotary_freqs=rotary_freqs)

        # Pooling обогащённых представлений
        semantic_inputs = self._pool_semantic_tokens(encoded, padding_mask=padding_mask)
        semantic_padding_mask = self._pool_semantic_padding_mask(padding_mask)

        projected_inputs = self.pre_quant_proj(semantic_inputs)
        quantized_fsq, indices = self.quantizer(projected_inputs)
        quantized = self.post_quant_proj(quantized_fsq)
        vq_loss = torch.tensor(0.0, device=quantized.device)

        tgt_tokens = bpe_tokens[:, :-1]
        tgt_emb = self.embedding(tgt_tokens)

        memory = self._apply_semantic_padding_mask(quantized, semantic_padding_mask)

        # Word Dropout on Decoder Input (prevents posterior collapse)
        if self.training and self.word_dropout_prob > 0.0:
            drop_mask = torch.rand(tgt_tokens.shape, device=tgt_tokens.device) < self.word_dropout_prob
            tgt_emb = torch.where(drop_mask.unsqueeze(-1), self.mask_token.expand_as(tgt_emb), tgt_emb)

        decoded, _, aux_loss = self._run_decoder(
            tgt_emb=tgt_emb,
            memory=memory,
            past_key_values=None,
            incremental=False,
        )
        del memory
        del tgt_emb

        valid_mask = ~padding_mask[:, :-1]
        active_decoded = decoded[valid_mask]
        del decoded
        active_targets = bpe_tokens[:, 1:][valid_mask]

        # 1. Chunked & Fused Label Smoothing Reconstruction Loss
        if HAS_LIGER and not self.config.use_fp8:
            recon_loss = _compute_liger_loss(
                self.lm_head.weight, active_decoded, active_targets, bias=self.lm_head.bias
            )
            active_logits = torch.empty(0, device=active_decoded.device)
        else:
            chunk_size = 4096
            if self.training and active_decoded.shape[0] > chunk_size:
                recon_loss = torch.tensor(0.0, device=active_decoded.device)

                def compute_chunk_loss(h, t):
                    chunk_logits = self.lm_head(h)
                    return F.cross_entropy(chunk_logits, t, reduction="sum", label_smoothing=0.1)

                for i in range(0, active_decoded.shape[0], chunk_size):
                    chunk_decoded = active_decoded[i : i + chunk_size]
                    chunk_targets = active_targets[i : i + chunk_size]
                    chunk_loss = checkpoint(
                        compute_chunk_loss, chunk_decoded, chunk_targets, use_reentrant=False
                    )
                    recon_loss = recon_loss + chunk_loss
                recon_loss = recon_loss / active_decoded.shape[0]
                active_logits = torch.empty(0, device=active_decoded.device)
            else:
                active_logits = self.lm_head(active_decoded)
                recon_loss = F.cross_entropy(
                    active_logits, active_targets, reduction="mean", label_smoothing=0.1
                )

        del active_decoded

        # 2. Codebook utilization metrics (no gradient)
        with torch.no_grad():
            flat_idx = indices.reshape(-1)
            cb_util = flat_idx.unique().numel() / max(1, self.num_fsq_codes)
            counts = torch.zeros(self.num_fsq_codes, device=flat_idx.device, dtype=torch.float32)
            counts.scatter_add_(0, flat_idx.long(), torch.ones_like(flat_idx, dtype=torch.float32))
            probs = counts / (counts.sum() + 1e-10)
            cb_entropy = -(probs * (probs + 1e-10).log()).sum()
            cb_ppl = cb_entropy.exp()

        total_loss = recon_loss + vq_loss + getattr(self.config, "router_aux_loss_coef", 0.01) * aux_loss

        loss_dict = {
            "recon": recon_loss.detach(),
            "cb_util": torch.tensor(cb_util, device=quantized.device),
            "cb_ppl": cb_ppl,
            "moe_aux_loss": aux_loss.detach(),
        }

        return active_logits, total_loss, loss_dict
