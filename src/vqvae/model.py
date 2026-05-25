from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = float(decay)
        self.epsilon = float(epsilon)

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.codebook.weight.data.clone())

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.codebook.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.codebook.weight).view(inputs.shape)

        if self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(dim=0)
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)

                # Граф больше не отслеживается, память свободна
                dw = torch.matmul(encodings.t(), flat_inputs)

                self.ema_w.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                ) * n
                self.codebook.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        return quantized, vq_loss, encoding_indices.view(inputs.shape[:-1])


class SemanticTextVQVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_semantic_tokens: int = 4096,
        semantic_sequence_length: int = 1,
        pad_token_id: int = 0,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = int(pad_token_id)
        self.semantic_sequence_length = max(1, int(semantic_sequence_length))
        self.max_position_embeddings = int(max_position_embeddings)

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(self.max_position_embeddings, hidden_size)

        # Энкодер архитектуры
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Квантизатор (Codebook)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_semantic_tokens, embedding_dim=hidden_size
        )

        # TASK-4: Компоненты декодера для деквантования и автоэнкодинга
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def _position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if int(seq_len) > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings="
                f"{self.max_position_embeddings}."
            )
        return torch.arange(seq_len, device=device).unsqueeze(0)

    def _pool_semantic_tokens(self, encoded: torch.Tensor) -> torch.Tensor:
        """Downsample encoder states to semantic token sequence length.

        Uses local average pooling with stride to preserve token-level locality
        before optional adaptive resampling to the configured semantic length.

        Args:
            encoded: Encoder output tensor with shape ``(B, T, H)``.

        Returns:
            Semantic feature tensor with shape ``(B, S, H)`` where
            ``S == self.semantic_sequence_length``.
        """
        source_len = int(encoded.shape[1])
        target_len = int(self.semantic_sequence_length)
        if source_len <= 0:
            raise ValueError("Encoded sequence length must be positive.")
        stride = max(1, ceil(source_len / target_len))
        pooled = F.avg_pool1d(
            encoded.transpose(1, 2),
            kernel_size=stride,
            stride=stride,
            ceil_mode=True,
        ).transpose(1, 2)
        if int(pooled.shape[1]) == target_len:
            return pooled
        return F.adaptive_avg_pool1d(pooled.transpose(1, 2), target_len).transpose(1, 2)

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
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        semantic_inputs = self._pool_semantic_tokens(encoded)
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

        semantic_features = self.quantizer.codebook(semantic_indices.long())
        memory = semantic_features

        batch_size = semantic_indices.shape[0]
        generated = torch.full(
            (batch_size, 1),
            int(bos_token_id),
            dtype=torch.long,
            device=semantic_indices.device,
        )

        for _ in range(int(max_length) - 1):
            positions = self._position_ids(generated.size(1), generated.device)
            tgt_emb = self.embedding(generated) + self.pos_embedding(positions)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1),
                device=generated.device,
            )
            decoded = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = self.lm_head(decoded)[:, -1, :] / float(temperature)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > float(top_p)
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            filtered_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            filtered_probs = F.softmax(filtered_logits, dim=-1)
            sampled_rank = torch.multinomial(filtered_probs, num_samples=1)
            next_token = sorted_indices.gather(dim=-1, index=sampled_rank).squeeze(-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            if eos_token_id is not None and bool((next_token == int(eos_token_id)).all()):
                break

        return generated

    # TASK-4: Реализация сквозного forward-цикла восстановления с расчетом лосса
    def forward(
        self, bpe_tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = self._resolve_padding_mask(bpe_tokens, padding_mask)
        # --- 1. ЭТАП ЭНКОДИНГА ---
        positions = self._position_ids(bpe_tokens.size(1), bpe_tokens.device)
        x = self.embedding(bpe_tokens) + self.pos_embedding(positions)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        semantic_inputs = self._pool_semantic_tokens(encoded)

        # --- 2. ЭТАП КВАНТОВАНИЯ ---
        quantized, vq_loss, semantic_idx = self.quantizer(semantic_inputs)

        # --- 3. ЭТАП ДЕКОДИРОВАНИЯ (Causal AR Reconstruction) ---
        # Сдвигаем токены для входа декодера, чтобы исключить читерство через self-attention
        tgt_tokens = bpe_tokens[:, :-1]
        tgt_positions = positions[:, :-1]
        tgt_emb = self.embedding(tgt_tokens) + self.pos_embedding(tgt_positions)

        # Превращаем квантованный вектор предложения в контекст (memory) для Cross-Attention
        memory = quantized  # Формат: (B, S_sem, hidden_size)

        # Казуальная маска для декодера
        device = bpe_tokens.device
        tgt_len = tgt_tokens.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)

        # Восстановление скрытых представлений и проекция в вокабуляр
        decoded = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits = self.lm_head(decoded)

        # --- 4. РАСЧЕТ РЕКОНСТРУКЦИИ (Reconstruction Loss) ---
        # Таргеты сдвинуты на +1 относительно входов декодера
        recon_targets = bpe_tokens[:, 1:]

        loss_mask = padding_mask[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), recon_targets.reshape(-1), reduction="none"
        )
        loss = loss.view(recon_targets.shape)
        loss = loss.masked_fill(loss_mask, 0.0)
        recon_loss = loss.sum() / (~loss_mask).sum().clamp(min=1)

        # Итоговый лосс для pre-train этапа
        total_loss = recon_loss + vq_loss
        return logits, total_loss
