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
            cluster_size = encodings.sum(dim=0)
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
            dw = torch.matmul(encodings.t(), flat_inputs)
            self.ema_w.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)
            n = self.ema_cluster_size.sum()
            cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon)) * n
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
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = int(pad_token_id)

        self.embedding = nn.Embedding(vocab_size, hidden_size)

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
        x = self.embedding(bpe_tokens)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sentence_vector = (encoded * mask_expanded).sum(dim=1) / (
            mask_expanded.sum(dim=1) + 1e-9
        )

        _, vq_loss, semantic_idx = self.quantizer(sentence_vector)
        return semantic_idx, vq_loss

    def decode_from_semantic_indices(
        self,
        semantic_indices: torch.Tensor,
        *,
        max_length: int,
        bos_token_id: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Decode semantic codebook indices into autoregressive BPE ids.

        Args:
            semantic_indices: Semantic token indices of shape ``(B,)`` or ``(B, S)``.
            max_length: Maximum number of BPE tokens to generate.
            bos_token_id: Token id used as first autoregressive token.
            eos_token_id: Optional early-stop token id.

        Returns:
            Tensor of generated BPE token ids with shape ``(B, L)``.

        Raises:
            ValueError: If semantic index tensor rank is invalid or max_length < 1.
        """
        if int(max_length) < 1:
            raise ValueError("max_length must be >= 1.")
        if semantic_indices.dim() == 1:
            semantic_indices = semantic_indices.unsqueeze(1)
        if semantic_indices.dim() != 2:
            raise ValueError("semantic_indices must have shape (B,) or (B, S).")

        semantic_features = self.quantizer.codebook(semantic_indices.long())
        memory = semantic_features.mean(dim=1, keepdim=True)

        batch_size = semantic_indices.shape[0]
        generated = torch.full(
            (batch_size, 1),
            int(bos_token_id),
            dtype=torch.long,
            device=semantic_indices.device,
        )

        for _ in range(int(max_length) - 1):
            tgt_emb = self.embedding(generated)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1),
                device=generated.device,
            )
            decoded = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits = self.lm_head(decoded)
            next_token = logits[:, -1, :].argmax(dim=-1)
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
        x = self.embedding(bpe_tokens)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        sentence_vector = (encoded * mask_expanded).sum(dim=1) / (
            mask_expanded.sum(dim=1) + 1e-9
        )

        # --- 2. ЭТАП КВАНТОВАНИЯ ---
        quantized, vq_loss, semantic_idx = self.quantizer(sentence_vector)

        # --- 3. ЭТАП ДЕКОДИРОВАНИЯ (Causal AR Reconstruction) ---
        # Сдвигаем токены для входа декодера, чтобы исключить читерство через self-attention
        tgt_tokens = bpe_tokens[:, :-1]
        tgt_emb = self.embedding(tgt_tokens)

        # Превращаем квантованный вектор предложения в контекст (memory) для Cross-Attention
        memory = quantized.unsqueeze(1)  # Формат: (B, 1, hidden_size)

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
