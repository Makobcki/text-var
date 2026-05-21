import torch
import torch.nn as nn
import torch.nn.functional as F

from var_branch.config import VARConfig


class VARTransformer(nn.Module):
    def __init__(self, cfg: VARConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.hidden_size)

        self.token_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden) for vocab_size in cfg.level_vocab_sizes]
        )
        self.scale_embedding = nn.Embedding(len(cfg.level_vocab_sizes), hidden)

        max_local_len = max(cfg.level_lengths)
        self.local_position_embedding = nn.Embedding(max_local_len, hidden)

        # Токен-заглушка для генерации целевых уровней, если current_level_input не передан
        self.target_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_size))

        # Базовая память (защита от падения Cross-Attention на 0-м уровне)
        self.null_token_embedding = nn.Embedding(1, hidden)
        self.null_token_embedding.weight.data.fill_(0.0)

        # Декодер (поддерживает tgt_mask для AR и memory для Cross-Attention)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden,
                    nhead=int(cfg.num_heads),
                    dim_feedforward=max(hidden, int(hidden * float(cfg.mlp_ratio))),
                    dropout=0.1,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(hidden)

        # Выходные головы (Классификаторы)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, vocab_size) for vocab_size in cfg.level_vocab_sizes]
        )

        # Early Exit классификаторы
        self.early_exit_heads = nn.ModuleDict()
        for layer_idx in cfg.exit_layers:
            for scale_idx, vocab_size in enumerate(cfg.level_vocab_sizes):
                head_key = f"layer_{layer_idx}_scale_{scale_idx}"
                self.early_exit_heads[head_key] = nn.Linear(hidden, vocab_size)

    def forward(
        self,
        prefix_inputs: list[torch.Tensor],
        *,
        target_level: int | None = None,
        current_level_input: torch.Tensor | None = None,
        batch_size: int | None = None,
        cfg_scale: float = 1.0,
        return_early_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:

        # --- Обработка CFG ---
        if cfg_scale != 1.0 and prefix_inputs:
            out_cond = self.forward(
                prefix_inputs,
                target_level=target_level,
                current_level_input=current_level_input,
                batch_size=batch_size,
                cfg_scale=1.0,
                return_early_outputs=return_early_outputs,
            )

            uncond_prefixes = [torch.zeros_like(p) for p in prefix_inputs]

            out_uncond = self.forward(
                uncond_prefixes,
                target_level=target_level,
                current_level_input=current_level_input,
                batch_size=batch_size,
                cfg_scale=1.0,
                return_early_outputs=return_early_outputs,
            )

            if return_early_outputs:
                logits_cond, early_cond = out_cond
                logits_uncond, early_uncond = out_uncond
                final_logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                final_early = [u + cfg_scale * (c - u) for c, u in zip(early_cond, early_uncond)]
                return final_logits, final_early

            return out_uncond + cfg_scale * (out_cond - out_uncond)

        # --- Инициализация ---
        if prefix_inputs:
            B = prefix_inputs[0].shape[0]
            device = prefix_inputs[0].device
        elif current_level_input is not None:
            B = current_level_input.shape[0]
            device = current_level_input.device
        else:
            B = batch_size if batch_size else 1
            device = self.target_token.device

        target_idx = target_level if target_level is not None else len(prefix_inputs)

        # --- Сборка Памяти (Уровни < target_idx) ---
        null_mem = self.null_token_embedding.weight.unsqueeze(0).expand(B, 1, -1)
        accumulated_memories = [null_mem]

        for s_idx, scale_input in enumerate(prefix_inputs):
            if s_idx >= target_idx:
                break

            emb = self.token_embeddings[s_idx](scale_input)
            emb = emb + self.scale_embedding.weight[s_idx].view(1, 1, -1)

            L = scale_input.shape[1]
            local_ids = torch.arange(L, device=device)
            emb = emb + self.local_position_embedding(local_ids).view(1, L, -1)

            current_context = torch.cat(accumulated_memories, dim=1)
            x_scale = emb
            for block in self.blocks:
                x_scale = block(tgt=x_scale, memory=current_context)

            accumulated_memories.append(self.norm(x_scale))

        final_memory = torch.cat(accumulated_memories, dim=1)

        # --- Подготовка Целевого Уровня ---
        if current_level_input is not None:
            x = self.token_embeddings[target_idx](current_level_input)
            target_len = current_level_input.shape[1]
        else:
            target_len = self.cfg.level_lengths[target_idx]
            x = self.target_token.expand(B, target_len, -1)

        x = x + self.scale_embedding.weight[target_idx].view(1, 1, -1)
        local_ids = torch.arange(target_len, device=device)
        x = x + self.local_position_embedding(local_ids).view(1, target_len, -1)

        tgt_mask = None
        if target_idx == 0:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len, device=device)

        # --- Прямой проход ---
        early_outputs = []
        for layer_idx, block in enumerate(self.blocks):
            x = block(
                tgt=x, memory=final_memory, tgt_mask=tgt_mask, tgt_is_causal=(target_idx == 0)
            )

            if layer_idx in self.cfg.exit_layers:
                target_features = self.norm(x)
                head_key = f"layer_{layer_idx}_scale_{target_idx}"
                logits = self.early_exit_heads[head_key](target_features)
                if return_early_outputs:
                    early_outputs.append(logits)

        full_encoded = self.norm(x)
        out_features = self.heads[target_idx](full_encoded)

        if return_early_outputs:
            return out_features, early_outputs
        return out_features
