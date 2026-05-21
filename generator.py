import torch
import torch.nn.functional as F

from model import VARTransformer


class RollbackEvent(RuntimeError):
    def __init__(self, block_start: int, block_end: int) -> None:
        super().__init__(f"Rollback requested for block [{block_start}, {block_end})")
        self.block_start = block_start
        self.block_end = block_end


def thermodynamic_sampling_with_stats(
    logits: torch.Tensor,
    alpha: float = 1.0,
    t_min: float = 0.1,
    t_max: float = 2.0,
    healthy_entropy_limit: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    topk_vals, _ = torch.topk(logits, k=2, dim=-1)
    delta_top2 = torch.clamp(topk_vals[..., 0] - topk_vals[..., 1], min=1e-5)

    t_base = t_min + (t_max - t_min) * torch.exp(-alpha * delta_top2)

    # Entropy-Penalty (Предохранитель от галлюцинаций)
    chaos_diff = torch.clamp(entropy - healthy_entropy_limit, min=0.0)
    chaos_penalty = torch.exp(-chaos_diff)

    t_dynamic = t_min + (t_base - t_min) * chaos_penalty
    t_dynamic = torch.clamp(t_dynamic, min=t_min, max=t_max).unsqueeze(-1)

    scaled_logits = logits / t_dynamic
    next_tokens = torch.distributions.Categorical(logits=scaled_logits).sample()

    return next_tokens, entropy, chaos_diff




def _decode_next_ar_token(
    model: VARTransformer,
    *,
    prefix_inputs: list[torch.Tensor],
    target_level: int,
    sequence: torch.Tensor,
    cfg_scale: float,
    alpha: float,
    healthy_entropy_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = sequence.shape[0]
    prompt = torch.full((batch, 1), model.cfg.mask_token_id, dtype=torch.long, device=sequence.device)
    ar_input = torch.cat([sequence, prompt], dim=1)
    logits = model(
        prefix_inputs=prefix_inputs,
        target_level=target_level,
        current_level_input=ar_input,
        batch_size=batch,
        cfg_scale=cfg_scale,
    )
    pos = sequence.shape[1]
    return thermodynamic_sampling_with_stats(
        logits[:, pos, :], alpha=alpha, healthy_entropy_limit=healthy_entropy_limit
    )


@torch.no_grad()
def hybrid_cascade_decode(
    model: VARTransformer,
    *,
    batch_size: int,
    device: torch.device,
    nar_steps: int = 4,
    cfg_scale: float = 1.0,
    alpha: float = 1.0,
    healthy_entropy_limit: float = 1.5,
) -> list[torch.Tensor]:

    batch = int(batch_size)
    out: list[torch.Tensor] = []

    len_lvl_0 = model.cfg.level_lengths[0]
    print(f"[HYBRID] Фаза 1: AR Генерация макро-плана ({len_lvl_0} шагов)...")
    lvl_0_sequence = torch.empty((batch, 0), dtype=torch.long, device=device)
    for _ in range(len_lvl_0):
        next_token, _, _ = _decode_next_ar_token(
            model,
            prefix_inputs=[],
            target_level=0,
            sequence=lvl_0_sequence,
            cfg_scale=cfg_scale,
            alpha=alpha,
            healthy_entropy_limit=healthy_entropy_limit,
        )
        lvl_0_sequence = torch.cat([lvl_0_sequence, next_token.unsqueeze(1)], dim=1)
    out.append(lvl_0_sequence)

    len_lvl_1 = model.cfg.level_lengths[1]
    print(f"[HYBRID] Фаза 2: AR Генерация структурного каркаса ({len_lvl_1} шагов)...")
    lvl_1_sequence = torch.empty((batch, 0), dtype=torch.long, device=device)
    for _ in range(len_lvl_1):
        next_token, _, _ = _decode_next_ar_token(
            model,
            prefix_inputs=out,
            target_level=1,
            sequence=lvl_1_sequence,
            cfg_scale=cfg_scale,
            alpha=alpha,
            healthy_entropy_limit=healthy_entropy_limit,
        )
        lvl_1_sequence = torch.cat([lvl_1_sequence, next_token.unsqueeze(1)], dim=1)
    out.append(lvl_1_sequence)

    len_lvl_2 = model.cfg.level_lengths[2]
    block_count = max(1, len_lvl_1)
    block_size = max(1, len_lvl_2 // block_count)
    print(f"[HYBRID] Фаза 3: Block-Local AR ({block_count} блоков, block_size={block_size})...")

    lvl_2_tokens = torch.full((batch, len_lvl_2), model.cfg.pad_token_id, dtype=torch.long, device=device)
    max_backtracks = 2
    chaos_streak_limit = 2

    for block_idx in range(block_count):
        start_idx = block_idx * block_size
        if start_idx >= len_lvl_2:
            break
        end_idx = len_lvl_2 if block_idx == block_count - 1 else min(len_lvl_2, start_idx + block_size)
        block_len = end_idx - start_idx

        state = "DECODE_BLOCK"
        attempt = 0
        while state != "COMMIT_BLOCK":
            if attempt > max_backtracks:
                state = "COMMIT_BLOCK"
                continue

            try:
                local_chunk = torch.full((batch, block_len), model.cfg.mask_token_id, dtype=torch.long, device=device)
                chaos_streak = 0

                for local_step in range(block_len):
                    # Дополнительный проход только по маскированному чанку (latent inpainting region)
                    chunk_context = local_chunk[:, : local_step + 1]
                    logits = model(
                        prefix_inputs=out,
                        target_level=2,
                        current_level_input=chunk_context,
                        cfg_scale=cfg_scale,
                        compact_memory_for_final_level=True,
                    )
                    next_token, _, chaos_diff = thermodynamic_sampling_with_stats(
                        logits[:, local_step, :], alpha=alpha, healthy_entropy_limit=healthy_entropy_limit
                    )

                    if torch.any(chaos_diff > 0):
                        chaos_streak += 1
                    else:
                        chaos_streak = 0

                    if chaos_streak >= chaos_streak_limit:
                        raise RollbackEvent(start_idx, end_idx)

                    local_chunk[:, local_step] = next_token

                lvl_2_tokens[:, start_idx:end_idx] = local_chunk
                state = "COMMIT_BLOCK"

            except RollbackEvent:
                # FSM rollback: откат к началу блока + маскирование блока
                lvl_2_tokens[:, start_idx:end_idx] = model.cfg.mask_token_id
                attempt += 1
                state = "DECODE_BLOCK"
    out.append(lvl_2_tokens)
    print("[HYBRID] Генерация завершена.")
    return out
