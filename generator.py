import math

import torch
import torch.nn.functional as F

from var_branch.model import VARTransformer


def thermodynamic_sampling(
    logits: torch.Tensor,
    alpha: float = 1.0,
    t_min: float = 0.1,
    t_max: float = 2.0,
    healthy_entropy_limit: float = 1.5,
) -> torch.Tensor:
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

    return next_tokens


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

    # ==========================================
    # ФАЗА 1: Уровень 0 (Сюжет) — Авторегрессия (AR)
    # ==========================================
    len_lvl_0 = model.cfg.level_lengths[0]
    print(f"[HYBRID] Фаза 1: AR Генерация ({len_lvl_0} шагов)...")

    lvl_0_sequence = torch.empty((batch, 0), dtype=torch.long, device=device)

    for step in range(len_lvl_0):
        logits = model(
            prefix_inputs=[],
            target_level=0,
            current_level_input=lvl_0_sequence if step > 0 else None,
            batch_size=batch,
            cfg_scale=cfg_scale,
        )

        next_token_logits = logits[:, -1, :]
        next_token = thermodynamic_sampling(
            next_token_logits, alpha=alpha, healthy_entropy_limit=healthy_entropy_limit
        )
        lvl_0_sequence = torch.cat([lvl_0_sequence, next_token.unsqueeze(1)], dim=1)

    out.append(lvl_0_sequence)

    # ==========================================
    # ФАЗА 2: Уровень 1 (Текст) — Итеративный NAR
    # ==========================================
    len_lvl_1 = model.cfg.level_lengths[1]
    mask_id = model.cfg.mask_token_id
    print(f"[HYBRID] Фаза 2: NAR Распаковка ({nar_steps} итераций)...")

    lvl_1_tokens = torch.full((batch, len_lvl_1), mask_id, dtype=torch.long, device=device)

    for step in range(nar_steps):
        logits = model(
            prefix_inputs=out, target_level=1, current_level_input=lvl_1_tokens, cfg_scale=cfg_scale
        )

        probs = F.softmax(logits, dim=-1)
        pred_tokens = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]

        if step == nar_steps - 1:
            lvl_1_tokens = pred_tokens
            break

        ratio = (step + 1) / nar_steps
        keep_ratio = math.cos(math.pi / 2 * (1 - ratio))
        num_to_keep = int(len_lvl_1 * keep_ratio)

        already_unmasked = lvl_1_tokens != mask_id
        confidences[already_unmasked] = float("inf")

        _, topk_indices = torch.topk(confidences, k=num_to_keep, dim=-1)

        new_mask = torch.ones((batch, len_lvl_1), dtype=torch.bool, device=device)
        new_mask.scatter_(dim=-1, index=topk_indices, value=False)

        lvl_1_tokens = torch.where(new_mask, torch.tensor(mask_id, device=device), pred_tokens)

    out.append(lvl_1_tokens)
    print("[HYBRID] Генерация завершена.")

    return out
