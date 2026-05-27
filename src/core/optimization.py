import math
from collections.abc import Callable
from typing import Any

import torch


def setup_blackwell_autotune(compile_mode: str = "default"):
    """
    Apply autotune profiles for triton and torch.compile
    specifically tuned for the Blackwell (and Hopper) architecture.
    """
    # Enable TF32 globally for massive throughput on modern Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        import torch._inductor.config as inductor_config
    except ImportError:
        return

    # Enable aggressive autotuning for matrix multiplications and pointwise ops
    inductor_config.max_autotune = True
    inductor_config.max_autotune_pointwise = True
    inductor_config.max_autotune_gemm_backends = "TRITON"

    # Coordinate descent tuning (finds optimal tile sizes on Blackwell)
    inductor_config.coordinate_descent_tuning = True

    # Use cudagraph trees to dramatically reduce CPU overhead for large graphs
    inductor_config.triton.cudagraph_trees = True

    # Enable multi-kernel generation (useful for autotuning)
    if hasattr(inductor_config.triton, "multi_kernel"):
        inductor_config.triton.multi_kernel = 1

    # Enable Tensor Memory Accelerator (TMA) which was introduced in Hopper
    # and heavily extended in Blackwell for asynchronous global-to-shared memory copies.
    if hasattr(inductor_config.triton, "use_tma"):
        inductor_config.triton.use_tma = True

    if compile_mode == "max-autotune":
        inductor_config.fx_graph_cache = True


def build_cosine_warmup_scheduler_lambda(
    max_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    """Build lambda for Linear Warmup + Cosine Decay schedule."""
    warmup_steps = max(0, min(max_steps - 1, warmup_steps)) if max_steps > 1 else 0
    min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))

    def _lr_lambda(current_step: int) -> float:
        if max_steps <= 1:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress_denominator = max(1, max_steps - warmup_steps)
        progress = min(
            1.0, max(0.0, float(current_step - warmup_steps) / float(progress_denominator))
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return _lr_lambda


def configure_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Separate model parameters into decayed and non-decayed groups."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
