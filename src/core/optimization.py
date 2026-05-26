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
