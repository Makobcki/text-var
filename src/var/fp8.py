import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_E4M3_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def _can_use_fp8_mm(out_features: int, in_features: int) -> bool:
    """Check if dimensions are compatible with torch._scaled_mm (multiples of 16)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):
        return False
    return out_features % 16 == 0 and in_features % 16 == 0


# ---------------------------------------------------------------------------
# Autograd Function – forward uses pre-quantized weights, only input is
# dynamically quantized.
@torch.library.custom_op("var::fp8_mm", mutates_args=())
def fp8_mm(
    input_e4m3: torch.Tensor,
    weight_fp32: torch.Tensor,
    inp_scale: torch.Tensor,
    weight_e4m3: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    input_2d = input_e4m3.view(-1, input_e4m3.shape[-1])
    out = torch._scaled_mm(
        input_2d,
        weight_e4m3.t(),
        scale_a=inp_scale,
        scale_b=weight_scale,
        out_dtype=weight_fp32.dtype,
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.view(*input_e4m3.shape[:-1], -1)
    return out

@fp8_mm.register_fake
def _(input_e4m3, weight_fp32, inp_scale, weight_e4m3, weight_scale):
    out_shape = list(input_e4m3.shape)
    out_shape[-1] = weight_e4m3.shape[0]
    return torch.empty(out_shape, dtype=weight_fp32.dtype, device=input_e4m3.device)

def fp8_mm_setup_context(ctx, inputs, output):
    input_e4m3, weight_fp32, inp_scale, _, _ = inputs
    ctx.save_for_backward(input_e4m3, weight_fp32, inp_scale)

def fp8_mm_backward(ctx, grad_output):
    input_e4m3, weight_fp32, inp_scale = ctx.saved_tensors

    input_deq = input_e4m3.to(grad_output.dtype) * inp_scale
    weight_c = weight_fp32.to(grad_output.dtype)

    grad_input = grad_output.matmul(weight_c)
    
    go2 = grad_output.reshape(-1, grad_output.shape[-1])
    in2 = input_deq.reshape(-1, input_deq.shape[-1])
    grad_weight = go2.t().matmul(in2)

    return grad_input, grad_weight, None, None, None

torch.library.register_autograd(
    "var::fp8_mm", fp8_mm_backward, setup_context=fp8_mm_setup_context
)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class FP8Linear(nn.Module):
    """Linear layer that caches FP8-quantized weights.

    Weights are quantized once and reused across all forward calls until
    ``refresh_fp8_weights()`` is called (should happen after optimizer.step()).
    Only the *input* is dynamically quantized on each forward – this is cheap
    because inputs are much smaller than weight matrices.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Decide once whether hardware FP8 matmul is feasible for this layer
        self._fp8_ok = _can_use_fp8_mm(out_features, in_features)

        # Cached quantized weight (populated lazily on first forward or
        # explicitly via refresh_fp8_weights).
        self.register_buffer("_weight_e4m3", None, persistent=False)
        self.register_buffer("_weight_scale", None, persistent=False)

    # -- call after optimizer.step() to refresh cached FP8 weights -----------
    @torch.no_grad()
    def refresh_fp8_weights(self) -> None:
        if not self._fp8_ok:
            return
        amax = self.weight.abs().max()
        scale = (amax / _E4M3_MAX).clamp(min=1e-12).float()
        new_e4m3 = (self.weight / scale).to(torch.float8_e4m3fn)
        
        if self._weight_e4m3 is None:
            self.register_buffer("_weight_e4m3", new_e4m3, persistent=False)
            self.register_buffer("_weight_scale", scale, persistent=False)
        else:
            self._weight_e4m3.copy_(new_e4m3)
            self._weight_scale.copy_(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self._fp8_ok:
            return F.linear(input, self.weight, self.bias)

        # Lazy init on first forward
        if self._weight_e4m3 is None:
            self.refresh_fp8_weights()

        # Dynamic input scaling moved here so torch.compile can fuse it
        inp_amax = input.abs().max()
        inp_scale = (inp_amax / _E4M3_MAX).clamp(min=1e-12).float()
        input_e4m3 = (input / inp_scale).to(torch.float8_e4m3fn)

        out = fp8_mm(
            input_e4m3,
            self.weight,
            inp_scale,
            self._weight_e4m3,
            self._weight_scale,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8={self._fp8_ok}"
        )


# ---------------------------------------------------------------------------
# Helper: refresh all FP8 caches in a model (call after optimizer.step)
# ---------------------------------------------------------------------------
def refresh_fp8_weights(model: nn.Module) -> None:
    """Walk the module tree and refresh cached FP8 weights on all FP8Linear layers."""
    for m in model.modules():
        if isinstance(m, FP8Linear):
            m.refresh_fp8_weights()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_linear_layer(
    in_features: int, out_features: int, bias: bool = True, use_fp8: bool = False
) -> nn.Module:
    if use_fp8:
        return FP8Linear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)
