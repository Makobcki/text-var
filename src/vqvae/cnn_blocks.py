import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ConvNeXT1DBlock(nn.Module):
    """1D implementation of ConvNeXT block for robust hierarchical feature extraction."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        # Depthwise convolution
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        # Pointwise convolutions (inverted bottleneck)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)

        # Drop path not strictly needed unless very deep, but adding for completeness
        self.drop_path_prob = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Simple drop path for training if prob > 0
        if self.training and self.drop_path_prob > 0.0:
            keep_prob = 1 - self.drop_path_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            x = x.div(keep_prob) * random_tensor

        return input_x + x


class HierarchicalDownsample1D(nn.Module):
    """Downsamples spatial dimension using a strided convolution followed by ConvNeXT blocks."""

    def __init__(self, dim: int, compression_rate: int = 4, num_blocks: int = 2, gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.input_norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.down_conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=compression_rate, stride=compression_rate
        )
        self.blocks = nn.ModuleList([ConvNeXT1DBlock(dim=dim) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.down_conv(x)
        use_ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


