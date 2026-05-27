import torch
import torch.nn as nn


class ConvNeXT1DBlock(nn.Module):
    """1D implementation of ConvNeXT block for robust hierarchical feature extraction."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        # Depthwise convolution
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Pointwise convolutions (inverted bottleneck)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Drop path not strictly needed unless very deep, but adding for completeness
        self.drop_path_prob = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.dwconv(x)
        # Permute (B, C, L) -> (B, L, C) for LayerNorm and Linear layers
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Permute back (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

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

    def __init__(self, dim: int, compression_rate: int = 4, num_blocks: int = 2):
        super().__init__()
        self.down_conv = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=compression_rate, stride=compression_rate
        )
        self.blocks = nn.Sequential(*[ConvNeXT1DBlock(dim=dim) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(x)
        x = self.blocks(x)
        return x


class HierarchicalUpsample1D(nn.Module):
    """Upsamples spatial dimension using ConvNeXT blocks followed by a transposed convolution."""

    def __init__(self, dim: int, compression_rate: int = 4, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNeXT1DBlock(dim=dim) for _ in range(num_blocks)])
        self.up_conv = nn.ConvTranspose1d(
            in_channels=dim, out_channels=dim, kernel_size=compression_rate, stride=compression_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.up_conv(x)
        return x
