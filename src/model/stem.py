"""
Convolutional Stem for OCR encoder.

Output stride 2×2: (B, C_in, 32, W) → (B, out_channels, 16, W/2)
Expands 1ch → 64 → out_channels with two strided convs + ResBlocks.
"""

import torch.nn as nn
from torch import Tensor

from src.data.color import INPUT_CHANNELS



class _ResBlock(nn.Module):
    """Residual block with two 3×3 convs + skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class ResNetStem(nn.Module):
    """ResNet stem with two strided convolutions.

    Output stride 2×2: (B, C_in, 32, W) → (B, out_channels, 16, W/2)
    Height reduced 2× by first conv, width reduced 2× by second conv.

      Conv 1→64, stride (2,1)           (32×W → 16×W)
      ResBlock 64                        (refine)
      Conv 64→out_channels, stride (1,2) (16×W → 16×W/2)
      ResBlock out_channels              (refine)
    """

    def __init__(self, in_channels: int = INPUT_CHANNELS, out_channels: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            # Downsample 2× height, expand channels
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            _ResBlock(64),
            # Downsample 2× width, expand channels
            nn.Conv2d(64, out_channels, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            _ResBlock(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
