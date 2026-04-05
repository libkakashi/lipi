"""
Convolutional Stems for OCR encoder.

Output stride 2×2: (B, C_in, 32, W) → (B, out_channels, 16, W/2)
Default in_channels comes from src.data.color.INPUT_CHANNELS.
"""

import torch.nn as nn
from torch import Tensor

from src.data.color import INPUT_CHANNELS


class ConvNeXtStem(nn.Module):
    """Lightweight stem — 4 convs, ~70K params.

    Good for 12.5M backbone where stem should be small.
    """

    def __init__(self, in_channels: int = INPUT_CHANNELS, out_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


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
    """Heavier stem with residual blocks.

    Extracts richer low-level features before attention stages.
    Better for larger backbones (25M+) where the stem cost is small
    relative to total params but the feature quality matters.

    Output stride 2×2: (B, C_in, 32, W) → (B, out_channels, 16, W/2)
    Height reduced 2× by first conv, width reduced 2× by second conv.

    depth=2 (default, ~500K params):
      Conv 3→64, stride (2,1)   (32×W → 16×W)
      ResBlock 64                (refine)
      Conv 64→128, stride (1,2)  (16×W → 16×W/2)
      ResBlock 128               (refine)
      Conv 128→out_channels      (project to stage 1 dim)

    depth=3 (~630K params):
      Same as depth=2, plus an extra ResBlock at out_channels.
      Larger receptive field helps LID distinguish similar scripts.
    """

    def __init__(self, in_channels: int = INPUT_CHANNELS, out_channels: int = 64,
                 depth: int = 2):
        super().__init__()
        layers = [
            # Downsample 2× height, expand channels
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            # Residual refinement
            _ResBlock(64),
            # Downsample 2× width, expand channels
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            # Residual refinement
            _ResBlock(128),
            # Project to output channels
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        ]
        if depth >= 3:
            layers.append(_ResBlock(out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
