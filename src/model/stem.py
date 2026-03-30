"""
Convolutional Stems for OCR encoder.

Two variants:
  ConvNeXtStem: lightweight (70K params) — for 12.5M backbone
  ResNetStem: heavier with residual blocks (500K+ params) — for 50M backbone

Both output stride 4×4: (B, 3, 32, W) → (B, out_channels, 8, W/4)
"""

import torch
import torch.nn as nn
from torch import Tensor


class ConvNeXtStem(nn.Module):
    """Lightweight stem — 4 convs, ~70K params.

    Good for 12.5M backbone where stem should be small.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
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
    """Heavier stem with residual blocks — ~500K params.

    Extracts richer low-level features before attention stages.
    Better for larger backbones (25M+) where the stem cost is small
    relative to total params but the feature quality matters.

    Structure:
      Conv 3→64, stride 2      (32×W → 16×W/2)
      ResBlock 64               (refine)
      Conv 64→128, stride 2     (16×W/2 → 8×W/4)
      ResBlock 128              (refine)
      Conv 128→out_channels     (project to stage 1 dim)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            # Downsample 2× and expand channels
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            # Residual refinement
            _ResBlock(64),
            # Downsample 2× and expand channels
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            # Residual refinement
            _ResBlock(128),
            # Project to output channels
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
