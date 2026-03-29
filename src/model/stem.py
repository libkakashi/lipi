"""
ConvNeXt-V2 Micro Stem.

4 conv layers with LayerNorm + GELU, total stride 4x4.
Input:  (B, 3, 32, W)
Output: (B, 64, 8, W/4)

Uses GroupNorm(1, C) as a channel-wise LayerNorm equivalent that works
on 4D tensors without reshaping. This is cleaner for ONNX export.
"""

import torch.nn as nn
from torch import Tensor


class ConvNeXtStem(nn.Module):
    """ConvNeXt-V2 inspired micro stem for OCR.

    Progressive downsampling: 32x W -> 16x W/2 -> 8x W/4.
    Four conv layers with GroupNorm + GELU activations.
    Output channels: 64.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            # Layer 1: 3 -> 32, stride 2x2 (32xW -> 16xW/2)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 32),  # LayerNorm equivalent for conv
            nn.GELU(),
            # Layer 2: 32 -> 32, stride 1x1 (refine features)
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            # Layer 3: 32 -> 64, stride 2x2 (16xW/2 -> 8xW/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            # Layer 4: 64 -> 64, stride 1x1 (refine features)
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, 32, W) — input image tensor.

        Returns:
            (B, 64, 8, W//4) — stem features.
        """
        return self.layers(x)
