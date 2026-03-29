"""
Learned Height Pooling.

Reduces the height dimension of feature maps using a learned linear projection
rather than a fixed stride or average pool. This lets the model learn which
vertical positions carry the most information for each channel.

Used twice in the encoder:
  - After Stage 1: h=8 -> h=4
  - After Stage 2: h=4 -> h=1 (full height collapse)
"""

import torch.nn as nn
from torch import Tensor


class LearnedHeightPooling(nn.Module):
    """Learned linear projection along the height axis.

    Reshapes (B, C, H, W) into (B, C*H, W), applies a linear layer
    to project from C*H_in to C*H_out, then reshapes back.

    This is equivalent to learning a weighted sum over height positions
    per channel.
    """

    def __init__(self, channels: int, h_in: int, h_out: int):
        """
        Args:
            channels: number of feature channels (C).
            h_in: input height.
            h_out: output height.
        """
        super().__init__()
        self.channels = channels
        self.h_in = h_in
        self.h_out = h_out
        # Project along height per-channel: learn a (h_out, h_in) weight per channel
        # Implemented as a grouped linear: treat each channel independently
        self.pool = nn.Conv1d(
            in_channels=channels * h_in,
            out_channels=channels * h_out,
            kernel_size=1,
            groups=channels,
            bias=True,
        )
        self.norm = nn.GroupNorm(1, channels * h_out)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H_in, W)

        Returns:
            (B, C, H_out, W)
        """
        B, C, H, W = x.shape
        assert H == self.h_in, f"Expected height {self.h_in}, got {H}"

        # Merge C and H into one dim: (B, C*H_in, W)
        x = x.reshape(B, C * H, W)

        # Grouped 1x1 conv: (B, C*H_in, W) -> (B, C*H_out, W)
        x = self.pool(x)
        x = self.norm(x)

        # Reshape back: (B, C, H_out, W)
        return x.reshape(B, C, self.h_out, W)
