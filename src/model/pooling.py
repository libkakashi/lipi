"""
Learned Height Pooling.

Reduces the height dimension of feature maps using a learned linear projection
rather than a fixed stride or average pool. This lets the model learn which
vertical positions carry the most information for each channel.

ExpertPooling wraps per-group height + width pooling for MoE routing.
"""

import torch
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


class ExpertPooling(nn.Module):
    """Per-group expert height + width pooling.

    Each group gets its own LearnedHeightPooling and width Conv1d.
    Routes samples by group_ids, pools independently, scatters back.
    """

    def __init__(self, channels: int, h_in: int, h_out: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.h_in = h_in
        self.h_out = h_out
        self.channels = channels

        self.height_pools = nn.ModuleList([
            LearnedHeightPooling(channels, h_in, h_out)
            for _ in range(num_groups)
        ])
        self.width_pools = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(1, channels),
                nn.GELU(),
            )
            for _ in range(num_groups)
        ])

    def forward(self, x: Tensor, h: int, w: int, group_ids: Tensor
                ) -> tuple[Tensor, int, int]:
        """
        Args:
            x: (B, h*w, C) — flattened 2D features.
            h, w: spatial dims.
            group_ids: (B,) — group index per sample.

        Returns:
            x: (B, h_out*w_out, C)
            h_out, w_out: new spatial dims.
        """
        B, _, C = x.shape

        # Sort by group for contiguous access
        sorted_idx = group_ids.argsort()
        counts = torch.bincount(group_ids, minlength=self.num_groups).tolist()
        x_sorted = x[sorted_idx]

        # Height pool: need (B, C, h, w) format
        x_sorted = x_sorted.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, h, w)
        h_pooled = torch.empty(B, C, self.h_out, w, device=x.device, dtype=x.dtype)

        start = 0
        for g in range(self.num_groups):
            end = start + counts[g]
            if start < end:
                h_pooled[start:end] = self.height_pools[g](x_sorted[start:end])
            start = end

        h_out = self.h_out

        # Width pool: reshape to (B*h_out, C, w) for Conv1d
        x_flat = h_pooled.reshape(B * h_out, C, w)
        # All groups produce the same w_out (stride=2), so we can batch the reshape
        # but need per-group Conv1d weights
        w_pooled_flat = torch.empty(B * h_out, C, (w + 1) // 2, device=x.device, dtype=x.dtype)

        start = 0
        for g in range(self.num_groups):
            n = counts[g]
            end = start + n
            if n > 0:
                # Each sample has h_out rows in the flattened tensor
                flat_s = start * h_out
                flat_e = end * h_out
                w_pooled_flat[flat_s:flat_e] = self.width_pools[g](x_flat[flat_s:flat_e])
            start = end

        w_out = w_pooled_flat.shape[2]

        # Reshape back to (B, h_out*w_out, C) and unsort
        x_out = w_pooled_flat.reshape(B, h_out, C, w_out)
        x_out = x_out.permute(0, 1, 3, 2).reshape(B, h_out * w_out, C)
        x_out = x_out[sorted_idx.argsort()]

        return x_out, h_out, w_out
