"""
Lipi Vision Encoder (Backbone).

Full architecture:
    Input: (B, 3, 32, W)
    -> Stem -> (B, 64, 16, W/2)              stride 2×2
    -> Channel projection 64->192
    -> Stage 1: 3x SWA blocks (window 4x4, C=192, 6 heads, RoPE-2D)
    -> Learned Height Pooling 16->4
    -> Channel projection 192->384
    -> Stage 2: 4x SWA blocks (window 4x8, C=384, 12 heads, RoPE-2D)
    -> Learned Height Pooling 4->2
    -> Fold h=2 into channels → (B, W/2, 768)
    -> Stage 3: 3x Global SA blocks (C=768, 12 heads, RoPE-1D)
    -> LayerNorm
    -> Output: (B, T, 768) where T = W/2

~35.4M parameters.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.model.stem import ConvNeXtStem, ResNetStem
from src.model.pooling import LearnedHeightPooling
from src.model.attention import SWABlock, GlobalBlock
from src.data.color import INPUT_CHANNELS


class LipiEncoder(nn.Module):
    """Lipi vision encoder backbone.

    Processes variable-width word crop images and produces a sequence
    of feature vectors suitable for RNN-T or CTC decoding.
    """

    def __init__(
        self,
        stem_channels: int = 64,
        stage1_dim: int = 192,
        stage1_heads: int = 6,
        stage1_blocks: int = 3,
        stage1_window_h: int = 4,
        stage1_window_w: int = 4,
        stage1_mlp_ratio: int = 3,
        stage2_dim: int = 384,
        stage2_heads: int = 12,
        stage2_blocks: int = 4,
        stage2_window_h: int = 4,
        stage2_window_w: int = 8,
        stage2_mlp_ratio: int = 3,
        stage3_dim: int = 384,
        stage3_heads: int = 12,
        stage3_blocks: int = 3,
        stage3_mlp_ratio: int = 4,
        stage3_window_w: int = 0,  # 0 = global attention (default), >0 = SWA with this width
        heavy_stem: bool = False,
    ):
        super().__init__()

        # Stage 0: Stem
        if heavy_stem:
            self.stem = ResNetStem(in_channels=INPUT_CHANNELS, out_channels=stem_channels)
        else:
            self.stem = ConvNeXtStem(in_channels=INPUT_CHANNELS, out_channels=stem_channels)

        # Channel projection: stem_channels -> stage1_dim
        self.proj1 = nn.Linear(stem_channels, stage1_dim)

        # Stage 1: SWA blocks — tight windows for single-character features
        self.stage1 = nn.ModuleList([
            SWABlock(
                dim=stage1_dim, num_heads=stage1_heads,
                window_h=stage1_window_h, window_w=stage1_window_w,
                shift=(i % 2 == 1), mlp_ratio=stage1_mlp_ratio,
            )
            for i in range(stage1_blocks)
        ])

        # Height pooling: 16 -> 4
        self.pool1 = LearnedHeightPooling(
            channels=stage1_dim, h_in=16, h_out=4
        )

        # Channel projection: stage1_dim -> stage2_dim
        self.proj2 = nn.Linear(stage1_dim, stage2_dim)

        # Stage 2: SWA blocks with alternating shift
        # Stage 2: SWA blocks — wider windows for neighbor context
        self.stage2 = nn.ModuleList([
            SWABlock(
                dim=stage2_dim, num_heads=stage2_heads,
                window_h=stage2_window_h, window_w=stage2_window_w,
                shift=(i % 2 == 1), mlp_ratio=stage2_mlp_ratio,
            )
            for i in range(stage2_blocks)
        ])

        # Height pooling: 4 -> 2 (fold h=2 into channels after this)
        self.pool2 = LearnedHeightPooling(
            channels=stage2_dim, h_in=4, h_out=2
        )

        # Stage 3: after height collapse (1D sequence)
        # Global (default) or wide SWA for larger models
        self._stage3_global = (stage3_window_w == 0)
        if self._stage3_global:
            self.stage3 = nn.ModuleList([
                GlobalBlock(
                    dim=stage3_dim, num_heads=stage3_heads,
                    mlp_ratio=stage3_mlp_ratio,
                )
                for _ in range(stage3_blocks)
            ])
        else:
            self.stage3 = nn.ModuleList([
                SWABlock(
                    dim=stage3_dim, num_heads=stage3_heads,
                    window_h=1, window_w=stage3_window_w,
                    shift=(i % 2 == 1), mlp_ratio=stage3_mlp_ratio,
                )
                for i in range(stage3_blocks)
            ])

        # Final layer norm
        self.norm = nn.LayerNorm(stage3_dim)

        # Store dims for external use
        self.output_dim = stage3_dim

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, 3, 32, W) — batch of height-normalized word crop images.

        Returns:
            features: (B, T, C) — encoded sequence features, T = W // 2.
            lengths: (B,) — valid sequence lengths (all equal to T for now,
                     but kept for future padding support).
        """
        B = x.shape[0]
        W = x.shape[3]
        T = W // 2  # Output sequence length

        # Stage 0: Stem
        # (B, 3, 32, W) -> (B, 64, 16, W/2)
        x = self.stem(x)
        _, C, h, w = x.shape  # h=16, w=W/2

        # Reshape to sequence: (B, C, h, w) -> (B, h*w, C)
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)

        # Channel projection 64 -> 192
        x = self.proj1(x)

        # Stage 1: SWA blocks (h=16, w=W/2)
        for block in self.stage1:
            x = block(x, h=h, w=w)

        # Height pooling 16 -> 4
        C1 = x.shape[-1]
        x = x.reshape(B, h, w, C1).permute(0, 3, 1, 2)
        x = self.pool1(x)
        h = 4  # New height

        # Channel projection 192 -> 384
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C1)
        x = self.proj2(x)

        # Stage 2: SWA blocks (h=4, w=W/2)
        for block in self.stage2:
            x = block(x, h=h, w=w)

        # Height pooling 4 -> 2, fold h=2 into channels
        C2 = x.shape[-1]
        x = x.reshape(B, h, w, C2).permute(0, 3, 1, 2)
        x = self.pool2(x)
        # (B, C2, 2, w) -> fold h into channels -> (B, w, C2*2)
        x = x.permute(0, 3, 2, 1).reshape(B, w, C2 * 2)

        # Stage 3: 1D sequence attention (global or wide SWA)
        for block in self.stage3:
            if self._stage3_global:
                x = block(x, seq_len=w)
            else:
                x = block(x, h=1, w=w)

        # Final norm
        x = self.norm(x)

        # Lengths (all T for uniform batches)
        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return x, lengths
