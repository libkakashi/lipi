"""
Lipi MoE Vision Encoder.

Architecture:
    Input: (B, 2, 32, W) — L+a from rgb_to_input
    -> ColorProjection: L+a → 1ch
    -> ResNet Stem: 1→64ch, stride 4×
    -> Shared SWA 4×4: universal features
    -> LID-1: 8-group classification
    -> Expert SWA 4×4: group-specific char features
    -> Height pool 8→4
    -> Expert SWA 4×16: group-specific sequence context
    -> Height pool 4→1
    -> LayerNorm
    -> Per-group BiLSTM CTC heads (8 groups, shared vocab)

No LID-2. Script identity comes from the CTC output characters themselves.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.data.color import ColorProjection
from src.model.stem import ResNetStem
from src.model.pooling import LearnedHeightPooling
from src.model.attention import SWABlock, SWABlockMoE
from src.model.lid import (
    LIDCoarse,
    SCRIPTS, NUM_SCRIPTS, NUM_GROUPS,
    GROUPS, SCRIPT_TO_GROUP, GROUP_TO_ID,
)


class GroupCTCHeads(nn.Module):
    """Per-group BiLSTM CTC heads.

    One head per group (8 total). Each head decodes all scripts in its group
    using a shared vocab. The output characters identify the script.
    """

    def __init__(self, enc_dim: int, vocab_size: int, num_groups: int = NUM_GROUPS,
                 hidden_dim: int = 246, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.ModuleDict({
                "lstm": nn.LSTM(
                    enc_dim, hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=True, batch_first=True,
                ),
                "proj": nn.Linear(hidden_dim * 2, vocab_size),
            })
            for _ in range(num_groups)
        ])

    def forward(self, features: Tensor, group_ids: Tensor) -> Tensor:
        B, T, C = features.shape
        logits = torch.zeros(B, T, self.heads[0]["proj"].out_features,
                             device=features.device, dtype=features.dtype)
        for g in range(len(self.heads)):
            mask = (group_ids == g)
            if not mask.any():
                continue
            head = self.heads[g]
            out, _ = head["lstm"](features[mask])
            logits[mask] = head["proj"](out).to(logits.dtype)
        return logits


class LipiMoEEncoder(nn.Module):
    """Full Lipi MoE encoder with shared SWA before LID routing."""

    def __init__(
        self,
        stem_channels: int = 64,
        stem_depth: int = 3,
        # Shared SWA (before LID)
        shared_dim: int = 288,
        shared_blocks: int = 3,
        shared_window_h: int = 4,
        shared_window_w: int = 4,
        shared_mlp_ratio: int = 4,
        # Expert SWA Stage 1 (after LID, 4×4 window)
        stage1_dim: int = 288,
        stage1_blocks: int = 5,
        stage1_window_h: int = 4,
        stage1_window_w: int = 4,
        stage1_mlp_ratio: int = 4,
        # Expert SWA Stage 2 (4×16 window)
        stage2_dim: int = 576,
        stage2_blocks: int = 9,
        stage2_window_h: int = 4,
        stage2_window_w: int = 16,
        stage2_mlp_ratio: int = 4,
        # Groups
        num_groups: int = NUM_GROUPS,
        vocab_size: int = 171,
        head_hidden: int = 246,
        head_layers: int = 2,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_groups = num_groups

        # Color projection: L+a → 1ch
        self.color_proj = ColorProjection()

        # Stem: 1→64ch, downsample 4×
        self.stem = ResNetStem(out_channels=stem_channels, depth=stem_depth)

        # Channel projection: stem → shared SWA dim
        self.proj_shared = nn.Linear(stem_channels, shared_dim)

        # Shared SWA blocks (universal features, feeds LID)
        self.shared_swa = nn.ModuleList([
            SWABlock(
                dim=shared_dim,
                num_heads=shared_dim // 32,
                window_h=shared_window_h, window_w=shared_window_w,
                shift=(i % 2 == 1), mlp_ratio=shared_mlp_ratio,
            )
            for i in range(shared_blocks)
        ])

        # LID-1: coarse group classification
        self.lid_coarse = LIDCoarse(in_channels=shared_dim, num_groups=num_groups)

        # Channel projection: shared → stage1
        self.proj1 = nn.Linear(shared_dim, stage1_dim) if shared_dim != stage1_dim else nn.Identity()

        # Expert SWA Stage 1: group-specific, 4×4 window
        self.stage1 = nn.ModuleList([
            SWABlockMoE(
                dim=stage1_dim,
                num_heads=stage1_dim // 32,
                num_groups=num_groups,
                window_h=stage1_window_h, window_w=stage1_window_w,
                shift=(i % 2 == 1), mlp_ratio=stage1_mlp_ratio,
            )
            for i in range(stage1_blocks)
        ])

        # Height pooling: 8 → 4
        self.pool1 = LearnedHeightPooling(channels=stage1_dim, h_in=8, h_out=4)

        # Channel projection: Stage 1 → Stage 2
        self.proj2 = nn.Linear(stage1_dim, stage2_dim)

        # Expert SWA Stage 2: group-specific, 4×16 window
        self.stage2 = nn.ModuleList([
            SWABlockMoE(
                dim=stage2_dim,
                num_heads=stage2_dim // 32,
                num_groups=num_groups,
                window_h=stage2_window_h, window_w=stage2_window_w,
                shift=(i % 2 == 1), mlp_ratio=stage2_mlp_ratio,
            )
            for i in range(stage2_blocks)
        ])

        # Height pooling: 4 → 1
        self.pool2 = LearnedHeightPooling(channels=stage2_dim, h_in=4, h_out=1)

        # Final norm
        self.norm = nn.LayerNorm(stage2_dim)

        # Per-group BiLSTM CTC heads
        self.ctc_heads = GroupCTCHeads(
            enc_dim=stage2_dim, vocab_size=vocab_size,
            num_groups=num_groups, hidden_dim=head_hidden,
            num_layers=head_layers, dropout=head_dropout,
        )

        self.output_dim = stage2_dim

    def forward(
        self,
        images: Tensor,
        group_ids: Tensor | None = None,
    ) -> dict:
        B = images.shape[0]
        W = images.shape[3]
        T = W // 4

        # Color projection: L+a → 1ch
        x = self.color_proj(images)

        # Stem
        x = self.stem(x)
        _, C, h, w = x.shape

        # Reshape to sequence + project to shared dim
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj_shared(x)

        # Shared SWA blocks
        for block in self.shared_swa:
            x = block(x, h=h, w=w)

        # LID-1: group classification
        group_logits = self.lid_coarse.classifier(x.mean(dim=1))
        if group_ids is None:
            group_ids = group_logits.argmax(dim=-1)

        # Project to stage1 dim
        x = self.proj1(x)

        # Expert SWA Stage 1
        for block in self.stage1:
            x = block(x, h=h, w=w, group_ids=group_ids)

        # Height pooling 8→4
        C1 = x.shape[-1]
        x = x.reshape(B, h, w, C1).permute(0, 3, 1, 2)
        x = self.pool1(x)
        h = 4

        # Channel projection to Stage 2
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C1)
        x = self.proj2(x)

        # Expert SWA Stage 2
        for block in self.stage2:
            x = block(x, h=h, w=w, group_ids=group_ids)

        # Height pooling 4→1
        C2 = x.shape[-1]
        x = x.reshape(B, h, w, C2).permute(0, 3, 1, 2)
        x = self.pool2(x)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, C2)

        # Final norm
        x = self.norm(x)

        # Per-group CTC heads
        logits = self.ctc_heads(x, group_ids)

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return {
            "logits": logits,
            "lengths": lengths,
            "group_logits": group_logits,
            "group_ids": group_ids,
        }
