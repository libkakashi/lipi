"""
Lipi MoE Vision Encoder.

Full hierarchical MoE architecture:
    Input: (B, 3, 32, W)
    -> LearnedColorProjection 3→1
    -> ResNet Stem 1→64ch, stride 4×4
    -> LID-1: coarse group classification (6 groups)
    -> Channel projection 64→dim1
    -> Stage 1 MoE: N × SWABlockMoE (shared attention, group-specific expert MLPs)
    -> Height pooling 8→4
    -> LID-2: fine script classification (17 scripts)
    -> Channel projection dim1→dim2
    -> Stage 2 MoE: M × SWABlockMoE (shared attention, group-specific expert MLPs)
    -> Height pooling 4→1
    -> LayerNorm
    -> Per-script BiLSTM CTC heads

Stage 1 MoE: 6 group-level experts (one per script family)
Stage 2 MoE: 6 group-level experts (script-level would be too expensive)
CTC heads: 17 per-script heads (cheap, ~3.4M each)

During training: ground-truth group/script IDs route to experts.
During inference: LID-1/LID-2 predictions route to experts.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.data.color import MODE, LearnedColorProjection
from src.model.stem import ResNetStem
from src.model.pooling import LearnedHeightPooling
from src.model.attention import SWABlockMoE
from src.model.lid import (
    LIDCoarse, LIDFine,
    SCRIPTS, NUM_SCRIPTS, NUM_GROUPS,
    SCRIPT_TO_GROUP, GROUP_TO_ID,
)


class ScriptCTCHeads(nn.Module):
    """Per-script BiLSTM CTC heads.

    Routes encoder output to the correct BiLSTM head based on script_ids.
    Each script has its own head because vocab sizes may differ.
    """

    def __init__(self, enc_dim: int, vocab_size: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleDict()
        for script in SCRIPTS:
            self.heads[script] = nn.ModuleDict({
                "lstm": nn.LSTM(
                    enc_dim, hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    bidirectional=True, batch_first=True,
                ),
                "proj": nn.Linear(hidden_dim * 2, vocab_size),
            })

    def forward(self, features: Tensor, script_ids: Tensor) -> Tensor:
        """
        Args:
            features: (B, T, C) encoder output.
            script_ids: (B,) script index per sample.

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T, C = features.shape
        logits = torch.zeros(B, T, self.heads[SCRIPTS[0]]["proj"].out_features,
                             device=features.device, dtype=features.dtype)

        for script_idx, script in enumerate(SCRIPTS):
            mask = (script_ids == script_idx)
            if not mask.any():
                continue
            head = self.heads[script]
            out, _ = head["lstm"](features[mask])
            logits[mask] = head["proj"](out).to(logits.dtype)

        return logits


class LipiMoEEncoder(nn.Module):
    """Full Lipi MoE encoder with hierarchical LID routing.

    Combines color projection, stem, LID classifiers, MoE stages,
    and per-script CTC heads into a single module.
    """

    def __init__(
        self,
        stem_channels: int = 64,
        stage1_dim: int = 288,
        stage1_heads: int = 9,
        stage1_blocks: int = 8,
        stage1_window_h: int = 4,
        stage1_window_w: int = 4,
        stage1_mlp_ratio: int = 4,
        stage2_dim: int = 576,
        stage2_heads: int = 18,
        stage2_blocks: int = 12,
        stage2_window_h: int = 4,
        stage2_window_w: int = 16,
        stage2_mlp_ratio: int = 4,
        num_groups: int = NUM_GROUPS,
        num_scripts: int = NUM_SCRIPTS,
        vocab_size: int = 171,
        head_hidden: int = 256,
        head_layers: int = 2,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_scripts = num_scripts

        # Color projection (learned 3→1 or identity for L+a mode)
        if MODE == "learned":
            self.color_proj = LearnedColorProjection()
        else:
            self.color_proj = None

        # Stem
        self.stem = ResNetStem(out_channels=stem_channels)

        # LID-1: coarse group classifier (reads stem output)
        self.lid_coarse = LIDCoarse(in_channels=stem_channels, num_groups=num_groups)

        # Channel projection: stem → Stage 1
        self.proj1 = nn.Linear(stem_channels, stage1_dim)

        # Stage 1 MoE: group-specific expert MLPs, shared SWA
        self.stage1 = nn.ModuleList([
            SWABlockMoE(
                dim=stage1_dim, num_heads=stage1_heads,
                num_groups=num_groups,
                window_h=stage1_window_h, window_w=stage1_window_w,
                shift=(i % 2 == 1), mlp_ratio=stage1_mlp_ratio,
            )
            for i in range(stage1_blocks)
        ])

        # Height pooling: 8 → 4
        self.pool1 = LearnedHeightPooling(channels=stage1_dim, h_in=8, h_out=4)

        # LID-2: fine script classifier (reads Stage 1 output)
        self.lid_fine = LIDFine(in_dim=stage1_dim, num_scripts=num_scripts)

        # Channel projection: Stage 1 → Stage 2
        self.proj2 = nn.Linear(stage1_dim, stage2_dim)

        # Stage 2 MoE: group-specific expert MLPs (not per-script — too expensive)
        self.stage2 = nn.ModuleList([
            SWABlockMoE(
                dim=stage2_dim, num_heads=stage2_heads,
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

        # Per-script BiLSTM CTC heads
        self.ctc_heads = ScriptCTCHeads(
            enc_dim=stage2_dim, vocab_size=vocab_size,
            hidden_dim=head_hidden, num_layers=head_layers,
            dropout=head_dropout,
        )

        self.output_dim = stage2_dim

        # Build script→group mapping tensor for deriving group_ids from script_ids
        # Register as buffer so it moves with .to(device)
        s2g = torch.zeros(num_scripts, dtype=torch.long)
        for script, group in SCRIPT_TO_GROUP.items():
            if script in SCRIPTS:
                s2g[SCRIPTS.index(script)] = GROUP_TO_ID[group]
        self.register_buffer("_script_to_group", s2g)

    def forward(
        self,
        images: Tensor,
        script_ids: Tensor | None = None,
        group_ids: Tensor | None = None,
    ) -> dict:
        """
        Args:
            images: (B, C_in, 32, W) — raw images (RGB if learned, L+a if fixed).
            script_ids: (B,) ground-truth script IDs (training) or None (inference).
            group_ids: (B,) ground-truth group IDs (training) or None (inference).

        Returns:
            dict with keys:
                "logits": (B, T, vocab_size) CTC logits
                "lengths": (B,) sequence lengths
                "group_logits": (B, num_groups) LID-1 logits
                "script_logits": (B, num_scripts) LID-2 logits
                "group_ids": (B,) group IDs used for routing
                "script_ids": (B,) script IDs used for routing
        """
        B = images.shape[0]
        W = images.shape[3]
        T = W // 4

        # Color projection
        if self.color_proj is not None:
            x = self.color_proj(images)
        else:
            x = images

        # Stem: (B, C_in, 32, W) → (B, 64, 8, W/4)
        stem_out = self.stem(x)
        _, C, h, w = stem_out.shape  # h=8, w=W/4

        # LID-1: coarse group classification
        group_logits = self.lid_coarse(stem_out)
        if group_ids is None:
            group_ids = group_logits.argmax(dim=-1)

        # Reshape to sequence + channel projection
        x = stem_out.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj1(x)

        # Stage 1 MoE
        for block in self.stage1:
            x = block(x, h=h, w=w, group_ids=group_ids)

        # LID-2: fine script classification (before height pooling, full spatial info)
        script_logits = self.lid_fine(x)
        if script_ids is None:
            script_ids = script_logits.argmax(dim=-1)

        # Derive group_ids from script_ids for Stage 2 routing
        stage2_group_ids = self._script_to_group.to(script_ids.device)[script_ids]

        # Height pooling 8→4
        C1 = x.shape[-1]
        x = x.reshape(B, h, w, C1).permute(0, 3, 1, 2)
        x = self.pool1(x)
        h = 4

        # Channel projection
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C1)
        x = self.proj2(x)

        # Stage 2 MoE (routed by group derived from script)
        for block in self.stage2:
            x = block(x, h=h, w=w, group_ids=stage2_group_ids)

        # Height pooling 4→1
        C2 = x.shape[-1]
        x = x.reshape(B, h, w, C2).permute(0, 3, 1, 2)
        x = self.pool2(x)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, C2)

        # Final norm
        x = self.norm(x)

        # Per-script CTC heads
        logits = self.ctc_heads(x, script_ids)

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return {
            "logits": logits,
            "lengths": lengths,
            "group_logits": group_logits,
            "script_logits": script_logits,
            "group_ids": group_ids,
            "script_ids": script_ids,
        }
