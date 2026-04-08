"""
Lipi MoE Vision Encoder.

Architecture:
    Input: (B, 2, 32, W) — L+a from rgb_to_input
    -> ColorProjection: L+a → 1ch
    -> ResNet Stem: 1→64ch, stride 2×2  → (B, 64, 16, W/2)
    -> Shared SWA 8×8: character-level universal features   (h=16, w=W/2)
    -> Shared SWA 8×32: sequence-level universal features
    -> LID-1: 13-group classification
    -> Expert SWA 8×8: group-specific features (high-res)   (h=16, w=W/2)
    -> Height pool 16→8 + Width pool 2×                    (h=8, w=W/4)
    -> Expert SWA 8×8: group-specific features (low-res)   (h=8, w=W/4)
    -> Expert SWA 8×8: group-specific sequence features    (h=8, w=W/4)
    -> Fold h=8 into channels → (B, W/4, C*8)
    -> LayerNorm
    -> LID-2: per-script classification (multi-script groups only)
    -> Per-script CTC heads (T=W/4, richer features from direct h=4 fold)
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util
from torch import Tensor

from src.data.color import ColorProjection
from src.model.stem import ResNetStem
from src.model.pooling import LearnedHeightPooling
from src.model.attention import SWABlock, FullyExpertSWABlock
from src.model.lid import LIDCoarse, NUM_GROUPS


class CTCHead(nn.Module):
    """Linear CTC head — SWA blocks already provide full context."""

    def __init__(self, enc_dim: int, vocab_size: int, **_kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Linear(enc_dim, vocab_size)

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(features)


class GroupCTCModule(nn.Module):
    """CTC module for one group. Handles both single-script and multi-script groups.

    Single-script: one CTC head, no LID-2.
    Multi-script: LID-2 classifier + per-script CTC heads.
    """

    def __init__(self, enc_dim: int, script_vocab_sizes: list[int],
                 script_names: list[str], **_kwargs):
        super().__init__()
        self.n_scripts = len(script_vocab_sizes)
        self.script_names = script_names
        self.multi_script = self.n_scripts > 1
        self.max_vocab = max(script_vocab_sizes)

        # Per-script CTC heads
        self.heads = nn.ModuleList([
            CTCHead(enc_dim, vs) for vs in script_vocab_sizes
        ])

        # LID-2 with learned spatial projection (only for multi-script groups)
        if self.multi_script:
            # Conv1d reduction: T → T//4 → 1 (adaptive pool handles any T)
            g = 16
            self.lid2_pool = nn.Sequential(
                nn.Conv1d(enc_dim, enc_dim, kernel_size=4, stride=4,
                          groups=g),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.lid2_classifier = nn.Sequential(
                nn.Linear(enc_dim, enc_dim // 4),
                nn.ReLU(),
                nn.Linear(enc_dim // 4, self.n_scripts),
            )
        else:
            self.lid2_pool = None
            self.lid2_classifier = None

    def forward(self, features: Tensor, script_ids: Tensor | None = None
                ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Args:
            features: (N, T, C) — encoder output for this group's samples.
            script_ids: (N,) — local script index within this group (0..n_scripts-1).
                        None at inference → uses LID-2 predictions.

        Returns:
            logits: (N, T, max_vocab)
            script_logits: (N, n_scripts) or None if single-script
            script_ids_used: (N,) script IDs used for routing
        """
        N, T, C = features.shape

        # LID-2 with learned spatial projection
        script_logits = None
        if self.multi_script:
            pooled = self.lid2_pool(features.permute(0, 2, 1)).squeeze(-1)  # (N, C)
            script_logits = self.lid2_classifier(pooled)             # (N, n_scripts)
            if script_ids is None:
                script_ids = script_logits.argmax(dim=-1)
        else:
            script_ids = torch.zeros(N, dtype=torch.long, device=features.device)

        # Route to per-script CTC heads
        logits = torch.zeros(N, T, self.max_vocab,
                             device=features.device, dtype=features.dtype)
        active_scripts = torch.bincount(
            script_ids, minlength=self.n_scripts).nonzero(as_tuple=True)[0].tolist()
        for s in active_scripts:
            mask = (script_ids == s)
            head_out = self.heads[s](features[mask]).to(logits.dtype)
            logits[mask, :, :head_out.shape[-1]] = head_out

        return logits, script_logits, script_ids


class LipiMoEEncoder(nn.Module):
    """Full Lipi MoE encoder with LID-1 group routing + LID-2 script routing."""

    def __init__(
        self,
        stem_channels: int = 64,
        stem_depth: int = 3,
        # Shared SWA
        shared_dim: int = 288,
        shared_blocks_4x4: int = 8,
        shared_blocks_4x16: int = 4,
        shared_mlp_ratio: int = 4,
        # Expert SWA Stage 1
        stage1_dim: int = 288,
        stage1_blocks: int = 12,
        stage1_downsample_after: int = 4,
        stage1_mlp_ratio: int = 4,
        # Expert SWA Stage 2
        stage2_dim: int = 576,
        stage2_blocks: int = 8,
        stage2_mlp_ratio: int = 4,
        # Groups and scripts
        num_groups: int = NUM_GROUPS,
        group_script_vocab_sizes: list[list[int]] | None = None,
        group_script_names: list[list[str]] | None = None,
        # Legacy: single vocab per group (no LID-2)
        vocab_sizes: list[int] | int | None = None,
        head_hidden: int = 384,
        head_layers: int = 2,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_groups = num_groups

        # Color projection
        self.color_proj = ColorProjection()

        # Stem
        self.stem = ResNetStem(out_channels=stem_channels, depth=stem_depth)

        # Channel projection
        self.proj_shared = nn.Linear(stem_channels, shared_dim)

        # Shared SWA: 8×8 (local) then 8×32 (wide context)
        # Window sizes scaled 2× from original 4×4/4×16 to match 2×2 stem
        self.shared_swa = nn.ModuleList()
        for i in range(shared_blocks_4x4):
            self.shared_swa.append(
                SWABlock(dim=shared_dim, num_heads=shared_dim // 32,
                         window_h=8, window_w=8,
                         shift=(i % 2 == 1), mlp_ratio=shared_mlp_ratio))
        for i in range(shared_blocks_4x16):
            self.shared_swa.append(
                SWABlock(dim=shared_dim, num_heads=shared_dim // 32,
                         window_h=8, window_w=32,
                         shift=(i % 2 == 1), mlp_ratio=shared_mlp_ratio))

        # LID-1
        self.lid_coarse = LIDCoarse(in_channels=shared_dim, num_groups=num_groups)

        # Channel projection
        self.proj1 = nn.Linear(shared_dim, stage1_dim) if shared_dim != stage1_dim else nn.Identity()

        # Expert SWA Stage 1 (8×8 windows, same scale as shared)
        self.stage1 = nn.ModuleList([
            FullyExpertSWABlock(dim=stage1_dim, num_heads=stage1_dim // 32,
                                num_groups=num_groups, window_h=8, window_w=8,
                                shift=(i % 2 == 1), mlp_ratio=stage1_mlp_ratio)
            for i in range(stage1_blocks)
        ])
        self.stage1_downsample_after = stage1_downsample_after

        # Height pool 16→8 + width pool 2×
        self.pool1 = LearnedHeightPooling(channels=stage1_dim, h_in=16, h_out=8)
        self.width_pool = nn.Sequential(
            nn.Conv1d(stage1_dim, stage1_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, stage1_dim),
            nn.GELU(),
        )

        # Channel projection
        self.proj2 = nn.Linear(stage1_dim, stage2_dim) if stage1_dim != stage2_dim else nn.Identity()

        # Expert SWA Stage 2 (8×8 windows at h=8, full vertical coverage)
        self.stage2 = nn.ModuleList([
            FullyExpertSWABlock(dim=stage2_dim, num_heads=stage2_dim // 32,
                                num_groups=num_groups, window_h=8, window_w=8,
                                shift=(i % 2 == 1), mlp_ratio=stage2_mlp_ratio)
            for i in range(stage2_blocks)
        ])

        # Fold h=8 directly into channels (no height pool — SWA already contextualized)
        self.enc_out_dim = stage2_dim * 8
        self.norm = nn.LayerNorm(self.enc_out_dim)

        # CTC heads: per-script within each group (with LID-2 for multi-script groups)
        if group_script_vocab_sizes is not None:
            # New: per-script CTC heads with LID-2
            if group_script_names is None:
                group_script_names = [[f"s{i}" for i in range(len(vs))]
                                      for vs in group_script_vocab_sizes]
            self.ctc_modules = nn.ModuleList([
                GroupCTCModule(
                    enc_dim=self.enc_out_dim,
                    script_vocab_sizes=group_script_vocab_sizes[g],
                    script_names=group_script_names[g],
                    hidden_dim=head_hidden,
                    num_layers=head_layers,
                    dropout=head_dropout,
                )
                for g in range(num_groups)
            ])
        else:
            # Legacy: single CTC head per group
            if isinstance(vocab_sizes, int):
                vocab_sizes = [vocab_sizes] * num_groups
            if vocab_sizes is None:
                vocab_sizes = [171] * num_groups
            self.ctc_modules = nn.ModuleList([
                GroupCTCModule(
                    enc_dim=self.enc_out_dim,
                    script_vocab_sizes=[vocab_sizes[g]],
                    script_names=[f"group{g}"],
                    hidden_dim=head_hidden,
                    num_layers=head_layers,
                    dropout=head_dropout,
                )
                for g in range(num_groups)
            ])

        self.output_dim = self.enc_out_dim

    def forward(
        self,
        images: Tensor,
        group_ids: Tensor | None = None,
        script_ids: Tensor | None = None,
        detach_for_experts: bool = False,
    ) -> dict:
        B = images.shape[0]

        # Color projection
        x = self.color_proj(images)

        # Stem
        x = self.stem(x)
        _, C, h, w = x.shape

        # Reshape + project
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj_shared(x)

        # Shared SWA
        for i, block in enumerate(self.shared_swa):
            if self.training and torch.is_grad_enabled():
                x = ckpt_util.checkpoint(block, x, h, w, use_reentrant=True)
            else:
                x = block(x, h=h, w=w)

        # LID-1
        group_logits = self.lid_coarse.forward_seq(x)
        if group_ids is None:
            group_ids = group_logits.argmax(dim=-1)

        if detach_for_experts:
            x = x.detach()

        x = self.proj1(x)

        # Expert SWA Stage 1 — before downsampling
        for i, block in enumerate(self.stage1[:self.stage1_downsample_after]):
            x = block(x, h=h, w=w, group_ids=group_ids)

        # Height pool 16→8, width pool 2×
        C1 = x.shape[-1]
        x = x.reshape(B, h, w, C1).permute(0, 3, 1, 2)
        x = self.pool1(x)
        h = 8
        x = x.reshape(B * h, C1, w)
        x = self.width_pool(x)
        w = x.shape[2]
        x = x.reshape(B, h, C1, w).permute(0, 1, 3, 2).reshape(B, h * w, C1)

        # Expert SWA Stage 1 — after downsampling
        for i, block in enumerate(self.stage1[self.stage1_downsample_after:]):
            x = block(x, h=h, w=w, group_ids=group_ids)

        x = self.proj2(x)

        # Expert SWA Stage 2
        for i, block in enumerate(self.stage2):
            x = block(x, h=h, w=w, group_ids=group_ids)

        # Fold h=8 into channels
        C2 = x.shape[-1]
        x = x.reshape(B, h, w, C2)
        x = x.permute(0, 2, 1, 3).reshape(B, w, C2 * h)
        T = w
        x = self.norm(x)

        # Per-group CTC with LID-2
        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)
        all_script_logits = []  # (group_idx, script_logits_tensor, mask)

        # Compute group counts on GPU, transfer once
        group_counts = torch.bincount(group_ids, minlength=self.num_groups)
        active_groups = group_counts.nonzero(as_tuple=True)[0].tolist()

        for g in active_groups:
            mask = (group_ids == g)

            # Get local script_ids for this group (if provided)
            local_script_ids = None
            if script_ids is not None:
                local_script_ids = script_ids[mask]

            g_logits, g_script_logits, g_script_ids = self.ctc_modules[g](
                x[mask], script_ids=local_script_ids)
            logits[mask, :, :g_logits.shape[-1]] = g_logits.to(logits.dtype)

            if g_script_logits is not None:
                all_script_logits.append((g, g_script_logits, mask))

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return {
            "logits": logits,
            "lengths": lengths,
            "group_logits": group_logits,
            "group_ids": group_ids,
            "script_logits_per_group": all_script_logits,  # for LID-2 loss
        }
