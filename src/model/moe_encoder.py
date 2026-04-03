"""
Lipi MoE Vision Encoder.

Architecture:
    Input: (B, 2, 32, W) — L+a from rgb_to_input
    -> ColorProjection: L+a → 1ch
    -> ResNet Stem: 1→64ch, stride 4×
    -> Shared SWA 4×4: character-level universal features
    -> Shared SWA 4×16: sequence-level universal features
    -> LID-1: 10-group classification
    -> Expert SWA 4×4: group-specific character features
    -> Height pool 8→4
    -> Expert SWA 4×16: group-specific sequence features
    -> Height pool 4→1
    -> LayerNorm
    -> LID-2: per-script classification (multi-script groups only)
    -> Per-script BiLSTM CTC heads
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
    """MLP CTC head for one script/group.

    The encoder (32 SWA blocks) already provides rich contextual features.
    A 3-layer MLP refines per-timestep features before projecting to vocab.
    No sequential LSTM bottleneck — fully parallel across timesteps.
    """

    def __init__(self, enc_dim: int, vocab_size: int,
                 hidden_dim: int = 384, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        mlp_hidden = enc_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, vocab_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features)


class GroupCTCModule(nn.Module):
    """CTC module for one group. Handles both single-script and multi-script groups.

    Single-script: one CTC head, no LID-2.
    Multi-script: LID-2 classifier + per-script CTC heads.
    """

    def __init__(self, enc_dim: int, script_vocab_sizes: list[int],
                 script_names: list[str],
                 hidden_dim: int = 384, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_scripts = len(script_vocab_sizes)
        self.script_names = script_names
        self.multi_script = self.n_scripts > 1
        self.max_vocab = max(script_vocab_sizes)

        # Per-script CTC heads
        self.heads = nn.ModuleList([
            CTCHead(enc_dim, vs, hidden_dim, num_layers, dropout)
            for vs in script_vocab_sizes
        ])

        # LID-2 with learned spatial projection (only for multi-script groups)
        if self.multi_script:
            # Conv1d reduction: T(=48) → 12 → 1. Direct gradient, no softmax.
            self.lid2_pool = nn.Sequential(
                nn.Conv1d(enc_dim, enc_dim, kernel_size=4, stride=4,
                          groups=enc_dim),                     # 48 → 12
                nn.GELU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=12,
                          groups=enc_dim),                     # 12 → 1
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
        for s in range(self.n_scripts):
            mask = (script_ids == s)
            if not mask.any():
                continue
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

        # Shared SWA 4×4
        self.shared_swa_4x4 = nn.ModuleList([
            SWABlock(dim=shared_dim, num_heads=shared_dim // 32,
                     window_h=4, window_w=4,
                     shift=(i % 2 == 1), mlp_ratio=shared_mlp_ratio)
            for i in range(shared_blocks_4x4)
        ])

        # Shared SWA 4×16
        self.shared_swa_4x16 = nn.ModuleList([
            SWABlock(dim=shared_dim, num_heads=shared_dim // 32,
                     window_h=4, window_w=16,
                     shift=(i % 2 == 1), mlp_ratio=shared_mlp_ratio)
            for i in range(shared_blocks_4x16)
        ])

        # LID-1
        self.lid_coarse = LIDCoarse(in_channels=shared_dim, num_groups=num_groups)

        # Channel projection
        self.proj1 = nn.Linear(shared_dim, stage1_dim) if shared_dim != stage1_dim else nn.Identity()

        # Expert SWA Stage 1
        self.stage1 = nn.ModuleList([
            FullyExpertSWABlock(dim=stage1_dim, num_heads=stage1_dim // 32,
                                num_groups=num_groups, window_h=4, window_w=4,
                                shift=(i % 2 == 1), mlp_ratio=stage1_mlp_ratio)
            for i in range(stage1_blocks)
        ])

        # Height pool 8→4
        self.pool1 = LearnedHeightPooling(channels=stage1_dim, h_in=8, h_out=4)

        # Channel projection
        self.proj2 = nn.Linear(stage1_dim, stage2_dim) if stage1_dim != stage2_dim else nn.Identity()

        # Expert SWA Stage 2
        self.stage2 = nn.ModuleList([
            FullyExpertSWABlock(dim=stage2_dim, num_heads=stage2_dim // 32,
                                num_groups=num_groups, window_h=4, window_w=16,
                                shift=(i % 2 == 1), mlp_ratio=stage2_mlp_ratio)
            for i in range(stage2_blocks)
        ])

        # Height pool 4→1
        self.pool2 = LearnedHeightPooling(channels=stage2_dim, h_in=4, h_out=1)

        # Final norm
        self.norm = nn.LayerNorm(stage2_dim)

        # CTC heads: per-script within each group (with LID-2 for multi-script groups)
        if group_script_vocab_sizes is not None:
            # New: per-script CTC heads with LID-2
            if group_script_names is None:
                group_script_names = [[f"s{i}" for i in range(len(vs))]
                                      for vs in group_script_vocab_sizes]
            self.ctc_modules = nn.ModuleList([
                GroupCTCModule(
                    enc_dim=stage2_dim,
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
                    enc_dim=stage2_dim,
                    script_vocab_sizes=[vocab_sizes[g]],
                    script_names=[f"group{g}"],
                    hidden_dim=head_hidden,
                    num_layers=head_layers,
                    dropout=head_dropout,
                )
                for g in range(num_groups)
            ])

        self.output_dim = stage2_dim

    def forward(
        self,
        images: Tensor,
        group_ids: Tensor | None = None,
        script_ids: Tensor | None = None,
        detach_for_experts: bool = False,
    ) -> dict:
        B = images.shape[0]
        W = images.shape[3]
        T = W // 4

        # Color projection
        x = self.color_proj(images)

        # Stem
        x = self.stem(x)
        _, C, h, w = x.shape

        # Reshape + project
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj_shared(x)

        # Shared SWA (with gradient checkpointing to save activation memory)
        for block in self.shared_swa_4x4:
            if self.training and torch.is_grad_enabled():
                x = ckpt_util.checkpoint(block, x, h, w, use_reentrant=False)
            else:
                x = block(x, h=h, w=w)
        for block in self.shared_swa_4x16:
            if self.training and torch.is_grad_enabled():
                x = ckpt_util.checkpoint(block, x, h, w, use_reentrant=False)
            else:
                x = block(x, h=h, w=w)

        # LID-1 (with learned attention pooling)
        group_logits = self.lid_coarse.forward_seq(x)
        if group_ids is None:
            group_ids = group_logits.argmax(dim=-1)

        # Optionally detach shared features before expert stages.
        # When True: LID-1 gradient → shared encoder, CTC gradient → experts only.
        # Prevents CTC from competing with LID-1 for shared features early in training.
        if detach_for_experts:
            x = x.detach()

        # Project to stage1
        x = self.proj1(x)

        # Expert SWA Stage 1
        for block in self.stage1:
            x = block(x, h=h, w=w, group_ids=group_ids)

        # Height pool 8→4
        C1 = x.shape[-1]
        x = x.reshape(B, h, w, C1).permute(0, 3, 1, 2)
        x = self.pool1(x)
        h = 4

        # Project to stage2
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C1)
        x = self.proj2(x)

        # Expert SWA Stage 2
        for block in self.stage2:
            x = block(x, h=h, w=w, group_ids=group_ids)

        # Height pool 4→1
        C2 = x.shape[-1]
        x = x.reshape(B, h, w, C2).permute(0, 3, 1, 2)
        x = self.pool2(x)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, C2)

        # Final norm
        x = self.norm(x)

        # Per-group CTC with LID-2
        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)
        all_script_logits = []  # (group_idx, script_logits_tensor, mask)

        for g in range(self.num_groups):
            mask = (group_ids == g)
            if not mask.any():
                continue

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
