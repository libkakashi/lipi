"""
Lipi v3 MoE Vision Encoder.

Architecture:
    Input: (B, 2, 32, W) — L+a
    -> HGNetV2 backbone (pretrained, OCR strides) → (B, 2048, 1, W/2)
    -> Project 2048 → dim                         → (B, W/2, dim)
    -> LID-1: 13-group classification
    -> Expert global attention blocks (1D)         → (B, W/2, dim)
    -> LID-2 + Per-script CTC heads (T=W/2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util
from torch import Tensor

import timm

from src.model.lid import LIDCoarse, NUM_GROUPS


class GlobalAttention(nn.Module):
    """Standard multi-head self-attention on 1D sequences."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ExpertBlock(nn.Module):
    """Per-group expert block with global attention + MLP.

    Each group has its own attention and MLP. Routes by group_ids,
    sorts for contiguous access, scatters back.
    """

    def __init__(self, dim: int, num_heads: int, num_groups: int, mlp_ratio: int = 4):
        super().__init__()
        self.num_groups = num_groups
        self.norm1 = nn.LayerNorm(dim)
        self.expert_attns = nn.ModuleList([
            GlobalAttention(dim, num_heads) for _ in range(num_groups)
        ])
        self.norm2 = nn.LayerNorm(dim)
        self.expert_mlps = nn.ModuleList([
            MLP(dim, mlp_ratio) for _ in range(num_groups)
        ])

    def forward(self, x: Tensor, group_ids: Tensor) -> Tensor:
        B = x.shape[0]
        sorted_idx = group_ids.argsort()
        counts = torch.bincount(group_ids, minlength=self.num_groups).tolist()
        x_sorted = x[sorted_idx]

        # Expert attention
        normed = self.norm1(x_sorted)
        attn_out = torch.empty_like(x_sorted)
        start = 0
        for g in range(self.num_groups):
            end = start + counts[g]
            if start < end:
                if self.training and torch.is_grad_enabled():
                    result = ckpt_util.checkpoint(
                        self.expert_attns[g], normed[start:end], use_reentrant=True)
                else:
                    result = self.expert_attns[g](normed[start:end])
                attn_out[start:end] = result.to(attn_out.dtype)
            start = end
        x_sorted = x_sorted + attn_out

        # Expert MLP
        normed = self.norm2(x_sorted)
        mlp_out = torch.empty_like(x_sorted)
        start = 0
        for g in range(self.num_groups):
            end = start + counts[g]
            if start < end:
                if self.training and torch.is_grad_enabled():
                    result = ckpt_util.checkpoint(
                        self.expert_mlps[g], normed[start:end], use_reentrant=True)
                else:
                    result = self.expert_mlps[g](normed[start:end])
                mlp_out[start:end] = result.to(mlp_out.dtype)
            start = end
        x_sorted = x_sorted + mlp_out

        return x_sorted[sorted_idx.argsort()]


class CTCHead(nn.Module):
    def __init__(self, enc_dim: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Linear(enc_dim, vocab_size)

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(features)


class GroupCTCModule(nn.Module):
    """Per-group CTC with optional LID-2 for multi-script groups."""

    def __init__(self, enc_dim: int, script_vocab_sizes: list[int],
                 script_names: list[str]):
        super().__init__()
        self.n_scripts = len(script_vocab_sizes)
        self.script_names = script_names
        self.multi_script = self.n_scripts > 1
        self.max_vocab = max(script_vocab_sizes)

        self.heads = nn.ModuleList([
            CTCHead(enc_dim, vs) for vs in script_vocab_sizes
        ])

        if self.multi_script:
            self.lid2_pool = nn.Sequential(
                nn.Conv1d(enc_dim, enc_dim, kernel_size=4, stride=4, groups=16),
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
        N, T, C = features.shape

        script_logits = None
        if self.multi_script:
            pooled = self.lid2_pool(features.permute(0, 2, 1)).squeeze(-1)
            script_logits = self.lid2_classifier(pooled)
            if script_ids is None:
                script_ids = script_logits.argmax(dim=-1)
        else:
            script_ids = torch.zeros(N, dtype=torch.long, device=features.device)

        logits = torch.zeros(N, T, self.max_vocab,
                             device=features.device, dtype=features.dtype)
        active_scripts = torch.bincount(
            script_ids, minlength=self.n_scripts).nonzero(as_tuple=True)[0].tolist()
        for s in active_scripts:
            mask = (script_ids == s)
            head_out = self.heads[s](features[mask]).to(logits.dtype)
            logits[mask, :, :head_out.shape[-1]] = head_out

        return logits, script_logits, script_ids


class LipiV3Encoder(nn.Module):
    """Lipi v3: HGNetV2 backbone + expert global attention + CTC.

    Replaces the v2 stem + shared SWA with a pretrained CNN backbone.
    Expert blocks use 1D global attention (no windows needed at h=1).
    """

    # OCR stride overrides: height-aggressive, width-conservative
    _OCR_STRIDES = {
        'stem.stem1.conv': (2, 1),      # h/2
        'stem.stem3.conv': (2, 1),       # h/4
        'stages_1.downsample.conv': (2, 2),  # h/8, w/2
        'stages_2.downsample.conv': (2, 1),  # h/16
        'stages_3.downsample.conv': (2, 1),  # h/32 = 1
    }

    def __init__(
        self,
        dim: int = 512,
        backbone: str = 'hgnetv2_b3',
        num_expert_blocks: int = 2,
        mlp_ratio: int = 2,
        num_groups: int = NUM_GROUPS,
        group_script_vocab_sizes: list[list[int]] | None = None,
        group_script_names: list[list[str]] | None = None,
        vocab_sizes: list[int] | int | None = None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.config = {
            "dim": dim,
            "backbone": backbone,
            "num_expert_blocks": num_expert_blocks,
            "mlp_ratio": mlp_ratio,
            "num_groups": num_groups,
            "group_script_vocab_sizes": group_script_vocab_sizes,
            "group_script_names": group_script_names,
        }

        # Input: L+a (2ch) → 3ch for pretrained backbone
        self.input_proj = nn.Conv2d(2, 3, kernel_size=1)

        # Pretrained backbone with OCR strides
        self.backbone = timm.create_model(
            f'{backbone}.ssld_stage1_in22k_in1k' if 'ssld' not in backbone else backbone,
            pretrained=True,
            features_only=True,
        )
        for name, mod in self.backbone.named_modules():
            if name in self._OCR_STRIDES:
                mod.stride = self._OCR_STRIDES[name]

        # Get backbone output channels (last stage)
        backbone_ch = self.backbone.feature_info.channels()[-1]

        # Project backbone features to dim
        self.proj = nn.Linear(backbone_ch, dim)

        # LID-1 on backbone features
        self.lid_coarse = LIDCoarse(in_channels=dim, num_groups=num_groups)

        # Expert global attention blocks (1D)
        self.expert_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_groups=num_groups,
                        mlp_ratio=mlp_ratio)
            for _ in range(num_expert_blocks)
        ])

        # Output
        self.enc_out_dim = dim
        self.norm = nn.LayerNorm(dim)

        # CTC heads
        if group_script_vocab_sizes is None:
            if isinstance(vocab_sizes, int):
                vocab_sizes = [vocab_sizes] * num_groups
            if vocab_sizes is None:
                vocab_sizes = [171] * num_groups
            group_script_vocab_sizes = [[vs] for vs in vocab_sizes]
            group_script_names = [[f"group{g}"] for g in range(num_groups)]
        elif group_script_names is None:
            group_script_names = [[f"s{i}" for i in range(len(vs))]
                                  for vs in group_script_vocab_sizes]

        self.ctc_modules = nn.ModuleList([
            GroupCTCModule(
                enc_dim=dim,
                script_vocab_sizes=group_script_vocab_sizes[g],
                script_names=group_script_names[g],
            )
            for g in range(num_groups)
        ])

    def forward(
        self,
        images: Tensor,
        group_ids: Tensor | None = None,
        script_ids: Tensor | None = None,
        detach_for_experts: bool = False,
    ) -> dict:
        B = images.shape[0]

        # Dequantize + adapt channels
        x = images.float() / 255.0 if images.dtype == torch.uint8 else images
        x = self.input_proj(x)

        # Backbone → last stage: (B, backbone_ch, 1, W/2)
        feats = self.backbone(x)
        x = feats[-1]

        # Squeeze h=1, project to dim: (B, W/2, dim)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, W/2, backbone_ch)
        x = self.proj(x)

        # LID-1
        group_logits = self.lid_coarse.forward_seq(x)
        if group_ids is None:
            group_ids = group_logits.argmax(dim=-1)

        if detach_for_experts:
            x = x.detach()

        # Expert global attention blocks
        for block in self.expert_blocks:
            x = block(x, group_ids)

        x = self.norm(x)
        T = x.shape[1]

        # Per-group CTC with LID-2
        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)
        all_script_logits = []

        group_counts = torch.bincount(group_ids, minlength=self.num_groups)
        active_groups = group_counts.nonzero(as_tuple=True)[0].tolist()

        for g in active_groups:
            mask = (group_ids == g)
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
            "script_logits_per_group": all_script_logits,
        }
