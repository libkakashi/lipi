"""
Lipi v3 MoE Vision Encoder.

Architecture:
    Input: (B, 3, 32, W) — RGB
    -> HGNetV2 backbone (pretrained, 1024ch)
       → (B, 1024, 2, W/2)
    -> LID-1: 13-group classification
    -> Project 1024 → dim
    -> Local expert blocks (h=2, 2×32 windows, per-character)
    -> Pool h=2→1
    -> Wide expert blocks (h=1, 1×128 windows, word-context)
    -> Aggregate local + wide features
    -> LID-2 + Per-script CTC heads (T=W/2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util
from torch import Tensor

import timm

from src.model.lid import LIDCoarse, NUM_GROUPS


class WindowedAttention(nn.Module):
    """Shifted window attention on flattened 2D sequences (Swin-style)."""

    def __init__(self, dim: int, num_heads: int,
                 window_h: int = 2, window_w: int = 32, shift: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_h = window_h
        self.window_w = window_w
        self.shift = shift
        self.shift_w = window_w // 2 if shift else 0
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        B, N, C = x.shape
        x = x.reshape(B, h, w, C)

        if self.shift_w > 0:
            x = torch.roll(x, shifts=-self.shift_w, dims=2)

        pad_w = (self.window_w - w % self.window_w) % self.window_w
        if pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w))
        wp = w + pad_w

        nH = h // self.window_h
        nW = wp // self.window_w
        x = x.reshape(B, nH, self.window_h, nW, self.window_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * nH * nW, self.window_h * self.window_w, C)

        qkv = self.qkv(x).reshape(-1, self.window_h * self.window_w, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.shift_w > 0:
            mask = self._build_shift_mask(h, wp, x.device)
            num_windows = nH * nW
            attn = attn.reshape(B, num_windows, self.num_heads, -1, attn.shape[-1])
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.reshape(-1, self.num_heads, attn.shape[-2], attn.shape[-1])

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * nH * nW, self.window_h * self.window_w, C)
        out = self.proj(out)

        out = out.reshape(B, nH, nW, self.window_h, self.window_w, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, nH * self.window_h, wp, C)
        out = out[:, :h, :w, :].reshape(B, h, w, C)

        if self.shift_w > 0:
            out = torch.roll(out, shifts=self.shift_w, dims=2)

        return out.reshape(B, h * w, C)

    def _build_shift_mask(self, h: int, wp: int, device: torch.device) -> Tensor:
        mask = torch.zeros(1, h, wp, 1, device=device)
        w_slices = [
            slice(0, -self.window_w),
            slice(-self.window_w, -self.shift_w),
            slice(-self.shift_w, None),
        ]
        region_id = 0
        for ws in w_slices:
            mask[:, :, ws, :] = region_id
            region_id += 1

        nH = h // self.window_h
        nW = wp // self.window_w
        mask = mask.reshape(1, nH, self.window_h, nW, self.window_w, 1)
        mask = mask.permute(0, 1, 3, 2, 4, 5).reshape(nH * nW, self.window_h * self.window_w)
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 2):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ExpertBlock(nn.Module):
    """Per-group expert block with shifted windowed attention + MLP."""

    def __init__(self, dim: int, num_heads: int, num_groups: int,
                 window_h: int = 2, window_w: int = 32, shift: bool = False,
                 mlp_ratio: int = 2):
        super().__init__()
        self.num_groups = num_groups
        self.norm1 = nn.LayerNorm(dim)
        self.expert_attns = nn.ModuleList([
            WindowedAttention(dim, num_heads, window_h, window_w, shift=shift)
            for _ in range(num_groups)
        ])
        self.norm2 = nn.LayerNorm(dim)
        self.expert_mlps = nn.ModuleList([
            MLP(dim, mlp_ratio) for _ in range(num_groups)
        ])

    def forward(self, x: Tensor, h: int, w: int, group_ids: Tensor) -> Tensor:
        B = x.shape[0]
        sorted_idx = group_ids.argsort()
        counts = torch.bincount(group_ids, minlength=self.num_groups).tolist()
        x_sorted = x[sorted_idx]

        normed = self.norm1(x_sorted)
        attn_out = torch.empty_like(x_sorted)
        start = 0
        for g in range(self.num_groups):
            end = start + counts[g]
            if start < end:
                if self.training and torch.is_grad_enabled():
                    result = ckpt_util.checkpoint(
                        self.expert_attns[g], normed[start:end], h, w,
                        use_reentrant=True)
                else:
                    result = self.expert_attns[g](normed[start:end], h, w)
                attn_out[start:end] = result.to(attn_out.dtype)
            start = end
        x_sorted = x_sorted + attn_out

        normed = self.norm2(x_sorted)
        mlp_out = torch.empty_like(x_sorted)
        start = 0
        for g in range(self.num_groups):
            end = start + counts[g]
            if start < end:
                if self.training and torch.is_grad_enabled():
                    result = ckpt_util.checkpoint(
                        self.expert_mlps[g], normed[start:end],
                        use_reentrant=True)
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


class LipiMoEEncoder(nn.Module):
    """Lipi v3: HGNetV2 backbone + dual-stream expert attention.

    Local stream: h=2, 2×32 windows — per-character features with vertical detail.
    Wide stream: h=1, 1×128 windows — word-level context after height pooling.
    Aggregation: concat local + wide, project back to dim.
    """

    _OCR_STRIDES = {
        'stem.stem1.conv': (2, 1),              # h/2
        'stem.stem3.conv': (2, 1),              # h/4
        'stages_1.downsample.conv': (2, 2),     # h/8, w/2  ← one width downsample
        'stages_2.downsample.conv': (2, 1),     # h/16
        'stages_3.downsample.conv': (1, 1),     # keep h=2
    }

    def __init__(
        self,
        dim: int = 256,
        backbone: str = 'hgnetv2_b3',
        num_local_blocks: int = 3,
        num_wide_blocks: int = 3,
        local_window_w: int = 16,
        wide_window_w: int = 64,
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
            "num_local_blocks": num_local_blocks,
            "num_wide_blocks": num_wide_blocks,
            "local_window_w": local_window_w,
            "wide_window_w": wide_window_w,
            "mlp_ratio": mlp_ratio,
            "num_groups": num_groups,
            "group_script_vocab_sizes": group_script_vocab_sizes,
            "group_script_names": group_script_names,
        }

        # Pretrained backbone with OCR strides
        self.backbone = timm.create_model(
            f'{backbone}.ssld_stage1_in22k_in1k' if 'ssld' not in backbone else backbone,
            pretrained=True,
            features_only=True,
        )
        for name, mod in self.backbone.named_modules():
            if name in self._OCR_STRIDES:
                mod.stride = self._OCR_STRIDES[name]

        # Remove stage 3's 1024→2048 expansion
        # aggregation is a Sequential with 2 ConvBNAct modules: [2304→1024, 1024→2048]
        # Keep only the first one so output stays at 1024ch
        for block in self.backbone.stages_3.blocks:
            agg_list = list(block.aggregation.children())
            block.aggregation = nn.Sequential(agg_list[0])

        backbone_ch = 1024

        # LID-1
        self.lid1 = LIDCoarse(in_channels=backbone_ch, num_groups=num_groups)

        # Project backbone → expert dim
        self.proj = nn.Linear(backbone_ch, dim)

        # Local expert blocks: h=2, 2×local_window_w windows
        self.local_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_groups=num_groups,
                        window_h=2, window_w=local_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio)
            for i in range(num_local_blocks)
        ])

        # Pool h=2→1 between local and wide streams
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))

        # Wide expert blocks: h=1, 1×wide_window_w windows
        self.wide_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_groups=num_groups,
                        window_h=1, window_w=wide_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio)
            for i in range(num_wide_blocks)
        ])

        # Per-group aggregation: concat local + wide → project
        self.expert_aggregates = nn.ModuleList([
            nn.Linear(dim * 2, dim) for _ in range(num_groups)
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

        x = images.float() / 255.0 if images.dtype == torch.uint8 else images

        # Backbone: (B, 1024, 2, W)
        feats = self.backbone(x)
        backbone_out = feats[-1]
        _, C, h, w = backbone_out.shape

        # LID-1
        group_logits = self.lid1(backbone_out)
        if group_ids is None:
            group_ids = group_logits.argmax(dim=-1)

        # Project: (B, 2W, dim)
        x = backbone_out.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj(x)

        if detach_for_experts:
            x = x.detach()

        # Local expert blocks at h=2
        for block in self.local_blocks:
            x = block(x, h, w, group_ids)

        # Pool h=2→1 for local features: (B, W, dim)
        dim = x.shape[-1]
        local_out = x.reshape(B, h, w, dim).permute(0, 3, 1, 2)  # (B, dim, h, w)
        local_out = self.h_pool(local_out).squeeze(2).permute(0, 2, 1)  # (B, W, dim)

        # Wide expert blocks at h=1
        x_wide = local_out
        for block in self.wide_blocks:
            x_wide = block(x_wide, 1, w, group_ids)

        # Per-group aggregation: concat local + wide, project
        group_counts = torch.bincount(group_ids, minlength=self.num_groups)
        active_groups = group_counts.nonzero(as_tuple=True)[0].tolist()

        combined = torch.cat([local_out, x_wide], dim=-1)  # (B, W, dim*2)
        x = torch.empty_like(local_out)  # (B, W, dim)
        for g in active_groups:
            mask = (group_ids == g)
            x[mask] = self.expert_aggregates[g](combined[mask])

        x = self.norm(x)
        T = x.shape[1]

        # Per-group CTC with LID-2
        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)
        all_script_logits = []

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
