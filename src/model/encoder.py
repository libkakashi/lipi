"""
Lipi v4 MoE Vision Encoder.

Architecture:
    Input: (B, 3, 32, W) — RGB
    -> HGNetV2 backbone (pretrained, 1024ch)
       → (B, 1024, 2, W/2)
    -> Project 1024 → dim, keep h=2
    -> 1 shared global attention block (h=2, cross-frame context)
    -> Frame-level group CE: per-frame script group classification
    -> Route frames to expert blocks by group
    -> Local expert blocks (h=2, 2×16 windows)
    -> Pool h=2→1
    -> Wide expert blocks (h=1, 1×64 windows)
    -> Per-group aggregation
    -> LID-2 + Per-script CTC heads (T=W/2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util
from torch import Tensor

import timm

from src.model.lid import NUM_GROUPS


class GlobalAttention(nn.Module):
    """Standard multi-head self-attention on flattened 2D sequences."""

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


class SharedBlock(nn.Module):
    """Shared (non-expert) attention + MLP block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GlobalAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class WindowedAttention(nn.Module):
    """Shifted window attention on flattened 2D sequences (Swin-style)."""

    def __init__(self, dim: int, num_heads: int,
                 window_h: int = 2, window_w: int = 16, shift: bool = False):
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
    """Per-group expert block with shifted windowed attention + MLP.

    Routes frames by group_ids — supports per-frame routing where
    different frames in the same image can go to different experts.
    """

    def __init__(self, dim: int, num_heads: int, num_groups: int,
                 window_h: int = 2, window_w: int = 16, shift: bool = False,
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
        """group_ids: (B,) — per-sample group routing."""
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
    """Lipi v4: HGNetV2 backbone + per-frame group CE + dual-stream experts.

    Per-frame group cross-entropy replaces per-image LID-1 classification.
    Each frame gets a group label — direct supervision at every position.
    One shared global attention block provides cross-frame context.
    Expert routing is per-sample (training) with future support for
    per-frame routing (mixed-script inference).
    """

    _OCR_STRIDES = {
        'stem.stem1.conv': (2, 1),
        'stem.stem3.conv': (2, 1),
        'stages_1.downsample.conv': (2, 2),     # one width downsample
        'stages_2.downsample.conv': (2, 1),
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
        for block in self.backbone.stages_3.blocks:
            agg_list = list(block.aggregation.children())
            block.aggregation = nn.Sequential(agg_list[0])

        backbone_ch = 1024

        # Project backbone → dim (keep h=2)
        self.proj = nn.Linear(backbone_ch, dim)

        # Shared global attention block (h=2, sees all frames for group context)
        self.shared_attn = SharedBlock(dim=dim, num_heads=dim // 64, mlp_ratio=mlp_ratio)

        # Frame-level group classifier: predicts group per frame
        # Pool h=2→1, then Linear to num_groups
        self.group_h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.group_head = nn.Linear(dim, num_groups)

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

        # Character CTC heads
        if group_script_names is None:
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

        # Backbone: (B, 1024, 2, W/2)
        feats = self.backbone(x)
        backbone_out = feats[-1]
        _, C, h, w = backbone_out.shape

        # Project to dim, keep h=2: (B, 2*W/2, dim)
        x = backbone_out.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj(x)

        # Shared global attention at h=2 (cross-frame context for group CTC)
        x = self.shared_attn(x)

        # Frame-level group prediction: pool h→1, classify each frame
        d = x.shape[-1]
        x_for_group = x.reshape(B, h, w, d).permute(0, 3, 1, 2)  # (B, dim, h, w)
        x_for_group = self.group_h_pool(x_for_group).squeeze(2).permute(0, 2, 1)  # (B, W/2, dim)
        group_logits = self.group_head(x_for_group)  # (B, W/2, num_groups)

        # Determine per-frame group assignments
        if group_ids is not None:
            # Training: ground truth — broadcast per-image to per-frame
            if group_ids.dim() == 1:
                frame_groups = group_ids.unsqueeze(1).expand(B, w)  # (B, W/2)
            else:
                frame_groups = group_ids  # already per-frame (B, W/2)
        else:
            # Inference: use per-frame predictions
            frame_groups = group_logits.argmax(dim=-1)  # (B, W/2)

        if detach_for_experts:
            x = x.detach()

        # Separate single-script (batchable) from mixed-script (per-image)
        is_single = (frame_groups == frame_groups[:, :1]).all(dim=1)  # (B,)
        x_out = torch.zeros(B, w, d, device=x.device, dtype=x.dtype)

        # Batched path: group single-script images by their group
        if is_single.any():
            single_idx = is_single.nonzero(as_tuple=True)[0]
            single_groups = frame_groups[single_idx, 0]  # (N,) each image's group
            for g in single_groups.unique().tolist():
                g_mask = (single_groups == g)
                batch_idx = single_idx[g_mask]  # indices into original batch
                x_batch = x[batch_idx]  # (N_g, h*w, dim)

                # Local expert blocks
                for block in self.local_blocks:
                    normed = block.norm1(x_batch)
                    attn_g = block.expert_attns[g]
                    if self.training and torch.is_grad_enabled():
                        attn_out = ckpt_util.checkpoint(
                            attn_g, normed, h, w, use_reentrant=True)
                    else:
                        attn_out = attn_g(normed, h, w)
                    x_batch = x_batch + attn_out.to(x_batch.dtype)
                    normed = block.norm2(x_batch)
                    mlp_g = block.expert_mlps[g]
                    if self.training and torch.is_grad_enabled():
                        mlp_out = ckpt_util.checkpoint(mlp_g, normed, use_reentrant=True)
                    else:
                        mlp_out = mlp_g(normed)
                    x_batch = x_batch + mlp_out.to(x_batch.dtype)

                # Pool h→1
                N_g = x_batch.shape[0]
                local_batch = x_batch.reshape(N_g, h, w, d).permute(0, 3, 1, 2)
                local_batch = self.h_pool(local_batch).squeeze(2).permute(0, 2, 1)

                # Wide expert blocks
                wide_batch = local_batch
                for block in self.wide_blocks:
                    normed = block.norm1(wide_batch)
                    attn_g = block.expert_attns[g]
                    if self.training and torch.is_grad_enabled():
                        attn_out = ckpt_util.checkpoint(
                            attn_g, normed, 1, w, use_reentrant=True)
                    else:
                        attn_out = attn_g(normed, 1, w)
                    wide_batch = wide_batch + attn_out.to(wide_batch.dtype)
                    normed = block.norm2(wide_batch)
                    mlp_g = block.expert_mlps[g]
                    if self.training and torch.is_grad_enabled():
                        mlp_out = ckpt_util.checkpoint(mlp_g, normed, use_reentrant=True)
                    else:
                        mlp_out = mlp_g(normed)
                    wide_batch = wide_batch + mlp_out.to(wide_batch.dtype)

                # Aggregate
                comb_batch = torch.cat([local_batch, wide_batch], dim=-1)
                x_out[batch_idx] = self.expert_aggregates[g](comb_batch)

        # Per-image path: mixed-script images
        for b in (~is_single).nonzero(as_tuple=True)[0]:
            fg = frame_groups[b]
            for g in fg.unique().tolist():
                seg_mask = (fg == g)
                seg_len = seg_mask.sum().item()
                if seg_len == 0:
                    continue

                x_2d = x[b].reshape(h, w, d)
                seg_2d = x_2d[:, seg_mask, :]
                x_seg = seg_2d.reshape(1, h * seg_len, d)

                for block in self.local_blocks:
                    normed = block.norm1(x_seg)
                    attn_g = block.expert_attns[g]
                    if self.training and torch.is_grad_enabled():
                        attn_out = ckpt_util.checkpoint(
                            attn_g, normed, h, seg_len, use_reentrant=True)
                    else:
                        attn_out = attn_g(normed, h, seg_len)
                    x_seg = x_seg + attn_out.to(x_seg.dtype)
                    normed = block.norm2(x_seg)
                    mlp_g = block.expert_mlps[g]
                    if self.training and torch.is_grad_enabled():
                        mlp_out = ckpt_util.checkpoint(mlp_g, normed, use_reentrant=True)
                    else:
                        mlp_out = mlp_g(normed)
                    x_seg = x_seg + mlp_out.to(x_seg.dtype)

                local_seg = x_seg.reshape(1, h, seg_len, d).permute(0, 3, 1, 2)
                local_seg = self.h_pool(local_seg).squeeze(2).permute(0, 2, 1)

                wide_seg = local_seg
                for block in self.wide_blocks:
                    normed = block.norm1(wide_seg)
                    attn_g = block.expert_attns[g]
                    if self.training and torch.is_grad_enabled():
                        attn_out = ckpt_util.checkpoint(
                            attn_g, normed, 1, seg_len, use_reentrant=True)
                    else:
                        attn_out = attn_g(normed, 1, seg_len)
                    wide_seg = wide_seg + attn_out.to(wide_seg.dtype)
                    normed = block.norm2(wide_seg)
                    mlp_g = block.expert_mlps[g]
                    if self.training and torch.is_grad_enabled():
                        mlp_out = ckpt_util.checkpoint(mlp_g, normed, use_reentrant=True)
                    else:
                        mlp_out = mlp_g(normed)
                    wide_seg = wide_seg + mlp_out.to(wide_seg.dtype)

                comb_seg = torch.cat([local_seg, wide_seg], dim=-1)
                agg_seg = self.expert_aggregates[g](comb_seg.squeeze(0))
                x_out[b, seg_mask] = agg_seg

        x = self.norm(x_out)
        T = x.shape[1]

        # Per-frame CTC routing: each frame's logits come from its group's CTC head
        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)
        all_script_logits = []

        group_counts = torch.bincount(frame_groups.reshape(-1), minlength=self.num_groups)
        active_groups = group_counts.nonzero(as_tuple=True)[0].tolist()

        for g in active_groups:
            frame_mask = (frame_groups == g)  # (B, T)
            sample_mask = frame_mask.any(dim=1)  # (B,)
            if not sample_mask.any():
                continue

            # Check if all samples with this group are fully this group (single-script)
            all_single = (frame_mask[sample_mask].all(dim=1)).all().item()

            if all_single:
                # Fast path: pass full sequences (all frames belong to this group)
                local_script_ids = None
                if script_ids is not None:
                    local_script_ids = script_ids[sample_mask]
                g_logits, g_script_logits, _ = self.ctc_modules[g](
                    x[sample_mask], script_ids=local_script_ids)
                for i, b_idx in enumerate(sample_mask.nonzero(as_tuple=True)[0]):
                    f_mask = frame_mask[b_idx]
                    logits[b_idx, f_mask, :g_logits.shape[-1]] = g_logits[i, f_mask].to(logits.dtype)
                if g_script_logits is not None:
                    all_script_logits.append((g, g_script_logits, sample_mask))
            else:
                # Mixed path: extract each sample's segment for this group
                for b_idx in sample_mask.nonzero(as_tuple=True)[0]:
                    f_mask = frame_mask[b_idx]  # (T,)
                    seg_len = f_mask.sum().item()
                    if seg_len == 0:
                        continue
                    seg_features = x[b_idx, f_mask].unsqueeze(0)  # (1, seg_len, dim)
                    local_sid = script_ids[b_idx:b_idx+1] if script_ids is not None else None
                    seg_logits, seg_script_logits, _ = self.ctc_modules[g](
                        seg_features, script_ids=local_sid)
                    logits[b_idx, f_mask, :seg_logits.shape[-1]] = seg_logits.squeeze(0).to(logits.dtype)
                    if seg_script_logits is not None:
                        seg_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
                        seg_mask[b_idx] = True
                        all_script_logits.append((g, seg_script_logits, seg_mask))

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return {
            "logits": logits,
            "lengths": lengths,
            "group_logits": group_logits,  # (B, W/2, num_groups) — frame-level
            "group_ids": frame_groups,    # (B, T) per-frame group assignments
            "script_logits_per_group": all_script_logits,
        }
