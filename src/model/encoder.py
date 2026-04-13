"""
Lipi v4 MoE Vision Encoder.

Architecture:
    Input: (B, 3, 32, W) — RGB
    -> HGNetV2 backbone (pretrained, 1024ch)
       → (B, 1024, 2, W/2)
    -> Project 1024 → dim, keep h=2
    -> 1 shared global attention block (h=2, cross-frame context)
    -> Frame-level LID-1: per-frame script group classification
    -> Route frames to group expert blocks by group_id
    -> 2 local group expert blocks (h=2, 2×16 windows, 13 experts)
    -> Pool h=2→1
    -> 2 wide group expert blocks (h=1, 1×64 windows, 13 experts)
    -> Per-group aggregation (concat local + wide → dim)
    -> Frame-level LID-2: per-frame script classification (multi-script groups)
    -> Route frames to script expert blocks by script_id
    -> 1 local script expert block (h=1, 1×16 windows, 26 experts)
    -> 1 wide script expert block (h=1, 1×64 windows, 26 experts)
    -> Per-script aggregation (concat local + wide → dim)
    -> Per-script CTC heads (T=W/2)
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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(attn.transpose(1, 2).reshape(B, N, C))


class SharedBlock(nn.Module):
    """Shared (non-expert) transformer block: attention + MLP."""

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

        # Pad width to multiple of window_w
        pad_w = (self.window_w - w % self.window_w) % self.window_w
        if pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w))
        wp = w + pad_w

        # Shift
        if self.shift and self.shift_w > 0:
            x = torch.roll(x, shifts=-self.shift_w, dims=2)

        # Window partition
        nH = h // self.window_h
        nW = wp // self.window_w
        x = x.reshape(B, nH, self.window_h, nW, self.window_w, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * nH * nW, self.window_h * self.window_w, C)

        # Attention
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn_mask = self._get_attn_mask(h, wp, x.device) if self.shift else None
        if attn_mask is not None:
            n_windows = nH * nW
            # Expand to (B*nH*nW, 1, win_size, win_size) for head broadcasting
            attn_mask = attn_mask.unsqueeze(1).repeat(x.shape[0] // n_windows, 1, 1, 1)

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
        out = self.proj(out)

        # Reverse window partition
        out = out.reshape(B, nH, nW, self.window_h, self.window_w, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, h, wp, C)

        # Reverse shift
        if self.shift and self.shift_w > 0:
            out = torch.roll(out, shifts=self.shift_w, dims=2)

        # Remove padding
        if pad_w > 0:
            out = out[:, :, :w, :]

        return out.reshape(B, N, C)

    def _get_attn_mask(self, h: int, wp: int, device) -> Tensor | None:
        if not self.shift:
            return None
        mask = torch.zeros(1, h, wp, device=device)
        mask[:, :, -self.shift_w:] = 1
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
    """Per-expert block with shifted windowed attention + MLP.

    Has N parallel attention+MLP experts. Each sample is routed to
    one expert based on its expert_id.
    """

    def __init__(self, dim: int, num_heads: int, num_experts: int,
                 window_h: int = 2, window_w: int = 16, shift: bool = False,
                 mlp_ratio: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.norm1 = nn.LayerNorm(dim)
        self.expert_attns = nn.ModuleList([
            WindowedAttention(dim, num_heads, window_h, window_w, shift=shift)
            for _ in range(num_experts)
        ])
        self.norm2 = nn.LayerNorm(dim)
        self.expert_mlps = nn.ModuleList([
            MLP(dim, mlp_ratio) for _ in range(num_experts)
        ])


class CTCHead(nn.Module):
    def __init__(self, enc_dim: int, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Linear(enc_dim, vocab_size)
        # Init near-uniform so CTC starts at random baseline, not worse
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(features)


class GroupCTCModule(nn.Module):
    """Per-group CTC heads — one per script in the group."""

    def __init__(self, enc_dim: int, script_vocab_sizes: list[int],
                 script_names: list[str]):
        super().__init__()
        self.n_scripts = len(script_vocab_sizes)
        self.script_names = script_names
        self.max_vocab = max(script_vocab_sizes)

        self.heads = nn.ModuleList([
            CTCHead(enc_dim, vs) for vs in script_vocab_sizes
        ])

    def forward(self, features: Tensor, script_ids: Tensor
                ) -> tuple[Tensor, Tensor]:
        """Route features to per-script CTC heads.

        Args:
            features: (N, T, dim)
            script_ids: (N,) — per-sample script assignment

        Returns:
            logits: (N, T, max_vocab) — zero-padded
            script_ids: (N,) — as provided
        """
        N, T, C = features.shape
        logits = torch.zeros(N, T, self.max_vocab,
                             device=features.device, dtype=features.dtype)
        for s in script_ids.unique().tolist():
            mask = (script_ids == s)
            if s < len(self.heads):
                head_out = self.heads[s](features[mask]).to(logits.dtype)
                logits[mask, :, :head_out.shape[-1]] = head_out
        return logits, script_ids


def _run_expert_block(block, x, expert_id, h, w, use_ckpt):
    """Run one sample through a specific expert in an ExpertBlock."""
    normed = block.norm1(x)
    attn = block.expert_attns[expert_id]
    if use_ckpt:
        attn_out = ckpt_util.checkpoint(attn, normed, h, w, use_reentrant=True)
    else:
        attn_out = attn(normed, h, w)
    x = x + attn_out.to(x.dtype)
    normed = block.norm2(x)
    mlp = block.expert_mlps[expert_id]
    if use_ckpt:
        mlp_out = ckpt_util.checkpoint(mlp, normed, use_reentrant=True)
    else:
        mlp_out = mlp(normed)
    x = x + mlp_out.to(x.dtype)
    return x


class LipiMoEEncoder(nn.Module):
    """Lipi v4: HGNetV2 backbone + LID-1 + group experts + LID-2 + script experts.

    Two-level expert routing:
      1. LID-1 classifies each frame into a script group (13 groups + blank)
      2. Group expert blocks process frames per-group (2 local + 2 wide)
      3. LID-2 classifies each frame into a script within its group
      4. Script expert blocks process frames per-script (1 local + 1 wide)
      5. Per-script CTC heads decode characters

    Single-script groups skip LID-2 (only 1 script, trivially assigned).
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
        num_group_local_blocks: int = 2,
        num_group_wide_blocks: int = 2,
        num_script_local_blocks: int = 1,
        num_script_wide_blocks: int = 1,
        local_window_w: int = 16,
        wide_window_w: int = 64,
        mlp_ratio: int = 2,
        num_groups: int = NUM_GROUPS,
        group_script_vocab_sizes: list[list[int]] | None = None,
        group_script_names: list[list[str]] | None = None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.blank_group_id = num_groups

        if group_script_vocab_sizes is None:
            group_script_vocab_sizes = [[100]] * num_groups
        if group_script_names is None:
            group_script_names = [[f"s{i}" for i in range(len(vs))]
                                  for vs in group_script_vocab_sizes]

        # Build flat script index: (group, local_script) → flat_id
        self.total_scripts = sum(len(vs) for vs in group_script_vocab_sizes)
        self._flat_script_id = {}  # (g, s) → flat
        self._flat_to_group_script = {}  # flat → (g, s)
        self._group_script_counts = [len(vs) for vs in group_script_vocab_sizes]
        flat = 0
        for g, vs in enumerate(group_script_vocab_sizes):
            for s in range(len(vs)):
                self._flat_script_id[(g, s)] = flat
                self._flat_to_group_script[flat] = (g, s)
                flat += 1

        # Which groups are multi-script
        self._multi_script_groups = {
            g for g, vs in enumerate(group_script_vocab_sizes) if len(vs) > 1
        }

        self.config = {
            "dim": dim, "backbone": backbone,
            "num_group_local_blocks": num_group_local_blocks,
            "num_group_wide_blocks": num_group_wide_blocks,
            "num_script_local_blocks": num_script_local_blocks,
            "num_script_wide_blocks": num_script_wide_blocks,
            "local_window_w": local_window_w,
            "wide_window_w": wide_window_w,
            "mlp_ratio": mlp_ratio, "num_groups": num_groups,
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

        # Shared global attention block
        self.shared_attn = SharedBlock(dim=dim, num_heads=dim // 64, mlp_ratio=mlp_ratio)

        # LID-1: per-frame group classification
        self.group_h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.group_head = nn.Linear(dim, num_groups + 1)  # +1 for blank

        # Group expert blocks (routed by group_id, 13 experts)
        self.group_local_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_experts=num_groups,
                        window_h=2, window_w=local_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio)
            for i in range(num_group_local_blocks)
        ])
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))  # pool h=2→1
        self.group_wide_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_experts=num_groups,
                        window_h=1, window_w=wide_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio)
            for i in range(num_group_wide_blocks)
        ])

        # Group aggregation: concat local + wide → dim
        # Init as average of local+wide (near-identity)
        self.group_aggregates = nn.ModuleList()
        for _ in range(num_groups):
            agg = nn.Linear(dim * 2, dim)
            nn.init.zeros_(agg.bias)
            with torch.no_grad():
                agg.weight.zero_()
                agg.weight[:, :dim] = 0.5 * torch.eye(dim)
                agg.weight[:, dim:] = 0.5 * torch.eye(dim)
            self.group_aggregates.append(agg)

        # LID-2: per-frame script classification within multi-script groups
        # One head per multi-script group
        self.lid2_heads = nn.ModuleDict()
        for g in range(num_groups):
            n_scripts = len(group_script_vocab_sizes[g])
            if n_scripts > 1:
                self.lid2_heads[str(g)] = nn.Linear(dim, n_scripts)

        # Script expert blocks (routed by flat script_id, 26 experts)
        # Initialize output projections near-zero so residual connections
        # pass features through initially (prevents randomly initialized
        # script experts from destroying group-expert features)
        self.script_local_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64,
                        num_experts=self.total_scripts,
                        window_h=1, window_w=local_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio)
            for i in range(num_script_local_blocks)
        ])
        self.script_wide_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64,
                        num_experts=self.total_scripts,
                        window_h=1, window_w=wide_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio)
            for i in range(num_script_wide_blocks)
        ])
        for block_list in [self.script_local_blocks, self.script_wide_blocks]:
            for block in block_list:
                for attn in block.expert_attns:
                    nn.init.zeros_(attn.proj.weight)
                    nn.init.zeros_(attn.proj.bias)
                for mlp in block.expert_mlps:
                    nn.init.zeros_(mlp.fc2.weight)
                    nn.init.zeros_(mlp.fc2.bias)

        # Script aggregation: concat local + wide → dim
        # Initialize as near-identity (average of local+wide) so untrained
        # script experts pass features through without destroying them
        self.script_aggregates = nn.ModuleList()
        for _ in range(self.total_scripts):
            agg = nn.Linear(dim * 2, dim)
            nn.init.zeros_(agg.bias)
            with torch.no_grad():
                agg.weight.zero_()
                agg.weight[:, :dim] = 0.5 * torch.eye(dim)
                agg.weight[:, dim:] = 0.5 * torch.eye(dim)
            self.script_aggregates.append(agg)

        # Output
        self.enc_out_dim = dim
        self.norm = nn.LayerNorm(dim)

        # Per-script CTC heads
        self.ctc_modules = nn.ModuleList([
            GroupCTCModule(
                enc_dim=dim,
                script_vocab_sizes=group_script_vocab_sizes[g],
                script_names=group_script_names[g],
            )
            for g in range(num_groups)
        ])

    def _get_flat_script_ids(self, group_ids, script_ids):
        """Convert (group_id, local_script_id) pairs to flat script indices.
        Blank/whitespace frames (group_id == blank_group_id) get flat_id = -1.
        """
        flat = torch.full_like(group_ids, -1)
        for (g, s), f in self._flat_script_id.items():
            mask = (group_ids == g) & (script_ids == s)
            flat[mask] = f
        return flat

    def forward(
        self,
        images: Tensor,
        group_ids: Tensor | None = None,
        script_ids: Tensor | None = None,
        detach_for_experts: bool = False,
    ) -> dict:
        B = images.shape[0]
        use_ckpt = self.training and torch.is_grad_enabled()

        x = images.float() / 255.0 if images.dtype == torch.uint8 else images

        # Backbone: (B, 1024, 2, W/2)
        feats = self.backbone(x)
        backbone_out = feats[-1]
        _, C, h, w = backbone_out.shape

        # Project to dim, keep h=2: (B, 2*W/2, dim)
        x = backbone_out.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj(x)

        # Shared global attention
        x = self.shared_attn(x)

        # LID-1: per-frame group prediction
        d = x.shape[-1]
        x_for_group = x.reshape(B, h, w, d).permute(0, 3, 1, 2)
        x_for_group = self.group_h_pool(x_for_group).squeeze(2).permute(0, 2, 1)
        group_logits = self.group_head(x_for_group)  # (B, W/2, num_groups+1)

        # Determine per-frame group assignments
        if group_ids is not None:
            if group_ids.dim() == 1:
                frame_groups = group_ids.unsqueeze(1).expand(B, w)
            else:
                frame_groups = group_ids
        else:
            frame_groups = group_logits.argmax(dim=-1)

        if detach_for_experts:
            x = x.detach()

        # =====================================================================
        # STAGE 1: Group expert blocks (2 local + 2 wide, routed by group_id)
        # =====================================================================

        # Separate single-group (batchable) from mixed-group (per-image)
        fg_filled = frame_groups.clone()
        fg_filled[fg_filled == self.blank_group_id] = -1
        primary = fg_filled.max(dim=1).values
        for b_idx in range(B):
            fg_filled[b_idx][fg_filled[b_idx] == -1] = primary[b_idx]
        is_single_group = (fg_filled == fg_filled[:, :1]).all(dim=1)

        # Output after group experts: (B, W/2, dim)
        x_after_group = torch.zeros(B, w, d, device=x.device, dtype=x.dtype)

        # Batched path for single-group images
        if is_single_group.any():
            single_idx = is_single_group.nonzero(as_tuple=True)[0]
            single_groups = primary[single_idx]
            for g in single_groups.unique().tolist():
                if g < 0 or g == self.blank_group_id:
                    continue
                g_mask = (single_groups == g)
                batch_idx = single_idx[g_mask]
                x_batch = x[batch_idx]

                # Local group expert blocks (h=2)
                for block in self.group_local_blocks:
                    x_batch = _run_expert_block(block, x_batch, g, h, w, use_ckpt)

                # Pool h=2→1
                N_g = x_batch.shape[0]
                local_batch = x_batch.reshape(N_g, h, w, d).permute(0, 3, 1, 2)
                local_batch = self.h_pool(local_batch).squeeze(2).permute(0, 2, 1)

                # Wide group expert blocks (h=1)
                wide_batch = local_batch
                for block in self.group_wide_blocks:
                    wide_batch = _run_expert_block(block, wide_batch, g, 1, w, use_ckpt)

                # Group aggregation
                comb = torch.cat([local_batch, wide_batch], dim=-1)
                x_after_group[batch_idx] = self.group_aggregates[g](comb)

        # Per-image path for mixed-group images
        for b_idx in (~is_single_group).nonzero(as_tuple=True)[0]:
            fg = frame_groups[b_idx]
            for g in fg.unique().tolist():
                if g == self.blank_group_id:
                    continue
                seg_mask = (fg == g)
                seg_len = seg_mask.sum().item()
                if seg_len == 0:
                    continue

                x_2d = x[b_idx].reshape(h, w, d)
                x_seg = x_2d[:, seg_mask, :].reshape(1, h * seg_len, d)

                for block in self.group_local_blocks:
                    x_seg = _run_expert_block(block, x_seg, g, h, seg_len, use_ckpt)

                local_seg = x_seg.reshape(1, h, seg_len, d).permute(0, 3, 1, 2)
                local_seg = self.h_pool(local_seg).squeeze(2).permute(0, 2, 1)

                wide_seg = local_seg
                for block in self.group_wide_blocks:
                    wide_seg = _run_expert_block(block, wide_seg, g, 1, seg_len, use_ckpt)

                comb = torch.cat([local_seg, wide_seg], dim=-1)
                x_after_group[b_idx, seg_mask] = self.group_aggregates[g](comb.squeeze(0))

        # =====================================================================
        # LID-2: per-frame script classification
        # =====================================================================

        # Collect LID-2 logits for multi-script groups
        lid2_logits_per_group = {}  # g → (B, T, n_scripts)
        for g_str, head in self.lid2_heads.items():
            g = int(g_str)
            lid2_logits_per_group[g] = head(x_after_group)  # (B, T, n_scripts)

        # Determine per-frame script assignments
        # frame_scripts: (B, T) — local script_id within each frame's group
        frame_scripts = torch.zeros(B, w, dtype=torch.long, device=x.device)

        if script_ids is not None:
            # Training: ground truth
            if script_ids.dim() == 1:
                frame_scripts = script_ids.unsqueeze(1).expand(B, w)
            else:
                frame_scripts = script_ids
        else:
            # Inference: predict from LID-2 for multi-script groups
            for g, lid2_log in lid2_logits_per_group.items():
                g_mask = (frame_groups == g)
                if g_mask.any():
                    pred = lid2_log.argmax(dim=-1)  # (B, T)
                    frame_scripts[g_mask] = pred[g_mask]

        # Convert to flat script IDs for script expert routing
        flat_scripts = self._get_flat_script_ids(frame_groups, frame_scripts)

        # =====================================================================
        # STAGE 2: Script expert blocks (1 local + 1 wide, routed by script_id)
        # =====================================================================

        x_after_script = torch.zeros(B, w, d, device=x.device, dtype=x.dtype)

        # Determine single-script images (all non-blank frames same script)
        fs_filled = flat_scripts.clone()
        blank_mask = (frame_groups == self.blank_group_id)
        fs_filled[blank_mask] = -1
        primary_script = fs_filled.max(dim=1).values
        for b_idx in range(B):
            fs_filled[b_idx][fs_filled[b_idx] == -1] = primary_script[b_idx]
        is_single_script = (fs_filled == fs_filled[:, :1]).all(dim=1)

        # Batched path for single-script images
        if is_single_script.any():
            single_idx = is_single_script.nonzero(as_tuple=True)[0]
            single_scripts = primary_script[single_idx]
            for s in single_scripts.unique().tolist():
                if s < 0:
                    continue
                s_mask = (single_scripts == s)
                batch_idx = single_idx[s_mask]
                x_batch = x_after_group[batch_idx]

                # Local script expert blocks (h=1)
                local_batch = x_batch
                for block in self.script_local_blocks:
                    local_batch = _run_expert_block(block, local_batch, s, 1, w, use_ckpt)

                # Wide script expert blocks (h=1)
                wide_batch = x_batch
                for block in self.script_wide_blocks:
                    wide_batch = _run_expert_block(block, wide_batch, s, 1, w, use_ckpt)

                # Script aggregation
                comb = torch.cat([local_batch, wide_batch], dim=-1)
                x_after_script[batch_idx] = self.script_aggregates[s](comb)

        # Per-image path for mixed-script images
        for b_idx in (~is_single_script).nonzero(as_tuple=True)[0]:
            fs = flat_scripts[b_idx]
            for s in fs.unique().tolist():
                if s < 0 or frame_groups[b_idx][flat_scripts[b_idx] == s][0] == self.blank_group_id:
                    continue
                seg_mask = (fs == s)
                seg_len = seg_mask.sum().item()
                if seg_len == 0:
                    continue

                x_seg = x_after_group[b_idx, seg_mask].unsqueeze(0)  # (1, seg_len, dim)

                local_seg = x_seg
                for block in self.script_local_blocks:
                    local_seg = _run_expert_block(block, local_seg, s, 1, seg_len, use_ckpt)

                wide_seg = x_seg
                for block in self.script_wide_blocks:
                    wide_seg = _run_expert_block(block, wide_seg, s, 1, seg_len, use_ckpt)

                comb = torch.cat([local_seg, wide_seg], dim=-1)
                x_after_script[b_idx, seg_mask] = self.script_aggregates[s](comb.squeeze(0))

        # =====================================================================
        # CTC heads
        # =====================================================================

        x = self.norm(x_after_script)
        T = x.shape[1]

        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)

        group_counts = torch.bincount(frame_groups.reshape(-1),
                                      minlength=self.num_groups + 1)
        active_groups = [g for g in group_counts.nonzero(as_tuple=True)[0].tolist()
                         if g != self.blank_group_id]

        for g in active_groups:
            frame_mask = (frame_groups == g)
            sample_mask = frame_mask.any(dim=1)
            if not sample_mask.any():
                continue

            # Get per-frame script IDs for this group
            g_frame_scripts = frame_scripts.clone()

            all_single = (frame_mask[sample_mask].all(dim=1)).all().item()
            if all_single:
                # All frames in these samples belong to this group
                # Use mode of script_ids as per-sample script
                sample_scripts = torch.zeros(sample_mask.sum(), dtype=torch.long,
                                             device=x.device)
                for i, b_idx in enumerate(sample_mask.nonzero(as_tuple=True)[0]):
                    s_ids = g_frame_scripts[b_idx][frame_mask[b_idx]]
                    sample_scripts[i] = s_ids[0]  # single-script: all same

                g_logits, _ = self.ctc_modules[g](x[sample_mask],
                                                   script_ids=sample_scripts)
                for i, b_idx in enumerate(sample_mask.nonzero(as_tuple=True)[0]):
                    f_mask = frame_mask[b_idx]
                    logits[b_idx, f_mask, :g_logits.shape[-1]] = \
                        g_logits[i, f_mask].to(logits.dtype)
            else:
                for b_idx in sample_mask.nonzero(as_tuple=True)[0]:
                    f_mask = frame_mask[b_idx]
                    seg_len = f_mask.sum().item()
                    if seg_len == 0:
                        continue
                    seg_features = x[b_idx, f_mask].unsqueeze(0)
                    s_id = g_frame_scripts[b_idx][f_mask][0]
                    seg_script = s_id.unsqueeze(0)
                    seg_logits, _ = self.ctc_modules[g](seg_features,
                                                         script_ids=seg_script)
                    logits[b_idx, f_mask, :seg_logits.shape[-1]] = \
                        seg_logits.squeeze(0).to(logits.dtype)

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return {
            "logits": logits,
            "lengths": lengths,
            "group_logits": group_logits,
            "group_ids": frame_groups,
            "lid2_logits_per_group": lid2_logits_per_group,
            "frame_scripts": frame_scripts,
            "flat_scripts": flat_scripts,
            "script_logits_per_group": [],  # backward compat — remove later
        }
