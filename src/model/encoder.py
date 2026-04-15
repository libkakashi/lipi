"""
Lipi v5 MoE Vision Encoder (from scratch — no pretrained backbone).

Architecture:
    Input: (B, 3, 32, W) — RGB
    -> ConvStem: two plain strided convs (no ResBlocks, small RF ~5px)
       → (B, 128, 8, W/2)
    -> Shared SWA-A: 2× windowed-attention blocks at (h=8, w=W/2)
       (window 8×8 covers full vertical extent in a single window,
        so downstream h>1 wouldn't add new vertical info)
    -> Pool h=8→1, proj 128→dim
    -> Shared SWA-B: 2× windowed-attention blocks at (h=1, w=W/2, dim)
    -> Frame-level LID-1: per-frame script group classification
    -> Route frames to group expert blocks by group_id
    -> 1 local group expert block  (h=1, window 1×16, 13 experts)
    -> 1 wide  group expert block  (h=1, window 1×64, 13 experts)
    -> Per-group aggregation (concat local + wide → dim)
    -> Frame-level LID-2: per-frame script classification (multi-script groups)
    -> Route frames to script expert blocks by script_id
    -> 1 local script expert block (h=1, 1×16 windows, 26 experts)
    -> 1 wide  script expert block (h=1, 1×64 windows, 26 experts)
    -> Per-script aggregation (concat local + wide → dim)
    -> Per-script CTC heads (T=W/2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util
from torch import Tensor

from src.model.lid import NUM_GROUPS


class DropPath(nn.Module):
    """Stochastic depth per sample (drop whole residual branches).

    Improves generalization. Standard in modern ViT/Swin. Zero cost at
    inference.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    """Learned per-channel scalar applied to a residual branch.

    Init near zero so the block starts near-identity. Stabilizes deep
    training; standard in CaiT/ConvNeXt.
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Cast gamma to x's dtype so autocast (bf16/fp16) doesn't upcast
        # the residual branch to fp32 at the multiply.
        return self.gamma.to(x.dtype) * x


class WindowedAttention(nn.Module):
    """Shifted window attention on flattened 2D sequences (Swin-style).

    Includes a learned relative position bias (per-head, per relative
    (Δh, Δw) offset within the window) and optional QK-norm on the
    query/key vectors before the dot product.
    """

    def __init__(self, dim: int, num_heads: int,
                 window_h: int = 2, window_w: int = 16, shift: bool = False,
                 qk_norm: bool = True):
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

        # QK-norm (applied per-head before the attention dot product)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Learned relative position bias table. Size = (2Wh-1)(2Ww-1) × heads.
        n_rel = (2 * window_h - 1) * (2 * window_w - 1)
        self.rel_pos_bias = nn.Parameter(torch.zeros(n_rel, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Index table (Wh*Ww, Wh*Ww) → offset into rel_pos_bias
        coords_h = torch.arange(window_h)
        coords_w = torch.arange(window_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flat = coords.flatten(1)  # (2, Wh*Ww)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_h - 1
        rel[:, :, 1] += window_w - 1
        rel[:, :, 0] *= 2 * window_w - 1
        rel_index = rel.sum(-1)  # (N, N)
        self.register_buffer("rel_pos_index", rel_index, persistent=False)

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
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Relative position bias: (win_size, win_size, heads) → (heads, win, win)
        win_size = self.window_h * self.window_w
        rel_bias = self.rel_pos_bias[self.rel_pos_index.view(-1)]
        rel_bias = rel_bias.view(win_size, win_size, -1).permute(2, 0, 1)
        # Broadcast to (1, heads, win, win) for SDPA
        rel_bias = rel_bias.unsqueeze(0).to(q.dtype)

        attn_mask = self._get_attn_mask(h, wp, x.device) if self.shift else None
        if attn_mask is not None:
            n_windows = nH * nW
            # Expand to (B*nH*nW, 1, win_size, win_size) for head broadcasting
            attn_mask = attn_mask.unsqueeze(1).repeat(x.shape[0] // n_windows, 1, 1, 1)
            attn_bias = attn_mask.to(rel_bias.dtype) + rel_bias
        else:
            attn_bias = rel_bias

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_bias)
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

    LayerScale + DropPath are applied per-sample on the residual branches
    (shared across experts — they affect the residual, not the expert op).
    """

    def __init__(self, dim: int, num_heads: int, num_experts: int,
                 window_h: int = 2, window_w: int = 16, shift: bool = False,
                 mlp_ratio: int = 2, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-4):
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
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path)


class ConvStem(nn.Module):
    """Small plain-conv stem with two strided convs. No ResBlocks.

    (B, 3, 32, W) → (B, out_ch, 8, W/2). Kernels are asymmetric (3 high,
    5 wide) — horizontal character strokes are wider than the vertical
    downsample budget, so a wider kernel captures more of a stroke in one
    shot. Receptive field after the stem:
        height: 7 px (as before)
        width:  ~9 px (slightly wider than the previous 5 px)
    Still an order of magnitude smaller than HGNet's ~100-200 px.
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 64, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=(3, 5), stride=(2, 1),
                      padding=(1, 2), bias=False),
            nn.GroupNorm(1, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=(3, 5), stride=(2, 2),
                      padding=(1, 2), bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SWABlock(nn.Module):
    """Windowed attention + MLP block (non-expert version of ExpertBlock).

    Used in the shared stem-post stages where all frames go through the
    same weights.
    """

    def __init__(self, dim: int, num_heads: int,
                 window_h: int, window_w: int, shift: bool,
                 mlp_ratio: int = 2, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(
            dim=dim, num_heads=num_heads,
            window_h=window_h, window_w=window_w, shift=shift)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), h, w)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


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


def _collect_segments(
    x: Tensor,
    mask: Tensor,
) -> tuple[Tensor | None, list[tuple[int, int]] | None]:
    """Collect per-sample frames matching `mask` into a padded batch.

    Args:
        x:    (B, T, d)
        mask: (B, T) boolean — which frames belong to this expert

    Returns:
        batch_x:   (N, max_seg_len, d)  — N = # samples with any frames in
                   this expert. Padded with zeros.
        batch_info: list of (b_idx, seg_len) — for scatter_segments.
        Both None if no samples have any frames in this expert.
    """
    B = x.shape[0]
    seg_lens = mask.sum(dim=1)  # (B,) — per-sample frame count
    has_g = seg_lens > 0
    b_indices = has_g.nonzero(as_tuple=True)[0]
    if b_indices.numel() == 0:
        return None, None

    lens_list = seg_lens[b_indices].tolist()
    b_list = b_indices.tolist()
    max_len = max(lens_list)
    N = len(b_list)
    d = x.shape[-1]

    batch_x = torch.zeros(N, max_len, d, device=x.device, dtype=x.dtype)
    for i, (b, sl) in enumerate(zip(b_list, lens_list)):
        batch_x[i, :sl] = x[b][mask[b]]

    return batch_x, list(zip(b_list, lens_list))


def _scatter_segments(
    x_out: Tensor,
    batch_out: Tensor,
    mask: Tensor,
    batch_info: list[tuple[int, int]],
) -> None:
    """Scatter packed batch results back to per-sample positions."""
    for i, (b, sl) in enumerate(batch_info):
        x_out[b, mask[b]] = batch_out[i, :sl]


def _run_expert_block(block, x, expert_id, h, w, use_ckpt):
    """Run one sample through a specific expert in an ExpertBlock.

    Applies LayerScale and DropPath on both residual branches (same as a
    standard modern transformer block, but the attn/mlp ops themselves are
    expert-specific).
    """
    normed = block.norm1(x)
    attn = block.expert_attns[expert_id]
    if use_ckpt:
        attn_out = ckpt_util.checkpoint(attn, normed, h, w, use_reentrant=True)
    else:
        attn_out = attn(normed, h, w)
    x = x + block.drop_path(block.ls1(attn_out.to(x.dtype)))
    normed = block.norm2(x)
    mlp = block.expert_mlps[expert_id]
    if use_ckpt:
        mlp_out = ckpt_util.checkpoint(mlp, normed, use_reentrant=True)
    else:
        mlp_out = mlp(normed)
    x = x + block.drop_path(block.ls2(mlp_out.to(x.dtype)))
    return x


class LipiMoEEncoder(nn.Module):
    """Lipi v5: ConvStem + shared SWA + group experts + LID-2 + script experts.

    Trained from scratch (no pretrained backbone). Small-RF stem keeps
    boundary contamination minimal before attention layers take over.

    Two-level expert routing:
      1. LID-1 classifies each frame into a script group (13 groups + blank)
      2. Group expert blocks process frames per-group (2 local + 2 wide)
      3. LID-2 classifies each frame into a script within its group
      4. Script expert blocks process frames per-script (1 local + 1 wide)
      5. Per-script CTC heads decode characters

    Single-script groups skip LID-2 (only 1 script, trivially assigned).
    """

    def __init__(
        self,
        dim: int = 256,
        stem_mid_ch: int = 64,
        stem_out_ch: int = 128,
        num_shared_a_blocks: int = 2,
        num_shared_b_blocks: int = 2,
        num_group_local_blocks: int = 1,
        num_group_wide_blocks: int = 1,
        num_script_local_blocks: int = 1,
        num_script_wide_blocks: int = 1,
        local_window_w: int = 16,
        wide_window_w: int = 64,
        mlp_ratio: int = 2,
        drop_path_rate: float = 0.1,
        # LayerScale init=1.0 is a no-op (identity). Reduce (e.g. 1e-2)
        # only if you see training instability; on top of identity-init
        # experts, small values scale expert gradients by the same factor
        # and can starve MoE experts that already see only 1/N of data.
        layer_scale_init: float = 1.0,
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
            "dim": dim,
            "stem_mid_ch": stem_mid_ch, "stem_out_ch": stem_out_ch,
            "num_shared_a_blocks": num_shared_a_blocks,
            "num_shared_b_blocks": num_shared_b_blocks,
            "num_group_local_blocks": num_group_local_blocks,
            "num_group_wide_blocks": num_group_wide_blocks,
            "num_script_local_blocks": num_script_local_blocks,
            "num_script_wide_blocks": num_script_wide_blocks,
            "local_window_w": local_window_w,
            "wide_window_w": wide_window_w,
            "mlp_ratio": mlp_ratio,
            "drop_path_rate": drop_path_rate,
            "layer_scale_init": layer_scale_init,
            "num_groups": num_groups,
            "group_script_vocab_sizes": group_script_vocab_sizes,
            "group_script_names": group_script_names,
        }

        # Drop-path schedule: linearly increase from 0 → drop_path_rate
        # across all residual stages along a sample's path.
        # Stages (parallel pairs count as one): shared_a, shared_b,
        # group (local/wide parallel), script (local/wide parallel).
        n_stages = (num_shared_a_blocks + num_shared_b_blocks
                    + max(num_group_local_blocks, num_group_wide_blocks)
                    + max(num_script_local_blocks, num_script_wide_blocks))
        dp_schedule = [drop_path_rate * i / max(n_stages - 1, 1)
                       for i in range(n_stages)]
        dp_iter = iter(dp_schedule)

        # Convolutional stem: (B, 3, 32, W) → (B, stem_out_ch, 8, W/2)
        self.stem = ConvStem(in_ch=3, mid_ch=stem_mid_ch, out_ch=stem_out_ch)

        # Shared SWA-A at (h=8, w=W/2), dim=stem_out_ch. Window 8×8 covers
        # the full height so attention sees vertical character extent.
        self.shared_a = nn.ModuleList([
            SWABlock(dim=stem_out_ch, num_heads=max(stem_out_ch // 64, 1),
                     window_h=8, window_w=8, shift=(i % 2 == 1),
                     mlp_ratio=mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=layer_scale_init)
            for i in range(num_shared_a_blocks)
        ])

        # Collapse h=8 → 1 via Swin-style patch merging: concatenate the 8
        # vertical tokens channel-wise (dim 128 → 1024) and project back to
        # dim. Strictly more general than average-pooling — the Linear can
        # learn per-row weighting (e.g., weight middle rows more for
        # x-height-dominant scripts).
        self._post_stem_h = 8  # stem downsamples 32px input by 4x
        self.proj_a = nn.Linear(stem_out_ch * self._post_stem_h, dim)

        # Shared SWA-B at (h=1, w=W/2), dim. Pure horizontal context.
        self.shared_b = nn.ModuleList([
            SWABlock(dim=dim, num_heads=max(dim // 64, 1),
                     window_h=1, window_w=16, shift=(i % 2 == 1),
                     mlp_ratio=mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=layer_scale_init)
            for i in range(num_shared_b_blocks)
        ])

        # Parallel stages share one drop-path rate per stage so local/wide
        # streams have matched residual scaling.
        group_dp = next(dp_iter) if max(num_group_local_blocks, num_group_wide_blocks) > 0 else 0.0
        script_dp = next(dp_iter) if max(num_script_local_blocks, num_script_wide_blocks) > 0 else 0.0

        # LID-1: per-frame group classification
        self.group_h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.group_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_groups + 1),
        )

        # Group expert blocks (routed by group_id, 13 experts)
        # Init output projections near-zero so residual connections pass
        # features through initially — experts learn to specialize gradually
        # without destroying features that CTC needs.
        # Both streams run at h=1; local/wide differentiate via window_w.
        self.group_local_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_experts=num_groups,
                        window_h=1, window_w=local_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop_path=group_dp,
                        layer_scale_init=layer_scale_init)
            for i in range(num_group_local_blocks)
        ])
        self.group_wide_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64, num_experts=num_groups,
                        window_h=1, window_w=wide_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop_path=group_dp,
                        layer_scale_init=layer_scale_init)
            for i in range(num_group_wide_blocks)
        ])
        for block_list in [self.group_local_blocks, self.group_wide_blocks]:
            for block in block_list:
                for attn in block.expert_attns:
                    nn.init.zeros_(attn.proj.weight)
                    nn.init.zeros_(attn.proj.bias)
                for mlp in block.expert_mlps:
                    nn.init.zeros_(mlp.fc2.weight)
                    nn.init.zeros_(mlp.fc2.bias)

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
                self.lid2_heads[str(g)] = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, n_scripts),
                )

        # Script expert blocks (routed by flat script_id, 26 experts)
        # Initialize output projections near-zero so residual connections
        # pass features through initially (prevents randomly initialized
        # script experts from destroying group-expert features)
        self.script_local_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64,
                        num_experts=self.total_scripts,
                        window_h=1, window_w=local_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop_path=script_dp,
                        layer_scale_init=layer_scale_init)
            for i in range(num_script_local_blocks)
        ])
        self.script_wide_blocks = nn.ModuleList([
            ExpertBlock(dim=dim, num_heads=dim // 64,
                        num_experts=self.total_scripts,
                        window_h=1, window_w=wide_window_w, shift=(i % 2 == 1),
                        mlp_ratio=mlp_ratio, drop_path=script_dp,
                        layer_scale_init=layer_scale_init)
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

        # Stem: (B, 3, 32, W) → (B, stem_out_ch, 8, W/2)
        x = self.stem(x)
        _, C, h, w = x.shape  # h=8, w=W/2

        # Shared SWA-A at (h=8, w=W/2)
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        for blk in self.shared_a:
            x = blk(x, h, w)

        # Patch-merge h=8 → 1: concatenate vertical tokens channel-wise,
        # then project to dim. All downstream stages run at h=1.
        assert h == self._post_stem_h, \
            f"expected post-stem height {self._post_stem_h}, got {h}"
        x = x.reshape(B, h, w, C).permute(0, 2, 1, 3).reshape(B, w, h * C)
        x = self.proj_a(x)  # (B, w, dim)
        h = 1

        # Shared SWA-B at (h=1, w=W/2)
        d = x.shape[-1]
        for blk in self.shared_b:
            x = blk(x, h, w)

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
            # Defensive: clamp invalid IDs (e.g. -100 padding) to blank.
            # Caller should already do this, but bincount/indexing crash otherwise.
            frame_groups = torch.where(
                (frame_groups >= 0) & (frame_groups <= self.blank_group_id),
                frame_groups, torch.full_like(frame_groups, self.blank_group_id))
        else:
            frame_groups = group_logits.argmax(dim=-1)

        if detach_for_experts:
            x = x.detach()

        # =====================================================================
        # STAGE 1: Group expert blocks (routed per-segment by group_id)
        # Iterates by group (≤ num_groups = 13) instead of (B, unique_groups)
        # to amortize kernel-launch overhead — all samples with the same
        # group are padded and processed in one batched call per expert.
        # =====================================================================

        x_after_group = torch.zeros(B, w, d, device=x.device, dtype=x.dtype)

        for g in range(self.num_groups):
            mask_g = (frame_groups == g)  # (B, w)
            if not mask_g.any():
                continue
            batch_x, batch_info = _collect_segments(x, mask_g)
            if batch_x is None:
                continue
            max_len = batch_x.shape[1]

            local = batch_x
            for block in self.group_local_blocks:
                local = _run_expert_block(block, local, g, 1, max_len, use_ckpt)

            wide = batch_x
            for block in self.group_wide_blocks:
                wide = _run_expert_block(block, wide, g, 1, max_len, use_ckpt)

            comb = torch.cat([local, wide], dim=-1)
            agg = self.group_aggregates[g](comb)
            _scatter_segments(x_after_group, agg, mask_g, batch_info)

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
            # Defensive: clamp invalid script_ids to 0
            frame_scripts = torch.where(
                frame_scripts >= 0, frame_scripts,
                torch.zeros_like(frame_scripts))
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
        # STAGE 2: Script expert blocks (routed per-segment by flat script_id)
        # =====================================================================

        x_after_script = torch.zeros(B, w, d, device=x.device, dtype=x.dtype)

        # Iterate by flat script-id (≤ total_scripts = 26) instead of
        # (B, unique_scripts). Same batching pattern as group experts.
        for s in range(self.total_scripts):
            mask_s = (flat_scripts == s)  # (B, w)
            if not mask_s.any():
                continue
            batch_x, batch_info = _collect_segments(x_after_group, mask_s)
            if batch_x is None:
                continue
            max_len = batch_x.shape[1]

            local = batch_x
            for block in self.script_local_blocks:
                local = _run_expert_block(block, local, s, 1, max_len, use_ckpt)

            wide = batch_x
            for block in self.script_wide_blocks:
                wide = _run_expert_block(block, wide, s, 1, max_len, use_ckpt)

            comb = torch.cat([local, wide], dim=-1)
            agg = self.script_aggregates[s](comb)
            _scatter_segments(x_after_script, agg, mask_s, batch_info)

        # =====================================================================
        # CTC heads (per-segment routing)
        # =====================================================================

        x = self.norm(x_after_script)
        T = x.shape[1]

        max_vocab = max(m.max_vocab for m in self.ctc_modules)
        logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)

        # CTC routing: iterate by group (≤ num_groups = 13). Within each
        # group, per-sample script_ids are passed to the GroupCTCModule
        # which handles per-script head routing internally.
        for g in range(self.num_groups):
            mask_g = (frame_groups == g)  # (B, T)
            if not mask_g.any():
                continue
            batch_feats, batch_info = _collect_segments(x, mask_g)
            if batch_feats is None:
                continue

            # One script_id per segment (all frames in a group segment share a script)
            batch_sids = torch.tensor(
                [frame_scripts[b][mask_g[b]][0].item() for b, _ in batch_info],
                dtype=torch.long, device=x.device)

            seg_logits, _ = self.ctc_modules[g](batch_feats, script_ids=batch_sids)
            vs = seg_logits.shape[-1]
            for i, (b, sl) in enumerate(batch_info):
                logits[b, mask_g[b], :vs] = seg_logits[i, :sl].to(logits.dtype)

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return {
            "logits": logits,
            "lengths": lengths,
            "group_logits": group_logits,
            "group_ids": frame_groups,
            "lid2_logits_per_group": lid2_logits_per_group,
            "frame_scripts": frame_scripts,
            "flat_scripts": flat_scripts,
        }
