"""
Lipi v5 MoE Vision Encoder (from scratch — no pretrained backbone).

Architecture:
    Input: (B, 3, 32, W) — RGB
    -> ConvStem: two plain strided convs (no ResBlocks, small RF ~5px)
       → (B, 128, 8, W/2)
    -> Shared SWA-A: 2× windowed-attention blocks at (h=8, w=W/2)
    -> Patch-merge 8→2, proj 128×4→dim
    -> Frame-level LID-0: per-frame super-group classification
       (5 super-groups: alphabetic, semitic, cjk, brahmic, other, + blank)
    -> Route frames to per-super-group SWA-B stacks
    -> 2× super-group SWA-B blocks per super-group (h=2, w=W/2, dim)
       (blank frames bypass super_b as identity)
    -> Frame-level LID-1: per-frame script group classification (15 groups)
    -> Patch-merge 2→1
    -> Route frames to group expert blocks by group_id
    -> 1 local group expert block  (h=1, window 1×16, 15 experts)
    -> 1 wide  group expert block  (h=1, window 1×64, 15 experts)
    -> Per-group aggregation (concat local + wide → dim)
    -> Frame-level LID-2: per-frame script classification (multi-script groups)
    -> Route frames to script expert blocks by script_id
    -> 1 local script expert block (h=1, 1×16 windows, 27 experts)
    -> 1 wide  script expert block (h=1, 1×64 windows, 27 experts)
    -> Per-script aggregation (concat local + wide → dim)
    -> Per-script CTC heads (T=W/2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import flex_attention as _raw_flex_attention
    import torch._dynamo.config as _dynamo_cfg
    # flex_attention requires torch.compile to generate the fused flash-style
    # kernel — without compile it materializes the full score matrix.
    # dynamic=True so variable W doesn't recompile, and bump the recompile
    # limit because each distinct (window_h, window_w, shift) combination
    # specializes the compiled function. We have ~8 unique attention shapes
    # across the model (stem SWA 8×16, 2×16, experts 1×16, 1×64) × shift
    # on/off = up to 16 variants.
    _dynamo_cfg.recompile_limit = max(getattr(_dynamo_cfg, "recompile_limit", 8), 32)
    flex_attention = torch.compile(_raw_flex_attention, dynamic=True)
    _HAS_FLEX_ATTENTION = True
except ImportError:
    flex_attention = None
    _HAS_FLEX_ATTENTION = False

from src.model.lid import (
    NUM_GROUPS, NUM_SUPER_GROUPS, GROUP_ID_TO_SUPER_GROUP_ID,
)


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
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            B * nH * nW, self.window_h * self.window_w, C)

        # Attention
        qkv = self.qkv(x).reshape(
            x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # flex_attention expects (B, H, S, D)
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()

        out_fa = self._attend(q_fa, k_fa, v_fa, nW)
        out = out_fa.transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
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

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, nW: int) -> Tensor:
        """Run attention with relpos bias + (optional) shifted-window mask.

        Uses flex_attention so the relpos bias goes through a score_mod
        closure and the mask goes through a mask_mod closure — both
        compile-friendly and let the flash-attention fast path stay on.

        Falls back to SDPA with an explicit attn_mask if flex_attention
        isn't available (older PyTorch).
        """
        rel_pos_bias = self.rel_pos_bias  # (n_rel, heads)
        rel_pos_index = self.rel_pos_index  # (win_size, win_size) long
        win_w = self.window_w
        shift_w = self.shift_w
        shift = self.shift

        # flex_attention requires CUDA for backward. Fall back to SDPA on CPU.
        use_flex = _HAS_FLEX_ATTENTION and q.device.type == "cuda"

        if use_flex:
            # Score mod: add per-head relative position bias at (q_idx, kv_idx).
            def score_mod(score, b, h_idx, q_idx, kv_idx):
                idx = rel_pos_index[q_idx, kv_idx]
                return score + rel_pos_bias[idx, h_idx].to(score.dtype)

            if shift and shift_w > 0:
                # Wrapped columns (came from the opposite side via torch.roll)
                # are the last shift_w positions of each row in each window.
                # Only the last window in a row contains wrapped positions —
                # i.e., the window whose index along width is nW-1. We flatten
                # batch so b encodes (image, window_col). nH is always 1 in v5,
                # so b % nW = window column.
                def mask_mod(b, h_idx, q_idx, kv_idx):
                    col_q = q_idx % win_w
                    col_k = kv_idx % win_w
                    # "Wrapped" if this is the last window AND col ∈ [win_w-shift_w, win_w)
                    is_last_window = (b % nW) == (nW - 1)
                    q_wrapped = is_last_window & (col_q >= (win_w - shift_w))
                    k_wrapped = is_last_window & (col_k >= (win_w - shift_w))
                    return q_wrapped == k_wrapped

                from torch.nn.attention.flex_attention import create_block_mask
                B, H, S, _ = q.shape
                block_mask = create_block_mask(
                    mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=q.device)
                return flex_attention(q, k, v, score_mod=score_mod,
                                      block_mask=block_mask)

            return flex_attention(q, k, v, score_mod=score_mod)

        # Fallback: SDPA with explicit additive mask (math backend).
        win_size = self.window_h * self.window_w
        rel_bias = rel_pos_bias[rel_pos_index.view(-1)]
        rel_bias = rel_bias.view(win_size, win_size, -1).permute(2, 0, 1)
        rel_bias = rel_bias.unsqueeze(0).to(q.dtype)

        if shift and shift_w > 0:
            # Build per-window shift mask as before
            # Wrapped positions: last shift_w columns of the last window.
            N_b = q.shape[0]  # B*nW*nH
            device = q.device
            mask = torch.zeros(nW, win_size, device=device, dtype=q.dtype)
            mask[-1, -shift_w:] = 1  # last window, last shift_w columns wrapped
            diff = mask.unsqueeze(2) - mask.unsqueeze(1)  # (nW, W, W)
            shift_bias = diff.masked_fill(diff != 0, -100.0).masked_fill(
                diff == 0, 0.0).to(q.dtype)
            # Repeat across sample batches: (B*nW, 1, W, W)
            reps = N_b // nW
            full_bias = rel_bias + shift_bias.unsqueeze(1).repeat(reps, 1, 1, 1)
        else:
            full_bias = rel_bias

        return F.scaled_dot_product_attention(q, k, v, attn_mask=full_bias)


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


def _patch_merge_h(x: Tensor, h: int, w: int, proj: nn.Linear) -> tuple[Tensor, int]:
    """Halve h by concatenating adjacent vertical row pairs, then project.

    Input:  x of shape (B, h*w, C), h must be even
    Output: (x_new, new_h) where x_new is (B, (h//2)*w, proj.out_features)

    This is Swin-style patch merging restricted to the H axis — width
    is preserved. Each new row combines two source rows channel-wise,
    then the linear projection learns which features to keep.
    """
    assert h % 2 == 0, f"h must be even for patch merge, got {h}"
    B, _, C = x.shape
    x = x.reshape(B, h // 2, 2, w, C).permute(0, 1, 3, 2, 4)
    x = x.reshape(B, (h // 2) * w, 2 * C)
    return proj(x), h // 2


class ConvStem(nn.Module):
    """Two-conv plain stem ending at dim=128. No ResBlocks.

    Spatial feature extraction in 2 convs (dim 3→64→out):
        Conv 1: 3  → 64,  stride (2, 1), 3×3 kernel  →  H 32→16
        Conv 2: 64 → out, stride (2, 2), 3×3 kernel  →  H 16→8, W→W/2

    3×3 kernels keep RF tight for clean routing boundaries:
    after stem RF ≈ 7 px H × 5 px W.

    Default out_ch=128 keeps SWA-A compute modest; capacity is added via
    shared_mlp_ratio=4 on the attention blocks instead of widening here.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=(2, 1),
                      padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv2d(64, out_ch, kernel_size=3, stride=(2, 2),
                      padding=1, bias=False),
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


def _per_sample_key_lens(frame_ids: Tensor, num_classes: int) -> Tensor:
    """(B, num_classes) per-sample counts of each key in frame_ids.

    Frames with id < 0 or >= num_classes don't contribute. Computed via one
    scatter_add — no Python loop. Returning the GPU tensor means the caller
    can do a single .tolist() for all keys at once instead of N .any() /
    .sum() / .tolist() syncs per key.
    """
    B, _ = frame_ids.shape
    counts = torch.zeros(B, num_classes, dtype=torch.long, device=frame_ids.device)
    valid = (frame_ids >= 0) & (frame_ids < num_classes)
    safe = torch.where(valid, frame_ids, torch.zeros_like(frame_ids))
    counts.scatter_add_(1, safe, valid.long())
    return counts


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

    Implementation note: vectorized via one nonzero + one cumsum + one
    index_put. Replaces the prior per-sample Python loop (which launched
    one kernel per active sample).
    """
    B, _, d = x.shape
    seg_lens = mask.sum(dim=1)  # (B,)
    has_g = seg_lens > 0
    b_indices = has_g.nonzero(as_tuple=True)[0]
    if b_indices.numel() == 0:
        return None, None

    # Single sync for the Python-side batch_info list.
    lens_list = seg_lens[b_indices].tolist()
    b_list = b_indices.tolist()
    max_len = max(lens_list)
    N = len(b_list)

    # Dense map b_idx → active_idx in [0, N); -1 for inactive samples.
    b_to_active = torch.full((B,), -1, dtype=torch.long, device=x.device)
    b_to_active[b_indices] = torch.arange(N, device=x.device)

    # Source (b, t) positions where mask is True.
    bs, ts = mask.nonzero(as_tuple=True)  # both (M,) where M = mask.sum()
    # Position within the sample's segment at each True (b, t).
    pos_in_seg = mask.long().cumsum(dim=1) - 1  # (B, T)
    p_dst = pos_in_seg[bs, ts]  # (M,)
    i_dst = b_to_active[bs]    # (M,)

    batch_x = torch.zeros(N, max_len, d, device=x.device, dtype=x.dtype)
    batch_x[i_dst, p_dst] = x[bs, ts]

    return batch_x, list(zip(b_list, lens_list))


def _scatter_segments(
    x_out: Tensor,
    batch_out: Tensor,
    mask: Tensor,
    batch_info: list[tuple[int, int]],
) -> None:
    """Scatter packed batch results back to per-sample positions.

    Vectorized: one nonzero + cumsum + index_put, no Python loop.
    """
    if not batch_info:
        return
    B = x_out.shape[0]
    N = len(batch_info)
    b_indices_cpu = torch.tensor(
        [b for b, _ in batch_info], dtype=torch.long, device=x_out.device)
    b_to_active = torch.full((B,), -1, dtype=torch.long, device=x_out.device)
    b_to_active[b_indices_cpu] = torch.arange(N, device=x_out.device)

    bs, ts = mask.nonzero(as_tuple=True)
    pos_in_seg = mask.long().cumsum(dim=1) - 1
    p_src = pos_in_seg[bs, ts]
    i_src = b_to_active[bs]

    x_out[bs, ts] = batch_out[i_src, p_src]


def _run_expert_block(block, x, expert_id, h, w):
    """Run one sample through a specific expert in an ExpertBlock.

    Applies LayerScale and DropPath on both residual branches (same as a
    standard modern transformer block, but the attn/mlp ops themselves are
    expert-specific).
    """
    normed = block.norm1(x)
    attn = block.expert_attns[expert_id]
    attn_out = attn(normed, h, w)
    x = x + block.drop_path(block.ls1(attn_out.to(x.dtype)))
    normed = block.norm2(x)
    mlp = block.expert_mlps[expert_id]
    mlp_out = mlp(normed)
    x = x + block.drop_path(block.ls2(mlp_out.to(x.dtype)))
    return x


class LipiMoEEncoder(nn.Module):
    """Lipi v5: ConvStem + LID-0 super-group routing + group experts + LID-2 + script experts.

    Trained from scratch (no pretrained backbone). Small-RF stem keeps
    boundary contamination minimal before attention layers take over.

    Three-level expert routing:
      1. LID-0 classifies each frame into a super-group (5 + blank)
      2. Per-super-group SWA-B stacks specialize features within each family
      3. LID-1 classifies each frame into a script group (15 + blank)
      4. Group expert blocks process frames per-group (local + wide streams)
      5. LID-2 classifies each frame into a script within its group
      6. Script expert blocks process frames per-script (local + wide streams)
      7. Per-script CTC heads decode characters

    Single-script groups skip LID-2 (only 1 script, trivially assigned).
    Blank frames bypass super_b (identity) and are zeroed by the group-
    expert stage.
    """

    def __init__(
        self,
        dim: int = 256,
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
        shared_mlp_ratio: int = 4,
        drop_path_rate: float = 0.1,
        # LayerScale init=1.0 is a no-op (identity). Reduce (e.g. 1e-2)
        # only if you see training instability; on top of identity-init
        # experts, small values scale expert gradients by the same factor
        # and can starve MoE experts that already see only 1/N of data.
        layer_scale_init: float = 1.0,
        num_groups: int = NUM_GROUPS,
        num_super_groups: int = NUM_SUPER_GROUPS,
        group_script_vocab_sizes: list[list[int]] | None = None,
        group_script_names: list[list[str]] | None = None,
    ):
        super().__init__()
        # num_super_groups is accepted as a kwarg so ckpt["model_config"] can be
        # replayed via LipiMoEEncoder(**cfg); the actual super-group count is
        # fixed by the lid.py constants and must match.
        assert num_super_groups == NUM_SUPER_GROUPS, (
            f"num_super_groups={num_super_groups} does not match "
            f"NUM_SUPER_GROUPS={NUM_SUPER_GROUPS} in lid.py")
        self.num_groups = num_groups
        self.blank_group_id = num_groups
        self.num_super_groups = NUM_SUPER_GROUPS
        self.blank_super_group_id = NUM_SUPER_GROUPS

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
            "stem_out_ch": stem_out_ch,
            "num_shared_a_blocks": num_shared_a_blocks,
            "num_shared_b_blocks": num_shared_b_blocks,
            "num_group_local_blocks": num_group_local_blocks,
            "num_group_wide_blocks": num_group_wide_blocks,
            "num_script_local_blocks": num_script_local_blocks,
            "num_script_wide_blocks": num_script_wide_blocks,
            "local_window_w": local_window_w,
            "wide_window_w": wide_window_w,
            "mlp_ratio": mlp_ratio,
            "shared_mlp_ratio": shared_mlp_ratio,
            "drop_path_rate": drop_path_rate,
            "layer_scale_init": layer_scale_init,
            "num_groups": num_groups,
            "num_super_groups": NUM_SUPER_GROUPS,
            "group_script_vocab_sizes": group_script_vocab_sizes,
            "group_script_names": group_script_names,
        }

        # Fixed mapping: group_id → super_group_id (buffer so it moves with
        # the model to GPU). Size is num_groups+1 so index num_groups (blank)
        # is always valid and maps to the blank super-group. For num_groups <
        # NUM_GROUPS (tests / partial models), truncate the canonical table.
        _g2sg_list = GROUP_ID_TO_SUPER_GROUP_ID[:num_groups]
        if len(_g2sg_list) < num_groups:
            # Extra groups beyond the canonical 15 fall back to the blank
            # super-group (they won't route to any specialized super_b stack).
            _g2sg_list = _g2sg_list + [NUM_SUPER_GROUPS] * (
                num_groups - len(_g2sg_list))
        _g2sg_list = _g2sg_list + [NUM_SUPER_GROUPS]  # blank group → blank super
        g2sg = torch.tensor(_g2sg_list, dtype=torch.long)
        self.register_buffer("group_to_super_group", g2sg, persistent=False)

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
        self.stem = ConvStem(in_ch=3, out_ch=stem_out_ch)

        # Shared SWA-A at (h=8, w=W/2), dim=stem_out_ch. Window 8×16:
        # full vertical extent × 2-character horizontal context.
        self.shared_a = nn.ModuleList([
            SWABlock(dim=stem_out_ch, num_heads=max(stem_out_ch // 64, 1),
                     window_h=8, window_w=16, shift=(i % 2 == 1),
                     mlp_ratio=shared_mlp_ratio, drop_path=next(dp_iter),
                     layer_scale_init=layer_scale_init)
            for i in range(num_shared_a_blocks)
        ])

        # Patch-merge (h=8 → 2): concat 4 adjacent rows channel-wise,
        # project to dim. Drops vertical resolution by 4× but keeps
        # learned weighting across the original rows.
        self._post_stem_h = 8  # stem downsamples 32px input by 4x
        self.merge_a = nn.Linear(stem_out_ch * 4, dim)  # 4 rows → dim

        # LID-0: per-frame super-group classification. Pools h=2→1 off the
        # merge_a output (before super_b) and predicts one of 5 script
        # families + blank. Same pooling + MLP pattern as group_head.
        self.super_h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lid0_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, NUM_SUPER_GROUPS + 1),
        )

        # Per-super-group SWA-B stacks. Each super-group gets its own copy
        # of what was previously a single shared SWA-B stack; frames route
        # to the stack matching their super-group. Same config as the old
        # shared_b (h=2, w=W/2, window 2×16). Drop-path rate is shared
        # across super-groups so a frame sees the same residual scaling
        # regardless of which super-group it routes to.
        sb_dp = [next(dp_iter) for _ in range(num_shared_b_blocks)]
        self.super_b = nn.ModuleList([
            nn.ModuleList([
                SWABlock(dim=dim, num_heads=max(dim // 64, 1),
                         window_h=2, window_w=16, shift=(i % 2 == 1),
                         mlp_ratio=shared_mlp_ratio, drop_path=sb_dp[i],
                         layer_scale_init=layer_scale_init)
                for i in range(num_shared_b_blocks)
            ])
            for _ in range(NUM_SUPER_GROUPS)
        ])

        # Patch-merge (h=2 → 1): concat 2 rows → project. Final collapse
        # to frame sequence before experts.
        self.merge_b = nn.Linear(dim * 2, dim)

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
        compute_until: str = "all",
    ) -> dict:
        """Run the encoder forward pass.

        compute_until: how far to run the pipeline before returning stubs
        for the remaining outputs. Lets staged training / inference skip
        compute that no active loss or consumer needs.

          "lid0"  — stop after lid0_head. Skips super_b, LID-1, merge_b,
                    group experts, LID-2, script experts, CTC.
          "lid1"  — stop after group_head (LID-1). Skips merge_b, group
                    experts, LID-2, script experts, CTC.
          "lid2"  — stop after LID-2 heads. Skips script experts + CTC.
                    (Equivalent to the old compute_ctc=False.)
          "all"   — full forward pass (default).
        """
        _STAGES = ("lid0", "lid1", "lid2", "all")
        assert compute_until in _STAGES, (
            f"compute_until must be one of {_STAGES}, got {compute_until!r}")
        _stage_idx = _STAGES.index(compute_until)
        B = images.shape[0]

        x = images.float() / 255.0 if images.dtype == torch.uint8 else images

        # Stem: (B, 3, 32, W) → (B, stem_out_ch, 8, W/2)
        x = self.stem(x)
        _, C, h, w = x.shape  # h=8, w=W/2

        # Shared SWA-A at (h=8, w=W/2)
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        for blk in self.shared_a:
            x = blk(x, h, w)

        # Patch-merge 8 → 2 (4× h-downsample): concat 4 adjacent rows,
        # project to dim. One intermediate stage (h=2) with attention
        # before the final collapse.
        assert h == self._post_stem_h, \
            f"expected post-stem height {self._post_stem_h}, got {h}"
        x = x.reshape(B, 2, 4, w, C).permute(0, 1, 3, 2, 4).reshape(B, 2 * w, 4 * C)
        x = self.merge_a(x)  # (B, 2*w, dim)
        h = 2

        d = x.shape[-1]

        # =====================================================================
        # LID-0: per-frame super-group classification
        #
        # Pooled h=2→1 view of the merge_a output predicts one of 5 script
        # families (+ blank). Runs BEFORE super_b so the routing decision is
        # made on family-agnostic features; super_b blocks then specialize
        # within the family.
        # =====================================================================
        x_for_super = x.reshape(B, h, w, d).permute(0, 3, 1, 2)  # (B, d, 2, w)
        x_for_super = self.super_h_pool(x_for_super).squeeze(2).permute(0, 2, 1)
        super_group_logits = self.lid0_head(x_for_super)  # (B, w, num_super+1)

        # Early-exit after LID-0 (compute_until="lid0"). Skips super_b +
        # group experts + LID-2 + script experts + CTC → ~3-5× faster per
        # step. All downstream keys are zero-filled stubs so callers that
        # expect them don't crash.
        if _stage_idx == 0:
            T = w
            max_vocab = max(m.max_vocab for m in self.ctc_modules)
            return {
                "logits": torch.zeros(B, T, max_vocab,
                                      device=x.device, dtype=x.dtype),
                "lengths": torch.full((B,), T, dtype=torch.long, device=x.device),
                "super_group_logits": super_group_logits,
                "super_group_ids": super_group_logits.argmax(dim=-1),
                "group_logits": torch.zeros(B, T, self.num_groups + 1,
                                            device=x.device, dtype=x.dtype),
                "group_ids": torch.zeros(B, T, dtype=torch.long, device=x.device),
                "lid2_logits_per_group": {},
                "frame_scripts": torch.zeros(B, T, dtype=torch.long, device=x.device),
                "flat_scripts": torch.full((B, T), -1, dtype=torch.long, device=x.device),
            }

        # Determine per-frame super-group assignments. During training with
        # GT group labels, map group→super_group via the fixed buffer.
        # During inference, use LID-0 argmax.
        if group_ids is not None:
            if group_ids.dim() == 1:
                _fg_early = group_ids.unsqueeze(1).expand(B, w)
            else:
                _fg_early = group_ids
            _fg_early = torch.where(
                (_fg_early >= 0) & (_fg_early <= self.blank_group_id),
                _fg_early, torch.full_like(_fg_early, self.blank_group_id))
            frame_super_groups = self.group_to_super_group[_fg_early]
        else:
            frame_super_groups = super_group_logits.argmax(dim=-1)

        # =====================================================================
        # Per-super-group SWA-B routing
        #
        # Frames route to one of NUM_SUPER_GROUPS SWA-B stacks based on
        # their super-group. Blank frames (super_id == blank_super_group_id)
        # bypass super_b as identity — they have no family, and the group
        # experts will zero them out downstream anyway.
        # =====================================================================
        x_2d = x.reshape(B, h, w, d)
        x_after_super = x_2d.clone()  # identity default for blank/unrouted frames

        # One upfront sync: (B, num_super_groups) per-sample per-sg counts.
        # Replaces per-iteration .any() + .sum() + .tolist() syncs.
        sg_lens_cpu = _per_sample_key_lens(
            frame_super_groups, self.num_super_groups).tolist()

        for sg in range(self.num_super_groups):
            # Python-only iteration: no syncs inside the loop head.
            b_list = [b for b in range(B) if sg_lens_cpu[b][sg] > 0]
            if not b_list:
                continue
            lens_list = [sg_lens_cpu[b][sg] for b in b_list]
            max_w_sg = max(lens_list)
            N = len(b_list)

            # Vectorized collect: same pattern as _collect_segments, but
            # preserves h=2 rows (each frame column carries 2 rows).
            mask_sg = (frame_super_groups == sg)  # (B, w)
            b_indices_gpu = torch.tensor(
                b_list, dtype=torch.long, device=x.device)
            b_to_active = torch.full(
                (B,), -1, dtype=torch.long, device=x.device)
            b_to_active[b_indices_gpu] = torch.arange(N, device=x.device)

            bs, ts = mask_sg.nonzero(as_tuple=True)  # (M,), (M,)
            pos_in_seg = mask_sg.long().cumsum(dim=1) - 1  # (B, w)
            p_dst = pos_in_seg[bs, ts]  # (M,)
            i_dst = b_to_active[bs]    # (M,)

            seg = torch.zeros(N, h, max_w_sg, d,
                              device=x.device, dtype=x.dtype)
            # Advanced indexing: x_2d[bs, :, ts] → (M, h, d) because the
            # non-contiguous advanced indices (bs, ts) broadcast to M and
            # get moved to the front, leaving the `:`-indexed h axis intact.
            seg[i_dst, :, p_dst] = x_2d[bs, :, ts]

            # Run per-super-group SWA-B stack at (h=2, w=max_w_sg)
            seg_flat = seg.reshape(N, h * max_w_sg, d)
            for block in self.super_b[sg]:
                seg_flat = block(seg_flat, h, max_w_sg)
            seg = seg_flat.reshape(N, h, max_w_sg, d)

            # Scatter back into x_after_super (reusing i_dst / p_dst).
            x_after_super[bs, :, ts] = seg[i_dst, :, p_dst]

        x = x_after_super.reshape(B, h * w, d)

        # LID-1 branches off BEFORE the final 2→1 merge. Two consequences:
        #   1. LID sees h=2 features (upper + lower half of each character)
        #      — richer script-identifying signal than a flattened h=1 view.
        #   2. merge_b below then only receives CTC gradient, so it can
        #      specialize for character discriminability without having to
        #      simultaneously serve LID's script-discrimination objective.
        x_for_group = x.reshape(B, h, w, d).permute(0, 3, 1, 2)  # (B, d, 2, w)
        x_for_group = self.group_h_pool(x_for_group).squeeze(2).permute(0, 2, 1)
        group_logits = self.group_head(x_for_group)  # (B, W/2, num_groups+1)

        # Early-exit after LID-1 (compute_until="lid1"). Skips merge_b +
        # group experts + LID-2 + script experts + CTC. For staged training
        # where only LID-0 and LID-1 have non-zero loss weights.
        if _stage_idx == 1:
            T = w
            max_vocab = max(m.max_vocab for m in self.ctc_modules)
            return {
                "logits": torch.zeros(B, T, max_vocab,
                                      device=x.device, dtype=x.dtype),
                "lengths": torch.full((B,), T, dtype=torch.long, device=x.device),
                "super_group_logits": super_group_logits,
                "super_group_ids": frame_super_groups,
                "group_logits": group_logits,
                "group_ids": group_logits.argmax(dim=-1),
                "lid2_logits_per_group": {},
                "frame_scripts": torch.zeros(B, T, dtype=torch.long, device=x.device),
                "flat_scripts": torch.full((B, T), -1, dtype=torch.long, device=x.device),
            }

        # Patch-merge 2 → 1: concat 2 rows, project to dim. Only touches
        # the CTC path from here on.
        x, h = _patch_merge_h(x, h, w, self.merge_b)

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

        # One upfront sync: per-sample per-group frame counts.
        group_lens_cpu = _per_sample_key_lens(
            frame_groups, self.num_groups).tolist()

        for g in range(self.num_groups):
            # Skip without a sync: checks a pre-transferred Python list.
            if not any(group_lens_cpu[b][g] > 0 for b in range(B)):
                continue
            mask_g = (frame_groups == g)  # (B, w)
            batch_x, batch_info = _collect_segments(x, mask_g)
            if batch_x is None:
                continue
            max_len = batch_x.shape[1]

            local = batch_x
            for block in self.group_local_blocks:
                local = _run_expert_block(block, local, g, 1, max_len)

            wide = batch_x
            for block in self.group_wide_blocks:
                wide = _run_expert_block(block, wide, g, 1, max_len)

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
            # Inference: predict from LID-2 for multi-script groups.
            # Reuse the Python-side group counts computed above to skip
            # empty groups without a sync.
            for g, lid2_log in lid2_logits_per_group.items():
                if not any(group_lens_cpu[b][g] > 0 for b in range(B)):
                    continue
                g_mask = (frame_groups == g)
                pred = lid2_log.argmax(dim=-1)  # (B, T)
                frame_scripts[g_mask] = pred[g_mask]

        # Convert to flat script IDs for script expert routing
        flat_scripts = self._get_flat_script_ids(frame_groups, frame_scripts)

        # Early-exit after LID-2 (compute_until="lid2"). Skips script
        # experts + final norm + CTC heads. Useful when ctc_weight=0 so
        # CTC loss isn't computed — the script expert forward would be
        # pure waste.
        if _stage_idx == 2:
            T = w
            max_vocab = max(m.max_vocab for m in self.ctc_modules)
            return {
                "logits": torch.zeros(B, T, max_vocab,
                                      device=x.device, dtype=x.dtype),
                "lengths": torch.full((B,), T, dtype=torch.long, device=x.device),
                "super_group_logits": super_group_logits,
                "super_group_ids": frame_super_groups,
                "group_logits": group_logits,
                "group_ids": frame_groups,
                "lid2_logits_per_group": lid2_logits_per_group,
                "frame_scripts": frame_scripts,
                "flat_scripts": flat_scripts,
            }

        # =====================================================================
        # STAGE 2: Script expert blocks (routed per-segment by flat script_id)
        # =====================================================================

        x_after_script = torch.zeros(B, w, d, device=x.device, dtype=x.dtype)

        # One upfront sync: per-sample per-script frame counts. flat_scripts
        # uses -1 for blank/unrouted, which _per_sample_key_lens filters.
        script_lens_cpu = _per_sample_key_lens(
            flat_scripts, self.total_scripts).tolist()

        # Iterate by flat script-id (≤ total_scripts = 26) instead of
        # (B, unique_scripts). Same batching pattern as group experts.
        for s in range(self.total_scripts):
            if not any(script_lens_cpu[b][s] > 0 for b in range(B)):
                continue
            mask_s = (flat_scripts == s)  # (B, w)
            batch_x, batch_info = _collect_segments(x_after_group, mask_s)
            if batch_x is None:
                continue
            max_len = batch_x.shape[1]

            local = batch_x
            for block in self.script_local_blocks:
                local = _run_expert_block(block, local, s, 1, max_len)

            wide = batch_x
            for block in self.script_wide_blocks:
                wide = _run_expert_block(block, wide, s, 1, max_len)

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

        # CTC routing: iterate by group. Within each group, per-sample
        # script_ids are passed to the GroupCTCModule which handles
        # per-script head routing internally.
        for g in range(self.num_groups):
            # Skip without a sync using the pre-transferred group counts.
            if not any(group_lens_cpu[b][g] > 0 for b in range(B)):
                continue
            mask_g = (frame_groups == g)  # (B, T)
            batch_feats, batch_info = _collect_segments(x, mask_g)
            if batch_feats is None:
                continue

            # One script_id per segment (all frames in a group segment share
            # a script). Build batch_sids on GPU via gather: first-True
            # column per sample, then index frame_scripts. Single H2D
            # transfer for b_tensor; no per-sample .item() sync.
            b_tensor = torch.tensor(
                [b for b, _ in batch_info], device=x.device, dtype=torch.long)
            first_col = mask_g[b_tensor].int().argmax(dim=1)  # (N,)
            batch_sids = frame_scripts[b_tensor, first_col]

            seg_logits, _ = self.ctc_modules[g](batch_feats, script_ids=batch_sids)
            vs = seg_logits.shape[-1]
            for i, (b, sl) in enumerate(batch_info):
                logits[b, mask_g[b], :vs] = seg_logits[i, :sl].to(logits.dtype)

        lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        return {
            "logits": logits,
            "lengths": lengths,
            "super_group_logits": super_group_logits,
            "super_group_ids": frame_super_groups,
            "group_logits": group_logits,
            "group_ids": frame_groups,
            "lid2_logits_per_group": lid2_logits_per_group,
            "frame_scripts": frame_scripts,
            "flat_scripts": flat_scripts,
        }
