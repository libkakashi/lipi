"""
Building blocks for the Lipi v5 MoE encoder.

Layer modules (DropPath, LayerScale, WindowedAttention, SWABlock, ExpertBlock,
CTCHead, GroupCTCModule, ConvStem, MLP) plus the small pure-function helpers
(_patch_merge_h, _per_sample_key_lens, _collect_segments, _scatter_segments,
_run_expert_block) that LipiMoEEncoder composes together.

The encoder itself lives in encoder.py.
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
    def __init__(self, dim: int, mlp_ratio: int = 4):
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
                 mlp_ratio: int = 4, drop_path: float = 0.0,
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
                 mlp_ratio: int = 4, drop_path: float = 0.0,
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
