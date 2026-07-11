"""
Building blocks for the Lipi v5 MoE encoder.

Layer modules (DropPath, LayerScale, WindowedAttention, SWABlock, MoELayer,
ConvNeXtBlock, BlurPool2d, CTCHead, GroupCTCModule, ConvStem, MLP) plus the
small pure-function helpers (_patch_merge_h, _per_sample_key_lens) that
LipiMoEEncoder composes together.

The encoder itself lives in encoder.py.

MoELayer is the workhorse of the expert stages: one shared windowed
attention over the full frame sequence + N per-frame routed MLPs + one
always-on shared MLP. Attention is script-agnostic (all scripts read as
horizontal 1D glyph sequences), so it lives once per layer; MLPs are the
per-script channel-transform work and are routed per frame.
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

# Escape hatch for GPUs where the flex_attention Triton kernel exceeds the
# device's shared-memory limit (H100 hits this at our head-dim × window
# combinations — the generated kernel wants ~278 KB, hardware caps at 232 KB).
# The SDPA fallback below produces the same numerics via an explicit mask;
# training is slower per step but correct.
import os as _os
if _os.environ.get("LIPI_DISABLE_FLEX_ATTENTION", "0") == "1":
    _HAS_FLEX_ATTENTION = False

# torch.compile escape hatch used by the SDPA mask builder below. Falls back
# to an identity decorator on builds without torch._dynamo (never compiled).
try:
    import torch._dynamo as _dynamo
    _dynamo_disable = _dynamo.disable
except (ImportError, AttributeError):
    def _dynamo_disable(fn):
        return fn


_mask_bias_cache: dict = {}


@_dynamo_disable
def _pad_wrap_mask_bias(nW: int, win_h: int, win_w: int, wp: int,
                        w_real: int, shift_w: int, n_batch: int,
                        device, dtype) -> Tensor:
    """Additive SDPA attention-mask bias for shift/pad columns.

    Built eagerly (torch._dynamo.disable): the boolean column algebra reasons
    over shape-derived tensors, and Inductor's dynamic-shape value-range pass
    rejects ordering comparisons on booleans ("A Boolean argument can only be
    used in Eq and Ne"), crashing compilation. The mask is a pure function of
    the integer window geometry — no dependence on q/k/v values and no
    gradient — so it is cached per geometry: static bucket shapes mean only
    a handful of keys exist for the life of a run, and rebuilding cost ~10
    kernel launches plus an up-to-19MB repeat per attention call. Returns
    (n_batch, 1, win_size, win_size).
    """
    key = (nW, win_h, win_w, wp, w_real, shift_w, str(device), dtype)
    cached = _mask_bias_cache.get(key)
    if cached is not None:
        # Cache holds the compact (nW, 1, S, S) form (~KBs per key); the
        # batch repeat is one cheap kernel. Caching post-repeat tensors
        # would pin tens of MB per bucket shape.
        return cached.repeat(n_batch // nW, 1, 1, 1)
    win_size = win_h * win_w
    # Per rolled column: pad = pre-roll column >= w_real;
    # wrap = the last shift_w rolled positions overall.
    cols = torch.arange(wp, device=device)
    pad_flag = ((cols + shift_w) % wp) >= w_real
    if shift_w > 0:
        wrap_flag = cols >= (wp - shift_w)
    else:
        wrap_flag = torch.zeros_like(pad_flag)
    # (nW, win_w) → tile rows to intra-window flattening (h, w)
    pad_win = pad_flag.reshape(nW, 1, win_w).expand(
        nW, win_h, win_w).reshape(nW, win_size)
    wrap_win = wrap_flag.reshape(nW, 1, win_w).expand(
        nW, win_h, win_w).reshape(nW, win_size)
    allowed = ((wrap_win[:, :, None] == wrap_win[:, None, :])
               & (~pad_win[:, None, :] | pad_win[:, :, None]))
    mask_bias = torch.zeros(nW, win_size, win_size, device=device, dtype=dtype)
    mask_bias.masked_fill_(~allowed, -100.0)
    compact = mask_bias.unsqueeze(1)  # (nW, 1, win_size, win_size)
    _mask_bias_cache[key] = compact
    # Repeat across sample batches: (B*nH*nW, 1, win_size, win_size)
    return compact.repeat(n_batch // nW, 1, 1, 1)


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

        out_fa = self._attend(q_fa, k_fa, v_fa, nW, wp, w)
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

    def _attend(self, q: Tensor, k: Tensor, v: Tensor,
                nW: int, wp: int, w_real: int) -> Tensor:
        """Run attention with relpos bias + shift/pad masking.

        Two mask rules, applied per (query, key) pair within a window:
          - wrap: after the roll, the last shift_w positions hold content
            from the opposite edge; queries and keys must have equal wrap
            status (standard Swin shift mask).
          - pad: positions whose pre-roll column is >= w_real are width
            padding (zeros). Real queries must not attend pad keys — the
            pad K/V are LayerNorm+bias artifacts, not content. Pad
            *queries* stay unmasked (their outputs are sliced away) so no
            softmax row is ever fully masked.

        Uses flex_attention so the relpos bias goes through a score_mod
        closure and the mask goes through a mask_mod closure — both
        compile-friendly and let the flash-attention fast path stay on.

        Falls back to SDPA with an explicit attn_mask if flex_attention
        isn't available (older PyTorch) or on CPU.
        """
        rel_pos_bias = self.rel_pos_bias  # (n_rel, heads)
        rel_pos_index = self.rel_pos_index  # (win_size, win_size) long
        win_w = self.window_w
        win_h = self.window_h
        shift_w = self.shift_w if self.shift else 0
        pad_w = wp - w_real

        # flex_attention requires CUDA for backward. Fall back to SDPA on CPU.
        use_flex = _HAS_FLEX_ATTENTION and q.device.type == "cuda"

        if use_flex:
            # Score mod: add per-head relative position bias at (q_idx, kv_idx).
            def score_mod(score, b, h_idx, q_idx, kv_idx):
                idx = rel_pos_index[q_idx, kv_idx]
                return score + rel_pos_bias[idx, h_idx].to(score.dtype)

            if shift_w > 0 or pad_w > 0:
                # Batch is flattened as (image, nH, window_col), so
                # b % nW = window column regardless of nH. Global rolled
                # column of an intra-window index = win_col*win_w + idx%win_w;
                # its pre-roll column is (col + shift_w) % wp.
                def mask_mod(b, h_idx, q_idx, kv_idx):
                    col_q = (b % nW) * win_w + (q_idx % win_w)
                    col_k = (b % nW) * win_w + (kv_idx % win_w)
                    q_pad = ((col_q + shift_w) % wp) >= w_real
                    k_pad = ((col_k + shift_w) % wp) >= w_real
                    q_wrap = col_q >= (wp - shift_w) if shift_w > 0 else (col_q < 0)
                    k_wrap = col_k >= (wp - shift_w) if shift_w > 0 else (col_k < 0)
                    return (q_wrap == k_wrap) & (~k_pad | q_pad)

                from torch.nn.attention.flex_attention import create_block_mask
                B, H, S, _ = q.shape
                block_mask = create_block_mask(
                    mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=q.device)
                return flex_attention(q, k, v, score_mod=score_mod,
                                      block_mask=block_mask)

            return flex_attention(q, k, v, score_mod=score_mod)

        # Fallback: SDPA with explicit additive mask. The gathered bias
        # must be contiguous: SDPA's mem-efficient backend rejects masks
        # whose last dim is strided (this permute leaves stride 6), which
        # silently forced the math backend on every call.
        win_size = win_h * win_w
        rel_bias = rel_pos_bias[rel_pos_index.view(-1)]
        rel_bias = rel_bias.view(win_size, win_size, -1).permute(2, 0, 1)
        rel_bias = rel_bias.contiguous().unsqueeze(0).to(q.dtype)

        if shift_w > 0 or pad_w > 0:
            # Constant additive mask built eagerly (see _pad_wrap_mask_bias);
            # rel_bias carries the learned gradient and stays in the graph.
            mask_bias = _pad_wrap_mask_bias(
                nW, win_h, win_w, wp, w_real, shift_w, q.shape[0],
                q.device, q.dtype)
            full_bias = rel_bias + mask_bias
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


class MoELayer(nn.Module):
    """DeepSeek-style MoE layer: shared attn + routed MLPs + shared MLP.

    Runs on the full unpacked frame sequence (B, T, dim). Every frame passes
    through the same windowed attention, then splits into two MLP branches
    that are summed:
      - routed: per-frame expert lookup; frame at (b, t) uses MLP indexed by
        expert_ids[b, t]. Frames whose id is out of range contribute zero.
      - shared: one MLP that always runs, on every frame.

    Attention is position-aware and sees true geometry (no packing/stitching
    across routing boundaries). MLPs are position-wise, so per-frame routing
    is just an indexed matmul — no sequence gather/scatter.

    Init:
      - LayerScale on both residual branches (small init, block starts
        near-identity).
      - Routed MLP output projections are zero-init so at step 0 the routed
        branch contributes nothing while the shared MLP already carries the
        MLP branch. Routed experts specialize gradually from there.
    """

    def __init__(self, dim: int, num_heads: int, num_experts: int,
                 window_w: int = 16, shift: bool = False,
                 routed_mlp_ratio: int = 4, shared_mlp_ratio: int = 2,
                 drop_path: float = 0.0, layer_scale_init: float = 1e-4):
        super().__init__()
        self.num_experts = num_experts

        # Shared attention branch (one set of weights for the whole layer).
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(
            dim, num_heads, window_h=1, window_w=window_w, shift=shift)
        self.ls1 = LayerScale(dim, layer_scale_init)

        # Routed + shared MLP branch.
        self.norm2 = nn.LayerNorm(dim)
        self.routed_mlps = nn.ModuleList([
            MLP(dim, routed_mlp_ratio) for _ in range(num_experts)
        ])
        self.shared_mlp = MLP(dim, shared_mlp_ratio)
        self.ls2 = LayerScale(dim, layer_scale_init)

        self.drop_path = DropPath(drop_path)

        # Zero-init the routed MLPs' output projections so the routed branch
        # contributes zero at step 0. The shared MLP carries the MLP branch
        # immediately; routed experts specialize as they receive gradient.
        for mlp in self.routed_mlps:
            nn.init.zeros_(mlp.fc2.weight)
            nn.init.zeros_(mlp.fc2.bias)

    def forward(self, x: Tensor, expert_ids: Tensor,
                expert_lens_cpu: list | None, T: int) -> Tensor:
        """Run one MoE layer on the full frame sequence.

        Args:
            x:               (B, T, dim) frame features.
            expert_ids:      (B, T) per-frame expert id. Values outside
                             [0, num_experts) route no MLP for that frame
                             (blank / unrouted frames).
            expert_lens_cpu: (B, num_experts) precomputed Python list of
                             frame counts, used to skip empty experts
                             without a GPU→CPU sync. If None, all experts
                             are attempted (uses mask.any() instead).
            T:               sequence length (== x.shape[1]).
        """
        # Shared attention on the full sequence (h=1).
        normed = self.norm1(x)
        attn_out = self.attn(normed, 1, T)
        x = x + self.drop_path(self.ls1(attn_out.to(x.dtype)))

        # MLP branch: routed + shared, summed then LayerScaled together.
        normed = self.norm2(x)
        routed_out = self._routed_mlp(normed, expert_ids, expert_lens_cpu)
        shared_out = self.shared_mlp(normed)
        x = x + self.drop_path(self.ls2((routed_out + shared_out).to(x.dtype)))
        return x

    def _routed_mlp(self, x: Tensor, expert_ids: Tensor,
                    expert_lens_cpu: list | None) -> Tensor:
        """Apply per-frame routed MLPs via a single gather/scatter.

        Frames are grouped by expert (one permutation), each expert runs on
        its contiguous slice, and the results are scattered back once. This
        replaces the old per-expert boolean-mask loop
        (``out[mask] = expert(x[mask])``): that touched the full (N, dim)
        tensor once per expert and, worse, its boolean-mask-assignment
        backward ran a full-size masked_fill per expert — ~27 of them per
        layer dominated the whole training step (profiled: backward was 90%
        of step-time, almost all add_/fill_/copy_). The expert MLP is
        pointwise across frames, so grouping is exact regardless of order.

        Frames with expert_id outside [0, num_experts) (blank / unrouted)
        get zero contribution — the shared MLP still runs on them upstream.
        """
        B, T, D = x.shape
        N = B * T
        xf = x.reshape(N, D)
        ef = expert_ids.reshape(N)
        out = torch.zeros(N, D, dtype=x.dtype, device=x.device)

        # Positions of routed frames, grouped into contiguous per-expert
        # runs. With CPU counts available, M is known without touching the
        # GPU: sort all N keys (invalid frames get key=num_experts, landing
        # past position M) instead of nonzero+argsort — nonzero forces a
        # GPU→CPU sync per call, one per MoE layer per step.
        valid = (ef >= 0) & (ef < self.num_experts)
        if expert_lens_cpu is not None:
            counts = [sum(expert_lens_cpu[b][e] for b in range(B))
                      for e in range(self.num_experts)]
            M = sum(counts)
            if M == 0:
                return out.reshape(B, T, D)
            key = torch.where(valid, ef, ef.new_full((), self.num_experts))
            order = torch.argsort(key, stable=True)
            pos = order[:M]
        else:
            pos = valid.nonzero(as_tuple=True)[0]  # (M,) flat positions
            if pos.numel() == 0:
                return out.reshape(B, T, D)
            order = torch.argsort(ef.index_select(0, pos))
            pos = pos.index_select(0, order)
            counts = torch.bincount(
                ef[valid], minlength=self.num_experts).tolist()
        xg = xf.index_select(0, pos)  # (M, D) gathered, grouped by expert

        chunks = []
        s = 0
        for e in range(self.num_experts):
            k = counts[e]
            if k == 0:
                continue
            chunks.append(self.routed_mlps[e](xg[s:s + k]).to(out.dtype))
            s += k
        yg = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        out = out.index_copy(0, pos, yg)  # one scatter; light backward
        return out.reshape(B, T, D)


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
    """Two-conv plain stem: (B, 3, 32, W) → (B, 96, 16, W/2).

    Spatial feature extraction in 2 convs (channels 3 → 64 → 96):
        Conv 1: 3  → 64,  stride (2, 2), 3×3 kernel  →  H 32→16, W→W/2
        Conv 2: 64 → 96,  stride (1, 1), 3×3 kernel  →  H 16, W/2 (no downsample)

    3×3 kernels keep RF tight for clean routing boundaries. The bulk of
    the vertical / horizontal downsampling happens later via BlurPool
    inside the ConvNeXt stage — the stem just prepares 96-ch features at
    the base spatial resolution ConvA operates on.
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 64, out_ch: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=(2, 2),
                      padding=1, bias=False),
            nn.GroupNorm(1, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=(1, 1),
                      padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt v1 block: dw 7×7 conv + LayerNorm + pw MLP (GELU) + residual.

    Operates in NCHW.  Depthwise 7×7 gives ~7-px spatial context per block
    at the current spatial resolution; the MLP mixes channels.  LayerScale +
    DropPath on the residual branch — the block starts near-identity.
    """

    def __init__(self, dim: int, mlp_ratio: int = 4,
                 drop_path: float = 0.0, layer_scale_init: float = 1e-4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * mlp_ratio
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, dim)
        self.ls = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        residual = x
        x = self.dwconv(x)
        # NCHW → NHWC for the LayerNorm + pw MLP (all channel-last ops)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.ls(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class BlurPool2d(nn.Module):
    """Anti-aliased strided downsampling (Zhang 2019) + optional 1×1 proj.

    Depthwise 3-tap [1,2,1] low-pass filter applied *before* the stride,
    then an optional 1×1 conv when in_ch != out_ch.  Strides can differ
    per axis, e.g. stride=(2,1) downsamples H by 2 while keeping W the
    same; stride=(2,2) downsamples both.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: tuple[int, int] = (2, 2)):
        super().__init__()
        self.in_ch = in_ch
        self.stride = stride
        # 3-tap triangle filter, normalized to sum to 1.
        a = torch.tensor([1., 2., 1.])
        kernel = a[:, None] * a[None, :]
        kernel = kernel / kernel.sum()
        # Depthwise: (in_ch, 1, 3, 3).
        filt = kernel.reshape(1, 1, 3, 3).repeat(in_ch, 1, 1, 1)
        self.register_buffer("filt", filt, persistent=False)
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # F.conv2d wants the filter in x's dtype for autocast paths.
        filt = self.filt.to(x.dtype)
        x = F.conv2d(x, filt, stride=self.stride, padding=1, groups=self.in_ch)
        return self.proj(x)


class SWABlock(nn.Module):
    """Windowed attention + MLP block — one set of weights for all frames.

    Used in the shared trunk (SWA-C, SWA-D) and for the LID-1 attention
    branch. The routed-MLP variant lives in `MoELayer`.
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


