"""
Shifted Window Attention and Global Self-Attention blocks.

Stage 1 & 2 use Shifted Window Attention (SWA) with 2D RoPE.
Stage 3 uses Global Self-Attention with 1D RoPE (after height collapse).

All ops are standard PyTorch — no custom CUDA kernels.
Designed for clean ONNX opset 17 export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_util
from torch import Tensor

from src.model.rope import RoPE2D, RoPE1D


class ShiftedWindowAttention(nn.Module):
    """Multi-head attention within shifted windows for 2D feature maps.

    Implements the Swin-style window partition + cyclic shift approach,
    with RoPE-2D positional encoding instead of learned relative bias.

    The window_h and window_w define the local attention window size.
    On alternating blocks, features are cyclically shifted by half the
    window size so that windows overlap different spatial regions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_h: int = 4,
        window_w: int = 4,
        shift: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_h = window_h
        self.window_w = window_w
        self.shift = shift
        self.shift_h = window_h // 2 if shift else 0
        self.shift_w = window_w // 2 if shift else 0

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rope = RoPE2D(dim=self.head_dim)

    def _partition_windows(self, x: Tensor, h: int, w: int) -> tuple[Tensor, int, int]:
        """Partition feature map into non-overlapping windows.

        Args:
            x: (B, h*w, C)
            h, w: spatial dimensions.

        Returns:
            windows: (B * num_windows, window_h * window_w, C)
            num_win_h, num_win_w: number of windows in each dim.
        """
        B, _, C = x.shape
        x = x.reshape(B, h, w, C)

        # Always pad so h and w are divisible by window size.
        # Using unconditional pad (even when pad=0) ensures a single code path
        # for torch.export/ONNX tracing — conditional padding creates trace-time
        # constants that break on different input sizes at runtime.
        pad_h = (self.window_h - h % self.window_h) % self.window_h
        pad_w = (self.window_w - w % self.window_w) % self.window_w
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w

        num_win_h = hp // self.window_h
        num_win_w = wp // self.window_w

        # Reshape into windows: (B, nH, wH, nW, wW, C)
        x = x.reshape(B, num_win_h, self.window_h, num_win_w, self.window_w, C)
        # -> (B, nH, nW, wH, wW, C) -> (B*nH*nW, wH*wW, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_h * self.window_w, C)
        return x, num_win_h, num_win_w

    def _merge_windows(
        self, x: Tensor, num_win_h: int, num_win_w: int, B: int, h: int, w: int
    ) -> Tensor:
        """Merge windows back into feature map.

        Args:
            x: (B * num_windows, window_h * window_w, C)

        Returns:
            (B, h*w, C) — cropped back to original spatial size.
        """
        C = x.shape[-1]
        # (B*nH*nW, wH*wW, C) -> (B, nH, nW, wH, wW, C)
        x = x.reshape(B, num_win_h, num_win_w, self.window_h, self.window_w, C)
        # -> (B, nH, wH, nW, wW, C) -> (B, H_padded, W_padded, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            B, num_win_h * self.window_h, num_win_w * self.window_w, C
        )
        # Crop to original size
        x = x[:, :h, :w, :].reshape(B, h * w, C)
        return x

    def _build_shift_mask(
        self, h: int, w: int, device: torch.device
    ) -> Tensor | None:
        """Build attention mask for shifted windows.

        In shifted mode, tokens from different original regions share a window.
        The mask prevents attention across region boundaries.

        Returns: (num_windows, window_size, window_size) or None if no shift.
        """
        if not self.shift:
            return None

        hp = h + (self.window_h - h % self.window_h) % self.window_h
        wp = w + (self.window_w - w % self.window_w) % self.window_w

        # Create region index map
        img_mask = torch.zeros(1, hp, wp, 1, device=device)
        h_slices = (
            slice(0, -self.window_h),
            slice(-self.window_h, -self.shift_h),
            slice(-self.shift_h, None),
        )
        w_slices = (
            slice(0, -self.window_w),
            slice(-self.window_w, -self.shift_w),
            slice(-self.shift_w, None),
        )
        region_id = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = region_id
                region_id += 1

        # Partition mask into windows
        num_win_h = hp // self.window_h
        num_win_w = wp // self.window_w
        mask_windows = img_mask.reshape(
            1, num_win_h, self.window_h, num_win_w, self.window_w, 1
        )
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).reshape(
            num_win_h * num_win_w, self.window_h * self.window_w
        )

        # (num_windows, ws, 1) - (num_windows, 1, ws) -> (num_windows, ws, ws)
        attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )
        return attn_mask

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        Args:
            x: (B, h*w, C) — flattened 2D feature map.
            h, w: spatial dimensions.

        Returns: (B, h*w, C)
        """
        B, N, C = x.shape

        # Cyclic shift
        if self.shift:
            x_shifted = x.reshape(B, h, w, C)
            x_shifted = torch.roll(x_shifted, shifts=(-self.shift_h, -self.shift_w), dims=(1, 2))
            x_shifted = x_shifted.reshape(B, h * w, C)
        else:
            x_shifted = x

        # Partition into windows
        x_win, nH, nW = self._partition_windows(x_shifted, h, w)
        # x_win: (B*num_windows, window_size, C)

        # QKV projection
        qkv = self.qkv(x_win)  # (B*nwin, ws, 3*C)
        qkv = qkv.reshape(-1, self.window_h * self.window_w, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nwin, heads, ws, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply RoPE-2D within each window
        q = self.rope(q, h=self.window_h, w=self.window_w)
        k = self.rope(k, h=self.window_h, w=self.window_w)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*nwin, heads, ws, ws)

        # Apply shift mask if needed
        mask = self._build_shift_mask(h, w, x.device)
        if mask is not None:
            num_windows = nH * nW
            # mask: (num_windows, ws, ws) -> broadcast over B and heads
            attn = attn.reshape(B, num_windows, self.num_heads, -1, attn.shape[-1])
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.reshape(-1, self.num_heads, attn.shape[-2], attn.shape[-1])

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B*nwin, heads, ws, head_dim)

        # Merge heads
        out = out.transpose(1, 2).reshape(-1, self.window_h * self.window_w, C)

        # Output projection
        out = self.proj(out)

        # Merge windows back
        out = self._merge_windows(out, nH, nW, B, h, w)

        # Reverse cyclic shift
        if self.shift:
            out = out.reshape(B, h, w, C)
            out = torch.roll(out, shifts=(self.shift_h, self.shift_w), dims=(1, 2))
            out = out.reshape(B, h * w, C)

        return out


class GlobalAttention(nn.Module):
    """Standard multi-head self-attention for 1D sequences.

    Used in Stage 3 after height collapse. Applies RoPE-1D.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rope = RoPE1D(dim=self.head_dim)

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        """
        Args:
            x: (B, T, C) — 1D sequence features.
            seq_len: T, the sequence length.

        Returns: (B, T, C)
        """
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply RoPE-1D
        q = self.rope(q, seq_len=T)
        k = self.rope(k, seq_len=T)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, heads, T, head_dim)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    """Two-layer MLP with GELU activation.

    Used in every attention block (both SWA and Global).
    """

    def __init__(self, dim: int, mlp_ratio: int = 3):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SWABlock(nn.Module):
    """Shifted Window Attention block with pre-norm residual.

    Pre-norm: LayerNorm before attention and MLP (smoother weight distributions,
    better for quantization).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_h: int = 4,
        window_w: int = 4,
        shift: bool = False,
        mlp_ratio: int = 3,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttention(dim, num_heads, window_h, window_w, shift)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        Args:
            x: (B, h*w, C)
            h, w: spatial dimensions.

        Returns: (B, h*w, C)
        """
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x))
        return x


class SWABlockMoE(nn.Module):
    """SWA block with shared + expert MLPs.

    Shared attention: spatial relationships are universal across scripts.
    Shared MLP: universal characters (digits, punctuation, common symbols).
    Expert MLPs: script-specific interpretation (one per group).

    All three are additive residuals on the same features.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        window_h: int = 4,
        window_w: int = 4,
        shift: bool = False,
        mlp_ratio: int = 3,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttention(dim, num_heads, window_h, window_w, shift)
        self.norm2 = nn.LayerNorm(dim)
        self.shared_mlp = MLP(dim, mlp_ratio=1)  # small — just digits/punctuation
        self.norm3 = nn.LayerNorm(dim)
        self.expert_mlps = nn.ModuleList([MLP(dim, mlp_ratio) for _ in range(num_groups)])

    def forward(self, x: Tensor, h: int, w: int, group_ids: Tensor) -> Tensor:
        """
        Args:
            x: (B, h*w, C)
            h, w: spatial dimensions.
            group_ids: (B,) integer group ID per sample.

        Returns: (B, h*w, C)
        """
        # Shared attention
        x = x + self.attn(self.norm1(x), h, w)

        # Shared MLP (universal characters — digits, punctuation)
        x = x + self.shared_mlp(self.norm2(x))

        # Expert MLP (script-specific interpretation)
        normed = self.norm3(x)
        expert_out = torch.zeros_like(x)
        for g in range(self.num_groups):
            mask = (group_ids == g)
            if mask.any():
                result = self.expert_mlps[g](normed[mask])
                expert_out[mask] = result.to(expert_out.dtype)
        x = x + expert_out
        return x


class FullyExpertSWABlock(nn.Module):
    """Fully expert SWA block — per-group attention + per-group MLP.

    Every component is specialized per group. No shared params except LayerNorms.
    Optimized: precomputes group indices once, sorts batch by group for
    contiguous memory access, scatters results back.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        window_h: int = 4,
        window_w: int = 4,
        shift: bool = False,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.norm1 = nn.LayerNorm(dim)
        self.expert_attns = nn.ModuleList([
            ShiftedWindowAttention(dim, num_heads, window_h, window_w, shift)
            for _ in range(num_groups)
        ])
        self.norm2 = nn.LayerNorm(dim)
        self.expert_mlps = nn.ModuleList([
            MLP(dim, mlp_ratio) for _ in range(num_groups)
        ])

    def _get_group_boundaries(self, group_ids: Tensor, B: int) -> list[tuple[int, int]]:
        """Sort by group and return (start, end) slices. Cached-friendly."""
        sorted_idx = group_ids.argsort()
        gids_sorted = group_ids[sorted_idx]

        bounds = []
        start = 0
        for g in range(self.num_groups):
            if start >= B:
                bounds.append((B, B))
                continue
            # Find end of this group
            end = start
            while end < B and gids_sorted[end] == g:
                end += 1
            bounds.append((start, end))
            start = end

        return sorted_idx, bounds

    def forward(self, x: Tensor, h: int, w: int, group_ids: Tensor) -> Tensor:
        B = x.shape[0]

        sorted_idx, bounds = self._get_group_boundaries(group_ids, B)
        x_sorted = x[sorted_idx]

        # Expert attention with gradient checkpointing
        normed = self.norm1(x_sorted)
        attn_out = torch.empty_like(x_sorted)
        for g in range(self.num_groups):
            s, e = bounds[g]
            if s < e:
                if self.training and torch.is_grad_enabled():
                    result = ckpt_util.checkpoint(
                        self.expert_attns[g], normed[s:e], h, w, use_reentrant=False)
                else:
                    result = self.expert_attns[g](normed[s:e], h, w)
                attn_out[s:e] = result.to(attn_out.dtype)
        x_sorted = x_sorted + attn_out

        # Expert MLP with gradient checkpointing
        normed = self.norm2(x_sorted)
        mlp_out = torch.empty_like(x_sorted)
        for g in range(self.num_groups):
            s, e = bounds[g]
            if s < e:
                if self.training and torch.is_grad_enabled():
                    result = ckpt_util.checkpoint(
                        self.expert_mlps[g], normed[s:e], use_reentrant=False)
                else:
                    result = self.expert_mlps[g](normed[s:e])
                mlp_out[s:e] = result.to(mlp_out.dtype)
        x_sorted = x_sorted + mlp_out

        return x_sorted[sorted_idx.argsort()]


class GlobalBlock(nn.Module):
    """Global self-attention block with pre-norm residual.

    Used in Stage 3 after height collapse.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GlobalAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        """
        Args:
            x: (B, T, C)
            seq_len: T.

        Returns: (B, T, C)
        """
        x = x + self.attn(self.norm1(x), seq_len)
        x = x + self.mlp(self.norm2(x))
        return x


class ParallelBlock(nn.Module):
    """VIPTR-style parallel local + global attention block.

    Runs SWA (local) and Global attention in parallel on the same input,
    then merges their outputs. Each frame gets both fine-grained local
    detail AND full-sequence context in a single block.

    2x compute per block vs SWA-only, but richer features.
    Use for 2D stages (before height collapse).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_h: int = 4,
        window_w: int = 4,
        shift: bool = False,
        mlp_ratio: int = 3,
    ):
        super().__init__()
        # Local branch (SWA)
        self.norm_local = nn.LayerNorm(dim)
        self.local_attn = ShiftedWindowAttention(dim, num_heads, window_h, window_w, shift)

        # Global branch
        self.norm_global = nn.LayerNorm(dim)
        self.global_attn = GlobalAttention(dim, num_heads)

        # Merge: project concatenated local+global back to dim
        self.merge = nn.Linear(dim * 2, dim)

        # Shared MLP after merge
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        Args:
            x: (B, h*w, C)
            h, w: spatial dimensions.

        Returns: (B, h*w, C)
        """
        # Parallel attention
        local_out = self.local_attn(self.norm_local(x), h, w)
        global_out = self.global_attn(self.norm_global(x), h * w)

        # Merge local + global
        merged = torch.cat([local_out, global_out], dim=-1)  # (B, h*w, 2*C)
        merged = self.merge(merged)  # (B, h*w, C)

        x = x + merged
        x = x + self.mlp(self.norm2(x))
        return x
