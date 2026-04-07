"""
RoPE (Rotary Position Embedding) for 2D sequences.

RoPE-2D: Applied where spatial height > 1.

All ops are standard PyTorch (no custom CUDA) for clean ONNX export.
Frequency tables are computed inline (no dict caching) to avoid
side-effect issues with torch.export/ONNX tracing.
"""

import torch
import torch.nn as nn
from torch import Tensor



def build_freqs_2d(h: int, w: int, dim: int, theta: float = 10000.0) -> Tensor:
    """Compute RoPE frequency table for a 2D grid.

    Splits dim in half: first half encodes row position, second half encodes column.
    Returns: (h*w, dim//2, 2) — cos and sin components.
    """
    half = dim // 2
    freqs_h = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
    freqs_w = 1.0 / (theta ** (torch.arange(0, half, 2, dtype=torch.float32) / half))

    rows = torch.arange(h, dtype=torch.float32)
    cols = torch.arange(w, dtype=torch.float32)

    angles_h = torch.outer(rows, freqs_h)  # (h, half//2)
    angles_w = torch.outer(cols, freqs_w)  # (w, half//2)

    # Expand to full grid
    angles_h = angles_h.unsqueeze(1).expand(-1, w, -1)  # (h, w, half//2)
    angles_w = angles_w.unsqueeze(0).expand(h, -1, -1)  # (h, w, half//2)

    # Concatenate height and width frequencies
    angles = torch.cat([angles_h, angles_w], dim=-1)  # (h, w, dim//2)
    angles = angles.reshape(h * w, dim // 2)

    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (h*w, dim//2, 2)


def apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: (..., seq_len, dim) — last dim must be even.
        freqs: (seq_len, dim//2, 2) — cos/sin pairs.

    Returns: (..., seq_len, dim) with rotary embeddings applied.
    """
    orig_shape = x.shape
    x_pairs = x.reshape(*orig_shape[:-1], orig_shape[-1] // 2, 2)

    cos = freqs[..., 0]  # (seq_len, dim//2)
    sin = freqs[..., 1]  # (seq_len, dim//2)

    x0 = x_pairs[..., 0]  # (..., seq_len, dim//2)
    x1 = x_pairs[..., 1]  # (..., seq_len, dim//2)

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    out = torch.stack([out0, out1], dim=-1)  # (..., seq_len, dim//2, 2)
    return out.reshape(orig_shape)


class RoPE2D(nn.Module):
    """Rotary Position Embedding for 2D spatial feature maps.

    Computes frequency tables on the fly. For training, the cost is
    negligible compared to attention computation. For inference, the
    window sizes are fixed so the same freqs are recomputed each call
    (a few microseconds — irrelevant vs the ~8ms total inference).
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        """Apply 2D RoPE.

        Args:
            x: (B, num_heads, h*w, head_dim) or (B, h*w, dim)
            h, w: spatial dimensions of the feature map.

        Returns: same shape as x, with rotary embeddings applied.
        """
        freqs = build_freqs_2d(h, w, self.dim, self.theta).to(
            device=x.device, dtype=x.dtype
        )
        return apply_rope(x, freqs)
