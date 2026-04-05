"""
Color space conversion for model input.

Pipeline: RGB → fixed L+a (2ch) → learned correction (2→1ch) → stem

The fixed L+a gives a proven-good encoding. The learned correction
compresses it to 1 channel, learning the optimal way to combine
luminance and chrominance for OCR. Zero-initialized so it starts
as simple averaging of L and a.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

INPUT_CHANNELS = 1  # stem sees 1 channel


def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    return np.where(arr > 0.04045, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)


def rgb_to_input(img: Image.Image) -> torch.Tensor:
    """Convert RGB PIL image to (2, H, W) tensor: fixed L + a channels."""
    arr = np.array(img, dtype=np.float32) / 255.0
    linear = _srgb_to_linear(arr)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b

    xn, yn = 0.95047, 1.0
    def f(t):
        return np.where(t > 0.008856, t ** (1 / 3), 7.787 * t + 16 / 116)

    fx, fy = f(x / xn), f(y / yn)
    L = (116 * fy - 16) / 100.0
    a = (500 * (fx - fy) + 128) / 255.0
    # Quantize to uint8 for 4× smaller shards; dequantize with dequantize_input()
    L_u8 = np.clip(L * 255, 0, 255).astype(np.uint8)
    a_u8 = np.clip(a * 255, 0, 255).astype(np.uint8)
    return torch.tensor(np.stack([L_u8, a_u8], axis=0), dtype=torch.uint8)


class ColorProjection(nn.Module):
    """Learned 2→1 projection on top of fixed L+a.

    Takes L+a (2ch), learns optimal compression to 1 channel.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, H, W) → (B, 1, H, W). Accepts uint8 or float32 input."""
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.net(x)
