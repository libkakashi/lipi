"""
Color space conversion for model input.

Pipeline: RGB → L+a (2ch) → stem directly

The L+a encoding separates luminance from chrominance, which is what
matters for OCR — text is defined by contrast. The stem's first conv
learns to combine the 2 channels optimally.
"""

import numpy as np
import torch
from PIL import Image


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
    # Quantize to uint8 for 4× smaller shards; dequantize with float() / 255.0
    L_u8 = np.clip(L * 255, 0, 255).astype(np.uint8)
    a_u8 = np.clip(a * 255, 0, 255).astype(np.uint8)
    return torch.tensor(np.stack([L_u8, a_u8], axis=0), dtype=torch.uint8)
