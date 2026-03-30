"""
Color space conversion for model input.

Two modes controlled by MODE:
  "la"      — Fixed L+a from CIELab. 2 channels. No learned params.
  "learned" — RGB passed to a learned 3→1 conv. 1 channel. Learns optimal projection.

Change MODE here. INPUT_CHANNELS updates accordingly.
Stem and all scripts read INPUT_CHANNELS from this module.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ---- Change this to switch modes ----
MODE = "learned"  # "la" or "learned"
# --------------------------------------

if MODE == "la":
    INPUT_CHANNELS = 2
elif MODE == "learned":
    INPUT_CHANNELS = 1  # stem sees 1ch from the learned projection
else:
    raise ValueError(f"Unknown color mode: {MODE}")


def _srgb_to_linear(arr: np.ndarray) -> np.ndarray:
    return np.where(arr > 0.04045, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)


def rgb_to_input(img: Image.Image) -> torch.Tensor:
    """Convert RGB PIL image to model input tensor.

    "la" mode: returns (2, H, W) — fixed L + a.
    "learned" mode: returns (3, H, W) — raw RGB for the learned conv.
    """
    arr = np.array(img, dtype=np.float32) / 255.0

    if MODE == "la":
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
        return torch.tensor(np.stack([L, a], axis=0), dtype=torch.float32)

    elif MODE == "learned":
        return torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32)


class LearnedColorProjection(nn.Module):
    """Learned 3→1 nonlinear color projection.

    Sits before the stem. Takes RGB, outputs 1 channel.
    The stem receives this single learned channel.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 1, H, W)"""
        return self.net(x)
