"""
Image preprocessing for model input.

Converts PIL RGB image to (3, H, W) uint8 tensor.
"""

import numpy as np
import torch
from PIL import Image


def rgb_to_input(img: Image.Image) -> torch.Tensor:
    """Convert RGB PIL image to (3, H, W) uint8 tensor."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr.transpose(2, 0, 1))
