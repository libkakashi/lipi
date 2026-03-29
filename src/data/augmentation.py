"""
Training Augmentations for OCR.

RandAugment-style augmentation pipeline with OCR-specific transforms.
All transforms preserve text readability while adding visual diversity.
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from typing import Callable


def jpeg_compress(img: Image.Image, quality_range: tuple = (30, 90)) -> Image.Image:
    """Apply JPEG compression artifacts."""
    import io
    quality = random.randint(*quality_range)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(img: Image.Image, sigma_range: tuple = (0.5, 2.0)) -> Image.Image:
    """Apply Gaussian blur."""
    sigma = random.uniform(*sigma_range)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def salt_pepper_noise(img: Image.Image, amount: float = 0.02) -> Image.Image:
    """Add salt and pepper noise."""
    arr = np.array(img)
    # Salt
    num_salt = int(amount * arr.size / 2)
    coords = tuple(np.random.randint(0, d, num_salt) for d in arr.shape[:2])
    arr[coords[0], coords[1]] = 255
    # Pepper
    coords = tuple(np.random.randint(0, d, num_salt) for d in arr.shape[:2])
    arr[coords[0], coords[1]] = 0
    return Image.fromarray(arr)


def brightness_jitter(img: Image.Image, factor_range: tuple = (0.7, 1.3)) -> Image.Image:
    """Adjust brightness."""
    factor = random.uniform(*factor_range)
    return ImageEnhance.Brightness(img).enhance(factor)


def contrast_jitter(img: Image.Image, factor_range: tuple = (0.7, 1.3)) -> Image.Image:
    """Adjust contrast."""
    factor = random.uniform(*factor_range)
    return ImageEnhance.Contrast(img).enhance(factor)


def rotation(img: Image.Image, max_angle: float = 3.0) -> Image.Image:
    """Slight rotation."""
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(255, 255, 255))


def perspective_warp(img: Image.Image, strength: float = 0.05) -> Image.Image:
    """Simulate camera angle perspective distortion."""
    w, h = img.size
    s = strength

    # Random perspective corners
    tl = (random.uniform(0, s * w), random.uniform(0, s * h))
    tr = (w - random.uniform(0, s * w), random.uniform(0, s * h))
    br = (w - random.uniform(0, s * w), h - random.uniform(0, s * h))
    bl = (random.uniform(0, s * w), h - random.uniform(0, s * h))

    coeffs = _find_perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [tl, tr, br, bl],
    )
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)


def _find_perspective_coeffs(src, dst):
    """Compute perspective transform coefficients."""
    import numpy as np
    matrix = []
    for s, d in zip(src, dst):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([s for pair in src for s in pair], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())


def shadow_gradient(img: Image.Image, direction: str = "random") -> Image.Image:
    """Apply uneven lighting / shadow gradient."""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    if direction == "random":
        direction = random.choice(["left", "right", "top", "bottom"])

    gradient = np.ones((h, w), dtype=np.float32)
    strength = random.uniform(0.3, 0.7)

    if direction == "left":
        gradient *= np.linspace(strength, 1.0, w)[np.newaxis, :]
    elif direction == "right":
        gradient *= np.linspace(1.0, strength, w)[np.newaxis, :]
    elif direction == "top":
        gradient *= np.linspace(strength, 1.0, h)[:, np.newaxis]
    elif direction == "bottom":
        gradient *= np.linspace(1.0, strength, h)[:, np.newaxis]

    arr = arr * gradient[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def downsample_upsample(img: Image.Image, scale_range: tuple = (0.5, 0.8)) -> Image.Image:
    """Simulate low resolution by downsampling then upsampling."""
    w, h = img.size
    scale = random.uniform(*scale_range)
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def invert(img: Image.Image) -> Image.Image:
    """Invert colors (simulate white-on-black or negative)."""
    return ImageOps.invert(img)


# All available augmentation transforms
AUGMENT_OPS: list[Callable] = [
    jpeg_compress,
    gaussian_blur,
    salt_pepper_noise,
    brightness_jitter,
    contrast_jitter,
    rotation,
    perspective_warp,
    shadow_gradient,
    downsample_upsample,
]


class RandAugmentOCR:
    """RandAugment-style augmentation for OCR training.

    Randomly applies N transforms from the pool, each with
    random intensity within its configured range.
    """

    def __init__(self, n_ops: int = 2, p: float = 0.5):
        """
        Args:
            n_ops: Number of augmentation ops to apply per image.
            p: Probability of applying augmentation at all.
        """
        self.n_ops = n_ops
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations.

        Args:
            img: PIL Image (RGB).

        Returns:
            Augmented PIL Image.
        """
        if random.random() > self.p:
            return img

        ops = random.sample(AUGMENT_OPS, min(self.n_ops, len(AUGMENT_OPS)))
        for op in ops:
            img = op(img)

        return img
