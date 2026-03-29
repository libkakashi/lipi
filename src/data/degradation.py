"""
Synthetic Degradation Pipeline.

Transforms clean word crops into realistic scanned-document appearances.
Applied to PDF-extracted crops (which have perfect labels) to create
training data that matches real-world scanning conditions.

Degradation types:
  - JPEG compression artifacts
  - Gaussian blur and motion blur
  - Salt-and-pepper noise
  - Perspective warp (camera angle)
  - Shadow gradients and uneven lighting
  - Paper texture overlay
  - Low resolution downsampling
  - Ink bleed / smudging
  - Slight rotation
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from src.data.augmentation import (
    jpeg_compress,
    gaussian_blur,
    salt_pepper_noise,
    brightness_jitter,
    contrast_jitter,
    rotation,
    perspective_warp,
    shadow_gradient,
    downsample_upsample,
)


def motion_blur(img: Image.Image, kernel_size: int = 0) -> Image.Image:
    """Apply horizontal motion blur (simulates scanner movement)."""
    if kernel_size == 0:
        kernel_size = random.choice([3, 5, 7])

    # Horizontal motion blur kernel
    kernel = [0] * (kernel_size * kernel_size)
    mid = kernel_size // 2
    for i in range(kernel_size):
        kernel[mid * kernel_size + i] = 1.0 / kernel_size

    return img.filter(ImageFilter.Kernel(
        size=(kernel_size, kernel_size),
        kernel=kernel,
        scale=1,
        offset=0,
    ))


def ink_bleed(img: Image.Image, radius: float = 0) -> Image.Image:
    """Simulate ink bleeding/spreading on paper."""
    if radius == 0:
        radius = random.uniform(0.3, 1.0)

    arr = np.array(img, dtype=np.float32)
    # Darken near-black pixels and spread them
    dark_mask = arr.mean(axis=2) < 128
    if dark_mask.any():
        blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
        # Blend: darker of original and blurred
        result = np.minimum(arr, blurred)
        return Image.fromarray(result.astype(np.uint8))
    return img


def paper_texture(img: Image.Image, intensity: float = 0) -> Image.Image:
    """Add paper texture noise (subtle uneven background)."""
    if intensity == 0:
        intensity = random.uniform(5, 20)

    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def uneven_lighting(img: Image.Image) -> Image.Image:
    """Apply non-uniform lighting (simulates desk lamp / window light)."""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    # Random gradient direction and intensity
    cx = random.uniform(0.2, 0.8) * w
    cy = random.uniform(0.2, 0.8) * h
    max_dist = np.sqrt(w**2 + h**2)

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    falloff = 1.0 - 0.3 * (dist / max_dist)
    falloff = falloff[:, :, np.newaxis]

    arr = arr * falloff
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# Degradation presets
LIGHT_DEGRADATION = [
    (jpeg_compress, 0.3),
    (gaussian_blur, 0.2),
    (brightness_jitter, 0.3),
    (contrast_jitter, 0.3),
    (rotation, 0.2),
]

MEDIUM_DEGRADATION = [
    (jpeg_compress, 0.5),
    (gaussian_blur, 0.4),
    (salt_pepper_noise, 0.3),
    (brightness_jitter, 0.4),
    (contrast_jitter, 0.4),
    (rotation, 0.3),
    (paper_texture, 0.3),
    (shadow_gradient, 0.2),
]

HEAVY_DEGRADATION = [
    (jpeg_compress, 0.7),
    (gaussian_blur, 0.5),
    (motion_blur, 0.3),
    (salt_pepper_noise, 0.4),
    (brightness_jitter, 0.5),
    (contrast_jitter, 0.5),
    (rotation, 0.4),
    (perspective_warp, 0.3),
    (paper_texture, 0.4),
    (shadow_gradient, 0.3),
    (ink_bleed, 0.3),
    (uneven_lighting, 0.2),
    (downsample_upsample, 0.3),
]


def apply_degradation(
    img: Image.Image,
    preset: str = "medium",
) -> Image.Image:
    """Apply a preset degradation pipeline.

    Args:
        img: Clean PIL Image.
        preset: One of 'light', 'medium', 'heavy'.

    Returns:
        Degraded PIL Image.
    """
    if preset == "light":
        ops = LIGHT_DEGRADATION
    elif preset == "medium":
        ops = MEDIUM_DEGRADATION
    elif preset == "heavy":
        ops = HEAVY_DEGRADATION
    else:
        raise ValueError(f"Unknown preset: {preset}")

    for op, prob in ops:
        if random.random() < prob:
            img = op(img)

    return img
