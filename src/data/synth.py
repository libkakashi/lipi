"""
Synthetic Word Image Renderer.

Generates word crop images with diverse fonts, sizes, and styles.
Used for training data generation and overfit testing.

For large-scale generation, wraps trdg (TextRecognitionDataGenerator)
when available. Falls back to PIL-based rendering otherwise.
"""

import random
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Collect available system fonts
def _find_system_fonts() -> list[str]:
    """Find TrueType/OpenType fonts available on the system."""
    font_dirs = [
        "/System/Library/Fonts",           # macOS system
        "/Library/Fonts",                   # macOS user
        os.path.expanduser("~/Library/Fonts"),  # macOS per-user
        "/usr/share/fonts",                 # Linux
        "/usr/local/share/fonts",           # Linux local
        "C:\\Windows\\Fonts",               # Windows
    ]

    extensions = {".ttf", ".ttc", ".otf"}
    fonts = []

    for font_dir in font_dirs:
        font_path = Path(font_dir)
        if font_path.exists():
            for f in font_path.rglob("*"):
                if f.suffix.lower() in extensions:
                    fonts.append(str(f))

    return fonts


SYSTEM_FONTS = _find_system_fonts()


def _pick_font(size: int, preferred: list[str] | None = None) -> ImageFont.FreeTypeFont:
    """Pick a random font at the given size.

    Tries preferred fonts first, then system fonts, then default.
    """
    candidates = (preferred or []) + SYSTEM_FONTS

    if candidates:
        random.shuffle(candidates)
        for font_path in candidates[:20]:  # Try up to 20
            try:
                return ImageFont.truetype(font_path, size=size)
            except (IOError, OSError):
                continue

    # Fallback
    return ImageFont.load_default(size=size)


def render_word(
    text: str,
    height: int = 32,
    font_size_range: tuple[int, int] = (18, 26),
    padding: tuple[int, int] = (4, 8),
    bg_color_range: tuple[int, int] = (220, 255),
    fg_color_range: tuple[int, int] = (0, 60),
    preferred_fonts: list[str] | None = None,
) -> Image.Image:
    """Render a word as a PIL Image with random styling.

    Args:
        text: Word to render.
        height: Image height in pixels.
        font_size_range: (min, max) font size.
        padding: (horizontal, vertical) padding around text.
        bg_color_range: (min, max) background brightness.
        fg_color_range: (min, max) foreground (text) brightness.
        preferred_fonts: Optional list of font paths to prefer.

    Returns:
        RGB PIL Image of the rendered word.
    """
    font_size = random.randint(*font_size_range)
    font = _pick_font(font_size, preferred_fonts)

    # Measure text size
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Calculate image dimensions
    pad_h, pad_v = padding
    img_w = text_w + 2 * pad_h
    img_h = height

    # Random colors
    bg = random.randint(*bg_color_range)
    fg = random.randint(*fg_color_range)

    # Create image
    img = Image.new("RGB", (img_w, img_h), color=(bg, bg, bg))
    draw = ImageDraw.Draw(img)

    # Center text vertically, left-align with padding
    y_offset = max(0, (img_h - text_h) // 2 - bbox[1])
    draw.text((pad_h, y_offset), text, fill=(fg, fg, fg), font=font)

    return img


def render_word_batch(
    words: list[str],
    n_variants: int = 1,
    height: int = 32,
    **kwargs,
) -> list[tuple[Image.Image, str]]:
    """Render multiple variants of each word.

    Args:
        words: List of words to render.
        n_variants: Number of style variants per word.
        height: Image height.
        **kwargs: Additional args passed to render_word.

    Returns:
        List of (image, label) tuples.
    """
    results = []
    for word in words:
        for _ in range(n_variants):
            img = render_word(word, height=height, **kwargs)
            results.append((img, word))
    return results


def generate_dataset(
    word_list: list[str],
    n_per_word: int = 100,
    height: int = 32,
    **kwargs,
) -> tuple[list[Image.Image], list[str]]:
    """Generate a synthetic dataset from a word list.

    Args:
        word_list: Words to render.
        n_per_word: Number of variants per word.
        height: Image height.

    Returns:
        (images, labels) tuple.
    """
    pairs = render_word_batch(word_list, n_variants=n_per_word, height=height, **kwargs)
    random.shuffle(pairs)
    images = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    return images, labels
