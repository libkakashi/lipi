"""
Image rendering and validation for OCR training data.

Handles:
- Word rendering with random ink/paper colors
- Single-character rendering with cmap-based font filtering
- Emoji rendering (synthetic colored blocks)
- Image validation (blank/faint detection)
- Image sizing (resize/pad to target dimensions)
"""

import random

import numpy as np
from PIL import Image

from src.data.text_renderer import render_text, font_has_codepoint


# ---------------------------------------------------------------------------
# Color generation
# ---------------------------------------------------------------------------

def random_bg_color() -> tuple[int, int, int]:
    """Random background color — biased toward light/paper tones."""
    r = random.random()
    if r < 0.5:
        # White/cream paper
        v = random.randint(220, 255)
        return (v, v - random.randint(0, 10), v - random.randint(0, 15))
    elif r < 0.75:
        # Colored paper
        return (random.randint(180, 240), random.randint(180, 240), random.randint(180, 240))
    else:
        # Wild colors (signs, banners, neon)
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def random_ink_color() -> tuple[int, int, int]:
    """Random ink color — biased toward dark tones."""
    r = random.random()
    if r < 0.6:
        # Black/dark ink
        v = random.randint(0, 40)
        return (v, v, v)
    elif r < 0.85:
        # Dark colored ink
        return (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))
    else:
        # Wild colors
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def image_has_ink(img: Image.Image, min_ink_pixels: int = 10) -> bool:
    """Check if a rendered image has visible content (not blank/faint).

    Only checks for blank or nearly-invisible renders. Tofu detection is
    handled upstream via font_has_codepoint (cmap check), which is definitive
    and avoids false positives on box-shaped characters like 口, ㅁ, ם, O.
    """
    arr = np.array(img)
    if arr.ndim == 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr.astype(float)

    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    bg = np.median(corners)

    ink_pixels = (np.abs(gray - bg) > 30).sum()
    return ink_pixels >= min_ink_pixels


# ---------------------------------------------------------------------------
# Image sizing
# ---------------------------------------------------------------------------

def resize_or_pad(img: Image.Image, height: int, max_width: int) -> Image.Image:
    """Cap width at max_width, preserve natural aspect ratio. No padding."""
    if img.width > max_width:
        img = img.resize((max_width, height), Image.BILINEAR)
    return img


# ---------------------------------------------------------------------------
# Word rendering
# ---------------------------------------------------------------------------

def font_covers_text(font_path: str, text: str) -> bool:
    """Check that a font has cmap entries for ALL characters in text.

    Prevents partial renders where some chars display and others show tofu.
    Skips shared chars (digits, punctuation, space) since all fonts have those.
    """
    for ch in text:
        cp = ord(ch)
        # Skip ASCII shared chars — every font has these
        if 0x0020 <= cp <= 0x007E:
            continue
        if not font_has_codepoint(font_path, ch):
            return False
    return True


def render_word(text: str, font_path: str, height: int = 32,
                clean: bool = False) -> Image.Image | None:
    """Render a word. Clean mode uses white bg + black ink."""
    if clean:
        bg = (255, 255, 255)
        ink = (0, 0, 0)
    else:
        for _ in range(5):
            bg = random_bg_color()
            ink = random_ink_color()
            bg_lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            ink_lum = 0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2]
            if abs(bg_lum - ink_lum) > 60:
                break

    font_size = random.randint(18, 26)
    return render_text(
        text, font_path, font_size, ink=ink, bg=bg, height=height,
        pad_x=random.randint(2, 8), pad_y=random.randint(2, 6),
    )


# ---------------------------------------------------------------------------
# Emoji rendering
# ---------------------------------------------------------------------------

def render_emoji(height: int = 32, max_width: int = 192) -> Image.Image:
    """Render a synthetic emoji-like colorful block image.

    Emojis are visually distinct: bright colors, round shapes, high contrast.
    Simulates with colored circles/rectangles on white backgrounds.
    """
    from PIL import ImageDraw

    w = random.randint(height, min(height * 3, max_width))
    img = Image.new("RGB", (w, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Random bright shapes
    n_shapes = random.randint(1, 4)
    for _ in range(n_shapes):
        color = (random.randint(100, 255), random.randint(50, 255), random.randint(50, 255))
        x1 = random.randint(0, w - 4)
        y1 = random.randint(0, height - 4)
        x2 = random.randint(x1 + 4, min(x1 + height, w))
        y2 = random.randint(y1 + 4, min(y1 + height, height))
        if random.random() < 0.5:
            draw.ellipse([x1, y1, x2, y2], fill=color)
        else:
            draw.rectangle([x1, y1, x2, y2], fill=color)

    return img


# ---------------------------------------------------------------------------
# Font filtering for single-char rendering
# ---------------------------------------------------------------------------

def filter_fonts_by_cmap(
    fonts: list[str],
    chars: list[str],
) -> dict[str, list[str]]:
    """Pre-filter fonts per character using cmap (definitive tofu prevention).

    Returns:
        char_fonts: {char: [weighted_font_paths]} for chars that have valid fonts.
        Characters with no valid fonts are omitted.
    """
    unique_fonts = list(dict.fromkeys(fonts))

    # Build cmap lookup: font -> set of renderable chars
    cmap_ok: dict[str, set[str]] = {}
    for f in unique_fonts:
        cmap_ok[f] = set()
        for ch in chars:
            if font_has_codepoint(f, ch):
                cmap_ok[f].add(ch)

    # Build per-char font list, preserving original weighting
    char_fonts: dict[str, list[str]] = {}
    for ch in chars:
        valid = [f for f in fonts if ch in cmap_ok.get(f, set())]
        if valid:
            char_fonts[ch] = valid

    return char_fonts
