"""
Fast text rendering using FreeType (C library).

Much faster than PIL's ImageFont for bulk rendering.
Supports all Unicode scripts via FreeType + any .ttf/.otf font.

Usage:
    from src.data.text_renderer import render_text
    img = render_text("Hello", "/path/to/font.ttf", height=32)
    # Returns PIL.Image.Image (RGB) or None on failure
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import freetype
    HAS_FREETYPE = True
except ImportError:
    HAS_FREETYPE = False

# Cache loaded faces to avoid reloading the same font
_face_cache: dict[str, "freetype.Face"] = {}


def _get_face(font_path: str, size: int) -> "freetype.Face":
    """Get or create a FreeType face, cached."""
    key = font_path
    if key not in _face_cache:
        _face_cache[key] = freetype.Face(font_path)
    face = _face_cache[key]
    face.set_pixel_sizes(0, size)
    return face


def render_text_freetype(text: str, font_path: str, font_size: int,
                         ink: tuple[int, ...], bg: tuple[int, ...],
                         height: int = 32, pad_x: int = 4, pad_y: int = 2) -> Image.Image | None:
    """Render text using FreeType. Returns RGB PIL Image or None."""
    try:
        face = _get_face(font_path, font_size)

        # First pass: measure total width and height bounds
        pen_x = 0
        min_y = 0
        max_y = 0
        positions = []

        for char in text:
            face.load_char(char, freetype.FT_LOAD_RENDER)
            glyph = face.glyph
            bitmap = glyph.bitmap

            if bitmap.width == 0 or bitmap.rows == 0:
                pen_x += glyph.advance.x >> 6
                continue

            x = pen_x + glyph.bitmap_left
            y = -glyph.bitmap_top

            positions.append((x, y, bitmap.width, bitmap.rows,
                            bytes(bitmap.buffer), bitmap.pitch))

            min_y = min(min_y, y)
            max_y = max(max_y, y + bitmap.rows)
            pen_x += glyph.advance.x >> 6

        if not positions or pen_x <= 0:
            return None

        text_w = pen_x
        text_h = max_y - min_y
        if text_w <= 0 or text_h <= 0:
            return None

        img_w = text_w + 2 * pad_x
        img_h = text_h + 2 * pad_y

        # Create RGB image with background color
        arr = np.full((img_h, img_w, 3), bg, dtype=np.uint8)

        # Render glyphs (vectorized — no Python pixel loops)
        ink_arr = np.array(ink, dtype=np.float32)
        for (x, y, w, h, buf, pitch) in positions:
            gx = x + pad_x
            gy = y - min_y + pad_y

            # Convert bitmap buffer to numpy array (single frombuffer call)
            if pitch == w:
                glyph_arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
            else:
                raw = np.frombuffer(buf, dtype=np.uint8).reshape(h, pitch)
                glyph_arr = raw[:, :w]

            # Clip to image bounds
            y0, y1 = max(0, gy), min(img_h, gy + h)
            x0, x1 = max(0, gx), min(img_w, gx + w)
            if y1 <= y0 or x1 <= x0:
                continue
            gy0, gx0 = y0 - gy, x0 - gx
            glyph_crop = glyph_arr[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]

            # Vectorized alpha blend (all 3 channels at once)
            alpha = glyph_crop.astype(np.float32)[..., np.newaxis] / 255.0
            region = arr[y0:y1, x0:x1].astype(np.float32)
            arr[y0:y1, x0:x1] = (region * (1 - alpha) + ink_arr * alpha).astype(np.uint8)

        # Resize to target height
        img = Image.fromarray(arr)
        scale = height / img_h
        new_w = max(4, int(img_w * scale))
        return img.resize((new_w, height), Image.BILINEAR)

    except Exception:
        return None


def render_text_pil(text: str, font_path: str, font_size: int,
                    ink: tuple[int, ...], bg: tuple[int, ...],
                    height: int = 32, pad_x: int = 4, pad_y: int = 2) -> Image.Image | None:
    """Fallback: render text using PIL."""
    try:
        from PIL import ImageFont, ImageDraw

        font = ImageFont.truetype(font_path, size=font_size)
        dummy = Image.new("RGB", (1, 1))
        bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= 0 or text_h <= 0:
            return None

        img_w = text_w + 2 * pad_x
        img_h = text_h + 2 * pad_y
        img = Image.new("RGB", (img_w, img_h), bg)
        ImageDraw.Draw(img).text(
            (pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink, font=font,
        )

        scale = height / img_h
        new_w = max(4, int(img_w * scale))
        return img.resize((new_w, height), Image.BILINEAR)
    except Exception:
        return None


# Auto-select the best renderer
if HAS_FREETYPE:
    render_text = render_text_freetype
else:
    render_text = render_text_pil


def font_has_codepoint(font_path: str, char: str) -> bool:
    """Check if a font's cmap contains the given character.

    Uses FreeType's get_char_index: returns 0 if the glyph is missing
    (which is what triggers tofu rendering). This is a definitive check —
    no pixel heuristics needed.

    Falls back to True if freetype is unavailable (let pixel check handle it).
    """
    if not HAS_FREETYPE:
        return True  # can't check, assume yes
    try:
        face = _get_face(font_path, 24)
        return face.get_char_index(ord(char)) != 0
    except Exception:
        return True  # on error, let rendering decide


def font_can_render(font_path: str, text: str, size: int = 24) -> bool:
    """Check if a font produces visible output for the given text."""
    img = render_text(text, font_path, size, ink=(0, 0, 0), bg=(255, 255, 255), height=32)
    if img is None:
        return False
    arr = np.array(img)
    return (arr < 200).sum() > len(text) * 3
