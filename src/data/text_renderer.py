"""
Shaped text rendering via PIL + Raqm (HarfBuzz + FriBiDi).

Raqm gives full OpenType shaping: Arabic joining + RTL ordering, Indic
matra reordering and conjuncts, Tamil split vowels, Thai mark positioning.
The previous FreeType `load_char` path drew one codepoint at a time, which
produced reversed/unjoined Arabic and mangled Indic — training data that
looked nothing like real text.

FreeType is still used for cmap checks (font_has_codepoint) — glyph
coverage is a font-table property and doesn't need a layout engine.

Usage:
    from src.data.text_renderer import render_text
    img = render_text("Hello", "/path/to/font.ttf", 24,
                      ink=(0, 0, 0), bg=(255, 255, 255), height=32)
    # Returns PIL.Image.Image (RGB) or None on failure
"""

import functools
import random
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

import freetype

_RAQM_AVAILABLE = features.check("raqm")
if not _RAQM_AVAILABLE:
    warnings.warn(
        "Pillow was built without Raqm — complex scripts (Arabic, Indic, "
        "Thai, ...) will render WRONG (unjoined/reordered). Install a "
        "Pillow build with libraqm before generating training data.",
        stacklevel=1,
    )

_LAYOUT_ENGINE = (ImageFont.Layout.RAQM if _RAQM_AVAILABLE
                  else ImageFont.Layout.BASIC)

# Cache PIL font objects: (path, size, weight_bucket) → FreeTypeFont.
# Bounded — each entry holds an open face.
_font_cache: dict[tuple, ImageFont.FreeTypeFont] = {}
_FONT_CACHE_MAX = 1024

# Cache variable-font weight axis info: path → (min, default, max) or None
_wght_axis_cache: dict[str, tuple[float, float, float] | None] = {}


def _wght_axis(font_path: str, font: ImageFont.FreeTypeFont):
    """Return the (min, default, max) of the wght axis, or None if static."""
    if font_path in _wght_axis_cache:
        return _wght_axis_cache[font_path]
    axis = None
    try:
        for ax in font.get_variation_axes():
            name = ax.get("name", b"")
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="ignore")
            if name.strip().lower() in ("weight", "wght"):
                axis = (float(ax["minimum"]), float(ax["default"]),
                        float(ax["maximum"]))
                break
    except OSError:
        axis = None  # not a variable font
    _wght_axis_cache[font_path] = axis
    return axis


def _pick_weight(font_path: str, font: ImageFont.FreeTypeFont) -> int:
    """Sample a weight bucket for a variable font (0 = default instance).

    Real documents are mostly regular with occasional bold; light weights
    are rare. Weights are quantized to 100s to keep the font cache small.
    """
    axis = _wght_axis(font_path, font)
    if axis is None:
        return 0
    lo, default, hi = axis
    r = random.random()
    if r < 0.60:
        return 0
    elif r < 0.90:
        target = default + random.choice([100, 200, 300])  # bold-ish
    else:
        target = default - 100  # light
    return int(np.clip(round(target / 100) * 100, lo, hi))


def _get_font(font_path: str, size: int, weight: int = 0) -> ImageFont.FreeTypeFont:
    """Get or create a PIL font, cached by (path, size, weight bucket)."""
    key = (font_path, size, weight)
    font = _font_cache.get(key)
    if font is None:
        font = ImageFont.truetype(font_path, size, layout_engine=_LAYOUT_ENGINE)
        if weight != 0:
            try:
                axes = font.get_variation_axes()
                values = []
                for ax in axes:
                    name = ax.get("name", b"")
                    if isinstance(name, bytes):
                        name = name.decode("ascii", errors="ignore")
                    if name.strip().lower() in ("weight", "wght"):
                        values.append(weight)
                    else:
                        values.append(ax["default"])
                font.set_variation_by_axes(values)
            except OSError:
                pass
        if len(_font_cache) >= _FONT_CACHE_MAX:
            _font_cache.pop(next(iter(_font_cache)))
        _font_cache[key] = font
    return font


_measure_draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def pick_weight(font_path: str, font_size: int) -> int:
    """Public weight sampler — lets callers fix one weight for a whole
    line instead of re-rolling per word."""
    return _pick_weight(font_path, _get_font(font_path, font_size))


@functools.lru_cache(maxsize=2048)
def render_word_baseline(text: str, font_path: str, font_size: int,
                         weight: int = 0, pad_x: int = 1,
                         ) -> tuple[Image.Image, int] | None:
    """Render text at natural metrics — no vertical rescaling.

    Returns (img, baseline_y) or None. The canvas spans the font's
    ascent..descent box (so every word at one size shares line geometry;
    'on' does NOT get stretched to the height of 'Apply'), expanded when
    shaped glyphs overflow it (stacked Indic/Thai marks). Metric boxes
    are clamped to 1.25/0.45 × size — some fonts report inflated
    ascent/descent that would shrink all their text.

    Always renders clean black-on-white (that's what makes the lru_cache
    valid — color and degradation are applied to the composed line).
    """
    try:
        font = _get_font(font_path, font_size, weight)
        ascent, descent = font.getmetrics()
        ascent = min(ascent, int(font_size * 1.25))
        descent = min(descent, int(font_size * 0.45))
        x0, y0, x1, y1 = _measure_draw.textbbox((0, 0), text, font=font)
        if x1 - x0 <= 0:
            return None
        top = min(0, y0)
        bottom = max(ascent + descent, y1)
        img = Image.new("RGB", ((x1 - x0) + 2 * pad_x, bottom - top),
                        (255, 255, 255))
        ImageDraw.Draw(img).text((pad_x - x0, -top), text,
                                 font=font, fill=(0, 0, 0))
        return img, ascent - top
    except Exception:
        return None


def render_text(text: str, font_path: str, font_size: int,
                ink: tuple[int, ...], bg: tuple[int, ...],
                height: int = 32, pad_x: int = 4, pad_y: int = 2,
                vary_weight: bool = True) -> Image.Image | None:
    """Render shaped text. Returns RGB PIL Image resized to `height`, or None.

    Direction is auto-detected (FriBiDi), so Arabic/Hebrew render RTL with
    correct joining. With vary_weight, variable fonts occasionally render
    at a bold/light instance for weight diversity.
    """
    try:
        weight = _pick_weight(font_path, _get_font(font_path, font_size)) \
            if vary_weight else 0
        font = _get_font(font_path, font_size, weight)

        x0, y0, x1, y1 = _measure_draw.textbbox((0, 0), text, font=font)
        text_w, text_h = x1 - x0, y1 - y0
        if text_w <= 0 or text_h <= 0:
            return None

        img_w = text_w + 2 * pad_x
        img_h = text_h + 2 * pad_y
        img = Image.new("RGB", (img_w, img_h), bg)
        ImageDraw.Draw(img).text((pad_x - x0, pad_y - y0), text,
                                 font=font, fill=ink)

        scale = height / img_h
        new_w = max(4, int(img_w * scale))
        return img.resize((new_w, height), Image.BILINEAR)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# cmap checks (FreeType — no layout engine needed)
# ---------------------------------------------------------------------------

_face_cache: dict[str, "freetype.Face"] = {}


def _get_face(font_path: str) -> "freetype.Face":
    if font_path not in _face_cache:
        _face_cache[font_path] = freetype.Face(font_path)
    return _face_cache[font_path]


def font_has_codepoint(font_path: str, char: str) -> bool:
    """Check if a font's cmap contains the given character.

    Uses FreeType's get_char_index: returns 0 if the glyph is missing
    (which is what triggers tofu rendering). This is a definitive check —
    no pixel heuristics needed.

    Returns False if the glyph is missing (cmap index 0 = .notdef).
    """
    try:
        face = _get_face(font_path)
        return face.get_char_index(ord(char)) != 0
    except Exception:
        return True  # on error, let rendering decide


def font_can_render(font_path: str, text: str, size: int = 24) -> bool:
    """Check if a font produces visible output for the given text."""
    img = render_text(text, font_path, size, ink=(0, 0, 0),
                      bg=(255, 255, 255), height=32, vary_weight=False)
    if img is None:
        return False
    arr = np.array(img)
    return (arr < 200).sum() > len(text) * 3
