"""
Font discovery for OCR training data generation.

Uses an explicit font-to-script mapping — no pattern matching.
Each font is mapped to the scripts it can actually render.
Prevents tofu rendering from fonts that pass cmap checks but
can't display the correct glyphs.

The registry data (which fonts, which scripts, which category) lives in
src.data.font_registry; this module holds the loading + selection logic.
"""

import os
from pathlib import Path

from src.data.font_registry import (
    FONT_TO_SCRIPTS, FONT_CATEGORIES, SCRIPT_TO_FONTS,
    HANDWRITING_KEYWORDS, DISPLAY_KEYWORDS,
)
from src.data.text_renderer import font_can_render, font_has_codepoint


# Common font directories by platform
_FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    "/System/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/Library/Fonts",
    # Project-local fonts
    str(Path(__file__).parent.parent.parent / "training_data" / "fonts"),
]


def find_system_fonts() -> list[str]:
    """Find all .ttf/.otf font files on the system."""
    fonts = []
    seen = set()
    for d in _FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf")):
                    path = os.path.join(root, f)
                    if path not in seen:
                        seen.add(path)
                        fonts.append(path)
    return fonts


def find_fonts_for_script(script: str) -> list[str]:
    """Find fonts that can render a given script.

    Uses explicit font-to-script mapping. Only returns fonts that are
    known to correctly render this script — no pattern matching.
    """
    allowed_names = SCRIPT_TO_FONTS.get(script, set())
    if not allowed_names:
        return []

    all_fonts = find_system_fonts()
    return [f for f in all_fonts if Path(f).name in allowed_names]


def build_weighted_font_list(
    fonts: list[str],
    sample_text: str,
    boosts: dict[str, float] | None = None,
) -> list[str]:
    """Build a weighted font list for training diversity.

    Weights: 70% clean/regular, 20% handwriting, 10% display.
    Validates each font can actually render the sample text.

    boosts maps a lowercase filename substring to a weight multiplier
    (result floored at 1 copy) — used to up-weight styles the model is
    weakest on (e.g. Nastaliq) without touching the registry.
    """
    valid = [f for f in fonts if font_can_render(f, sample_text)]
    if not valid:
        return []

    weighted = []
    for f in valid:
        fname = Path(f).name
        cat = FONT_CATEGORIES.get(fname)
        if cat is None:
            # Fallback to keyword matching
            name_lower = fname.lower()
            if any(k in name_lower for k in HANDWRITING_KEYWORDS):
                cat = "handwriting"
            elif any(k in name_lower for k in DISPLAY_KEYWORDS):
                cat = "display"
            else:
                cat = "sans"
        if cat == "handwriting":
            base = 2  # 20% weight
        elif cat == "display":
            base = 1  # 10% weight
        else:
            base = 7  # 70% weight
        mult = 1.0
        if boosts:
            for sub, m in boosts.items():
                if sub in fname.lower():
                    mult *= m
        weighted.extend([f] * max(1, round(base * mult)))

    return weighted
