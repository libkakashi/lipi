"""
Font discovery and validation for OCR training data generation.

Finds system fonts that can render each script, validates with cmap
checks, and builds weighted font lists (70% clean, 20% handwriting,
10% display) for diverse training data.
"""

import os
from pathlib import Path

from src.data.renderer import font_can_render, font_has_codepoint


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

# Keywords for font style weighting
_HANDWRITING_KEYWORDS = [
    "caveat", "dancing", "indie", "patrick", "shadow", "kalam",
    "nanumpen", "chilanka", "handwrit", "cursive", "script",
]
_DISPLAY_KEYWORDS = [
    "permanent", "amatic", "lobster", "pacifico", "special", "display",
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
    """Find fonts that contain glyphs for a given script."""
    from src.data.script_detect import _SCRIPT_RANGES

    if script not in _SCRIPT_RANGES:
        return []

    # Get a sample codepoint from the script's range
    ranges = _SCRIPT_RANGES[script]
    sample_cp = None
    for start, end in ranges:
        mid = (start + end) // 2
        sample_cp = chr(mid)
        break

    if sample_cp is None:
        return []

    all_fonts = find_system_fonts()
    valid = []
    for f in all_fonts:
        if font_has_codepoint(f, sample_cp):
            valid.append(f)

    return valid


def build_weighted_font_list(
    fonts: list[str],
    sample_text: str,
) -> list[str]:
    """Build a weighted font list for training diversity.

    Weights: 70% clean/regular, 20% handwriting, 10% display.
    Validates each font can actually render the sample text.
    """
    valid = [f for f in fonts if font_can_render(f, sample_text)]
    if not valid:
        return []

    weighted = []
    for f in valid:
        name = Path(f).name.lower()
        if any(k in name for k in _HANDWRITING_KEYWORDS):
            weighted.extend([f] * 2)  # 20% weight
        elif any(k in name for k in _DISPLAY_KEYWORDS):
            weighted.extend([f] * 1)  # 10% weight
        else:
            weighted.extend([f] * 7)  # 70% weight

    return weighted
