"""
Text shaping regression tests.

The training renderer must produce SHAPED text (Raqm: HarfBuzz + FriBiDi).
A per-codepoint renderer silently produces reversed/unjoined Arabic and
mangled Indic — training data that looks nothing like real text. These
tests pin the observable consequences of shaping so a renderer regression
fails loudly instead of poisoning the next data generation run.

Run: pytest tests/test_shaping.py -v
"""

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import text_renderer
from src.data.text_renderer import render_text

FONT_DIR = Path(__file__).parent.parent / "training_data" / "fonts"


def _find_font(prefix: str) -> str | None:
    if not FONT_DIR.exists():
        return None
    hits = sorted(p for p in FONT_DIR.iterdir()
                  if p.name.lower().startswith(prefix.lower()))
    return str(hits[0]) if hits else None


def _render_width(text: str, font: str) -> int:
    img = render_text(text, font, 32, ink=(0, 0, 0), bg=(255, 255, 255),
                      height=32, pad_x=0, pad_y=1, vary_weight=False)
    assert img is not None, f"render failed: {text!r}"
    return img.width


def test_raqm_layout_engine_available():
    """Without Raqm every complex script renders wrong — hard requirement."""
    assert text_renderer._RAQM_AVAILABLE, (
        "Pillow lacks Raqm; do NOT generate training data with this build"
    )


@pytest.mark.skipif(not FONT_DIR.exists(), reason="fonts not downloaded")
def test_arabic_joining():
    """Joined Arabic is much narrower than the sum of isolated letters."""
    font = _find_font("Amiri-Regular") or _find_font("NotoNaskh")
    if font is None:
        pytest.skip("no Arabic font")
    word = "كتاب"
    joined = _render_width(word, font)
    isolated = sum(_render_width(ch, font) for ch in word)
    assert joined < isolated * 0.8, (
        f"Arabic not joining: word width {joined} vs isolated sum {isolated}"
    )


@pytest.mark.skipif(not FONT_DIR.exists(), reason="fonts not downloaded")
def test_devanagari_conjunct():
    """क + ् + ष must ligate to क्ष — narrower than the unshaped sequence."""
    font = _find_font("NotoSansDevanagari")
    if font is None:
        pytest.skip("no Devanagari font")
    conjunct = _render_width("क्ष", font)
    parts = _render_width("क", font) + _render_width("ष", font)
    assert conjunct < parts * 0.85, (
        f"Devanagari conjunct not forming: {conjunct} vs parts {parts}"
    )


@pytest.mark.skipif(not FONT_DIR.exists(), reason="fonts not downloaded")
def test_devanagari_matra_reorder():
    """The i-matra must add width (it occupies its own column left of क).

    Measured as raw text bbox at a fixed font size — the rendered-image
    widths aren't comparable because height normalization shrinks the
    taller matra cluster.
    """
    font_path = _find_font("NotoSansDevanagari")
    if font_path is None:
        pytest.skip("no Devanagari font")
    font = text_renderer._get_font(font_path, 32)
    d = text_renderer._measure_draw

    def bbox_w(t):
        x0, _, x1, _ = d.textbbox((0, 0), t, font=font)
        return x1 - x0

    assert bbox_w("कि") > bbox_w("क"), "i-matra rendered zero-width"
