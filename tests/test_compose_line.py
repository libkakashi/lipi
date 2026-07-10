"""Baseline line-composition tests.

Verifies the typographic realism of the new compositor: words rendered
at natural metrics (no per-word vertical stretch), composed on a shared
baseline, labels aligned after whole-line scaling.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
FONT_DIR = REPO / "training_data" / "fonts"

from src.data.rendering import compose_line_baseline, font_covers_text
from src.data.text_renderer import render_word_baseline
from src.taxonomy import NUM_GROUPS


def _latin_font():
    for f in sorted(FONT_DIR.glob("*.ttf")):
        if font_covers_text(str(f), "Apply on"):
            return str(f)
    return None

FONT = _latin_font() if FONT_DIR.exists() else None
needs_font = pytest.mark.skipif(FONT is None, reason="no latin font available")


def _ink_rows(img, x0=0, x1=None):
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    if x1 is not None:
        arr = arr[:, x0:x1]
    rows = np.where((arr < 128).any(axis=1))[0]
    return rows


@needs_font
class TestRenderWordBaseline:

    def test_same_metric_box_no_per_word_stretch(self):
        """'on' and 'Apply' at one size share the metric box height and
        baseline — 'on' must NOT be scaled up to fill the canvas."""
        r1 = render_word_baseline("on", FONT, 40)
        r2 = render_word_baseline("Apply", FONT, 40)
        assert r1 is not None and r2 is not None
        (img1, b1), (img2, b2) = r1, r2
        assert img1.height == img2.height  # same ascent+descent box
        assert b1 == b2                     # same baseline position
        # x-height word has strictly less ink extent than asc+desc word
        assert len(_ink_rows(img1)) < len(_ink_rows(img2))

    def test_cache_returns_same_object(self):
        a = render_word_baseline("cache", FONT, 40)
        b = render_word_baseline("cache", FONT, 40)
        assert a is b


@needs_font
class TestComposeLine:

    def test_shared_baseline_placement(self):
        blocks = []
        for text in ["on", "Apply", "mom"]:
            img, baseline = render_word_baseline(text, FONT, 40)
            blocks.append((img, baseline, text, 0, 0, img.width))
        # gap block
        blocks.insert(1, (None, 0, "", 15, 0, 12))
        out = compose_line_baseline(blocks, 32)
        assert out is not None
        line, placed, scale = out
        assert line.height == 32
        # all text blocks placed so y + baseline is constant (one baseline)
        baselines = [y + b[1] for b, (t, g, s, x, w, y)
                     in zip(blocks, placed) if b[0] is not None]
        assert len(set(baselines)) == 1
        # widths scale consistently
        assert line.width == max(4, round(sum(b[5] for b in blocks) * scale)) \
            or abs(line.width - sum(b[5] for b in blocks) * scale) < 2

    def test_x_height_preserved_in_final_image(self):
        """In the composed line, 'mom' (x-height only) occupies fewer ink
        rows than 'Adam' (ascenders, no descenders) — per-word
        normalization would stretch both to the same extent. Both words
        end at the shared baseline."""
        b1 = render_word_baseline("mom", FONT, 40)
        b2 = render_word_baseline("Adam", FONT, 40)
        blocks = [
            (b1[0], b1[1], "mom", 0, 0, b1[0].width),
            (None, 0, "", 15, 0, 14),
            (b2[0], b2[1], "Adam", 0, 0, b2[0].width),
        ]
        line, placed, scale = compose_line_baseline(blocks, 32)
        x_mom_end = round(blocks[0][5] * scale)
        x_adam_start = round((blocks[0][5] + 14) * scale)
        rows_mom = _ink_rows(line, 0, x_mom_end)
        rows_adam = _ink_rows(line, x_adam_start, line.width)
        assert len(rows_mom) < len(rows_adam)
        # shared baseline: ink bottoms align (neither has descenders),
        # tops differ (ascenders on 'Adam' only)
        assert abs(int(rows_mom.max()) - int(rows_adam.max())) <= 2
        assert int(rows_adam.min()) < int(rows_mom.min())


@needs_font
class TestRenderPlanLine:

    @classmethod
    def setup_class(cls):
        spec = importlib.util.spec_from_file_location(
            "genmod", REPO / "scripts" / "data" / "generate.py")
        cls.gen = importlib.util.module_from_spec(spec)
        sys.modules["genmod"] = cls.gen
        spec.loader.exec_module(cls.gen)

    def test_line_render_end_to_end(self):
        import random
        random.seed(0)
        plan = [
            {"text": "hello", "script": "latin"},
            {"text": " ", "script": "whitespace"},
            {"text": "world", "script": "latin"},
        ]
        out = self.gen._render_plan_line(plan, {"latin": [FONT]}, 32, 1280)
        assert out is not None
        img, label, gl, segments = out
        assert img.height == 32
        assert label == "helloworld"
        assert len(segments) == 2
        assert gl.shape[0] == img.width
        for seg in segments:
            assert 0 <= seg["offset"] < img.width
            assert seg["offset"] + seg["width"] <= img.width
            # per-pixel labels match the segment's group
            mid = seg["offset"] + seg["width"] // 2
            assert gl[mid] == seg["group_id"]
        # gap between the words is blank-labeled
        gap_x = (segments[0]["offset"] + segments[0]["width"]
                 + segments[1]["offset"]) // 2
        assert gl[gap_x] == NUM_GROUPS
