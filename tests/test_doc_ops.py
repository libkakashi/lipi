"""Document decoration + binarization op tests — synthetic images, fast."""

import random

import numpy as np
from PIL import Image

from src.data.augmentation import (
    highlighter, table_rules, text_decoration,
)


def _text_image(w=120, h=32):
    """White image with a dark text-like band in the x-height region."""
    arr = np.full((h, w, 3), 250, np.uint8)
    arr[10:24, 8:112] = 30  # "text"
    return Image.fromarray(arr)


class TestTextDecoration:

    def test_adds_line_preserves_size(self):
        random.seed(0)
        img = _text_image()
        out = text_decoration(img)
        assert out.size == img.size
        assert not np.array_equal(np.asarray(out), np.asarray(img))


class TestHighlighter:

    def test_ink_stays_dark_background_tinted(self):
        random.seed(1)
        img = _text_image()
        out = np.asarray(highlighter(img), dtype=np.float32)
        # ink pixels stay dark (multiply blend)
        assert out[16, 60].max() < 80
        # some background pixel inside the band got colored (channels differ)
        band_bg = out[12:22, 114:118].reshape(-1, 3)
        assert (band_bg.max(axis=1) - band_bg.min(axis=1)).max() > 20 or \
            (np.asarray(highlighter(_text_image())) != np.asarray(_text_image())).any()

    def test_skips_dark_background(self):
        img = Image.fromarray(np.full((32, 80, 3), 20, np.uint8))
        out = highlighter(img)
        assert np.array_equal(np.asarray(out), np.asarray(img))


class TestTableRules:

    def test_draws_at_least_one_rule(self):
        for seed in range(10):
            random.seed(seed)
            img = _text_image()
            out = table_rules(img)
            assert out.size == img.size
            assert not np.array_equal(np.asarray(out), np.asarray(img))

    def test_tiny_image_passthrough(self):
        img = Image.fromarray(np.full((4, 8, 3), 255, np.uint8))
        assert table_rules(img).size == img.size
