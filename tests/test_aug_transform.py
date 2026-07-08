"""
Augmentation x-transform tests.

partial_crop / pad_with_border shift content horizontally; their
*_with_transform variants report the (a, b) affine so training labels
(segment offsets, per-pixel group labels) can follow the content.
"""

import random

import numpy as np
from PIL import Image

from src.data.augmentation import (
    RandAugmentOCR,
    pad_with_border,
    pad_with_border_with_transform,
    partial_crop,
    partial_crop_with_transform,
)
from src.training.dataloader import shift_labels_x, shift_segments_x


def _stripe_image(w=200, h=32, x0=100, x1=104):
    """White image with a black vertical stripe at [x0, x1)."""
    arr = np.full((h, w, 3), 255, np.uint8)
    arr[:, x0:x1] = 0
    return Image.fromarray(arr)


def _stripe_center(img):
    """Darkness-weighted x centroid — where the stripe landed."""
    arr = np.asarray(img.convert("L"), dtype=np.float64)
    dark = 255.0 - arr.mean(axis=0)
    xs = np.arange(arr.shape[1])
    return float((dark * xs).sum() / max(dark.sum(), 1e-6))


class TestOpTransforms:

    def test_partial_crop_transform_tracks_content(self):
        for seed in range(20):
            random.seed(seed)
            out, (a, b) = partial_crop_with_transform(_stripe_image())
            expected = a * 102 + b
            assert abs(_stripe_center(out) - expected) < 3.0

    def test_pad_with_border_transform_tracks_content(self):
        for seed in range(20):
            random.seed(seed)
            out, (a, b) = pad_with_border_with_transform(_stripe_image())
            expected = a * 102 + b
            assert abs(_stripe_center(out) - expected) < 3.0

    def test_plain_ops_match_transform_variants(self):
        """partial_crop / pad_with_border must stay byte-identical to
        their transform-returning variants under the same RNG state."""
        for op, op_t in [(partial_crop, partial_crop_with_transform),
                         (pad_with_border, pad_with_border_with_transform)]:
            random.seed(7)
            plain = op(_stripe_image())
            random.seed(7)
            with_t, _ = op_t(_stripe_image())
            assert np.array_equal(np.asarray(plain), np.asarray(with_t))


class TestRandAugmentTransform:

    def test_identity_when_skipped(self):
        aug = RandAugmentOCR(p=0.0)
        img = _stripe_image()
        out, (a, b) = aug.apply_with_transform(img)
        assert (a, b) == (1.0, 0.0)
        assert np.array_equal(np.asarray(out), np.asarray(img))

    def test_composed_transform_tracks_content(self):
        """Both x-shifting ops applied in sequence: the composed (a, b)
        must still predict where the stripe lands."""
        aug = RandAugmentOCR(n_ops=2, p=1.0,
                             ops=[partial_crop, pad_with_border], chains=[])
        for seed in range(20):
            random.seed(seed)
            out, (a, b) = aug.apply_with_transform(_stripe_image())
            expected = a * 102 + b
            if 2 <= expected <= 198:  # stripe still in frame
                assert abs(_stripe_center(out) - expected) < 4.0

    def test_call_returns_image_only(self):
        aug = RandAugmentOCR(n_ops=2, p=1.0,
                             ops=[partial_crop, pad_with_border], chains=[])
        random.seed(3)
        out = aug(_stripe_image())
        assert isinstance(out, Image.Image)


class TestLabelRemap:

    def test_shift_labels_identity(self):
        gl = np.arange(20, dtype=np.int64) % 5
        out = shift_labels_x(gl, 1.0, 0.0, fill=15)
        assert np.array_equal(out, gl)

    def test_shift_labels_shrink_and_shift(self):
        # Content shrunk 2x and shifted right by 5 (pad_with_border-like).
        gl = np.full(20, 15, dtype=np.int64)
        gl[8:12] = 3
        out = shift_labels_x(gl, 0.5, 5.0, fill=15)
        # Block [8, 12) maps to [9, 11); everything else is fill.
        assert set(np.where(out == 3)[0]) == {9, 10}
        assert (out[:9] == 15).all() and (out[11:] == 15).all()

    def test_shift_labels_out_of_range_gets_fill(self):
        gl = np.zeros(10, dtype=np.int64)
        # Left shift by 5: output pixels [5, 10) map past the source end.
        out = shift_labels_x(gl, 1.0, -5.0, fill=15)
        assert (out[:5] == 0).all()
        assert (out[5:] == 15).all()

    def test_shift_segments_maps_offsets(self):
        segs = [{"offset": 10, "width": 20, "text": "ab"}]
        out = shift_segments_x(segs, 0.5, 8.0, width=100)
        assert out[0]["offset"] == 13 and out[0]["width"] == 10

    def test_shift_segments_drops_cropped_out(self):
        segs = [{"offset": 0, "width": 10, "text": "a"},
                {"offset": 50, "width": 20, "text": "b"}]
        # Hard left crop: first segment lands fully outside.
        out = shift_segments_x(segs, 1.0, -40.0, width=100)
        assert len(out) == 1
        assert out[0]["text"] == "b"
        assert out[0]["offset"] == 10 and out[0]["width"] == 20

    def test_shift_segments_clamps_to_image(self):
        segs = [{"offset": 80, "width": 30, "text": "c"}]
        out = shift_segments_x(segs, 1.0, 0.0, width=100)
        assert out[0]["offset"] == 80 and out[0]["width"] == 20
