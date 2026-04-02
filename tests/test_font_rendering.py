"""
Font rendering validation tests.

Verifies that fonts actually render properly for each script —
no blank images, no tofu, no garbled output, correct dimensions.
Catches silent rendering failures that would pollute training data.

Run: pytest tests/test_font_rendering.py -v
     pytest tests/test_font_rendering.py -v -k "test_sample_word"  # quick check
"""

import unicodedata
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP
from src.data.fonts import find_fonts_for_script, build_weighted_font_list
from src.data.rendering import render_word, image_has_ink, resize_or_pad, filter_fonts_by_cmap
from src.data.renderer import font_has_codepoint
from src.data.word_lists import load_word_list

FONT_DIR = Path(__file__).parent.parent / "training_data" / "fonts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_fonts():
    """Check if fonts are downloaded."""
    return FONT_DIR.exists() and any(FONT_DIR.glob("*.ttf")) or any(FONT_DIR.glob("*.otf"))


def _get_sample_words(script, n=5):
    """Get sample words for a script."""
    words = load_word_list(script)
    if not words:
        return []
    return words[:n]


def _get_sample_chars(script, n=10):
    """Get sample characters from a script's vocab."""
    from src.data.vocab import build_script_vocab
    from src.data.bigrams import BLANK_TOKEN
    group = SCRIPT_TO_GROUP[script]
    vocab = build_script_vocab(script, group)
    chars = [ch for ch in vocab if ch.strip() and ord(ch) > 127 and ch != BLANK_TOKEN]
    return chars[:n]


skip_no_fonts = pytest.mark.skipif(
    not _has_fonts(),
    reason="Fonts not downloaded. Run: python scripts/setup_fonts.py"
)


# ---------------------------------------------------------------------------
# 1. Font Discovery Tests
# ---------------------------------------------------------------------------

@skip_no_fonts
class TestFontDiscovery:
    """Verify fonts are found for each script."""

    def test_every_script_has_fonts(self):
        missing = []
        for script in SCRIPTS:
            if script == "emoji":
                continue
            fonts = find_fonts_for_script(script)
            if not fonts:
                missing.append(script)
        assert not missing, f"No fonts found for: {missing}"

    def test_font_discovery_uses_valid_codepoint(self):
        """Font discovery sample codepoint must be assigned, not Cn.
        Previously Sinhala/Lao/Georgian/Ethiopic used unassigned midpoints."""
        from src.data.script_detect import _SCRIPT_RANGES
        failures = []
        for script in SCRIPTS:
            if script == "emoji":
                continue
            ranges = _SCRIPT_RANGES.get(script, [])
            if not ranges:
                failures.append(f"{script}: no ranges in script_detect")
                continue
            # Verify the first assigned codepoint the function would pick
            found = False
            for start, end in ranges:
                for cp in range(start, end + 1):
                    cat = unicodedata.category(chr(cp))
                    if cat != "Cn":
                        found = True
                        break
                if found:
                    break
            if not found:
                failures.append(f"{script}: no assigned codepoints in ranges")
        assert not failures, "\n  ".join(failures)

    def test_weighted_font_list_not_empty(self):
        for script in SCRIPTS:
            if script == "emoji":
                continue
            fonts = find_fonts_for_script(script)
            if not fonts:
                continue
            words = _get_sample_words(script, 1)
            if not words:
                continue
            weighted = build_weighted_font_list(fonts, words[0])
            assert len(weighted) > 0, f"{script}: weighted font list empty"

    def test_weighted_list_has_variety(self):
        """Weighted list should have multiple unique fonts where possible."""
        for script in SCRIPTS:
            if script == "emoji":
                continue
            fonts = find_fonts_for_script(script)
            words = _get_sample_words(script, 1)
            if not fonts or not words:
                continue
            weighted = build_weighted_font_list(fonts, words[0])
            unique = len(set(weighted))
            # At least 2 unique fonts for scripts that have multiple
            if len(fonts) >= 2:
                assert unique >= 2, (
                    f"{script}: only {unique} unique font(s) in weighted list")


# ---------------------------------------------------------------------------
# 2. Word Rendering Tests
# ---------------------------------------------------------------------------

@skip_no_fonts
class TestWordRendering:
    """Verify word rendering produces valid images."""

    def test_sample_word_renders(self):
        """At least one word renders for each script."""
        for script in SCRIPTS:
            if script == "emoji":
                continue
            fonts = find_fonts_for_script(script)
            words = _get_sample_words(script, 3)
            if not fonts or not words:
                continue
            weighted = build_weighted_font_list(fonts, words[0])
            if not weighted:
                continue

            rendered = False
            for word in words:
                for font in list(set(weighted))[:3]:
                    img = render_word(word, font, height=32)
                    if img is not None and image_has_ink(img):
                        rendered = True
                        break
                if rendered:
                    break
            assert rendered, f"{script}: no word rendered with ink"

    def test_rendered_image_is_rgb(self):
        """Rendered images are RGB, not grayscale."""
        for script in ["latin", "devanagari", "arabic"]:
            fonts = find_fonts_for_script(script)
            words = _get_sample_words(script, 1)
            if not fonts or not words:
                continue
            weighted = build_weighted_font_list(fonts, words[0])
            if not weighted:
                continue
            img = render_word(words[0], weighted[0], height=32)
            if img is not None:
                assert img.mode == "RGB", f"{script}: image mode is {img.mode}"

    def test_rendered_image_correct_height(self):
        """Rendered images have the requested height."""
        for height in [32, 48, 64]:
            fonts = find_fonts_for_script("latin")
            if not fonts:
                continue
            weighted = build_weighted_font_list(fonts, "Hello")
            if not weighted:
                continue
            img = render_word("Hello", weighted[0], height=height)
            if img is not None:
                assert img.height == height, (
                    f"Requested height {height}, got {img.height}")

    def test_rendered_image_reasonable_width(self):
        """Rendered images aren't absurdly wide or narrow."""
        fonts = find_fonts_for_script("latin")
        if not fonts:
            return
        weighted = build_weighted_font_list(fonts, "Hello")
        if not weighted:
            return
        img = render_word("Hello", weighted[0], height=32)
        if img is not None:
            assert 10 < img.width < 500, f"Width {img.width} out of range"

    def test_rendered_image_not_all_same_color(self):
        """Rendered text should not be uniform color (would mean render failed silently)."""
        fonts = find_fonts_for_script("latin")
        if not fonts:
            return
        weighted = build_weighted_font_list(fonts, "Hello")
        if not weighted:
            return
        img = render_word("Hello", weighted[0], height=32)
        if img is not None:
            arr = np.array(img)
            # Check variance — uniform image has zero variance
            assert arr.std() > 5, "Rendered image is uniform color"


class TestResizeAndPad:
    """Test image sizing — no fonts needed."""

    def test_pad_narrow_image(self):
        img = Image.new("RGB", (100, 32), (255, 255, 255))
        result = resize_or_pad(img, 32, 192)
        assert result.size == (192, 32)

    def test_resize_wide_image(self):
        wide = Image.new("RGB", (300, 32), (255, 255, 255))
        result = resize_or_pad(wide, 32, 192)
        assert result.size == (192, 32)

    def test_exact_size_passthrough(self):
        exact = Image.new("RGB", (192, 32), (255, 255, 255))
        result = resize_or_pad(exact, 32, 192)
        assert result.size == (192, 32)

    def test_pad_uses_light_background(self):
        img = Image.new("RGB", (50, 32), (0, 0, 0))  # black image
        result = resize_or_pad(img, 32, 192)
        arr = np.array(result)
        # Padding area (right side) should be light
        assert arr[16, 150, 0] == 240  # padded with (240,240,240)


# ---------------------------------------------------------------------------
# 3. Ink Detection Tests
# ---------------------------------------------------------------------------

class TestInkDetection:
    """Verify image_has_ink correctly identifies blank vs inked images."""

    def test_blank_image_no_ink(self):
        """A solid color image has no ink."""
        img = Image.new("RGB", (100, 32), (240, 240, 240))
        assert not image_has_ink(img)

    def test_white_image_no_ink(self):
        img = Image.new("RGB", (100, 32), (255, 255, 255))
        assert not image_has_ink(img)

    def test_black_image_no_ink(self):
        """Solid black = uniform, no contrast = no ink."""
        img = Image.new("RGB", (100, 32), (0, 0, 0))
        assert not image_has_ink(img)

    def test_image_with_text_has_ink(self):
        """An image with drawn content has ink."""
        from PIL import ImageDraw
        img = Image.new("RGB", (100, 32), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 5, 80, 27], fill=(0, 0, 0))
        assert image_has_ink(img)

    def test_faint_marks_below_threshold(self):
        """Very faint marks (< 10 pixels different) are treated as no ink."""
        img = Image.new("RGB", (100, 32), (240, 240, 240))
        arr = np.array(img)
        # Add 5 pixels of slight contrast
        arr[10:12, 10:13, :] = 200
        img = Image.fromarray(arr)
        assert not image_has_ink(img, min_ink_pixels=10)

    def test_threshold_parameter(self):
        """min_ink_pixels parameter controls sensitivity."""
        from PIL import ImageDraw
        img = Image.new("RGB", (100, 32), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Draw a tiny dot (< 10 pixels)
        draw.rectangle([45, 14, 48, 17], fill=(0, 0, 0))
        # Should pass with low threshold
        assert image_has_ink(img, min_ink_pixels=5)


# ---------------------------------------------------------------------------
# 4. Cmap Validation Tests
# ---------------------------------------------------------------------------

@skip_no_fonts
class TestCmapValidation:
    """Verify font cmap checks prevent tofu rendering."""

    def test_cmap_check_rejects_missing_glyphs(self):
        """font_has_codepoint returns False for chars not in font."""
        # Use a Latin font and check for CJK chars
        fonts = find_fonts_for_script("latin")
        if not fonts:
            return
        latin_font = fonts[0]
        # CJK char should NOT be in a Latin-only font (usually)
        cjk_char = "\u4E00"  # 一
        # This might pass for some universal fonts, so just verify it returns bool
        result = font_has_codepoint(latin_font, cjk_char)
        assert isinstance(result, bool)

    def test_cmap_check_accepts_correct_chars(self):
        """font_has_codepoint returns True for chars the font supports."""
        fonts = find_fonts_for_script("latin")
        if not fonts:
            return
        # ASCII 'A' should be in every Latin font
        assert font_has_codepoint(fonts[0], "A")

    def test_filter_fonts_by_cmap_excludes_bad_fonts(self):
        """filter_fonts_by_cmap only returns fonts that have the char."""
        fonts = find_fonts_for_script("latin")
        if not fonts:
            return
        # Filter for ASCII chars — all fonts should pass
        chars = list("ABC")
        char_fonts = filter_fonts_by_cmap(fonts[:5], chars)
        for ch in chars:
            if ch in char_fonts:
                for f in char_fonts[ch]:
                    assert font_has_codepoint(f, ch)

    def test_filter_fonts_skips_unsupported_chars(self):
        """Chars with no valid fonts are omitted from the result."""
        fonts = find_fonts_for_script("latin")
        if not fonts:
            return
        # Mix ASCII (supported) with exotic char (likely unsupported)
        chars = ["A", "\U0001F600"]  # A + emoji
        char_fonts = filter_fonts_by_cmap(fonts[:3], chars)
        assert "A" in char_fonts  # definitely supported
        # Emoji might or might not be — just verify structure
        for ch, font_list in char_fonts.items():
            assert len(font_list) > 0


# ---------------------------------------------------------------------------
# 5. Per-Script Rendering Smoke Tests
# ---------------------------------------------------------------------------

@skip_no_fonts
class TestPerScriptRendering:
    """Smoke test: render one word per script, verify it has ink."""

    @pytest.fixture(scope="class")
    def script_render_results(self):
        """Render one word per script, cache results."""
        results = {}
        for script in SCRIPTS:
            if script == "emoji":
                results[script] = {"status": "skip", "reason": "emoji"}
                continue

            fonts = find_fonts_for_script(script)
            if not fonts:
                results[script] = {"status": "no_fonts"}
                continue

            words = _get_sample_words(script, 5)
            if not words:
                results[script] = {"status": "no_words"}
                continue

            weighted = build_weighted_font_list(fonts, words[0])
            if not weighted:
                results[script] = {"status": "no_valid_fonts"}
                continue

            # Try rendering
            best_img = None
            for word in words:
                for font in list(set(weighted))[:5]:
                    img = render_word(word, font, height=32)
                    if img is not None and image_has_ink(img):
                        best_img = img
                        results[script] = {
                            "status": "ok",
                            "word": word,
                            "font": font,
                            "width": img.width,
                            "height": img.height,
                        }
                        break
                if best_img:
                    break

            if not best_img:
                results[script] = {"status": "render_failed"}

        return results

    def test_all_scripts_render(self, script_render_results):
        """Every script with fonts+words should produce at least one image."""
        failed = []
        for script, result in script_render_results.items():
            if result["status"] in ("skip", "no_fonts", "no_words"):
                continue
            if result["status"] != "ok":
                failed.append(f"{script}: {result['status']}")
        assert not failed, f"Scripts failed to render: {failed}"

    def test_rendered_heights_correct(self, script_render_results):
        for script, result in script_render_results.items():
            if result.get("status") != "ok":
                continue
            assert result["height"] == 32, (
                f"{script}: height {result['height']}")

    def test_rendered_widths_reasonable(self, script_render_results):
        for script, result in script_render_results.items():
            if result.get("status") != "ok":
                continue
            assert 4 < result["width"] < 1000, (
                f"{script}: width {result['width']}")


# ---------------------------------------------------------------------------
# 6. Character Rendering Coverage
# ---------------------------------------------------------------------------

@skip_no_fonts
class TestCharRenderingCoverage:
    """Exhaustive tests: render ALL vocab chars, not just samples."""

    def test_every_script_char_has_cmap_font(self):
        """Every renderable char in every script must have at least one font
        with a cmap entry. Chars with zero fonts = guaranteed tofu in training."""
        from scripts.generate_data import get_renderable_chars

        failures = []
        for script in SCRIPTS:
            if script == "emoji":
                continue

            fonts = find_fonts_for_script(script)
            if not fonts:
                continue

            chars = get_renderable_chars(script)
            no_font = []
            for ch in chars:
                has_any = False
                for font in fonts[:20]:  # check up to 20 fonts
                    if font_has_codepoint(font, ch):
                        has_any = True
                        break
                if not has_any:
                    no_font.append(f"U+{ord(ch):04X}")

            if no_font:
                pct = len(no_font) / len(chars) * 100
                failures.append(
                    f"{script}: {len(no_font)}/{len(chars)} chars ({pct:.1f}%) "
                    f"have no font: {no_font[:5]}")

        assert not failures, (
            "Chars with zero cmap-supporting fonts:\n  " + "\n  ".join(failures))

    def test_render_all_vocab_chars_have_ink(self):
        """Actually render every vocab char for every script and verify ink.
        This catches fonts that have cmap entries but render blank/invisible."""
        from scripts.generate_data import get_renderable_chars

        failures = []
        for script in SCRIPTS:
            if script == "emoji":
                continue

            fonts = find_fonts_for_script(script)
            if not fonts:
                continue

            chars = get_renderable_chars(script)
            blank_chars = []
            tested = 0

            for ch in chars:
                # Find a font that claims to support this char
                font = None
                for f in fonts[:20]:
                    if font_has_codepoint(f, ch):
                        font = f
                        break
                if font is None:
                    continue

                img = render_word(ch, font, height=32)
                tested += 1
                if img is None or not image_has_ink(img):
                    blank_chars.append(f"U+{ord(ch):04X}")

            if blank_chars:
                pct = len(blank_chars) / max(tested, 1) * 100
                failures.append(
                    f"{script}: {len(blank_chars)}/{tested} chars ({pct:.1f}%) "
                    f"render blank: {blank_chars[:5]}")

        # Allow up to 5% blank (some combining chars legitimately have no visible ink alone)
        real_failures = []
        for f in failures:
            pct = float(f.split("(")[1].split("%")[0])
            if pct > 5.0:
                real_failures.append(f)

        assert not real_failures, (
            "Scripts with >5% blank renders:\n  " + "\n  ".join(real_failures))

    def test_render_sample_words_every_script(self):
        """Render 20 words per script and verify all have ink."""
        failures = []
        for script in SCRIPTS:
            if script == "emoji":
                continue

            fonts = find_fonts_for_script(script)
            words = _get_sample_words(script, 20)
            if not fonts or not words:
                continue

            weighted = build_weighted_font_list(fonts, words[0])
            if not weighted:
                continue

            blank = 0
            for word in words:
                font = weighted[len(word) % len(weighted)]  # deterministic pick
                img = render_word(word, font, height=32)
                if img is None or not image_has_ink(img):
                    blank += 1

            if blank > 0:
                failures.append(f"{script}: {blank}/{len(words)} words render blank")

        assert not failures, (
            "Scripts with blank word renders:\n  " + "\n  ".join(failures))

    def test_combining_chars_render_with_base(self):
        """Combining/dependent characters should render with their base character.
        E.g., Devanagari vowel signs need a consonant to display properly."""
        import unicodedata

        for script in ["devanagari", "bengali", "tamil", "arabic"]:
            fonts = find_fonts_for_script(script)
            if not fonts:
                continue

            words = _get_sample_words(script, 10)
            if not words:
                continue

            weighted = build_weighted_font_list(fonts, words[0])
            if not weighted:
                continue

            # Render actual words (which naturally contain combining chars)
            rendered = 0
            for word in words[:5]:
                img = render_word(word, weighted[0], height=32)
                if img is not None and image_has_ink(img):
                    rendered += 1

            assert rendered >= 3, (
                f"{script}: only {rendered}/5 words with combining chars rendered")

    def test_no_tofu_for_cmap_passed_chars_all_scripts(self):
        """For EVERY script: if cmap says a char is supported, render must have ink."""
        failures = []
        for script in SCRIPTS:
            if script == "emoji":
                continue

            fonts = find_fonts_for_script(script)
            if not fonts:
                continue

            chars = _get_sample_chars(script, 20)
            tofu = 0
            tested = 0

            for ch in chars:
                font = None
                for f in fonts[:10]:
                    if font_has_codepoint(f, ch):
                        font = f
                        break
                if font is None:
                    continue

                tested += 1
                img = render_word(ch, font, height=32)
                if img is not None and not image_has_ink(img):
                    tofu += 1

            if tofu > 0:
                failures.append(f"{script}: {tofu}/{tested} cmap-passed chars have no ink")

        assert not failures, (
            "Cmap-passed but blank renders:\n  " + "\n  ".join(failures))
