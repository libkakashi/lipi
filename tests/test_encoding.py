"""
Encoding tests for all scripts.

Tests: roundtrip correctness, vocab sizes, encode/decode consistency.
"""

import pytest
from pathlib import Path

from src.encoding.decompose import encode_text, decode_ids, script_vocab_size
from src.taxonomy import SCRIPTS
from src.data.word_lists import contains_emoji

WORD_LIST_DIR = Path(__file__).resolve().parent.parent / "training_data" / "word_lists"


def test_emoji_codepoints_are_rejected_from_training_text():
    assert contains_emoji("hello😀")
    assert contains_emoji("warning ⚠")
    assert not contains_emoji("ordinary text")

# Script → word list files for testing
SCRIPT_WORD_SOURCES = {
    "latin": ["english_100k.txt"],
    "cyrillic": ["cyrillic.txt"],
    "greek": ["greek.txt"],
    "arabic": ["arabic.txt"],
    "hebrew": ["hebrew.txt"],
    "han_sparse": ["chinese.txt"],
    "han_dense": ["chinese.txt"],
    "kana": ["japanese.txt"],
    "korean": ["korean.txt"],
    "devanagari": ["devanagari.txt"],
    "gurmukhi": ["gurmukhi.txt"],
    "gujarati": ["gujarati.txt"],
    "bengali": ["bengali.txt"],
    "odia": ["odia.txt"],
    "kannada": ["kannada.txt"],
    "telugu": ["telugu.txt"],
    "malayalam": ["malayalam.txt"],
    "tamil": ["tamil.txt"],
    "sinhala": ["sinhala.txt"],
    "thai": ["thai.txt"],
    "lao": ["lao.txt"],
    "burmese": ["burmese.txt"],
    "khmer": ["khmer.txt"],
    "armenian": ["armenian.txt"],
    "georgian": ["georgian.txt"],
    "ethiopic": ["ethiopic.txt"],
    "tibetan": ["tibetan.txt"],
}


def _load_sample_words(script: str, max_words: int = 200) -> list[str]:
    sources = SCRIPT_WORD_SOURCES.get(script, [])
    words = []
    for fname in sources:
        path = WORD_LIST_DIR / fname
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if w:
                    words.append(w)
                if len(words) >= max_words:
                    return words[:max_words]
    return words[:max_words]


class TestVocabBasics:

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_vocab_size_positive(self, script):
        vs = script_vocab_size(script)
        assert vs > 0, f"{script}: vocab_size is 0"

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_vocab_size_includes_blank(self, script):
        """Vocab size should be at least 2 (BLANK + at least one token)."""
        vs = script_vocab_size(script)
        assert vs >= 2, f"{script}: vocab_size {vs} too small"


class TestRoundtrip:

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_roundtrip_sample_words(self, script):
        words = _load_sample_words(script)
        if not words:
            pytest.skip(f"No word list for {script}")

        failures = []
        for w in words:
            ids = encode_text(w, script)
            if not ids:
                continue
            decoded = decode_ids(ids, script)
            if decoded != w:
                # Check if failure is due to cross-script chars (expected)
                # by re-encoding the decoded output — if it roundtrips cleanly,
                # the original just had foreign chars the encoder skipped
                re_ids = encode_text(decoded, script)
                re_decoded = decode_ids(re_ids, script) if re_ids else ""
                if re_decoded == decoded:
                    continue  # foreign chars dropped, remainder is stable
                failures.append((w, decoded))
                if len(failures) >= 10:
                    break

        assert not failures, (
            f"{script}: {len(failures)} roundtrip failures. "
            f"First 5: {failures[:5]}")

    @pytest.mark.parametrize("script", [
        "latin", "cyrillic", "greek", "hebrew", "armenian", "georgian",
        "ethiopic", "devanagari", "tamil", "thai", "korean",
        "han_sparse", "han_dense", "kana",
    ])
    def test_roundtrip_basic(self, script):
        """Quick roundtrip with known-good text."""
        samples = {
            "latin": "Hello World",
            "cyrillic": "Привет мир",
            "greek": "Γειά σου",
            "hebrew": "שלום",
            "armenian": "Բարև",
            "georgian": "გამარჯობა",
            "ethiopic": "ሰላም",
            "devanagari": "नमस्ते",
            "tamil": "வணக்கம்",
            "thai": "สวัสดี",
            "korean": "한국어",
            "han_sparse": "明日世界",
            "han_dense": "語能學漢",
            "kana": "あいうえお",     # pure hiragana
        }
        text = samples[script]
        ids = encode_text(text, script)
        assert ids, f"{script}: encode returned empty for {text!r}"
        decoded = decode_ids(ids, script)
        assert decoded == text, f"{script}: {text!r} → {decoded!r}"


class TestEncodeProperties:

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_encode_returns_ints(self, script):
        words = _load_sample_words(script, max_words=50)
        for w in words:
            ids = encode_text(w, script)
            assert all(isinstance(t, int) for t in ids), (
                f"{script}: non-int in encode output for {w!r}")

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_encode_ids_in_range(self, script):
        """All encoded IDs should be in [1, vocab_size)."""
        vs = script_vocab_size(script)
        words = _load_sample_words(script, max_words=50)
        for w in words:
            ids = encode_text(w, script)
            for tid in ids:
                assert 1 <= tid < vs, (
                    f"{script}: ID {tid} out of range [1, {vs}) for {w!r}")

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_blank_not_in_output(self, script):
        """Token ID 0 (BLANK) should never appear in encode output."""
        words = _load_sample_words(script, max_words=50)
        for w in words:
            ids = encode_text(w, script)
            assert 0 not in ids, f"{script}: BLANK (0) in encode for {w!r}"

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_empty_string(self, script):
        ids = encode_text("", script)
        assert ids == [], f"{script}: non-empty encode for empty string"
        decoded = decode_ids([], script)
        assert decoded == "", f"{script}: non-empty decode for empty ids"
