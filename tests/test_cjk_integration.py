"""
CJK codec integration tests.

Covers: vocab loading, encode/decode round-trip, token ID ranges,
        full coverage, visual similarity mapping, CTC head compatibility,
        and eval decode path simulation.
"""

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_codec():
    from src.encoding.config import get_cjk_codec
    return get_cjk_codec()


def _vocab_size():
    return _get_codec().vocab_size


# ---------------------------------------------------------------------------
# 1. Codec Loading
# ---------------------------------------------------------------------------

class TestCJKCodecLoading:

    def test_codec_loads(self):
        codec = _get_codec()
        assert codec.vocab_size > 200, f"Vocab suspiciously small: {codec.vocab_size}"

    def test_blank_at_zero(self):
        codec = _get_codec()
        assert codec.decode(0) == ""

    def test_vocab_size_matches_config(self):
        from src.encoding.config import _CJK_VOCAB_SIZE
        codec = _get_codec()
        assert codec.vocab_size == _CJK_VOCAB_SIZE

    def test_no_duplicate_tokens(self):
        codec = _get_codec()
        assert len(codec.tokens) == len(set(codec.tokens)), "Duplicate tokens in codec"


# ---------------------------------------------------------------------------
# 2. Encode/Decode Round-trip
# ---------------------------------------------------------------------------

class TestCJKRoundtrip:

    def test_cjk_roundtrip(self):
        codec = _get_codec()
        words = ["学校", "日本語", "東京", "人工知能"]
        for w in words:
            ids = codec.encode_text(w)
            result = codec.decode_ids(ids)
            assert result == w, f"CJK roundtrip failed: {w!r} -> {result!r}"

    def test_kana_roundtrip(self):
        codec = _get_codec()
        words = ["ひらがな", "カタカナ", "あいうえお"]
        for w in words:
            ids = codec.encode_text(w)
            result = codec.decode_ids(ids)
            assert result == w, f"Kana roundtrip failed: {w!r} -> {result!r}"

    def test_mixed_kana_cjk_roundtrip(self):
        codec = _get_codec()
        text = "こんにちは世界"
        ids = codec.encode_text(text)
        result = codec.decode_ids(ids)
        assert result == text

    def test_punctuation_roundtrip(self):
        codec = _get_codec()
        text = "「你好!」"  # ASCII ! (NFKC normalizes fullwidth ！→!)
        ids = codec.encode_text(text)
        result = codec.decode_ids(ids)
        assert result == text

    def test_empty_string(self):
        codec = _get_codec()
        assert codec.encode_text("") == []
        assert codec.decode_ids([]) == ""

    def test_nfkc_normalization(self):
        """Halfwidth katakana should normalize to fullwidth."""
        codec = _get_codec()
        hw = "ｱｲｳ"
        ids = codec.encode_text(hw)
        result = codec.decode_ids(ids)
        assert result == "アイウ"


# ---------------------------------------------------------------------------
# 3. Token ID Ranges
# ---------------------------------------------------------------------------

class TestTokenIDRanges:

    def test_encode_ids_in_range(self):
        codec = _get_codec()
        vs = codec.vocab_size
        texts = ["的一是不了在人有我他", "学校日本語東京", "こんにちは世界"]
        for text in texts:
            ids = codec.encode_text(text)
            for i in ids:
                assert 0 < i < vs, f"ID {i} out of range (vocab_size={vs})"

    def test_all_cjk_chars_encode_within_range(self):
        """All 27,584 CJK chars produce IDs within vocab_size."""
        codec = _get_codec()
        vs = codec.vocab_size
        all_cjk = [chr(cp) for cp in range(0x4E00, 0xA000)] + \
                  [chr(cp) for cp in range(0x3400, 0x4DC0)]
        overflow = []
        for char in all_cjk:
            ids = codec.encode_text(char)
            if not ids:
                overflow.append((char, "no encoding"))
            for i in ids:
                if i >= vs:
                    overflow.append((char, i))
        assert not overflow, f"{len(overflow)} chars have issues: {overflow[:5]}"


# ---------------------------------------------------------------------------
# 4. Full Coverage
# ---------------------------------------------------------------------------

class TestFullCoverage:

    def test_all_cjk_chars_roundtrip(self):
        """Every CJK char must round-trip through encode/decode."""
        codec = _get_codec()
        all_cjk = [chr(cp) for cp in range(0x4E00, 0xA000)] + \
                  [chr(cp) for cp in range(0x3400, 0x4DC0)]
        failures = []
        for char in all_cjk:
            ids = codec.encode_text(char)
            if not ids:
                failures.append((char, "no encoding"))
                continue
            result = codec.decode_ids(ids)
            if result != char:
                failures.append((char, result))
        assert not failures, (
            f"{len(failures)} roundtrip failures. First 10: {failures[:10]}")

    def test_distinct_chars_have_distinct_encodings(self):
        """Different CJK chars must produce different token sequences."""
        codec = _get_codec()
        chars = list("的一是不了在人有我他国学日本語")
        seqs = [tuple(codec.encode_text(c)) for c in chars]
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                assert seqs[i] != seqs[j], (
                    f"{chars[i]!r} and {chars[j]!r} have same encoding")


# ---------------------------------------------------------------------------
# 5. Visual Similarity Mapping
# ---------------------------------------------------------------------------

class TestVisualMapping:

    def test_mapping_file_exists(self):
        mapping = Path(__file__).parent.parent / "training_data" / "corpora" / "cjk_visual_mapping.tsv"
        assert mapping.exists()

    def test_vocab_file_exists(self):
        vocab = Path(__file__).parent.parent / "training_data" / "corpora" / "cjk_vocab.txt"
        assert vocab.exists()

    def test_two_token_chars_use_alt(self):
        """Chars not in single-token vocab should encode as 2 tokens (leaf + ALT)."""
        codec = _get_codec()
        # Pick a rare char that's likely in the mapping
        rare_chars = [chr(cp) for cp in range(0x4E00, 0x4E10)]
        for char in rare_chars:
            ids = codec.encode_text(char)
            if len(ids) == 2:
                # Second token should be an ALT token
                assert ids[1] in codec._alt_id_set, (
                    f"2-token char {char!r} second token not ALT")
                return  # found at least one
        # If all were single-token, that's fine too


# ---------------------------------------------------------------------------
# 6. CTC Head Compatibility
# ---------------------------------------------------------------------------

class TestCTCHeadCompatibility:

    def test_ctc_head_accepts_vocab_size(self):
        from src.model.encoder import CTCHead
        vs = _vocab_size()
        head = CTCHead(enc_dim=384, vocab_size=vs)
        assert head.vocab_size == vs
        assert head.proj.out_features == vs

    def test_group_ctc_module_accepts_vocab_size(self):
        from src.model.encoder import GroupCTCModule
        vs = _vocab_size()
        gcm = GroupCTCModule(
            enc_dim=384,
            script_vocab_sizes=[vs],
            script_names=["han_kana"],
        )
        assert gcm.max_vocab == vs


# ---------------------------------------------------------------------------
# 7. Eval Decode Path Simulation
# ---------------------------------------------------------------------------

class TestEvalDecodePath:

    def test_ctc_greedy_decode_simulation(self):
        """Simulate CTC greedy decode -> decode_ids."""
        codec = _get_codec()
        test_cases = ["学校", "東京タワー", "こんにちは世界", "的"]
        for text in test_cases:
            ids = codec.encode_text(text)
            # Simulate CTC output: add blanks and repeats
            ctc_output = []
            for i in ids:
                ctc_output.extend([0, i, i, 0])
            # CTC greedy decode: collapse repeats, remove blanks
            collapsed = []
            prev = -1
            for i in ctc_output:
                if i != prev:
                    if i != 0:
                        collapsed.append(i)
                    prev = i
            result = codec.decode_ids(collapsed)
            assert result == text, f"Eval path failed: {text!r} -> {result!r}"

    def test_decode_with_only_blanks(self):
        codec = _get_codec()
        assert codec.decode_ids([0, 0, 0]) == ""

    def test_decode_skips_blank_correctly(self):
        codec = _get_codec()
        ids = codec.encode_text("日")
        ids_with_blanks = [0] + [val for i in ids for val in (i, 0)]
        result_clean = codec.decode_ids(ids)
        result_blanks = codec.decode_ids(ids_with_blanks)
        assert result_clean == result_blanks


# ---------------------------------------------------------------------------
# 8. Unified API
# ---------------------------------------------------------------------------

class TestUnifiedAPI:

    def test_encode_decode_via_decompose(self):
        from src.encoding.decompose import encode_text, decode_ids, script_vocab_size
        text = "東京タワー"
        ids = encode_text(text, "han_kana")
        result = decode_ids(ids, "han_kana")
        assert result == text

    def test_vocab_size_via_decompose(self):
        from src.encoding.decompose import script_vocab_size
        vs = script_vocab_size("han_kana")
        assert vs == _vocab_size()
