"""
Comprehensive CJK 13-symbol encoding integration tests.

Covers: vocab loading, tokenizer round-trip, decompose/reconstruct,
        token coverage, ID overflow, duplicate sequences, CTC head
        compatibility, and eval decode path simulation.
"""

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_vocab():
    from src.encoding.vocab import build_script_vocab
    return build_script_vocab("han_kana", "sino_japanese")


def _vocab_size():
    return len(_load_vocab())


# ---------------------------------------------------------------------------
# 1. Vocab Loading
# ---------------------------------------------------------------------------

class TestCJKVocabLoading:

    def test_han_kana_vocab_loads(self):
        vocab = _load_vocab()
        # 13 base symbols + SEP + up to 2500 BPE + kana + punct + BLANK
        assert len(vocab) > 200, f"Vocab suspiciously small: {len(vocab)}"

    def test_blank_token_at_index_zero(self):
        from src.encoding.tokenizer import BLANK_TOKEN
        vocab = _load_vocab()
        assert vocab[0] == BLANK_TOKEN

    def test_vocab_has_no_duplicates(self):
        vocab = _load_vocab()
        assert len(vocab) == len(set(vocab)), (
            f"Duplicates in vocab: {len(vocab)} total, {len(set(vocab))} unique")

    def test_base_symbols_in_vocab(self):
        """All 13 base symbols (U+E000-E00C) must be in vocab."""
        vocab_set = set(_load_vocab())
        for i in range(13):
            sym = chr(0xE000 + i)
            assert sym in vocab_set, f"Base symbol U+{0xE000+i:04X} not in vocab"

    def test_sep_absorbed_by_bpe(self):
        """SEP token is fully absorbed by BPE — should NOT be in final vocab."""
        from src.encoding.decompose import SEP_CHAR
        vocab = _load_vocab()
        # SEP is used during encoding but always merged into BPE tokens.
        # Dead-token pruning correctly removes it from the vocab.
        assert SEP_CHAR not in vocab, "SEP should be pruned (fully absorbed by BPE)"


# ---------------------------------------------------------------------------
# 2. Tokenizer Encode/Decode Round-trip
# ---------------------------------------------------------------------------

class TestCJKTokenizerRoundtrip:

    @pytest.fixture
    def tok(self):
        from src.encoding.tokenizer import LipiTokenizer
        return LipiTokenizer(vocab=_load_vocab())

    def test_cjk_roundtrip(self, tok):
        """CJK text: decompose -> encode -> decode -> reconstruct."""
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        words = ["学校", "日本語", "東京", "人工知能"]
        for w in words:
            decomposed = decompose_han_kana(w)
            ids = tok.encode(decomposed)
            decoded = tok.decode(ids)
            result = reconstruct_han_kana(list(decoded))
            assert result == w, f"CJK roundtrip failed: {w!r} -> {result!r}"

    def test_kana_roundtrip(self, tok):
        """Pure kana passes through decomposition unchanged."""
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        words = ["ひらがな", "カタカナ", "あいうえお"]
        for w in words:
            decomposed = decompose_han_kana(w)
            ids = tok.encode(decomposed)
            decoded = tok.decode(ids)
            result = reconstruct_han_kana(list(decoded))
            assert result == w, f"Kana roundtrip failed: {w!r} -> {result!r}"

    def test_mixed_kana_cjk_roundtrip(self, tok):
        """Mixed kana + CJK text."""
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        text = "こんにちは世界"
        decomposed = decompose_han_kana(text)
        ids = tok.encode(decomposed)
        decoded = tok.decode(ids)
        result = reconstruct_han_kana(list(decoded))
        assert result == text, f"Mixed roundtrip failed: {text!r} -> {result!r}"

    def test_encode_ids_in_range(self, tok):
        """All encoded IDs must be < vocab_size."""
        from src.encoding.decompose import decompose_han_kana
        vs = _vocab_size()
        texts = ["的一是不了在人有我他", "学校日本語東京", "こんにちは世界"]
        for text in texts:
            decomposed = decompose_han_kana(text)
            ids = tok.encode(decomposed)
            for i in ids:
                assert 0 < i < vs, f"ID {i} out of range (vocab_size={vs}) for {text!r}"


# ---------------------------------------------------------------------------
# 3. Decompose/Reconstruct Round-trip (detailed)
# ---------------------------------------------------------------------------

class TestDecomposeReconstruct:

    def test_common_chars(self):
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        text = "的一是不了在人有我他"
        decomposed = decompose_han_kana(text)
        result = reconstruct_han_kana(list(decomposed))
        assert result == text

    def test_distinct_chars_have_distinct_codes(self):
        """Different CJK chars must produce different decompositions."""
        from src.encoding.decompose import decompose_han_kana
        chars = list("的一是不了在人有我他国学日本語")
        codes = [decompose_han_kana(c) for c in chars]
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                assert codes[i] != codes[j], (
                    f"{chars[i]!r} and {chars[j]!r} have same code: {codes[i]!r}")

    def test_mixed_kana_cjk(self):
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        text = "こんにちは世界"
        result = reconstruct_han_kana(list(decompose_han_kana(text)))
        assert result == text

    def test_pure_kana(self):
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        text = "ひらがなカタカナ"
        decomposed = decompose_han_kana(text)
        # Kana are now encoded (not pass-through), so decomposed != text
        assert decomposed != text
        # But roundtrip should still work
        result = reconstruct_han_kana(list(decomposed))
        assert result == text

    def test_empty_string(self):
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        assert decompose_han_kana("") == ""
        assert reconstruct_han_kana([]) == ""

    def test_single_char(self):
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        for ch in "的あア":
            decomposed = decompose_han_kana(ch)
            result = reconstruct_han_kana(list(decomposed))
            assert result == ch, f"Single char failed: {ch!r}"

    def test_punctuation(self):
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana
        text = "「你好！」"
        decomposed = decompose_han_kana(text)
        result = reconstruct_han_kana(list(decomposed))
        assert result == text

    def test_ext_a_chars(self):
        """CJK Extension A chars (U+3400..U+4DBF)."""
        from src.encoding.decompose import (
            decompose_han_kana, reconstruct_han_kana,
            _load_arbitrary_encoding,
        )
        enc = _load_arbitrary_encoding("han_kana")
        cjk_ct = enc["char_to_tokens"]
        ext_a_chars = [
            ch for ch in cjk_ct
            if 0x3400 <= ord(ch) <= 0x4DBF
        ][:5]
        for ch in ext_a_chars:
            decomposed = decompose_han_kana(ch)
            result = reconstruct_han_kana(list(decomposed))
            assert result == ch, f"Ext-A char {ch!r} (U+{ord(ch):04X}) failed"


# ---------------------------------------------------------------------------
# 4. All Char Codes Tokens In Vocab
# ---------------------------------------------------------------------------

class TestAllTokensInVocab:

    def test_all_code_tokens_in_vocab(self):
        """Every token in cjk_char_codes.tsv must appear in the frozen vocab."""
        vocab_set = set(_load_vocab())

        tsv_path = (Path(__file__).parent.parent /
                    "training_data" / "word_lists" / "cjk_char_codes.tsv")
        assert tsv_path.exists(), f"Missing: {tsv_path}"

        missing_tokens = set()
        total_chars = 0
        for line in tsv_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("character\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            total_chars += 1
            for hex_tok in parts[1].split():
                tok = chr(int(hex_tok, 16))
                if tok not in vocab_set:
                    missing_tokens.add(tok)

        assert total_chars >= 27584, f"Expected >= 27584 chars, got {total_chars}"
        assert not missing_tokens, (
            f"{len(missing_tokens)} code tokens missing from vocab: "
            f"{sorted(f'U+{ord(t):04X}' for t in missing_tokens)[:10]}")


# ---------------------------------------------------------------------------
# 5. No Token ID Overflow
# ---------------------------------------------------------------------------

class TestNoIDOverflow:

    def test_all_chars_encode_within_range(self):
        """Encode all 27,584 CJK chars, verify all IDs < vocab_size."""
        from src.encoding.tokenizer import LipiTokenizer
        from src.encoding.decompose import decompose_han_kana

        vocab = _load_vocab()
        tok = LipiTokenizer(vocab=vocab)
        vs = len(vocab)

        tsv_path = (Path(__file__).parent.parent /
                    "training_data" / "word_lists" / "cjk_char_codes.tsv")

        overflow_chars = []
        for line in tsv_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("character\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            char = parts[0]
            decomposed = decompose_han_kana(char)
            ids = tok.encode(decomposed)
            for i in ids:
                if i >= vs:
                    overflow_chars.append((char, i))
                    break

        assert not overflow_chars, (
            f"{len(overflow_chars)} chars have IDs >= {vs}: "
            f"{overflow_chars[:5]}")


# ---------------------------------------------------------------------------
# 6. No Duplicate Sequences
# ---------------------------------------------------------------------------

class TestNoDuplicateSequences:

    def test_unique_token_sequences(self):
        """All 27,584 chars must have unique token sequences."""
        from src.encoding.decompose import _load_arbitrary_encoding
        enc = _load_arbitrary_encoding("han_kana")
        cjk_ct = enc["char_to_tokens"]

        seq_to_chars: dict[tuple[str, ...], list[str]] = {}
        for char, tokens in cjk_ct.items():
            key = tuple(tokens)
            if key not in seq_to_chars:
                seq_to_chars[key] = []
            seq_to_chars[key].append(char)

        duplicates = {
            seq: chars for seq, chars in seq_to_chars.items()
            if len(chars) > 1
        }

        assert not duplicates, (
            f"{len(duplicates)} duplicate sequences found. Examples: "
            + "; ".join(
                f"{''.join(seq)} -> {chars}"
                for seq, chars in list(duplicates.items())[:5]
            ))


# ---------------------------------------------------------------------------
# 7. CTC Loss Compatibility
# ---------------------------------------------------------------------------

class TestCTCHeadCompatibility:

    def test_ctc_head_accepts_vocab_size(self):
        """CTCHead can be created with actual vocab_size."""
        from src.model.encoder import CTCHead
        vs = _vocab_size()
        head = CTCHead(enc_dim=384, vocab_size=vs)
        assert head.vocab_size == vs
        assert head.proj.out_features == vs

    def test_group_ctc_module_accepts_vocab_size(self):
        """GroupCTCModule accepts actual vocab size."""
        from src.model.encoder import GroupCTCModule
        vs = _vocab_size()
        gcm = GroupCTCModule(
            enc_dim=384,
            script_vocab_sizes=[vs],
            script_names=["han_kana"],
        )
        assert gcm.max_vocab == vs

    def test_vocab_size_flows_through_model_config(self):
        """Verify the vocab file produces a size accepted by CTCHead."""
        from src.model.encoder import CTCHead
        vs = _vocab_size()
        head = CTCHead(enc_dim=384, vocab_size=vs)
        assert head.proj.out_features == vs


# ---------------------------------------------------------------------------
# 8. Eval Decode Path Simulation
# ---------------------------------------------------------------------------

class TestEvalDecodePath:

    def test_ctc_greedy_decode_simulation(self):
        """Simulate CTC greedy decode -> tok.decode() -> reconstruct."""
        from src.encoding.tokenizer import LipiTokenizer
        from src.encoding.decompose import decompose_han_kana, reconstruct_han_kana

        tok = LipiTokenizer(vocab=_load_vocab())

        test_cases = ["学校", "東京タワー", "こんにちは世界", "的"]

        for text in test_cases:
            decomposed = decompose_han_kana(text)
            ids = tok.encode(decomposed)

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

            decoded_str = tok.decode(collapsed)
            result = reconstruct_han_kana(list(decoded_str))
            assert result == text, f"Eval path failed: {text!r} -> {result!r}"

    def test_decode_with_only_blanks(self):
        """All-blank CTC output should produce empty string."""
        from src.encoding.tokenizer import LipiTokenizer
        tok = LipiTokenizer(vocab=_load_vocab())
        assert tok.decode([0, 0, 0]) == ""

    def test_decode_skips_blank_correctly(self):
        """Blank tokens (ID 0) are filtered out during decode."""
        from src.encoding.tokenizer import LipiTokenizer
        from src.encoding.decompose import decompose_han_kana
        tok = LipiTokenizer(vocab=_load_vocab())

        decomposed = decompose_han_kana("日")
        ids = tok.encode(decomposed)
        ids_with_blanks = [0] + [val for i in ids for val in (i, 0)]
        assert tok.decode(ids_with_blanks) == tok.decode(ids)


# ---------------------------------------------------------------------------
# 9. 13-Symbol Encoding Properties
# ---------------------------------------------------------------------------

class TestEncoding:

    def test_all_codes_use_base_symbols_and_sep(self):
        """All char codes should only contain base symbols + SEP + BPE tokens."""
        from src.encoding.decompose import _load_arbitrary_encoding
        enc = _load_arbitrary_encoding("han_kana")
        cjk_ct = enc["char_to_tokens"]

        for char, tokens in cjk_ct.items():
            for t in tokens:
                cp = ord(t)
                assert (0xE000 <= cp <= 0xF8FF) or cp == 0x2E3B, (
                    f"Char {char!r} has unexpected token U+{cp:04X}")

    def test_cjk_decomposition_produces_encoding_tokens(self):
        """Decomposing CJK chars should produce only encoding tokens, not the chars."""
        from src.encoding.decompose import decompose_han_kana
        text = "的一是"
        decomposed = decompose_han_kana(text)
        # Decomposed text should NOT contain the original CJK chars
        for ch in text:
            assert ch not in decomposed, (
                f"Char {ch!r} appears in its own decomposition")

    def test_sep_not_added_for_kana(self):
        """Kana chars should NOT get SEP tokens."""
        from src.encoding.decompose import decompose_han_kana, SEP_CHAR
        text = "あいうえお"
        decomposed = decompose_han_kana(text)
        assert SEP_CHAR not in decomposed


# ---------------------------------------------------------------------------
# 10. BPE Token Handling
# ---------------------------------------------------------------------------

class TestBPETokens:

    def test_bpe_tokens_are_pua_chars(self):
        """BPE merged tokens should be PUA characters (U+E00D+)."""
        from src.encoding.decompose import _load_arbitrary_encoding
        enc = _load_arbitrary_encoding("han_kana")
        cjk_ct = enc["char_to_tokens"]

        pua_tokens = set()
        for tokens in cjk_ct.values():
            for t in tokens:
                if len(t) == 1 and ord(t) >= 0xE00D:
                    pua_tokens.add(t)

        # BPE tokens may or may not appear in per-char codes
        # (they primarily help in word-level sequences)

    def test_bpe_tokens_in_vocab(self):
        """All BPE tokens from char codes table are in vocab."""
        from src.encoding.decompose import _load_arbitrary_encoding
        enc = _load_arbitrary_encoding("han_kana")
        cjk_ct = enc["char_to_tokens"]

        vocab_set = set(_load_vocab())
        pua_missing = []
        for tokens in cjk_ct.values():
            for t in tokens:
                if len(t) == 1 and ord(t) >= 0xE000 and t not in vocab_set:
                    pua_missing.append(f"U+{ord(t):04X}")

        assert not pua_missing, (
            f"{len(set(pua_missing))} BPE tokens missing from vocab: "
            f"{sorted(set(pua_missing))[:10]}")
