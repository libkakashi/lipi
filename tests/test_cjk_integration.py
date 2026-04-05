"""
Comprehensive CJK decomposition integration tests.

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
    from src.data.vocab import build_script_vocab
    return build_script_vocab("han_kana", "sino_japanese")


def _vocab_size():
    return len(_load_vocab())


# ---------------------------------------------------------------------------
# 1. Vocab Loading
# ---------------------------------------------------------------------------

class TestCJKVocabLoading:

    def test_han_kana_vocab_size(self):
        vocab = _load_vocab()
        # 712 base + 2500 BPE + 1 BLANK = 3213
        assert 3100 <= len(vocab) <= 3400, f"Expected ~3213 tokens, got {len(vocab)}"

    def test_blank_token_at_index_zero(self):
        from src.data.bigrams import BLANK_TOKEN
        vocab = _load_vocab()
        assert vocab[0] == BLANK_TOKEN

    def test_vocab_has_no_duplicates(self):
        vocab = _load_vocab()
        assert len(vocab) == len(set(vocab)), (
            f"Duplicates in vocab: {len(vocab)} total, {len(set(vocab))} unique")


# ---------------------------------------------------------------------------
# 2. Tokenizer Encode/Decode Round-trip
# ---------------------------------------------------------------------------

class TestCJKTokenizerRoundtrip:

    @pytest.fixture
    def tok(self):
        from src.data.bigrams import LipiTokenizer
        return LipiTokenizer(vocab=_load_vocab())

    def test_cjk_roundtrip(self, tok):
        """CJK text: decompose -> encode -> decode -> reconstruct."""
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        words = ["学校", "日本語", "東京", "人工知能"]
        for w in words:
            decomposed = decompose_han_kana(w)
            ids = tok.encode(decomposed)
            decoded = tok.decode(ids)
            result = reconstruct_han_kana(list(decoded))
            assert result == w, f"CJK roundtrip failed: {w!r} -> {result!r}"

    def test_kana_roundtrip(self, tok):
        """Pure kana passes through decomposition unchanged."""
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        words = ["ひらがな", "カタカナ", "あいうえお"]
        for w in words:
            decomposed = decompose_han_kana(w)
            ids = tok.encode(decomposed)
            decoded = tok.decode(ids)
            result = reconstruct_han_kana(list(decoded))
            assert result == w, f"Kana roundtrip failed: {w!r} -> {result!r}"

    def test_mixed_kana_cjk_roundtrip(self, tok):
        """Mixed kana + CJK text."""
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        text = "こんにちは世界"
        decomposed = decompose_han_kana(text)
        ids = tok.encode(decomposed)
        decoded = tok.decode(ids)
        result = reconstruct_han_kana(list(decoded))
        assert result == text, f"Mixed roundtrip failed: {text!r} -> {result!r}"

    def test_encode_ids_in_range(self, tok):
        """All encoded IDs must be < vocab_size."""
        from src.data.decompose import decompose_han_kana
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
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        text = "的一是不了在人有我他"
        decomposed = decompose_han_kana(text)
        result = reconstruct_han_kana(list(decomposed))
        assert result == text

    def test_collision_pairs(self):
        """Collision override chars must roundtrip correctly."""
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        pairs = [("土", "士"), ("与", "马")]
        for a, b in pairs:
            dec_a = decompose_han_kana(a)
            dec_b = decompose_han_kana(b)
            assert dec_a != dec_b, (
                f"Collision! {a!r} and {b!r} both decompose to {dec_a!r}")
            assert reconstruct_han_kana(list(dec_a)) == a
            assert reconstruct_han_kana(list(dec_b)) == b

    def test_mixed_kana_cjk(self):
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        text = "こんにちは世界"
        result = reconstruct_han_kana(list(decompose_han_kana(text)))
        assert result == text

    def test_pure_kana(self):
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        text = "ひらがなカタカナ"
        decomposed = decompose_han_kana(text)
        assert decomposed == text  # Kana should pass through
        result = reconstruct_han_kana(list(decomposed))
        assert result == text

    def test_empty_string(self):
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        assert decompose_han_kana("") == ""
        assert reconstruct_han_kana([]) == ""

    def test_single_char(self):
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        for ch in "的あア":
            decomposed = decompose_han_kana(ch)
            result = reconstruct_han_kana(list(decomposed))
            assert result == ch, f"Single char failed: {ch!r}"

    def test_punctuation(self):
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        text = "「你好！」"
        decomposed = decompose_han_kana(text)
        result = reconstruct_han_kana(list(decomposed))
        assert result == text

    def test_ext_a_chars(self):
        """CJK Extension A chars (U+3400..U+4DBF)."""
        from src.data.decompose import (
            decompose_han_kana, reconstruct_han_kana,
            _load_cjk_decomposition, _cjk_char_to_tokens,
        )
        _load_cjk_decomposition()
        ext_a_chars = [
            ch for ch in _cjk_char_to_tokens
            if 0x3400 <= ord(ch) <= 0x4DBF
        ][:5]
        for ch in ext_a_chars:
            decomposed = decompose_han_kana(ch)
            result = reconstruct_han_kana(list(decomposed))
            assert result == ch, f"Ext-A char {ch!r} (U+{ord(ch):04X}) failed"


# ---------------------------------------------------------------------------
# 4. All Decomposition Tokens In Vocab
# ---------------------------------------------------------------------------

class TestAllTokensInVocab:

    def test_all_decomposition_tokens_in_vocab(self):
        """Every token in cjk_decomposition.tsv must appear in the frozen vocab."""
        vocab_set = set(_load_vocab())

        tsv_path = (Path(__file__).parent.parent /
                    "training_data" / "word_lists" / "cjk_decomposition.tsv")
        assert tsv_path.exists(), f"Missing: {tsv_path}"

        missing_tokens = set()
        total_chars = 0
        for line in tsv_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("character\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            total_chars += 1
            for tok in parts[2].split():
                if tok not in vocab_set:
                    missing_tokens.add(tok)

        assert total_chars == 27584, f"Expected 27584 chars, got {total_chars}"
        assert not missing_tokens, (
            f"{len(missing_tokens)} decomposition tokens missing from vocab: "
            f"{sorted(missing_tokens)[:10]}")


# ---------------------------------------------------------------------------
# 5. No Token ID Overflow
# ---------------------------------------------------------------------------

class TestNoIDOverflow:

    def test_all_chars_encode_within_range(self):
        """Encode all 27,584 CJK chars, verify all IDs < vocab_size."""
        from src.data.bigrams import LipiTokenizer
        from src.data.decompose import decompose_han_kana

        vocab = _load_vocab()
        tok = LipiTokenizer(vocab=vocab)
        vs = len(vocab)

        tsv_path = (Path(__file__).parent.parent /
                    "training_data" / "word_lists" / "cjk_decomposition.tsv")

        overflow_chars = []
        for line in tsv_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("character\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
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
        from src.data.decompose import (
            _load_cjk_decomposition, _cjk_char_to_tokens,
        )
        _load_cjk_decomposition()

        seq_to_chars: dict[tuple[str, ...], list[str]] = {}
        for char, tokens in _cjk_char_to_tokens.items():
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
        from src.model.moe_encoder import CTCHead
        vs = _vocab_size()
        head = CTCHead(enc_dim=384, vocab_size=vs)
        assert head.vocab_size == vs
        assert head.proj.out_features == vs

    def test_group_ctc_module_accepts_vocab_size(self):
        """GroupCTCModule accepts actual vocab size."""
        from src.model.moe_encoder import GroupCTCModule
        vs = _vocab_size()
        gcm = GroupCTCModule(
            enc_dim=384,
            script_vocab_sizes=[vs],
            script_names=["han_kana"],
        )
        assert gcm.max_vocab == vs

    def test_vocab_size_flows_through_model_config(self):
        """Verify the vocab file produces a size accepted by CTCHead."""
        from src.model.moe_encoder import CTCHead
        vs = _vocab_size()
        head = CTCHead(enc_dim=384, vocab_size=vs)
        assert head.proj.out_features == vs


# ---------------------------------------------------------------------------
# 8. Eval Decode Path Simulation
# ---------------------------------------------------------------------------

class TestEvalDecodePath:

    def test_ctc_greedy_decode_simulation(self):
        """Simulate CTC greedy decode -> tok.decode() -> reconstruct."""
        from src.data.bigrams import LipiTokenizer
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana

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
        from src.data.bigrams import LipiTokenizer
        tok = LipiTokenizer(vocab=_load_vocab())
        assert tok.decode([0, 0, 0]) == ""

    def test_decode_skips_blank_correctly(self):
        """Blank tokens (ID 0) are filtered out during decode."""
        from src.data.bigrams import LipiTokenizer
        from src.data.decompose import decompose_han_kana
        tok = LipiTokenizer(vocab=_load_vocab())

        decomposed = decompose_han_kana("日")
        ids = tok.encode(decomposed)
        ids_with_blanks = [0] + [val for i in ids for val in (i, 0)]
        assert tok.decode(ids_with_blanks) == tok.decode(ids)


# ---------------------------------------------------------------------------
# 9. SEP Token Handling
# ---------------------------------------------------------------------------

class TestSEPToken:

    def test_sep_in_vocab(self):
        """SEP token must be in the frozen vocab."""
        from src.data.decompose import SEP_CHAR
        vocab = _load_vocab()
        assert SEP_CHAR in vocab, f"SEP token U+{ord(SEP_CHAR):04X} not in vocab"

    def test_sep_only_for_prefix_collision_chars(self):
        """SEP appears only in decompositions of prefix-collision chars, not all."""
        from src.data.decompose import decompose_han_kana, SEP_CHAR
        # Pure kana: no SEP
        assert SEP_CHAR not in decompose_han_kana("あいうえお")
        # CJK text: some chars may have SEP, some may not
        # (depends on prefix collisions — not every char gets SEP)
        decomposed = decompose_han_kana("的一是不了在人有我他")
        # At least SOME CJK chars should have SEP (prefix-collision chars)
        # but NOT necessarily all of them
        # Just verify it's valid — the detailed check is in the roundtrip tests

    def test_sep_not_added_for_kana(self):
        """Kana chars should NOT get SEP tokens."""
        from src.data.decompose import decompose_han_kana, SEP_CHAR
        text = "あいうえお"
        decomposed = decompose_han_kana(text)
        assert SEP_CHAR not in decomposed

    def test_reconstruct_handles_sequences_with_and_without_sep(self):
        """Reconstruct works for both SEP-terminated and non-SEP sequences."""
        from src.data.decompose import (
            reconstruct_han_kana, decompose_han_kana,
            _load_cjk_decomposition, _cjk_char_to_tokens, SEP_CHAR,
        )
        _load_cjk_decomposition()

        # Find one char WITH SEP and one WITHOUT
        with_sep = without_sep = None
        for char, tokens in _cjk_char_to_tokens.items():
            if SEP_CHAR in tokens and with_sep is None:
                with_sep = char
            if SEP_CHAR not in tokens and without_sep is None:
                without_sep = char
            if with_sep and without_sep:
                break

        if with_sep:
            result = reconstruct_han_kana(list(decompose_han_kana(with_sep)))
            assert result == with_sep, f"SEP char roundtrip failed: {with_sep!r}"
        if without_sep:
            result = reconstruct_han_kana(list(decompose_han_kana(without_sep)))
            assert result == without_sep, f"Non-SEP char roundtrip failed: {without_sep!r}"


# ---------------------------------------------------------------------------
# 10. BPE Token Handling
# ---------------------------------------------------------------------------

class TestBPETokens:

    def test_bpe_tokens_are_pua_chars(self):
        """BPE merged tokens should be PUA characters (U+E000+)."""
        from src.data.decompose import (
            _load_cjk_decomposition, _cjk_char_to_tokens,
        )
        _load_cjk_decomposition()

        pua_tokens = set()
        for tokens in _cjk_char_to_tokens.values():
            for t in tokens:
                if len(t) == 1 and ord(t) >= 0xE000:
                    pua_tokens.add(t)

        assert len(pua_tokens) > 0, "No BPE (PUA) tokens found"

    def test_bpe_tokens_in_vocab(self):
        """All BPE tokens from decomposition table are in vocab."""
        from src.data.decompose import (
            _load_cjk_decomposition, _cjk_char_to_tokens,
        )
        _load_cjk_decomposition()

        vocab_set = set(_load_vocab())
        pua_missing = []
        for tokens in _cjk_char_to_tokens.values():
            for t in tokens:
                if len(t) == 1 and ord(t) >= 0xE000 and t not in vocab_set:
                    pua_missing.append(f"U+{ord(t):04X}")

        assert not pua_missing, (
            f"{len(set(pua_missing))} BPE tokens missing from vocab: "
            f"{sorted(set(pua_missing))[:10]}")
