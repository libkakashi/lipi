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
    from src.encoding.config import get_han_codec
    return get_han_codec()


def _vocab_size():
    return _get_codec().vocab_size


def _get_split_codec(script):
    from src.encoding.config import get_han_codec
    return get_han_codec(script)


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
        # Han codec is CJK vocab minus kana; still capped at _CJK_VOCAB_SIZE.
        # Actual size depends on how many kana entries existed in cjk_vocab.txt.
        assert codec.vocab_size <= _CJK_VOCAB_SIZE
        assert codec.vocab_size > 1000, "Han codec suspiciously small"

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
        """Kana uses its own codec now (NoFusionCodec), not han."""
        from src.encoding.decompose import encode_text, decode_ids
        words = ["ひらがな", "カタカナ", "あいうえお"]
        for w in words:
            ids = encode_text(w, "kana")
            result = decode_ids(ids, "kana")
            assert result == w, f"Kana roundtrip failed: {w!r} -> {result!r}"

    def test_mixed_kana_cjk_via_segments(self):
        """Mixed Japanese text: kana via kana codec, kanji via han codec."""
        from src.encoding.decompose import encode_text, decode_ids
        kana_part = "こんにちは"
        han_part = "世界"
        # Each segment encodes/decodes through its own codec
        kana_ids = encode_text(kana_part, "kana")
        han_ids = encode_text(han_part, "han")
        assert decode_ids(kana_ids, "kana") == kana_part
        assert decode_ids(han_ids, "han") == han_part

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

    def test_fullwidth_katakana_roundtrip(self):
        """Fullwidth katakana should roundtrip cleanly through kana codec."""
        from src.encoding.decompose import encode_text, decode_ids
        fw = "アイウ"
        ids = encode_text(fw, "kana")
        result = decode_ids(ids, "kana")
        assert result == fw


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
        from src.model.blocks import CTCHead
        vs = _vocab_size()
        head = CTCHead(enc_dim=384, vocab_size=vs)
        assert head.vocab_size == vs
        assert head.proj.out_features == vs

    def test_group_ctc_module_accepts_vocab_size(self):
        from src.model.blocks import GroupCTCModule
        vs = _vocab_size()
        gcm = GroupCTCModule(
            enc_dim=384,
            script_vocab_sizes=[vs],
            script_names=["han"],
        )
        assert gcm.max_vocab == vs


# ---------------------------------------------------------------------------
# 7. Eval Decode Path Simulation
# ---------------------------------------------------------------------------

class TestEvalDecodePath:

    def test_ctc_greedy_decode_simulation(self):
        """Simulate CTC greedy decode -> decode_ids."""
        codec = _get_codec()
        # Pure kanji test cases (han codec doesn't handle kana anymore)
        test_cases = ["学校", "東京都", "人工知能", "的"]
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

    def test_encode_decode_han(self):
        from src.encoding.decompose import encode_text, decode_ids, script_vocab_size
        text = "東京都"  # all kanji
        ids = encode_text(text, "han")
        result = decode_ids(ids, "han")
        assert result == text

    def test_encode_decode_kana(self):
        from src.encoding.decompose import encode_text, decode_ids
        text = "タワー"  # all kana
        ids = encode_text(text, "kana")
        result = decode_ids(ids, "kana")
        assert result == text

    def test_vocab_size_via_decompose(self):
        from src.encoding.decompose import script_vocab_size
        vs = script_vocab_size("han")
        assert vs == _vocab_size()


# ---------------------------------------------------------------------------
# 9. Sparse / dense Han routing
# ---------------------------------------------------------------------------

class TestHanComplexitySplit:

    def test_taxonomy_names_and_flatten_ids(self):
        from src.taxonomy import GROUP_SCRIPTS, SCRIPT_TO_ID
        assert GROUP_SCRIPTS["han"] == ["han_sparse", "han_dense"]
        assert SCRIPT_TO_ID["han_sparse"] == 5
        assert SCRIPT_TO_ID["han_dense"] == 6
        assert SCRIPT_TO_ID["kana"] == 7

    def test_visual_complexity_examples(self):
        from src.encoding.han_split import (
            HAN_DENSE, HAN_SPARSE, han_script_for_char,
        )
        assert all(han_script_for_char(c) == HAN_SPARSE for c in "一山川明日")
        assert all(han_script_for_char(c) == HAN_DENSE for c in "語學漢鬱龍龜")

    def test_split_codecs_are_smaller_than_legacy_head(self):
        legacy_size = _get_codec().vocab_size
        sparse_size = _get_split_codec("han_sparse").vocab_size
        dense_size = _get_split_codec("han_dense").vocab_size
        assert sparse_size < legacy_size
        assert dense_size < legacy_size
        assert max(sparse_size, dense_size) < 2500

    def test_all_cjk_chars_roundtrip_through_assigned_head(self):
        from src.encoding.han_split import han_script_for_char
        failures = []
        for cp in list(range(0x3400, 0x4DC0)) + list(range(0x4E00, 0xA000)):
            char = chr(cp)
            script = han_script_for_char(char)
            codec = _get_split_codec(script)
            ids = codec.encode_text(char)
            if not ids or codec.decode_ids(ids) != char:
                failures.append((char, script, ids))
        assert not failures, failures[:10]

    def test_visual_alt_pairs_stay_inside_one_head(self):
        from src.encoding.han_split import han_script_for_char
        mapping = (Path(__file__).parent.parent / "training_data" / "corpora"
                   / "cjk_visual_mapping.tsv")
        failures = []
        for line in mapping.read_text(encoding="utf-8").splitlines()[1:]:
            char, _slot, match, *_ = line.split("\t")
            if han_script_for_char(char) != han_script_for_char(match):
                failures.append((char, match))
        assert not failures, failures[:10]

    def test_mixed_han_and_kana_split_into_maximal_runs(self):
        from src.data.script_detect import split_by_script
        assert split_by_script("山語川", "han_sparse") == [
            ("山", "han_sparse"),
            ("語", "han_dense"),
            ("川", "han_sparse"),
        ]
        assert split_by_script("山かな語", "han_sparse") == [
            ("山", "han_sparse"),
            ("かな", "kana"),
            ("語", "han_dense"),
        ]
        assert split_by_script("かな。", "kana") == [
            ("かな", "kana"),
            ("。", "han_sparse"),
        ]

    def test_group_ctc_module_has_two_heads(self):
        from src.model.blocks import GroupCTCModule
        sizes = [_get_split_codec(s).vocab_size
                 for s in ("han_sparse", "han_dense")]
        module = GroupCTCModule(
            enc_dim=32,
            script_vocab_sizes=sizes,
            script_names=["han_sparse", "han_dense"],
        )
        assert len(module.heads) == 2
        assert [head.vocab_size for head in module.heads] == sizes

    def test_legacy_checkpoint_warm_starts_both_heads(self):
        import torch
        from src.training.taxonomy_checkpoint import migrate_taxonomy_state

        old = _get_codec()
        sparse = _get_split_codec("han_sparse")
        dense = _get_split_codec("han_dense")
        old_weight = torch.arange(old.vocab_size * 2, dtype=torch.float32).reshape(-1, 2)
        old_bias = torch.arange(old.vocab_size, dtype=torch.float32)
        old_expert = torch.ones(2, 2)
        state = {
            "ctc_modules.0.heads.0.proj.weight": old_weight,
            "ctc_modules.0.heads.0.proj.bias": old_bias,
            "script_layers.0.routed_mlps.0.fc1.weight": old_expert,
        }
        current = {
            "ctc_modules.0.heads.0.proj.weight": torch.zeros(sparse.vocab_size, 2),
            "ctc_modules.0.heads.0.proj.bias": torch.zeros(sparse.vocab_size),
            "ctc_modules.0.heads.1.proj.weight": torch.zeros(dense.vocab_size, 2),
            "ctc_modules.0.heads.1.proj.bias": torch.zeros(dense.vocab_size),
            "script_layers.0.routed_mlps.6.fc1.weight": torch.zeros(2, 2),
        }
        changed = migrate_taxonomy_state(
            state,
            current,
            {"group_script_names": [["han"]]},
            {"group_script_names": [["han_sparse", "han_dense"]]},
        )
        assert changed == 5
        assert torch.equal(state["ctc_modules.0.heads.0.proj.weight"][0], old_weight[0])
        assert torch.equal(state["ctc_modules.0.heads.1.proj.weight"][0], old_weight[0])
        dense_token = dense.tokens.index("語") + 1
        old_token = old.tokens.index("語") + 1
        assert torch.equal(
            state["ctc_modules.0.heads.1.proj.weight"][dense_token],
            old_weight[old_token],
        )
        assert torch.equal(
            state["script_layers.0.routed_mlps.6.fc1.weight"],
            old_expert,
        )
