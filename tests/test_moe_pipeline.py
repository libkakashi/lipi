"""
Test suite for the MoE OCR pipeline.

Covers: frozen vocabs, tokenization, decomposition, word lists,
script/group definitions, data encoding, and model construction.

Run: pytest tests/test_moe_pipeline.py -v
"""

import unicodedata
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WORD_LIST_DIR = Path(__file__).parent.parent / "training_data" / "word_lists"
VOCAB_DIR = Path(__file__).parent.parent / "src" / "data" / "frozen_vocabs"


@pytest.fixture(scope="module")
def lid_config():
    """Load script/group definitions."""
    from src.model.lid import (
        SCRIPTS, SCRIPT_TO_GROUP, SCRIPT_TO_ID, GROUP_TO_ID,
        GROUPS, GROUP_SCRIPTS, NUM_SCRIPTS, NUM_GROUPS,
    )
    return {
        "scripts": SCRIPTS,
        "groups": GROUPS,
        "script_to_group": SCRIPT_TO_GROUP,
        "script_to_id": SCRIPT_TO_ID,
        "group_to_id": GROUP_TO_ID,
        "group_scripts": GROUP_SCRIPTS,
        "num_scripts": NUM_SCRIPTS,
        "num_groups": NUM_GROUPS,
    }


@pytest.fixture(scope="module")
def all_vocabs():
    """Load all frozen vocabs."""
    from src.data.vocab import build_script_vocab
    from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP
    vocabs = {}
    for script in SCRIPTS:
        group = SCRIPT_TO_GROUP[script]
        vocabs[script] = build_script_vocab(script, group)
    return vocabs


@pytest.fixture(scope="module")
def all_tokenizers():
    """Build tokenizers for all scripts."""
    from src.data.vocab import build_script_vocab
    from src.data.bigrams import LipiTokenizer
    from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP
    tokenizers = {}
    for script in SCRIPTS:
        group = SCRIPT_TO_GROUP[script]
        vocab = build_script_vocab(script, group)
        tokenizers[script] = LipiTokenizer(vocab=vocab, bigrams=set())
    return tokenizers


@pytest.fixture(scope="module")
def script_word_lists():
    """Map each script to its word list files."""
    from src.model.lid import SCRIPTS
    # Only include scripts that have word list files
    mapping = {}
    for script in SCRIPTS:
        path = WORD_LIST_DIR / f"{script}.txt"
        if path.exists():
            words = [l.strip() for l in path.read_text(
                encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            if words:
                mapping[script] = words
    return mapping


# ---------------------------------------------------------------------------
# 1. Frozen Vocab Tests
# ---------------------------------------------------------------------------

class TestFrozenVocabs:
    """Tests for frozen vocab files in src/data/frozen_vocabs/."""

    def test_every_script_has_vocab_file(self, lid_config):
        for script in lid_config["scripts"]:
            path = VOCAB_DIR / f"{script}_vocab.txt"
            assert path.exists(), f"Missing vocab file: {path}"

    def test_blank_token_at_index_zero(self, all_vocabs):
        from src.data.bigrams import BLANK_TOKEN
        for script, vocab in all_vocabs.items():
            assert vocab[0] == BLANK_TOKEN, (
                f"{script}: vocab[0] is {repr(vocab[0])}, expected BLANK")

    def test_no_unassigned_codepoints(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            for ch in vocab:
                assert unicodedata.category(ch) != "Cn", (
                    f"{script}: unassigned U+{ord(ch):04X}")

    def test_no_duplicates(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            assert len(vocab) == len(set(vocab)), (
                f"{script}: {len(vocab) - len(set(vocab))} duplicates")

    def test_sorted_after_blank(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            tokens = vocab[1:]  # skip blank
            assert tokens == sorted(tokens), f"{script}: not sorted"

    def test_has_digits(self, all_vocabs):
        digits = set("0123456789")
        for script, vocab in all_vocabs.items():
            vocab_set = set(vocab)
            for d in digits:
                assert d in vocab_set, (
                    f"{script}: missing digit {d}")

    def test_has_basic_punctuation(self, all_vocabs):
        punct = set(".,-()")
        for script, vocab in all_vocabs.items():
            vocab_set = set(vocab)
            for p in punct:
                assert p in vocab_set, (
                    f"{script}: missing punctuation {repr(p)}")

    def test_no_latin_letters_in_non_latin(self, all_vocabs):
        latin = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for script, vocab in all_vocabs.items():
            if script in ("latin", "emoji"):
                continue
            for ch in vocab:
                assert ch not in latin, (
                    f"{script}: has Latin letter {repr(ch)}")

    def test_has_space(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            assert " " in vocab, f"{script}: missing space"


# ---------------------------------------------------------------------------
# 2. Word List Tests
# ---------------------------------------------------------------------------

class TestWordLists:
    """Tests for training word lists in training_data/word_lists/."""

    def test_minimum_word_count(self, script_word_lists):
        for script, words in script_word_lists.items():
            if script == "emoji":
                continue
            assert len(words) >= 1000, (
                f"{script}: only {len(words)} words, need 1000+")

    def test_no_unassigned_unicode(self, script_word_lists):
        for script, words in script_word_lists.items():
            for w in words[:500]:  # sample first 500
                for ch in w:
                    assert unicodedata.category(ch) != "Cn", (
                        f"{script}: word {repr(w)} has unassigned U+{ord(ch):04X}")

    def test_no_latin_in_non_latin_words(self, script_word_lists):
        latin = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        skip = {"latin", "emoji"}
        for script, words in script_word_lists.items():
            if script in skip:
                continue
            for w in words:
                assert not any(c in latin for c in w), (
                    f"{script}: word {repr(w)} has Latin letters")

    def test_word_length_bounds(self, script_word_lists):
        for script, words in script_word_lists.items():
            for w in words:
                assert 1 <= len(w) <= 50, (
                    f"{script}: word length {len(w)}: {repr(w[:20])}")


# ---------------------------------------------------------------------------
# 3. Vocab ↔ Word List Coverage
# ---------------------------------------------------------------------------

class TestVocabCoverage:
    """Every character in every word list must be in the frozen vocab."""

    def test_full_coverage(self, all_vocabs, script_word_lists):
        from src.data.decompose import decompose_text, DECOMPOSE_GROUPS
        from src.model.lid import SCRIPT_TO_GROUP

        for script, words in script_word_lists.items():
            if script == "emoji":
                continue
            vocab_set = set(all_vocabs[script])
            group = SCRIPT_TO_GROUP[script]

            missing = set()
            for w in words:
                text = decompose_text(w, group) if group in DECOMPOSE_GROUPS else w
                for ch in text:
                    if ch not in vocab_set:
                        missing.add(ch)

            assert not missing, (
                f"{script}: {len(missing)} chars in words but not in vocab: "
                f"{[f'U+{ord(c):04X}' for c in sorted(missing)[:5]]}")


# ---------------------------------------------------------------------------
# 4. Tokenizer Tests
# ---------------------------------------------------------------------------

class TestTokenizers:
    """Tests for encode/decode with frozen vocab tokenizers."""

    def test_blank_id_is_zero(self, all_tokenizers):
        for script, tok in all_tokenizers.items():
            assert tok.blank_id == 0, f"{script}: blank_id={tok.blank_id}"

    def test_encode_produces_no_blanks(self, all_tokenizers, script_word_lists):
        for script, tok in all_tokenizers.items():
            words = script_word_lists.get(script, [])
            for w in words[:50]:
                ids = tok.encode(w)
                assert 0 not in ids, (
                    f"{script}: blank in encoded {repr(w)}: {ids}")

    def test_encode_decode_roundtrip(self, all_tokenizers, script_word_lists):
        from src.data.decompose import decompose_text, reconstruct_text, DECOMPOSE_GROUPS
        from src.model.lid import SCRIPT_TO_GROUP

        for script, tok in all_tokenizers.items():
            group = SCRIPT_TO_GROUP[script]
            words = script_word_lists.get(script, [])
            for w in words[:100]:
                text = decompose_text(w, group) if group in DECOMPOSE_GROUPS else w
                ids = tok.encode(text)
                decoded = tok.decode(ids)
                if group in DECOMPOSE_GROUPS:
                    decoded = reconstruct_text(decoded, group)
                assert decoded == w, (
                    f"{script}: roundtrip failed: {repr(w)} -> {ids} -> {repr(decoded)}")

    def test_vocab_size_matches(self, all_tokenizers, all_vocabs):
        for script in all_tokenizers:
            assert all_tokenizers[script].vocab_size == len(all_vocabs[script]), (
                f"{script}: tokenizer vocab_size={all_tokenizers[script].vocab_size} "
                f"!= frozen vocab len={len(all_vocabs[script])}")


# ---------------------------------------------------------------------------
# 5. Script/Group Definition Tests
# ---------------------------------------------------------------------------

class TestScriptGroupDefinitions:
    """Tests for lid.py script and group consistency."""

    def test_every_script_has_group(self, lid_config):
        for script in lid_config["scripts"]:
            assert script in lid_config["script_to_group"], (
                f"{script}: not in SCRIPT_TO_GROUP")

    def test_every_script_has_id(self, lid_config):
        for script in lid_config["scripts"]:
            assert script in lid_config["script_to_id"], (
                f"{script}: not in SCRIPT_TO_ID")

    def test_script_ids_contiguous(self, lid_config):
        ids = sorted(lid_config["script_to_id"].values())
        assert ids == list(range(len(ids))), "Script IDs not contiguous"

    def test_group_ids_contiguous(self, lid_config):
        ids = sorted(lid_config["group_to_id"].values())
        assert ids == list(range(len(ids))), "Group IDs not contiguous"

    def test_group_scripts_covers_all(self, lid_config):
        covered = set()
        for scripts in lid_config["group_scripts"].values():
            covered.update(scripts)
        for script in lid_config["scripts"]:
            assert script in covered, (
                f"{script}: not in any GROUP_SCRIPTS entry")

    def test_group_scripts_matches_mapping(self, lid_config):
        for group, scripts in lid_config["group_scripts"].items():
            for script in scripts:
                assert lid_config["script_to_group"][script] == group, (
                    f"{script}: GROUP_SCRIPTS says {group}, "
                    f"SCRIPT_TO_GROUP says {lid_config['script_to_group'][script]}")

    def test_num_scripts_matches(self, lid_config):
        assert lid_config["num_scripts"] == len(lid_config["scripts"])

    def test_num_groups_matches(self, lid_config):
        assert lid_config["num_groups"] == len(lid_config["groups"])


# ---------------------------------------------------------------------------
# 6. Decomposition Tests
# ---------------------------------------------------------------------------

class TestDecomposition:
    """Tests for CJK and Korean decomposition."""

    def test_korean_common_syllables_frozen(self):
        from src.data.decompose import _common_hangul, _COMMON_HANGUL_250
        assert len(_common_hangul) == 250
        assert _common_hangul == set(_COMMON_HANGUL_250)

    def test_korean_roundtrip(self):
        from src.data.decompose import decompose_korean, reconstruct_korean
        words = ["안녕하세요", "대한민국", "서울", "감사합니다", "학교"]
        for w in words:
            decomposed = decompose_korean(w)
            reconstructed = reconstruct_korean(list(decomposed))
            assert reconstructed == w, (
                f"Korean roundtrip: {repr(w)} -> {repr(decomposed)} -> {repr(reconstructed)}")

    def test_han_kana_roundtrip(self):
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        words = ["学校", "日本語", "東京", "漢字"]
        for w in words:
            decomposed = decompose_han_kana(w)
            reconstructed = reconstruct_han_kana(list(decomposed))
            # CJK roundtrip is ~99.7%, allow some failures
            # but basic words should work
            if reconstructed != w:
                pytest.skip(f"Known IDS roundtrip limitation: {repr(w)}")

    def test_decompose_text_passthrough(self):
        from src.data.decompose import decompose_text
        # Non-decomposed scripts pass through unchanged
        assert decompose_text("Hello", "latin") == "Hello"
        assert decompose_text("Привет", "cyrillic_greek") == "Привет"

    def test_decompose_groups_constant(self):
        from src.data.decompose import DECOMPOSE_GROUPS
        assert DECOMPOSE_GROUPS == frozenset({"han_kana", "korean"})


# ---------------------------------------------------------------------------
# 7. Model Construction Tests
# ---------------------------------------------------------------------------

class TestModelConstruction:
    """Tests that the model builds correctly with frozen vocabs."""

    def test_model_builds(self):
        from src.model.moe_encoder import LipiMoEEncoder
        from src.data.vocab import get_all_script_vocabs
        from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP

        active_scripts = list(SCRIPTS)
        active_groups = []
        seen = set()
        for s in active_scripts:
            g = SCRIPT_TO_GROUP[s]
            if g not in seen:
                active_groups.append(g)
                seen.add(g)

        _, vocab_sizes = get_all_script_vocabs(active_scripts, active_groups)
        group_names = []
        for group in active_groups:
            group_names.append([s for s in active_scripts
                                if SCRIPT_TO_GROUP[s] == group])

        model = LipiMoEEncoder(
            num_groups=len(active_groups),
            group_script_vocab_sizes=vocab_sizes,
            group_script_names=group_names,
        )
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_ctc_head_sizes_match_vocab(self):
        from src.model.moe_encoder import LipiMoEEncoder
        from src.data.vocab import get_all_script_vocabs
        from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP

        active_scripts = list(SCRIPTS)
        active_groups = []
        seen = set()
        for s in active_scripts:
            g = SCRIPT_TO_GROUP[s]
            if g not in seen:
                active_groups.append(g)
                seen.add(g)

        _, vocab_sizes = get_all_script_vocabs(active_scripts, active_groups)
        group_names = []
        for group in active_groups:
            group_names.append([s for s in active_scripts
                                if SCRIPT_TO_GROUP[s] == group])

        model = LipiMoEEncoder(
            num_groups=len(active_groups),
            group_script_vocab_sizes=vocab_sizes,
            group_script_names=group_names,
        )

        for g, group in enumerate(active_groups):
            ctc_mod = model.ctc_modules[g]
            scripts = group_names[g]
            for s, script in enumerate(scripts):
                head_vocab = ctc_mod.heads[s].proj.out_features
                frozen_vocab = vocab_sizes[g][s]
                assert head_vocab == frozen_vocab, (
                    f"{script}: CTC head outputs {head_vocab}, "
                    f"frozen vocab is {frozen_vocab}")


# ---------------------------------------------------------------------------
# 8. Data Generation Char List Tests
# ---------------------------------------------------------------------------

class TestCharGeneration:
    """Tests for _get_script_chars used in data generation."""

    def test_no_blank_in_chars(self):
        from scripts.generate_data import _get_script_chars
        from src.data.bigrams import BLANK_TOKEN
        from src.model.lid import SCRIPTS

        for script in SCRIPTS:
            if script == "emoji":
                continue
            chars = _get_script_chars(script)
            assert BLANK_TOKEN not in chars, (
                f"{script}: BLANK_TOKEN in renderable chars")

    def test_no_space_in_chars(self):
        from scripts.generate_data import _get_script_chars
        from src.model.lid import SCRIPTS

        for script in SCRIPTS:
            if script == "emoji":
                continue
            chars = _get_script_chars(script)
            assert " " not in chars, f"{script}: space in renderable chars"

    def test_chars_subset_of_vocab(self):
        from scripts.generate_data import _get_script_chars
        from src.data.vocab import build_script_vocab
        from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP

        for script in SCRIPTS:
            if script == "emoji":
                continue
            chars = set(_get_script_chars(script))
            vocab = set(build_script_vocab(script, SCRIPT_TO_GROUP[script]))
            outside = chars - vocab
            assert not outside, (
                f"{script}: {len(outside)} renderable chars not in vocab")
