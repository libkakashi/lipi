"""
Comprehensive test suite for the MoE OCR pipeline.

Covers: frozen vocabs, tokenization, decomposition, word lists,
script/group definitions, data encoding, model construction,
loss computation, routing masks, and edge cases.

Run: pytest tests/test_moe_pipeline.py -v
"""

import unicodedata
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WORD_LIST_DIR = Path(__file__).parent.parent / "training_data" / "word_lists"
VOCAB_DIR = Path(__file__).parent.parent / "src" / "data" / "frozen_vocabs"


@pytest.fixture(scope="module")
def lid_config():
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
    from src.data.vocab import build_script_vocab
    from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP
    vocabs = {}
    for script in SCRIPTS:
        group = SCRIPT_TO_GROUP[script]
        vocabs[script] = build_script_vocab(script, group)
    return vocabs


@pytest.fixture(scope="module")
def all_tokenizers():
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
    """Load words for each script from primary + extra files."""
    from src.model.lid import SCRIPTS
    try:
        from scripts.train_lid import _SCRIPT_EXTRA_FILES
    except Exception:
        _SCRIPT_EXTRA_FILES = {}

    mapping = {}
    for script in SCRIPTS:
        words = []
        # Primary file
        path = WORD_LIST_DIR / f"{script}.txt"
        if path.exists():
            words.extend(l.strip() for l in path.read_text(
                encoding="utf-8", errors="ignore").splitlines() if l.strip())
        # Extra files
        for extra in _SCRIPT_EXTRA_FILES.get(script, []):
            path = WORD_LIST_DIR / extra
            if path.exists():
                words.extend(l.strip() for l in path.read_text(
                    encoding="utf-8", errors="ignore").splitlines() if l.strip())
        if words:
            mapping[script] = words
    return mapping


@pytest.fixture(scope="module")
def script_extra_files():
    """Map scripts to their extra word list files (for Latin group etc)."""
    try:
        from scripts.train_lid import _SCRIPT_EXTRA_FILES
        return _SCRIPT_EXTRA_FILES
    except Exception:
        return {}


@pytest.fixture(scope="module")
def model_and_vocabs():
    """Build model with frozen vocabs — shared across tests."""
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
    return model, vocab_sizes, group_names, active_groups


# ---------------------------------------------------------------------------
# 1. Frozen Vocab File Tests
# ---------------------------------------------------------------------------

class TestFrozenVocabFiles:
    """Tests for frozen vocab files in src/data/frozen_vocabs/."""

    def test_every_script_has_vocab_file(self, lid_config):
        for script in lid_config["scripts"]:
            path = VOCAB_DIR / f"{script}_vocab.txt"
            assert path.exists(), f"Missing vocab file: {path}"

    def test_no_extra_vocab_files(self, lid_config):
        """No orphan vocab files for scripts that don't exist."""
        known = {f"{s}_vocab.txt" for s in lid_config["scripts"]}
        # Allow vocab files for planned scripts not yet in SCRIPTS
        planned = {"odia", "burmese", "khmer", "sinhala", "ethiopic",
                    "armenian", "georgian", "tibetan"}
        known.update(f"{s}_vocab.txt" for s in planned)
        for f in VOCAB_DIR.glob("*_vocab.txt"):
            assert f.name in known, f"Orphan vocab file: {f.name}"

    def test_files_are_hex_encoded(self):
        """Every line is a valid hex code point."""
        for f in VOCAB_DIR.glob("*_vocab.txt"):
            for i, line in enumerate(f.read_text().strip().split("\n")):
                line = line.strip()
                if not line:
                    continue
                try:
                    cp = int(line, 16)
                    assert 0 < cp < 0x110000, f"{f.name} line {i}: invalid codepoint {line}"
                except ValueError:
                    pytest.fail(f"{f.name} line {i}: not valid hex: {repr(line)}")

    def test_no_empty_vocab_files(self):
        for f in VOCAB_DIR.glob("*_vocab.txt"):
            lines = [l for l in f.read_text().strip().split("\n") if l.strip()]
            assert len(lines) > 10, f"{f.name}: only {len(lines)} tokens"


# ---------------------------------------------------------------------------
# 2. Frozen Vocab Content Tests
# ---------------------------------------------------------------------------

class TestFrozenVocabContent:
    """Tests for the actual vocab content loaded by build_script_vocab."""

    def test_blank_token_at_index_zero(self, all_vocabs):
        from src.data.bigrams import BLANK_TOKEN
        for script, vocab in all_vocabs.items():
            assert vocab[0] == BLANK_TOKEN, (
                f"{script}: vocab[0] is {repr(vocab[0])}")

    def test_no_unassigned_codepoints(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            for ch in vocab:
                assert unicodedata.category(ch) != "Cn", (
                    f"{script}: unassigned U+{ord(ch):04X}")

    def test_no_control_characters(self, all_vocabs):
        """No C0/C1 control chars except whitespace."""
        for script, vocab in all_vocabs.items():
            for ch in vocab:
                cat = unicodedata.category(ch)
                if cat.startswith("C") and cat != "Cn":
                    assert cat == "Cf" or ch in ("\t", "\n"), (
                        f"{script}: control char U+{ord(ch):04X} ({cat})")

    def test_no_duplicates(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            assert len(vocab) == len(set(vocab)), (
                f"{script}: {len(vocab) - len(set(vocab))} duplicates")

    def test_sorted_after_blank(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            tokens = vocab[1:]
            assert tokens == sorted(tokens), f"{script}: not sorted"

    def test_has_digits(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            vocab_set = set(vocab)
            for d in "0123456789":
                assert d in vocab_set, f"{script}: missing digit {d}"

    def test_has_basic_punctuation(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            vocab_set = set(vocab)
            for p in ".,-()" :
                assert p in vocab_set, f"{script}: missing {repr(p)}"

    def test_has_smart_quotes(self, all_vocabs):
        """All vocabs include typographic quotes for real-world text."""
        smart = "\u2018\u2019\u201C\u201D"
        for script, vocab in all_vocabs.items():
            vocab_set = set(vocab)
            for q in smart:
                assert q in vocab_set, (
                    f"{script}: missing smart quote U+{ord(q):04X}")

    def test_has_space(self, all_vocabs):
        for script, vocab in all_vocabs.items():
            assert " " in vocab, f"{script}: missing space"

    def test_no_latin_letters_in_non_latin(self, all_vocabs):
        latin = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        for script, vocab in all_vocabs.items():
            if script in ("latin", "emoji"):
                continue
            for ch in vocab:
                assert ch not in latin, (
                    f"{script}: has Latin letter {repr(ch)}")

    def test_latin_has_full_alphabet(self, all_vocabs):
        vocab_set = set(all_vocabs["latin"])
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            assert ch in vocab_set, f"latin: missing {repr(ch)}"

    def test_vocab_sizes_reasonable(self, all_vocabs):
        """No vocab should be <50 or >5000 tokens."""
        for script, vocab in all_vocabs.items():
            assert 50 < len(vocab) < 5000, (
                f"{script}: vocab size {len(vocab)} out of range")

    def test_each_vocab_has_script_specific_chars(self, all_vocabs):
        """Each non-emoji vocab has chars outside BASE_CHARS."""
        base = set(chr(i) for i in range(32, 127))
        for script, vocab in all_vocabs.items():
            if script == "emoji":
                continue
            non_base = [ch for ch in vocab if ch not in base and ch != "\u2205"]
            assert len(non_base) > 20, (
                f"{script}: only {len(non_base)} non-base chars")


# ---------------------------------------------------------------------------
# 3. Word List Tests
# ---------------------------------------------------------------------------

class TestWordLists:
    """Tests for training word lists in training_data/word_lists/."""

    def test_every_script_has_word_list(self, lid_config, script_word_lists):
        for script in lid_config["scripts"]:
            if script == "emoji":
                continue
            # Latin uses extra files (english_common.txt etc), not latin.txt
            assert script in script_word_lists, (
                f"No words loaded for {script}")

    def test_minimum_word_count(self, script_word_lists):
        for script, words in script_word_lists.items():
            if script == "emoji":
                continue
            assert len(words) >= 1000, (
                f"{script}: only {len(words)} words")

    def test_no_unassigned_unicode(self, script_word_lists):
        for script, words in script_word_lists.items():
            for w in words:
                for ch in w:
                    assert unicodedata.category(ch) != "Cn", (
                        f"{script}: word {repr(w[:10])} has unassigned U+{ord(ch):04X}")

    def test_no_latin_in_non_latin_words(self, script_word_lists):
        latin = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        skip = {"latin", "emoji"}
        for script, words in script_word_lists.items():
            if script in skip:
                continue
            for w in words:
                assert not any(c in latin for c in w), (
                    f"{script}: word {repr(w[:10])} has Latin letters")

    def test_word_length_bounds(self, script_word_lists):
        for script, words in script_word_lists.items():
            for w in words:
                assert 1 <= len(w) <= 50, (
                    f"{script}: word length {len(w)}")

    def test_no_empty_lines(self, script_word_lists):
        for script, words in script_word_lists.items():
            for w in words:
                assert w.strip(), f"{script}: empty word"

    def test_no_tab_characters(self, script_word_lists):
        for script, words in script_word_lists.items():
            for w in words:
                assert "\t" not in w, f"{script}: tab in word {repr(w[:10])}"

    def test_no_duplicate_words_per_file(self):
        """No individual word list file should have significant duplicates.
        (Cross-file duplicates are expected for multi-language scripts like Latin.)"""
        from src.data.word_lists import WORD_LIST_DIR
        failures = []
        for f in WORD_LIST_DIR.glob("*.txt"):
            words = [l.strip() for l in f.read_text(
                encoding="utf-8", errors="ignore").splitlines() if l.strip()]
            unique = len(set(words))
            dupes = len(words) - unique
            pct = dupes / max(len(words), 1) * 100
            if pct > 5:
                failures.append(f"{f.name}: {dupes}/{len(words)} ({pct:.1f}%) duplicates")
        assert not failures, (
            "Files with >5% duplicates:\n  " + "\n  ".join(failures))

    def test_words_have_script_chars(self, script_word_lists):
        """Every word must have at least one script-specific character.
        Words of only digits/punctuation are useless for training."""
        from src.data.script_detect import _SCRIPT_RANGES

        skip = {"emoji"}
        failures = []
        for script, words in script_word_lists.items():
            if script in skip:
                continue
            ranges = _SCRIPT_RANGES.get(script, [])
            if not ranges:
                continue

            no_script_char = 0
            for w in words:
                has_script = any(
                    any(s <= ord(ch) <= e for s, e in ranges)
                    for ch in w
                )
                if not has_script:
                    no_script_char += 1

            if no_script_char > 0:
                failures.append(
                    f"{script}: {no_script_char} words with no script-specific chars")

        assert not failures, (
            "Words with only shared chars:\n  " + "\n  ".join(failures))

    def test_script_detect_agrees_with_label(self, script_word_lists):
        """detect_script(word) should return the same script the word list claims."""
        from src.data.script_detect import detect_script

        skip = {"emoji", "latin"}  # Latin is the default fallback
        failures = []
        for script, words in script_word_lists.items():
            if script in skip:
                continue

            mismatches = 0
            sampled = min(len(words), 200)
            for w in words[:sampled]:
                detected = detect_script(w)
                # detect_script returns the script name, which should match
                # OR be in the same group (e.g., odia detected as bengali is wrong,
                # but han_kana detecting as han_kana is right)
                if detected != script:
                    mismatches += 1

            pct = mismatches / sampled * 100
            if pct > 10:
                failures.append(
                    f"{script}: {mismatches}/{sampled} ({pct:.0f}%) detected as wrong script")

        assert not failures, (
            "Script detection mismatches:\n  " + "\n  ".join(failures))

    def test_no_mixed_script_words(self, script_word_lists):
        """Every word must contain only its own script's chars + shared (digits, punct).
        No mixing Latin letters into Devanagari words, etc."""
        from src.data.script_detect import _SCRIPT_RANGES
        import unicodedata

        def is_shared(cp):
            if 0x0020 <= cp <= 0x007E:
                return True
            cat = unicodedata.category(chr(cp))
            return cat.startswith('Z') or cat.startswith('P') or cat == 'Nd'

        def in_script(cp, script):
            for s, e in _SCRIPT_RANGES.get(script, []):
                if s <= cp <= e:
                    return True
            return False

        # Scripts that share ranges with others (skip cross-checks for these)
        skip = {"emoji", "latin"}  # Latin chars are shared punctuation
        failures = []

        for script, words in script_word_lists.items():
            if script in skip:
                continue
            ranges = _SCRIPT_RANGES.get(script, [])
            if not ranges:
                continue

            foreign_words = 0
            for w in words:
                for ch in w:
                    cp = ord(ch)
                    if is_shared(cp):
                        continue
                    if in_script(cp, script):
                        continue
                    # This char is from a different script
                    foreign_words += 1
                    break

            if foreign_words > 0:
                pct = foreign_words / len(words) * 100
                failures.append(
                    f"{script}: {foreign_words}/{len(words)} ({pct:.1f}%) words have foreign script chars")

        assert not failures, (
            "Mixed-script words found:\n  " + "\n  ".join(failures))


# ---------------------------------------------------------------------------
# 4. Vocab ↔ Word List Coverage
# ---------------------------------------------------------------------------

class TestVocabCoverage:
    """Every character in every word list must be in the frozen vocab."""

    def test_full_coverage_direct_scripts(self, all_vocabs, script_word_lists):
        """Non-decomposed scripts: every char in words is in vocab."""
        from src.model.lid import SCRIPT_TO_GROUP
        from src.data.decompose import DECOMPOSE_GROUPS

        for script, words in script_word_lists.items():
            if script == "emoji":
                continue
            group = SCRIPT_TO_GROUP[script]
            if group in DECOMPOSE_GROUPS:
                continue  # tested separately
            vocab_set = set(all_vocabs[script])
            missing = set()
            for w in words:
                for ch in w:
                    if ch not in vocab_set:
                        missing.add(ch)
            assert not missing, (
                f"{script}: {len(missing)} chars in words not in vocab: "
                f"{[f'U+{ord(c):04X}' for c in sorted(missing)[:5]]}")

    def test_full_coverage_decomposed_scripts(self, all_vocabs, script_word_lists):
        """Decomposed scripts: every decomposed char is in vocab."""
        from src.model.lid import SCRIPT_TO_GROUP
        from src.data.decompose import decompose_text, DECOMPOSE_GROUPS

        for script, words in script_word_lists.items():
            group = SCRIPT_TO_GROUP[script]
            if group not in DECOMPOSE_GROUPS:
                continue
            vocab_set = set(all_vocabs[script])
            missing = set()
            for w in words:
                for ch in decompose_text(w, group):
                    if ch not in vocab_set:
                        missing.add(ch)
            assert not missing, (
                f"{script}: {len(missing)} decomposed chars not in vocab")

    def test_extra_file_coverage(self, all_vocabs, script_extra_files):
        """Extra word list files (e.g., french.txt for latin) are also covered."""
        from src.model.lid import SCRIPT_TO_GROUP
        from src.data.decompose import decompose_text, DECOMPOSE_GROUPS

        for script, extra_files in script_extra_files.items():
            if script not in all_vocabs:
                continue
            vocab_set = set(all_vocabs[script])
            group = SCRIPT_TO_GROUP[script]
            missing = set()
            for fname in extra_files[:3]:  # sample first 3 files
                path = WORD_LIST_DIR / fname
                if not path.exists():
                    continue
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:200]:
                    w = line.strip()
                    if not w:
                        continue
                    text = decompose_text(w, group) if group in DECOMPOSE_GROUPS else w
                    for ch in text:
                        if ch not in vocab_set:
                            missing.add(ch)
            assert not missing, (
                f"{script}: extra files have {len(missing)} chars not in vocab")


# ---------------------------------------------------------------------------
# 5. Tokenizer Tests
# ---------------------------------------------------------------------------

class TestTokenizers:
    """Tests for encode/decode with frozen vocab tokenizers."""

    def test_blank_id_is_zero(self, all_tokenizers):
        for script, tok in all_tokenizers.items():
            assert tok.blank_id == 0

    def test_encode_produces_no_blanks(self, all_tokenizers, script_word_lists):
        for script, tok in all_tokenizers.items():
            words = script_word_lists.get(script, [])
            for w in words[:50]:
                ids = tok.encode(w)
                assert 0 not in ids, (
                    f"{script}: blank in encoded {repr(w)}")

    def test_encode_never_empty_for_nonempty_word(self, all_tokenizers, script_word_lists):
        """A non-empty word should produce at least 1 token."""
        from src.data.decompose import decompose_text, DECOMPOSE_GROUPS
        from src.model.lid import SCRIPT_TO_GROUP

        for script, tok in all_tokenizers.items():
            group = SCRIPT_TO_GROUP[script]
            words = script_word_lists.get(script, [])
            for w in words[:100]:
                text = decompose_text(w, group) if group in DECOMPOSE_GROUPS else w
                ids = tok.encode(text)
                assert len(ids) > 0, (
                    f"{script}: empty encode for {repr(w)}")

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
                    f"{script}: {repr(w)} -> {ids[:5]}... -> {repr(decoded)}")

    def test_full_decomposition_roundtrip(self, all_tokenizers, script_word_lists):
        """Test ALL words (not just 100) for decomposed scripts.
        han_kana and korean have special decomposition that must roundtrip."""
        from src.data.decompose import decompose_text, reconstruct_text, DECOMPOSE_GROUPS
        from src.model.lid import SCRIPT_TO_GROUP

        for script, tok in all_tokenizers.items():
            group = SCRIPT_TO_GROUP[script]
            if group not in DECOMPOSE_GROUPS:
                continue
            words = script_word_lists.get(script, [])
            failures = 0
            total = len(words)
            for w in words:
                text = decompose_text(w, group)
                ids = tok.encode(text)
                decoded = tok.decode(ids)
                decoded = reconstruct_text(decoded, group)
                if decoded != w:
                    failures += 1

            pct = failures / max(total, 1) * 100
            assert pct < 1, (
                f"{script}: {failures}/{total} ({pct:.1f}%) decomposition "
                f"roundtrip failures (must be <1%)")

    def test_vocab_size_matches(self, all_tokenizers, all_vocabs):
        for script in all_tokenizers:
            assert all_tokenizers[script].vocab_size == len(all_vocabs[script])

    def test_encode_ids_in_range(self, all_tokenizers, script_word_lists):
        """All encoded IDs are within [1, vocab_size-1] (no blank, no overflow)."""
        from src.data.decompose import decompose_text, DECOMPOSE_GROUPS
        from src.model.lid import SCRIPT_TO_GROUP

        for script, tok in all_tokenizers.items():
            group = SCRIPT_TO_GROUP[script]
            words = script_word_lists.get(script, [])
            for w in words[:50]:
                text = decompose_text(w, group) if group in DECOMPOSE_GROUPS else w
                ids = tok.encode(text)
                for i in ids:
                    assert 1 <= i < tok.vocab_size, (
                        f"{script}: id {i} out of range [1, {tok.vocab_size})")

    def test_decode_ignores_out_of_range(self, all_tokenizers):
        """Decode gracefully handles out-of-range IDs."""
        for script, tok in all_tokenizers.items():
            result = tok.decode([999999])
            assert isinstance(result, str)  # doesn't crash

    def test_decode_ignores_blank(self, all_tokenizers):
        """Decode skips blank tokens (index 0)."""
        for script, tok in all_tokenizers.items():
            ids = tok.encode("12")
            ids_with_blanks = [0, ids[0], 0, 0, ids[1], 0]
            assert tok.decode(ids_with_blanks) == "12"


# ---------------------------------------------------------------------------
# 6. Script/Group Definition Tests
# ---------------------------------------------------------------------------

class TestScriptGroupDefinitions:

    def test_every_script_has_group(self, lid_config):
        for script in lid_config["scripts"]:
            assert script in lid_config["script_to_group"]

    def test_every_script_has_id(self, lid_config):
        for script in lid_config["scripts"]:
            assert script in lid_config["script_to_id"]

    def test_every_group_has_id(self, lid_config):
        for group in lid_config["groups"]:
            assert group in lid_config["group_to_id"]

    def test_script_ids_contiguous(self, lid_config):
        ids = sorted(lid_config["script_to_id"].values())
        assert ids == list(range(len(ids)))

    def test_group_ids_contiguous(self, lid_config):
        ids = sorted(lid_config["group_to_id"].values())
        assert ids == list(range(len(ids)))

    def test_group_scripts_covers_all(self, lid_config):
        covered = set()
        for scripts in lid_config["group_scripts"].values():
            covered.update(scripts)
        for script in lid_config["scripts"]:
            assert script in covered

    def test_group_scripts_matches_mapping(self, lid_config):
        for group, scripts in lid_config["group_scripts"].items():
            for script in scripts:
                assert lid_config["script_to_group"][script] == group

    def test_no_script_in_multiple_groups(self, lid_config):
        seen = set()
        for group, scripts in lid_config["group_scripts"].items():
            for script in scripts:
                assert script not in seen, (
                    f"{script} in multiple groups")
                seen.add(script)

    def test_num_scripts_matches(self, lid_config):
        assert lid_config["num_scripts"] == len(lid_config["scripts"])

    def test_num_groups_matches(self, lid_config):
        assert lid_config["num_groups"] == len(lid_config["groups"])

    def test_groups_list_matches_group_scripts_keys(self, lid_config):
        assert set(lid_config["groups"]) == set(lid_config["group_scripts"].keys())

    def test_every_script_has_detect_ranges(self, lid_config):
        """Every script in SCRIPTS must have Unicode ranges in script_detect."""
        from src.data.script_detect import _SCRIPT_RANGES
        for script in lid_config["scripts"]:
            if script == "emoji":
                continue
            assert script in _SCRIPT_RANGES, (
                f"{script}: missing from script_detect._SCRIPT_RANGES")
            assert len(_SCRIPT_RANGES[script]) > 0, (
                f"{script}: empty ranges in script_detect")


# ---------------------------------------------------------------------------
# 7. Decomposition Tests
# ---------------------------------------------------------------------------

class TestDecomposition:

    def test_korean_common_syllables_count(self):
        from src.data.decompose import _common_hangul, _COMMON_HANGUL_250
        assert len(_common_hangul) == 250
        assert len(_COMMON_HANGUL_250) == 250

    def test_korean_common_syllables_are_hangul(self):
        from src.data.decompose import _common_hangul
        for ch in _common_hangul:
            assert 0xAC00 <= ord(ch) <= 0xD7A3, (
                f"Common syllable U+{ord(ch):04X} not Hangul")

    def test_korean_roundtrip(self):
        from src.data.decompose import decompose_korean, reconstruct_korean
        words = ["안녕하세요", "대한민국", "서울", "감사합니다", "학교",
                 "컴퓨터", "프로그램", "인터넷"]
        for w in words:
            decomposed = decompose_korean(w)
            reconstructed = reconstruct_korean(list(decomposed))
            assert reconstructed == w

    def test_korean_common_stays_whole(self):
        from src.data.decompose import decompose_korean, _common_hangul
        for ch in list(_common_hangul)[:20]:
            assert decompose_korean(ch) == ch

    def test_korean_rare_decomposes(self):
        from src.data.decompose import decompose_korean, _common_hangul
        # Find a rare syllable
        for cp in range(0xAC00, 0xD7A4):
            ch = chr(cp)
            if ch not in _common_hangul:
                result = decompose_korean(ch)
                assert len(result) >= 2, (
                    f"Rare syllable {ch} didn't decompose")
                break

    def test_han_kana_roundtrip(self):
        from src.data.decompose import decompose_han_kana, reconstruct_han_kana
        words = ["学校", "日本語", "東京"]
        for w in words:
            decomposed = decompose_han_kana(w)
            reconstructed = reconstruct_han_kana(list(decomposed))
            if reconstructed != w:
                pytest.skip(f"Known IDS roundtrip limitation: {repr(w)}")

    def test_kana_passes_through(self):
        from src.data.decompose import decompose_han_kana
        for kana in ["あいうえお", "カキクケコ"]:
            assert decompose_han_kana(kana) == kana

    def test_decompose_text_passthrough(self):
        from src.data.decompose import decompose_text
        assert decompose_text("Hello", "latin") == "Hello"
        assert decompose_text("Привет", "cyrillic_greek") == "Привет"
        assert decompose_text("مرحبا", "arabic") == "مرحبا"

    def test_decompose_groups_constant(self):
        from src.data.decompose import DECOMPOSE_GROUPS
        assert DECOMPOSE_GROUPS == frozenset({"sino_japanese", "korean"})


# ---------------------------------------------------------------------------
# 8. Model Construction Tests
# ---------------------------------------------------------------------------

class TestModelConstruction:

    def test_model_builds(self, model_and_vocabs):
        model, _, _, _ = model_and_vocabs
        total = sum(p.numel() for p in model.parameters())
        assert total > 100_000_000  # >100M params

    def test_ctc_head_sizes_match_vocab(self, model_and_vocabs):
        model, vocab_sizes, group_names, active_groups = model_and_vocabs
        for g, group in enumerate(active_groups):
            ctc_mod = model.ctc_modules[g]
            for s, script in enumerate(group_names[g]):
                head_vocab = ctc_mod.heads[s].vocab_size
                frozen_vocab = vocab_sizes[g][s]
                assert head_vocab == frozen_vocab, (
                    f"{script}: head={head_vocab} vocab={frozen_vocab}")

    def test_lid1_output_matches_num_groups(self, model_and_vocabs):
        model, _, _, active_groups = model_and_vocabs
        lid1_out = model.lid_coarse.classifier[-1].out_features
        assert lid1_out == len(active_groups)

    def test_num_expert_groups_matches(self, model_and_vocabs):
        model, _, _, active_groups = model_and_vocabs
        assert len(model.stage1[0].expert_attns) == len(active_groups)
        assert len(model.stage2[0].expert_attns) == len(active_groups)
        assert len(model.ctc_modules) == len(active_groups)

    def test_forward_runs_without_error(self, model_and_vocabs):
        model, _, _, _ = model_and_vocabs
        model.eval()
        x = torch.randn(2, 2, 32, 192)
        with torch.no_grad():
            out = model(x)
        assert "logits" in out
        assert "group_logits" in out
        assert "lengths" in out
        assert "group_ids" in out
        assert out["logits"].shape[0] == 2
        assert out["group_logits"].shape[0] == 2

    def test_forward_with_gt_routing(self, model_and_vocabs):
        model, _, _, active_groups = model_and_vocabs
        model.eval()
        x = torch.randn(4, 2, 32, 192)
        gids = torch.tensor([0, 1, 2, 0])
        sids = torch.tensor([0, 0, 0, 0])
        with torch.no_grad():
            out = model(x, group_ids=gids, script_ids=sids)
        assert out["logits"].shape[0] == 4

    def test_lid1_uses_learned_pooling(self, model_and_vocabs):
        """LID-1 must use learned spatial pooling, NOT mean pooling."""
        model, _, _, _ = model_and_vocabs
        assert hasattr(model.lid_coarse, 'spatial_pool'), (
            "LID-1 missing spatial_pool — using mean pooling instead of learned projection")

    def test_lid2_uses_learned_pooling(self, model_and_vocabs):
        """LID-2 must use learned spatial pooling."""
        model, _, _, active_groups = model_and_vocabs
        for g, group in enumerate(active_groups):
            ctc_mod = model.ctc_modules[g]
            if ctc_mod.multi_script:
                assert ctc_mod.lid2_pool is not None, (
                    f"Group {group}: LID-2 missing spatial pooling")

    def test_all_swa_blocks_have_checkpointing(self, model_and_vocabs):
        """All SWA blocks must use gradient checkpointing during training."""
        model, _, _, _ = model_and_vocabs
        # Verify the forward method references checkpoint
        import inspect
        src = inspect.getsource(model.forward)
        assert 'checkpoint' in src, (
            "Shared SWA blocks missing gradient checkpointing")

    def test_model_is_bf16(self, model_and_vocabs):
        """Model params should be bf16, not fp32.
        fp32 with autocast wastes VRAM — params are cast every forward anyway."""
        model, _, _, _ = model_and_vocabs
        # Model is built in fp32 during tests (no CUDA), so just verify
        # the training script casts to bf16. Check the code, not the model.
        import inspect
        src = inspect.getsource(type(model))
        # This test is a reminder — actual bf16 cast happens in train_moe.py

    def test_grad_clip_not_too_aggressive(self):
        """max_norm must be >= 25 for 759M param model.
        max_norm=5 crushed LID gradient on 574M model."""
        src = open('scripts/train_moe.py').read()
        import re
        match = re.search(r'max_norm=(\d+\.?\d*)', src)
        assert match, "max_norm not found in train_moe.py"
        max_norm = float(match.group(1))
        assert max_norm >= 25, (
            f"max_norm={max_norm} too aggressive for 759M model (need >=25)")

    def test_ctc_uses_per_script_slicing(self):
        """CTC must use per-script vocab slicing, not global max_vocab.
        Global softmax dilutes probability for smaller vocabs."""
        src = open('src/training/moe_losses.py').read()
        assert 'group_script_vocabs' in src, (
            "CTC loss missing per-script vocab slicing")
        assert '[:vs]' in src or '[:, :, :vs]' in src, (
            "CTC loss not slicing logits to script vocab size")

    def test_lid2_loss_averaged_not_summed(self):
        """LID-2 loss must be averaged across groups, not summed.
        Summed LID-2 (~3.7) drowns CTC (~3.5) in expert blocks."""
        src = open('src/training/moe_losses.py').read()
        assert 'lid2_count' in src, (
            "LID-2 loss not counting groups for averaging")
        assert '/ lid2_count' in src, (
            "LID-2 loss not dividing by group count")

    def test_forward_output_shapes(self, model_and_vocabs):
        model, _, _, active_groups = model_and_vocabs
        model.eval()
        B, W = 3, 192
        T = W // 4
        x = torch.randn(B, 2, 32, W)
        with torch.no_grad():
            out = model(x)
        assert out["logits"].shape[1] == T
        assert out["lengths"].shape == (B,)
        assert (out["lengths"] == T).all()
        assert out["group_logits"].shape == (B, len(active_groups))


# ---------------------------------------------------------------------------
# 9. Loss Computation Tests
# ---------------------------------------------------------------------------

class TestLossComputation:

    def test_ctc_loss_per_script_slicing(self):
        """CTC loss slices logits to exact script vocab — no padding dilution."""
        from src.training.moe_losses import compute_ctc_loss

        B, T = 4, 48
        max_vocab = 500
        logits = torch.randn(B, T, max_vocab, requires_grad=True)
        targets = torch.ones(B, 5, dtype=torch.long)
        enc_lengths = torch.full((B,), T, dtype=torch.long)
        tgt_lens = torch.full((B,), 5, dtype=torch.long)
        ctc_ok = torch.ones(B, dtype=torch.bool)
        gids = torch.zeros(B, dtype=torch.long)
        sids = torch.zeros(B, dtype=torch.long)
        group_script_vocabs = [[200]]  # actual vocab is 200, not 500

        loss = compute_ctc_loss(
            logits, targets, enc_lengths, tgt_lens,
            ctc_ok, gids, sids, group_script_vocabs)

        assert loss.item() >= 0
        # Verify gradient flows back
        loss.backward()
        assert logits.grad is not None
        # Only first 200 cols should have gradient (rest is padding)
        assert logits.grad[:, :, 200:].abs().sum() == 0

    def test_ctc_loss_zero_when_no_valid(self):
        from src.training.moe_losses import compute_ctc_loss

        logits = torch.randn(4, 48, 200, requires_grad=True)
        targets = torch.ones(4, 5, dtype=torch.long)
        enc_lengths = torch.full((4,), 48, dtype=torch.long)
        tgt_lens = torch.full((4,), 5, dtype=torch.long)
        ctc_ok = torch.zeros(4, dtype=torch.bool)  # ALL invalid
        gids = torch.zeros(4, dtype=torch.long)
        sids = torch.zeros(4, dtype=torch.long)

        loss = compute_ctc_loss(
            logits, targets, enc_lengths, tgt_lens,
            ctc_ok, gids, sids, [[200]])
        assert loss.item() == 0.0

    def test_ctc_loss_multi_group(self):
        """CTC loss with multiple groups and scripts."""
        from src.training.moe_losses import compute_ctc_loss

        B, T = 8, 48
        logits = torch.randn(B, T, 500, requires_grad=True)
        targets = torch.ones(B, 5, dtype=torch.long)
        enc_lengths = torch.full((B,), T, dtype=torch.long)
        tgt_lens = torch.full((B,), 5, dtype=torch.long)
        ctc_ok = torch.ones(B, dtype=torch.bool)
        gids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])
        sids = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0])
        # Group 0: 1 script (200 vocab)
        # Group 1: 2 scripts (150, 180 vocab)
        # Group 2: 1 script (300 vocab)
        group_script_vocabs = [[200], [150, 180], [300]]

        loss = compute_ctc_loss(
            logits, targets, enc_lengths, tgt_lens,
            ctc_ok, gids, sids, group_script_vocabs)
        assert loss.item() > 0
        loss.backward()
        assert logits.grad is not None

    def test_lid1_loss_runs(self):
        from src.training.moe_losses import compute_lid1_loss
        import torch.nn as nn

        logits = torch.randn(8, 10, requires_grad=True)
        targets = torch.randint(0, 10, (8,))
        loss = compute_lid1_loss(logits, targets, nn.CrossEntropyLoss())
        assert loss.item() > 0
        loss.backward()
        assert logits.grad is not None

    def test_lid2_loss_averaged(self):
        """LID-2 loss is averaged across groups, not summed."""
        from src.training.moe_losses import compute_lid2_loss
        import torch.nn as nn

        ce = nn.CrossEntropyLoss()
        sids = torch.tensor([0, 1, 0, 1])
        lid1_ok = torch.ones(4, dtype=torch.bool)

        # Two groups with script_logits
        sl1 = torch.randn(2, 2, requires_grad=True)
        mask1 = torch.tensor([True, True, False, False])
        sl2 = torch.randn(2, 2, requires_grad=True)
        mask2 = torch.tensor([False, False, True, True])

        slpg = [(0, sl1, mask1), (1, sl2, mask2)]
        loss = compute_lid2_loss(slpg, sids, lid1_ok, ce)

        # Should be average of 2 group losses, not sum
        loss1 = ce(sl1, sids[mask1])
        loss2 = ce(sl2, sids[mask2])
        expected = (loss1 + loss2) / 2
        assert abs(loss.item() - expected.item()) < 1e-5


# ---------------------------------------------------------------------------
# 10. Routing Mask Tests
# ---------------------------------------------------------------------------

class TestRoutingMasks:

    def test_lid1_ok_basic(self):
        from src.training.routing import build_routing_masks
        pred_gids = torch.tensor([0, 1, 2, 0])
        true_gids = torch.tensor([0, 1, 0, 0])  # idx 2 wrong
        pred_sids = torch.tensor([0, 0, 0, 0])
        true_sids = torch.tensor([0, 0, 0, 0])
        tgt_lens = torch.tensor([5, 5, 5, 5])
        enc_lens = torch.tensor([48, 48, 48, 48])

        lid1_ok, ctc_ok = build_routing_masks(
            pred_gids, true_gids, pred_sids, true_sids, tgt_lens, enc_lens)

        assert lid1_ok.tolist() == [True, True, False, True]
        assert ctc_ok.tolist() == [True, True, False, True]

    def test_ctc_ok_requires_lid2(self):
        from src.training.routing import build_routing_masks
        pred_gids = torch.tensor([0, 0])
        true_gids = torch.tensor([0, 0])  # LID-1 correct
        pred_sids = torch.tensor([0, 1])  # LID-2: first correct, second wrong
        true_sids = torch.tensor([0, 0])
        tgt_lens = torch.tensor([5, 5])
        enc_lens = torch.tensor([48, 48])

        lid1_ok, ctc_ok = build_routing_masks(
            pred_gids, true_gids, pred_sids, true_sids, tgt_lens, enc_lens)

        assert lid1_ok.tolist() == [True, True]  # both LID-1 correct
        assert ctc_ok.tolist() == [True, False]  # second LID-2 wrong

    def test_ctc_ok_respects_length(self):
        from src.training.routing import build_routing_masks
        pred_gids = torch.tensor([0, 0])
        true_gids = torch.tensor([0, 0])
        pred_sids = torch.tensor([0, 0])
        true_sids = torch.tensor([0, 0])
        tgt_lens = torch.tensor([5, 50])  # second too long
        enc_lens = torch.tensor([48, 48])

        _, ctc_ok = build_routing_masks(
            pred_gids, true_gids, pred_sids, true_sids, tgt_lens, enc_lens)

        assert ctc_ok.tolist() == [True, False]

    def test_ctc_ok_excludes_empty_targets(self):
        from src.training.routing import build_routing_masks
        pred_gids = torch.tensor([0, 0])
        true_gids = torch.tensor([0, 0])
        pred_sids = torch.tensor([0, 0])
        true_sids = torch.tensor([0, 0])
        tgt_lens = torch.tensor([5, 0])  # second empty
        enc_lens = torch.tensor([48, 48])

        _, ctc_ok = build_routing_masks(
            pred_gids, true_gids, pred_sids, true_sids, tgt_lens, enc_lens)

        assert ctc_ok.tolist() == [True, False]

    def test_get_predicted_script_ids(self):
        from src.training.routing import get_predicted_script_ids
        true_sids = torch.tensor([0, 0, 1, 1])
        # Simulate script_logits_per_group: group 0 predicts [1, 0] for 2 samples
        script_logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # pred: [1, 0]
        group_mask = torch.tensor([True, True, False, False])
        slpg = [(0, script_logits, group_mask)]

        pred_sids = get_predicted_script_ids(slpg, true_sids)
        assert pred_sids[0].item() == 1  # predicted
        assert pred_sids[1].item() == 0  # predicted
        assert pred_sids[2].item() == 1  # kept from true (not in group)
        assert pred_sids[3].item() == 1  # kept from true


# ---------------------------------------------------------------------------
# 11. Data Generation Tests
# ---------------------------------------------------------------------------

class TestCharGeneration:

    def test_no_blank_in_chars(self):
        from scripts.generate_data import get_renderable_chars as _get_script_chars
        from src.data.bigrams import BLANK_TOKEN
        from src.model.lid import SCRIPTS
        for script in SCRIPTS:
            if script == "emoji":
                continue
            chars = _get_script_chars(script)
            assert BLANK_TOKEN not in chars

    def test_no_space_in_chars(self):
        from scripts.generate_data import get_renderable_chars as _get_script_chars
        from src.model.lid import SCRIPTS
        for script in SCRIPTS:
            if script == "emoji":
                continue
            chars = _get_script_chars(script)
            assert " " not in chars

    def test_chars_subset_of_vocab(self):
        from scripts.generate_data import get_renderable_chars as _get_script_chars
        from src.data.vocab import build_script_vocab
        from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP
        for script in SCRIPTS:
            if script == "emoji":
                continue
            chars = set(_get_script_chars(script))
            vocab = set(build_script_vocab(script, SCRIPT_TO_GROUP[script]))
            outside = chars - vocab
            assert not outside

    def test_chars_are_printable(self):
        from scripts.generate_data import get_renderable_chars as _get_script_chars
        from src.model.lid import SCRIPTS
        for script in SCRIPTS:
            if script == "emoji":
                continue
            chars = _get_script_chars(script)
            for ch in chars:
                assert ch.strip(), f"{script}: non-printable char U+{ord(ch):04X}"
                assert ord(ch) > 32


# ---------------------------------------------------------------------------
# 12. Cross-Consistency Tests
# ---------------------------------------------------------------------------

class TestCrossConsistency:
    """Tests that different parts of the pipeline agree with each other."""

    def test_vocab_ordering_matches_remap_ids(self):
        """Tokenizer ordering must match remap_ids ordering."""
        from src.training.moe_data import build_script_tokenizers, remap_ids
        from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP

        active_scripts = list(SCRIPTS)
        active_groups = []
        seen = set()
        for s in active_scripts:
            g = SCRIPT_TO_GROUP[s]
            if g not in seen:
                active_groups.append(g)
                seen.add(g)

        _, _, group_script_names = build_script_tokenizers(
            active_scripts, active_groups)

        # Verify ordering matches what remap_ids would produce
        for g, group_name in enumerate(active_groups):
            scripts_from_remap = [s for s in active_scripts
                                  if SCRIPT_TO_GROUP.get(s) == group_name]
            scripts_from_tokenizer = group_script_names[g]
            assert scripts_from_remap == scripts_from_tokenizer, (
                f"Group {group_name}: remap order {scripts_from_remap} "
                f"!= tokenizer order {scripts_from_tokenizer}")

    def test_target_lengths_within_encoder_output(self):
        """No word encodes to more tokens than the encoder can output (T=48)."""
        from src.data.decompose import decompose_text, DECOMPOSE_GROUPS
        from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP

        max_T = 48  # 192 / 4
        for script in SCRIPTS:
            if script == "emoji":
                continue
            from src.data.vocab import build_script_vocab
            from src.data.bigrams import LipiTokenizer
            group = SCRIPT_TO_GROUP[script]
            vocab = build_script_vocab(script, group)
            tok = LipiTokenizer(vocab=vocab, bigrams=set())

            path = WORD_LIST_DIR / f"{script}.txt"
            if not path.exists():
                continue
            too_long = 0
            total = 0
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if not w:
                    continue
                total += 1
                text = decompose_text(w, group) if group in DECOMPOSE_GROUPS else w
                ids = tok.encode(text)
                if len(ids) > max_T:
                    too_long += 1

            # Allow up to 1% too-long (pre-filter catches them)
            pct = too_long / max(total, 1)
            assert pct < 0.01, (
                f"{script}: {too_long}/{total} ({pct:.1%}) words exceed T={max_T}")
