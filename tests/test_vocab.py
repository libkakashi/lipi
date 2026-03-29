"""
Tests for character + bigram vocabulary system.
"""

import pytest
import tempfile
from pathlib import Path

from src.data.bigrams import (
    LipiTokenizer, LATIN_CHARS, PUNCTUATION, BLANK_TOKEN,
    DEVANAGARI_CHARS, SCRIPT_CHARSETS,
    build_bigram_vocab, tokenize, detokenize,
)


@pytest.fixture
def eng_tok():
    return LipiTokenizer.build_character_level("en")


@pytest.fixture
def hindi_tok():
    return LipiTokenizer.build_character_level("hi")


class TestCharacterLevel:

    def test_blank_token(self, eng_tok):
        assert eng_tok.blank_id == 0
        assert eng_tok.vocab[0] == BLANK_TOKEN

    def test_vocab_contains_all_latin(self, eng_tok):
        for char in LATIN_CHARS:
            assert char in eng_tok.vocab, f"Missing Latin char: {char}"

    def test_vocab_contains_punctuation(self, eng_tok):
        for char in PUNCTUATION:
            assert char in eng_tok.vocab, f"Missing punctuation: {char}"

    def test_hindi_contains_devanagari(self, hindi_tok):
        for char in DEVANAGARI_CHARS[:20]:
            assert char in hindi_tok.vocab, f"Missing Devanagari char: {char}"

    def test_hindi_contains_latin(self, hindi_tok):
        for char in LATIN_CHARS[:10]:
            assert char in hindi_tok.vocab

    def test_no_bigrams_in_char_level(self, eng_tok):
        for token in eng_tok.vocab:
            assert len(token) <= 1 or token == BLANK_TOKEN


class TestRoundtrip:

    def test_english_words(self, eng_tok):
        words = [
            "Hello", "World", "Test", "OCR", "Lipi",
            "Section", "Court", "Order", "Petition", "Judgment",
        ]
        for word in words:
            ids = eng_tok.encode(word)
            decoded = eng_tok.decode(ids)
            assert decoded == word, f"Roundtrip failed: '{word}' -> {ids} -> '{decoded}'"

    def test_numbers(self, eng_tok):
        for num in ["12345", "2024", "149", "0", "987654"]:
            ids = eng_tok.encode(num)
            assert eng_tok.decode(ids) == num

    def test_mixed_case(self, eng_tok):
        for word in ["WP(C)", "No.123/2024", "Section-149", "A.I.R."]:
            ids = eng_tok.encode(word)
            assert eng_tok.decode(ids) == word

    def test_hindi_words(self, hindi_tok):
        for word in ["भारत", "न्यायालय", "आदेश", "दंड", "संहिता"]:
            ids = hindi_tok.encode(word)
            decoded = hindi_tok.decode(ids)
            assert decoded == word, f"Hindi roundtrip failed: '{word}'"

    def test_mixed_script(self, hindi_tok):
        for word in ["Section", "149", "Hello"]:
            ids = hindi_tok.encode(word)
            assert hindi_tok.decode(ids) == word

    def test_blank_not_in_encoded(self, eng_tok):
        ids = eng_tok.encode("Hello")
        assert 0 not in ids

    @pytest.mark.parametrize("script_id", ["en", "hi", "ta", "te", "kn", "bn_as"])
    def test_all_scripts_build(self, script_id):
        tok = LipiTokenizer.build_character_level(script_id)
        assert tok.vocab_size > 0
        assert tok.blank_id == 0


class TestBigramTokenizer:

    def test_bigram_tokenization(self):
        bigrams = {"th", "he", "er"}
        assert tokenize("there", bigrams) == ["th", "er", "e"]
        assert tokenize("the", bigrams) == ["th", "e"]
        assert tokenize("a", bigrams) == ["a"]

    def test_no_bigrams(self):
        assert tokenize("hello", set()) == ["h", "e", "l", "l", "o"]

    def test_detokenize(self):
        assert detokenize(["th", "e", "r", "e"]) == "there"
        assert detokenize([]) == ""

    def test_roundtrip_with_bigrams(self):
        bigrams = {"th", "he", "in", "er"}
        word = "there"
        tokens = tokenize(word, bigrams)
        assert detokenize(tokens) == word

    def test_build_from_word_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for _ in range(100):
                f.write("the\nthere\nthen\nthing\n")
            path = f.name

        try:
            bigrams = build_bigram_vocab(
                [path], set(LATIN_CHARS + PUNCTUATION), max_bigrams=10
            )
            assert "th" in bigrams  # should be top bigram
            assert len(bigrams) <= 10
            assert all(len(bg) == 2 for bg in bigrams)
        finally:
            Path(path).unlink()

    def test_build_for_script_with_bigrams(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            words = ["the", "there", "then", "thing", "this"] * 100
            f.write("\n".join(words))
            path = f.name

        try:
            tok = LipiTokenizer.build_for_script("en", word_lists=[path], max_bigrams=10)
            assert tok.vocab_size > 83  # chars + some bigrams
            # Verify bigrams are at most 2 chars
            for token in tok.vocab:
                assert len(token) <= 2 or token == BLANK_TOKEN

            # Roundtrip still works
            ids = tok.encode("there")
            assert tok.decode(ids) == "there"
        finally:
            Path(path).unlink()

    def test_max_token_length_is_two(self):
        """Bigram vocab never produces tokens longer than 2 characters."""
        tok = LipiTokenizer.build_character_level("en")
        for token in tok.vocab:
            assert len(token) <= 2 or token == BLANK_TOKEN


class TestCuratedBigrams:

    def test_curated_builds(self):
        tok = LipiTokenizer.build_with_curated_bigrams("en")
        assert tok.vocab_size > 83  # more than just chars
        assert tok.vocab_size < 250  # reasonable upper bound

    def test_curated_roundtrips(self):
        tok = LipiTokenizer.build_with_curated_bigrams("en")
        for word in ["Hello", "World", "Section", "12345", "WP(C)", "information"]:
            assert tok.decode(tok.encode(word)) == word

    def test_curated_no_digit_bigrams(self):
        tok = LipiTokenizer.build_with_curated_bigrams("en")
        digits = set("0123456789")
        for token in tok.vocab:
            if len(token) == 2:
                assert not (token[0] in digits and token[1] in digits), \
                    f"Digit bigram found: {token}"

    def test_curated_compresses(self):
        tok = LipiTokenizer.build_with_curated_bigrams("en")
        char_tok = LipiTokenizer.build_character_level("en")
        # "the" should compress: 3 chars -> 2 tokens (th + e)
        assert len(tok.encode("the")) < len(char_tok.encode("the"))

    def test_numbers_stay_character_level(self):
        tok = LipiTokenizer.build_with_curated_bigrams("en")
        ids = tok.encode("12345")
        assert len(ids) == 5  # one token per digit


class TestSaveLoad:

    def test_save_load_roundtrip(self, eng_tok):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        eng_tok.save(path)
        loaded = LipiTokenizer.load(path)

        assert loaded.vocab_size == eng_tok.vocab_size
        assert loaded.vocab == eng_tok.vocab
        ids = loaded.encode("Hello")
        assert loaded.decode(ids) == "Hello"

        Path(path).unlink()

    def test_hindi_save_load(self, hindi_tok):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        hindi_tok.save(path)
        loaded = LipiTokenizer.load(path)
        ids = loaded.encode("भारत")
        assert loaded.decode(ids) == "भारत"

        Path(path).unlink()


class TestEdgeCases:

    def test_empty_string(self, eng_tok):
        ids = eng_tok.encode("")
        assert ids == []
        assert eng_tok.decode([]) == ""

    def test_single_char(self, eng_tok):
        ids = eng_tok.encode("A")
        assert eng_tok.decode(ids) == "A"

    def test_decode_with_blanks(self, eng_tok):
        ids = eng_tok.encode("Hi")
        ids_with_blanks = [0, ids[0], 0, 0, ids[1], 0]
        assert eng_tok.decode(ids_with_blanks) == "Hi"

    def test_space_handling(self, eng_tok):
        assert " " in eng_tok.vocab
