"""
Character + Bigram Vocabulary System.

Vocabulary structure:
  Base (always present, every adapter inherits this):
    [0]: blank (RNN-T)
    [1..95]: all printable ASCII (keyboard characters)

  Per-adapter extension:
    [96..N]: script-specific characters (Devanagari, Tamil, etc.)
    [N+1..end]: bigrams (75 Latin base + script-specific)

The base English/Latin vocabulary is NOT an adapter — it's the foundation.
Adapters add script-specific characters and bigrams on top.

Bigrams are curated from linguistic frequency data (Gutenberg corpus,
560K words), not derived from training data. This avoids baking
training distribution bias into the vocabulary.

No digit-digit bigrams — numbers stay character-level.
"""

import json
from collections import Counter
from pathlib import Path


BLANK_TOKEN = "\u2205"  # ∅ blank for RNN-T

# Base character set: all printable ASCII (32-126).
# This is the foundation every adapter inherits.
# 95 characters: 26 lower + 26 upper + 10 digits + 1 space + 32 symbols
BASE_CHARS = [chr(i) for i in range(32, 127)]

# Top 75 Latin-script bigrams by frequency.
# Source: case-sensitive bigram counts from 560K words of English literature
# (Gutenberg: Pride & Prejudice, Alice in Wonderland, Frankenstein,
# Sherlock Holmes, Moby Dick). No digit-digit pairs.
# These cover 51% compression on English text (half the decode steps).
LATIN_BIGRAMS = [
    "th", "he", "in", "er", "an", "re", "nd", "ha", "at", "ou",
    "ed", "on", "en", "ng", "hi", "is", "to", "it", "es", "as",
    "or", "ar", "te", "st", "of", "le", "ve", "se", "ea", "me",
    "al", "ne", "nt", "ll", "ti", "de", "be", "li", "wh", "wa",
    "no", "ho", "ro", "ur", "co", "el", "ce", "sh", "ch", "ee",
    "ri", "om", "ut", "wi", "ow", "ly", "ma", "ad", "ot", "fo",
    "et", "so", "il", "ai", "us", "ra", "la", "pe", "si", "ic",
    "we", "lo", "ta", "un", "io",
]

# Script-specific character ranges (added on top of BASE_CHARS per adapter)
DEVANAGARI_CHARS = [chr(c) for c in range(0x0900, 0x0980) if chr(c).strip()]
TAMIL_CHARS = [chr(c) for c in range(0x0B80, 0x0C00) if chr(c).strip()]
TELUGU_CHARS = [chr(c) for c in range(0x0C00, 0x0C80) if chr(c).strip()]
KANNADA_CHARS = [chr(c) for c in range(0x0C80, 0x0D00) if chr(c).strip()]
BENGALI_CHARS = [chr(c) for c in range(0x0980, 0x0A00) if chr(c).strip()]
GUJARATI_CHARS = [chr(c) for c in range(0x0A80, 0x0B00) if chr(c).strip()]
ODIA_CHARS = [chr(c) for c in range(0x0B00, 0x0B80) if chr(c).strip()]
GURMUKHI_CHARS = [chr(c) for c in range(0x0A00, 0x0A80) if chr(c).strip()]
MALAYALAM_CHARS = [chr(c) for c in range(0x0D00, 0x0D80) if chr(c).strip()]
URDU_CHARS = [chr(c) for c in range(0x0600, 0x0700) if chr(c).strip()]
URDU_CHARS += [chr(c) for c in range(0xFB50, 0xFE00) if chr(c).strip()]

SCRIPT_CHARSETS: dict[str, list[str]] = {
    "en": [],  # English uses BASE_CHARS only
    "hi": DEVANAGARI_CHARS,
    "ta": TAMIL_CHARS,
    "te": TELUGU_CHARS,
    "kn": KANNADA_CHARS,
    "bn_as": BENGALI_CHARS,
    "or": ODIA_CHARS,
    "gu": GUJARATI_CHARS,
    "pa": GURMUKHI_CHARS,
    "ml": MALAYALAM_CHARS,
    "ur": URDU_CHARS,
}

# Top 75 Hindi/Devanagari bigrams by frequency.
# Source: 110K words from 19 Hindi Wikipedia articles (general topics).
# Excludes danda (।) bigrams. 37.7% compression on Hindi text.
HINDI_BIGRAMS = [
    "्र", "के", "है", "ें", "ार", "मे", "का", "्य", "र्", "रा",
    "या", "त्", "स्", "प्", "ान", "ों", "ता", "्त", "िक", "से",
    "न्", "की", "क्", "िय", "ने", "वि", "ना", "वा", "्व", "और",
    "भा", "रत", "को", "द्", "ित", "मा", "ाज", "कर", "ात",
    "कि", "ाल", "ति", "सा", "था", "यो", "ैं", "हा", "्ष", "री",
    "जा", "्थ", "पर", "रि", "सं", "नि", "ला", "ती", "हो", "लि",
    "दि", "िल", "दे", "िन", "सम", "्द", "ले", "इस", "ास", "बा",
    "्ट", "ां", "ड़", "्म",
]

# Curated bigrams per script family.
# Latin: from Gutenberg English corpus (top 75).
# Hindi: from Hindi Wikipedia (top 73, excl. danda bigrams).
# Tamil, Telugu, etc.: to be added from Wikipedia frequency data.
CURATED_BIGRAMS: dict[str, list[str]] = {
    "latin": LATIN_BIGRAMS,
    "hi": HINDI_BIGRAMS,
}


def build_bigram_vocab(
    word_lists: list[str | Path],
    script_chars: set[str],
    max_bigrams: int = 75,
) -> list[str]:
    """Count character bigrams across word lists, keep top N.

    No digit-digit bigrams — numbers stay character-level.

    Args:
        word_lists: Paths to text files, one word per line.
        script_chars: Set of valid characters for this script.
        max_bigrams: How many bigrams to keep.

    Returns:
        List of bigram strings, ordered by frequency.
    """
    bigram_counts: Counter[str] = Counter()
    digits = set("0123456789")

    for path in word_lists:
        path = Path(path)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                for i in range(len(word) - 1):
                    bigram = word[i : i + 2]
                    if bigram[0] in digits and bigram[1] in digits:
                        continue
                    if bigram[0] in script_chars and bigram[1] in script_chars:
                        bigram_counts[bigram] += 1

    return [bg for bg, _ in bigram_counts.most_common(max_bigrams)]


def tokenize(word: str, bigram_set: set[str]) -> list[str]:
    """Greedy left-to-right tokenization.

    Try to match bigram first, fall back to single character.
    """
    tokens: list[str] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i : i + 2] in bigram_set:
            tokens.append(word[i : i + 2])
            i += 2
        else:
            tokens.append(word[i])
            i += 1
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Trivial — just concatenate."""
    return "".join(tokens)


class LipiTokenizer:
    """Character + bigram tokenizer for Lipi OCR.

    Base vocabulary (always present):
      [0]: blank (RNN-T)
      [1..95]: printable ASCII

    Adapters extend with:
      [96..N]: script-specific characters
      [N+1..end]: curated bigrams
    """

    def __init__(self, vocab: list[str], bigrams: set[str] | None = None):
        self._vocab = vocab
        self._token_to_id = {t: i for i, t in enumerate(vocab)}

        if bigrams is not None:
            self._bigrams = bigrams
        else:
            self._bigrams = {t for t in vocab if len(t) == 2}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    @property
    def blank_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs using greedy bigram matching."""
        tokens = tokenize(text, self._bigrams)
        ids = []
        for t in tokens:
            if t in self._token_to_id:
                ids.append(self._token_to_id[t])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text. Blank tokens (ID 0) are skipped."""
        return "".join(
            self._vocab[i]
            for i in ids
            if i != self.blank_id and 0 < i < len(self._vocab)
        )

    def save(self, path: str | Path):
        """Save vocabulary to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LipiTokenizer":
        """Load vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)

    @classmethod
    def build_for_script(
        cls,
        script_id: str,
        word_lists: list[str | Path] | None = None,
        max_bigrams: int = 75,
    ) -> "LipiTokenizer":
        """Build vocabulary from base + script chars + counted bigrams.

        For building bigrams from custom word lists (e.g., training data).
        For production, prefer build_with_curated_bigrams().
        """
        script_chars = SCRIPT_CHARSETS.get(script_id, [])
        all_chars = BASE_CHARS + script_chars

        seen: set[str] = set()
        unique_chars: list[str] = []
        for c in all_chars:
            if c not in seen:
                seen.add(c)
                unique_chars.append(c)

        bigram_list: list[str] = []
        if word_lists:
            valid_chars = set(unique_chars)
            bigram_list = build_bigram_vocab(word_lists, valid_chars, max_bigrams)

        vocab = [BLANK_TOKEN] + unique_chars + bigram_list
        bigram_set = set(bigram_list)
        return cls(vocab=vocab, bigrams=bigram_set)

    @classmethod
    def build_character_level(cls, script_id: str) -> "LipiTokenizer":
        """Build character-level vocabulary (no bigrams).

        Useful for Phase 1 CTC training and as a baseline.
        """
        return cls.build_for_script(script_id, word_lists=None, max_bigrams=0)

    @classmethod
    def build_with_curated_bigrams(cls, script_id: str) -> "LipiTokenizer":
        """Build vocabulary using pre-curated bigram lists.

        Uses linguistically-derived bigram frequency data rather than
        counting from a training corpus.

        Latin adapter: 95 ASCII chars + 75 curated bigrams = 171 tokens.
        Other adapters: 95 ASCII + script chars + curated bigrams.
        """
        script_chars = SCRIPT_CHARSETS.get(script_id, [])
        all_chars = BASE_CHARS + script_chars

        seen: set[str] = set()
        unique_chars: list[str] = []
        for c in all_chars:
            if c not in seen:
                seen.add(c)
                unique_chars.append(c)

        # Select curated bigrams
        if script_id in ("en",) or not script_chars:
            bigram_list = list(CURATED_BIGRAMS.get("latin", []))
        else:
            # Non-Latin: use Latin bigrams as base + script-specific if available
            bigram_list = list(CURATED_BIGRAMS.get("latin", []))
            bigram_list += list(CURATED_BIGRAMS.get(script_id, []))

        vocab = [BLANK_TOKEN] + unique_chars + bigram_list
        bigram_set = set(bigram_list)
        return cls(vocab=vocab, bigrams=bigram_set)
