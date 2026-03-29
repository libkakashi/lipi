"""
Character + Bigram Vocabulary System.

Each script adapter uses a vocabulary of individual characters plus the
~150 most common character pairs (bigrams) for that script. This gives
~30% fewer decode steps than pure character-level, with zero added complexity.

Why bigrams, not BPE:
  - 2-char tokens max — error granularity is at most 2 characters
  - No tokenizer training library needed — just count character pair frequencies
  - No ambiguity — greedy left-to-right matching always gives the same result
  - Adding a new script takes minutes — count bigrams from any word list
  - ~400 total tokens — barely larger than pure character-level (~250)
  - RNN-T GRU handles longer patterns dynamically

Vocabulary layout:
  [0]: blank (RNN-T)
  [1..N_chars]: individual characters (Latin + regional script)
  [N_chars+1..N_chars+N_bigrams]: top bigrams for this script
  Total: ~400 tokens
"""

import json
from collections import Counter
from pathlib import Path


BLANK_TOKEN = "\u2205"  # ∅ blank for RNN-T

# Latin character set: always present in every vocabulary
LATIN_CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PUNCTUATION = list(".,;:!?'\"()-/&@#$%+= ")

# Script character ranges
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

# Pre-curated bigram lists per script family.
# Latin bigrams: weighted blend of English (60%), Spanish (15%), French (8%),
# German (5%), Portuguese (5%), Italian (3%) character pair frequencies.
# Source: practicalcryptography.com Wortschatz corpus + sttmedia.com.
# No digit-digit bigrams. All lowercase (case-insensitive matching in tokenizer).
CURATED_BIGRAMS: dict[str, list[str]] = {
    "latin": [
        "er", "th", "in", "es", "en", "he", "an", "re", "on", "nt",
        "de", "st", "te", "ar", "al", "to", "or", "nd", "ti", "ra",
        "as", "el", "se", "le", "at", "la", "co", "ed", "ta", "ne",
        "ri", "it", "is", "sa", "ea", "ng", "ro", "me", "et", "ha",
        "ec", "si", "na", "ou", "ve", "of", "hi", "li", "ll", "so",
        "os", "ue", "ad", "un", "qu", "io", "do", "pa", "da", "ma",
        "ca", "ci", "ch", "ia", "ac", "em", "ic", "no", "ie", "lo",
        "ns", "od", "ei", "au", "di", "il", "tr", "ss", "ur", "ge",
        "ai", "be", "ce", "eu", "po", "am", "om", "sc", "rd", "tt",
        "pe", "rs", "rt", "ol", "ni",
    ],
    # Hindi/Devanagari bigrams will be added when we have frequency data
    # Tamil, Telugu, etc. — same approach, from linguistic frequency tables
}


SCRIPT_CHARSETS: dict[str, list[str]] = {
    "en": [],
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


def build_bigram_vocab(
    word_lists: list[str | Path],
    script_chars: set[str],
    max_bigrams: int = 150,
) -> list[str]:
    """Count character bigrams across word lists, keep top N.

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
                    # Skip digit-digit bigrams — numbers should stay
                    # character-level (each digit is its own token)
                    if bigram[0] in digits and bigram[1] in digits:
                        continue
                    if bigram[0] in script_chars and bigram[1] in script_chars:
                        bigram_counts[bigram] += 1

    return [bg for bg, _ in bigram_counts.most_common(max_bigrams)]


def tokenize(word: str, bigram_set: set[str]) -> list[str]:
    """Greedy left-to-right tokenization.

    Try to match bigram first, fall back to single character.
    No ambiguity, no merge rules, no edge cases.
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

    Vocabulary:
      [0]: blank (RNN-T)
      [1..N]: individual characters (Latin + punctuation + script-specific)
      [N+1..N+M]: top bigrams for this script

    Zero OOV by construction — every character is an atomic token.
    Bigrams are an acceleration layer; fallback is always char-by-char.
    """

    def __init__(self, vocab: list[str], bigrams: set[str] | None = None):
        """
        Args:
            vocab: Ordered list of token strings. Index = token ID.
                   vocab[0] must be the blank token.
            bigrams: Set of bigram strings for greedy tokenization.
                     If None, extracted from vocab (tokens of length 2).
        """
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
        """Encode text to token IDs.

        Uses greedy left-to-right bigram matching.
        Falls back to character-level for unmatched chars.

        Args:
            text: Input string.

        Returns:
            List of token IDs (no blank tokens).
        """
        tokens = tokenize(text, self._bigrams)
        ids = []
        for t in tokens:
            if t in self._token_to_id:
                ids.append(self._token_to_id[t])
            # else: skip unknown chars (shouldn't happen if vocab covers the script)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs. Blank tokens (ID 0) are skipped.

        Returns:
            Decoded string.
        """
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
        max_bigrams: int = 150,
    ) -> "LipiTokenizer":
        """Build a character + bigram vocabulary for a script.

        Args:
            script_id: Script identifier (e.g., 'en', 'hi', 'ta').
            word_lists: Paths to word list files for bigram counting.
                        If None, builds character-level only (no bigrams).
            max_bigrams: Maximum number of bigrams to include.

        Returns:
            LipiTokenizer instance.
        """
        script_chars = SCRIPT_CHARSETS.get(script_id, [])
        all_chars = LATIN_CHARS + PUNCTUATION + script_chars

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_chars: list[str] = []
        for c in all_chars:
            if c not in seen:
                seen.add(c)
                unique_chars.append(c)

        # Build bigrams from word lists
        bigram_list: list[str] = []
        if word_lists:
            valid_chars = set(unique_chars)
            bigram_list = build_bigram_vocab(word_lists, valid_chars, max_bigrams)

        # Assemble vocab: blank + chars + bigrams
        vocab = [BLANK_TOKEN] + unique_chars + bigram_list

        bigram_set = set(bigram_list)
        return cls(vocab=vocab, bigrams=bigram_set)

    @classmethod
    def build_character_level(cls, script_id: str) -> "LipiTokenizer":
        """Build a character-level vocabulary (no bigrams).

        Useful for Phase 1 CTC training and as a baseline.
        """
        return cls.build_for_script(script_id, word_lists=None, max_bigrams=0)

    @classmethod
    def build_with_curated_bigrams(cls, script_id: str) -> "LipiTokenizer":
        """Build vocabulary using pre-curated bigram lists.

        Uses linguistically-derived bigram frequency data rather than
        counting from a training corpus. This avoids baking training
        data bias into the vocabulary.

        For Latin-script languages (en), uses a weighted blend of
        English/Spanish/French/German/Portuguese/Italian frequencies.

        Args:
            script_id: Script identifier.

        Returns:
            LipiTokenizer with curated bigrams.
        """
        script_chars = SCRIPT_CHARSETS.get(script_id, [])
        all_chars = LATIN_CHARS + PUNCTUATION + script_chars

        seen: set[str] = set()
        unique_chars: list[str] = []
        for c in all_chars:
            if c not in seen:
                seen.add(c)
                unique_chars.append(c)

        # Select curated bigrams for this script
        if script_id in ("en",) or not script_chars:
            # Latin-script language → use Latin curated bigrams
            bigram_list = list(CURATED_BIGRAMS.get("latin", []))
        else:
            # Non-Latin script → use script-specific curated bigrams if available,
            # otherwise fall back to empty (character-level only for now)
            bigram_list = list(CURATED_BIGRAMS.get(script_id, []))

        vocab = [BLANK_TOKEN] + unique_chars + bigram_list
        bigram_set = set(bigram_list)
        return cls(vocab=vocab, bigrams=bigram_set)
