"""
Word list loading for training data generation.

Loads word lists from training_data/word_lists/ for each script,
merging primary and extra files (e.g., latin.txt + english_common.txt + french.txt).
"""

import bisect
import itertools
import random
import re
import unicodedata
from pathlib import Path

WORD_LIST_DIR = Path(__file__).parent.parent.parent / "training_data" / "word_lists"
CORPORA_DIR = Path(__file__).parent.parent.parent / "training_data" / "corpora"

_EMOJI_RE = re.compile(
    r"[\u2600-\u27BF\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]")


def contains_emoji(text: str) -> bool:
    """Return whether text contains an unsupported emoji code point."""
    return _EMOJI_RE.search(text) is not None


class WordSampler:
    """Word sampler mixing uniform coverage with frequency realism.

    Uniform sampling maximizes character/shape coverage but gives the
    model a flat implicit prior — the common-word shapes that dominate
    real pages are undertrained. With `freq_frac` probability a word is
    drawn from the script's Wikipedia frequency table
    (training_data/corpora/{script}_word_freq.tsv, tempered by count^0.5
    so 'the/of/and'-class words don't swamp everything); otherwise
    uniform over the merged word list. Frequency entries are restricted
    to words already in the word list, so font-coverage and encoding
    behavior are unchanged. Scripts without a table sample uniformly.
    """

    def __init__(self, script: str, words: list[str], freq_frac: float = 0.3,
                 temper: float = 0.5, max_freq_words: int = 50_000):
        self.words = words
        self.freq_frac = freq_frac
        self._freq_words: list[str] = []
        self._cum: list[float] = []

        path = CORPORA_DIR / f"{script}_word_freq.tsv"
        if not path.exists() or not words:
            return
        known = set(words)
        weights = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 2:
                        continue
                    w, count = parts
                    if w not in known:
                        continue
                    try:
                        weight = float(count) ** temper
                    except ValueError:
                        continue
                    self._freq_words.append(w)
                    weights.append(weight)
                    if len(self._freq_words) >= max_freq_words:
                        break
        except OSError:
            self._freq_words = []
            return
        if len(self._freq_words) < 100:
            self._freq_words = []  # too little overlap — stay uniform
            return
        self._cum = list(itertools.accumulate(weights))

    @property
    def has_freq(self) -> bool:
        return bool(self._freq_words)

    def sample(self) -> str:
        if self._freq_words and random.random() < self.freq_frac:
            r = random.uniform(0, self._cum[-1])
            return self._freq_words[
                min(bisect.bisect_left(self._cum, r),
                    len(self._freq_words) - 1)]
        return random.choice(self.words)

# Extra word list files per script (in addition to {script}.txt)
_SCRIPT_EXTRA_FILES = {
    "latin": [
        "english_common.txt", "french.txt", "german.txt", "spanish.txt",
        "turkish.txt", "vietnamese.txt", "italian.txt", "portuguese.txt",
        "polish.txt", "dutch.txt", "romanian.txt", "czech.txt",
        "hungarian.txt", "swedish.txt", "norwegian.txt", "danish.txt",
        "finnish.txt", "croatian.txt", "indonesian.txt", "malay.txt",
        "swahili.txt", "afrikaans.txt", "albanian.txt", "basque.txt",
        "catalan.txt", "estonian.txt", "galician.txt", "icelandic.txt",
        "latvian.txt", "lithuanian.txt", "maltese.txt", "slovak.txt",
        "slovenian.txt", "welsh.txt", "irish.txt", "tagalog.txt",
    ],
    "cyrillic": ["ukrainian.txt"],
    "devanagari": ["marathi.txt", "hindi_legal.txt"],
    "arabic": ["persian.txt", "urdu.txt"],
    # Han reads Chinese/Japanese sources. The generator splits each sampled
    # word into maximal Han/Kana runs.
    "han": ["chinese.txt", "japanese.txt"],
    # Kana uses Japanese word lists — words get split at script boundaries;
    # kana portions become kana segments.
    "kana": ["japanese.txt"],
}

_RTL_ALLOWED_CHARS: dict[str, frozenset[str]] = {}


def _rtl_word_is_encodable(word: str, script: str) -> bool:
    """Reject RTL entries whose rendered characters would be dropped.

    ASCII is allowed because generation splits it into a Latin segment.
    Non-ASCII characters must belong to the corresponding codec. Format
    controls are intentionally rejected: they affect bidi/shaping but have no
    visible CTC alignment and were previously discarded from the target.
    """
    if script not in {"arabic", "hebrew"}:
        return True

    from src.encoding.config import FUSION_BASE_CHARS, NO_FUSION_SCRIPTS

    allowed = _RTL_ALLOWED_CHARS.get(script)
    if allowed is None:
        chars = (FUSION_BASE_CHARS[script] if script == "arabic"
                 else NO_FUSION_SCRIPTS[script].chars)
        allowed = frozenset(chars)
        _RTL_ALLOWED_CHARS[script] = allowed

    for char in word:
        if unicodedata.category(char) == "Cf":
            return False
        cp = ord(char)
        if char.isspace() or 0x21 <= cp <= 0x7E:
            continue
        if char not in allowed:
            return False
    return True


def load_word_list(script: str) -> list[str]:
    """Load and merge all word list files for a script.

    Returns shuffled list of words (2-15 chars, no leading digits).
    """
    files = [WORD_LIST_DIR / f"{script}.txt"]
    for extra in _SCRIPT_EXTRA_FILES.get(script, []):
        files.append(WORD_LIST_DIR / extra)

    words = []
    for path in files:
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = unicodedata.normalize("NFC", line.strip())
                if (2 <= len(w) <= 15 and not w[0].isdigit()
                        # Scrape artifacts starting with a combining mark
                        # render as dotted-circle placeholders — unlearnable.
                        and unicodedata.category(w[0]) not in ("Mn", "Mc", "Me")
                        and not contains_emoji(w)
                        and _rtl_word_is_encodable(w, script)):
                    words.append(w)

    if words:
        words = list(set(words))
        if script == "han":
            from src.data.script_detect import split_by_script
            words = [word for word in words
                     if any(seg_script == script
                            for _, seg_script in split_by_script(word, script))]
        random.shuffle(words)
    return words


def load_all_word_lists(scripts: list[str]) -> dict[str, list[str]]:
    """Load word lists for all given scripts, in parallel.

    Loading is filter-heavy (NFC-normalize + encodability checks over
    millions of lines — ~10s for the largest scripts), and the per-script
    loads are independent, so the full-taxonomy load drops from minutes to
    the cost of the largest single script.

    Returns:
        {script_name: [words]}. Scripts with no words get empty list.
    """
    if len(scripts) <= 1:
        return {s: load_word_list(s) for s in scripts}
    import os
    from concurrent.futures import ProcessPoolExecutor
    workers = min(len(scripts), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        loaded = list(ex.map(load_word_list, scripts))
    return dict(zip(scripts, loaded))
