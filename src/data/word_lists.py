"""
Word list loading for training data generation.

Loads word lists from training_data/word_lists/ for each script,
merging primary and extra files (e.g., latin.txt + english_common.txt + french.txt).
"""

import random
from pathlib import Path

WORD_LIST_DIR = Path(__file__).parent.parent.parent / "training_data" / "word_lists"

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
    "han_kana": ["chinese.txt", "japanese.txt"],
}


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
                w = line.strip()
                if 2 <= len(w) <= 15 and w and not w[0].isdigit():
                    words.append(w)

    if words:
        words = list(set(words))
        random.shuffle(words)
    return words


def load_all_word_lists(scripts: list[str]) -> dict[str, list[str]]:
    """Load word lists for all given scripts.

    Returns:
        {script_name: [words]}. Scripts with no words get empty list.
    """
    result = {}
    for script in scripts:
        if script == "emoji":
            result[script] = ["emoji"]
            continue
        words = load_word_list(script)
        if words:
            result[script] = words
        else:
            result[script] = []
    return result
