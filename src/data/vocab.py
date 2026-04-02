"""
Fixed vocabulary definitions per script.

Vocabs are determined by Unicode ranges and decomposition rules — NOT by
training data. This prevents dirty word lists from inflating vocab sizes.

Each vocab includes:
  - BLANK token (index 0, for CTC)
  - BASE_CHARS (printable ASCII 32-126)
  - Script-specific characters from Unicode ranges
  - For decomposed scripts: decomposition tokens instead of raw chars
"""

from pathlib import Path

from src.data.bigrams import BASE_CHARS, BLANK_TOKEN
from src.data.decompose import DECOMPOSE_GROUPS

# Unicode ranges per script (same as script_detect.py)
_SCRIPT_RANGES = {
    "latin": [
        (0x0041, 0x024F),  # Basic Latin + Latin Extended
        (0x1E00, 0x1EFF),  # Latin Extended Additional
    ],
    "cyrillic": [
        (0x0400, 0x04FF),  # Cyrillic
        (0x0500, 0x052F),  # Cyrillic Supplement
    ],
    "greek": [
        (0x0370, 0x03FF),  # Greek and Coptic
        (0x1F00, 0x1FFF),  # Greek Extended
    ],
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "bengali": [(0x0980, 0x09FF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "arabic": [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ],
    "hebrew": [
        (0x0590, 0x05FF),  # Hebrew
        (0xFB1D, 0xFB4F),  # Hebrew Presentation Forms
    ],
    "thai": [(0x0E00, 0x0E7F)],
    "lao": [(0x0E80, 0x0EFF)],
    "emoji": [],  # emoji uses synthetic renders, not character vocab
}

# Frozen decomposition vocabs — loaded once from pre-computed files.
# These were generated from the IDS database + Korean jamo/syllable analysis.
# They do NOT depend on training data.
_FROZEN_VOCAB_DIR = Path(__file__).parent / "frozen_vocabs"
_frozen_cache: dict[str, list[str]] = {}


def _load_frozen_vocab(group: str) -> list[str]:
    """Load frozen vocab tokens for a decomposed group."""
    if group not in _frozen_cache:
        path = _FROZEN_VOCAB_DIR / f"{group}_vocab.txt"
        _frozen_cache[group] = path.read_text(encoding="utf-8").strip().split("\n")
    return _frozen_cache[group]


def _chars_from_ranges(ranges: list[tuple[int, int]]) -> list[str]:
    """Get all assigned, printable characters from Unicode ranges."""
    import unicodedata
    chars = []
    for start, end in ranges:
        for cp in range(start, end + 1):
            ch = chr(cp)
            cat = unicodedata.category(ch)
            # Skip unassigned (Cn), control (Cc), and whitespace
            if cat != 'Cn' and not cat.startswith('C') and ch.strip():
                chars.append(ch)
    return chars


def build_script_vocab(script: str, group: str) -> list[str]:
    """Build the fixed vocab for a script.

    Returns:
        Sorted list of tokens: [BLANK, base_chars..., script_chars...]
    """
    tokens = set(BASE_CHARS)

    if group in DECOMPOSE_GROUPS:
        # Decomposed: vocab from frozen pre-computed files
        tokens.update(_load_frozen_vocab(group))
    else:
        # Non-decomposed: vocab defined by Unicode ranges
        if script in _SCRIPT_RANGES:
            tokens.update(_chars_from_ranges(_SCRIPT_RANGES[script]))

    if script == "emoji":
        # Emoji has minimal vocab (just the label "emoji")
        tokens.add("e")
        tokens.add("m")
        tokens.add("o")
        tokens.add("j")
        tokens.add("i")

    return [BLANK_TOKEN] + sorted(tokens)


def get_all_script_vocabs(
    active_scripts: list[str],
    active_groups: list[str],
) -> tuple[list[list[str]], list[list[int]]]:
    """Build fixed vocabs for all active scripts, organized by group.

    Returns:
        group_vocabs[g][s]: token list for script s in group g
        group_vocab_sizes[g][s]: vocab size
    """
    from src.model.lid import SCRIPT_TO_GROUP

    group_vocabs = []
    group_vocab_sizes = []

    for g, group_name in enumerate(active_groups):
        scripts_in_group = [s for s in active_scripts
                            if SCRIPT_TO_GROUP.get(s) == group_name]
        vocabs = []
        sizes = []
        for script in scripts_in_group:
            vocab = build_script_vocab(script, group_name)
            vocabs.append(vocab)
            sizes.append(len(vocab))
            print(f"    Group {g} ({group_name}) / {script}: {len(vocab)} tokens")
        group_vocabs.append(vocabs)
        group_vocab_sizes.append(sizes)

    return group_vocabs, group_vocab_sizes
