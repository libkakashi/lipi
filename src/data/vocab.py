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

from src.data.bigrams import BASE_CHARS, BLANK_TOKEN
from src.data.decompose import get_vocab_tokens, DECOMPOSE_GROUPS

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

# Groups that use decomposition (vocab from decompose.py, not Unicode ranges)
# han_kana: IDS depth-2 components + kana
# korean: jamo + top-250 common syllables
_DECOMPOSED_GROUPS = {"han_kana", "korean"}


def _chars_from_ranges(ranges: list[tuple[int, int]]) -> list[str]:
    """Get all printable characters from Unicode ranges."""
    chars = []
    for start, end in ranges:
        for cp in range(start, end + 1):
            ch = chr(cp)
            if ch.strip():  # skip whitespace/control chars
                chars.append(ch)
    return chars


def build_script_vocab(script: str, group: str) -> list[str]:
    """Build the fixed vocab for a script.

    Returns:
        Sorted list of tokens: [BLANK, base_chars..., script_chars...]
    """
    tokens = set(BASE_CHARS)

    if group in _DECOMPOSED_GROUPS:
        # Decomposed: vocab defined by decomposition rules
        tokens.update(get_vocab_tokens(group))
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
