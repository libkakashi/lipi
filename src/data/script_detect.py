"""
Unicode-based script detection.

Detects which script a text string belongs to by examining Unicode code points.
Used to auto-label training samples with script/group IDs when LMDB
datasets don't store script labels explicitly.
"""

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, SCRIPT_TO_ID, GROUP_TO_ID


# Unicode block ranges for each script
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
    "cjk": [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Extension A
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
        (0x3000, 0x303F),   # CJK Symbols
    ],
    "korean": [
        (0xAC00, 0xD7AF),  # Hangul Syllables
        (0x1100, 0x11FF),  # Hangul Jamo
        (0x3130, 0x318F),  # Hangul Compatibility Jamo
    ],
    "thai": [(0x0E00, 0x0E7F)],
    "lao": [(0x0E80, 0x0EFF)],
    "emoji": [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
        (0x1F680, 0x1F6FF),  # Transport and Map
        (0x1F900, 0x1F9FF),  # Supplemental Symbols
        (0x2600, 0x26FF),    # Misc Symbols
        (0x2700, 0x27BF),    # Dingbats
    ],
}


def _char_to_script(ch: str) -> str | None:
    """Map a single character to its script name, or None if unknown."""
    cp = ord(ch)
    for script, ranges in _SCRIPT_RANGES.items():
        for start, end in ranges:
            if start <= cp <= end:
                return script
    return None


def detect_script(text: str) -> str:
    """Detect the script of a text string by majority vote.

    Args:
        text: Input string.

    Returns:
        Script name from SCRIPTS list. Defaults to "latin" if undetected.
    """
    counts = {}
    for ch in text:
        if ch.isspace() or ch in '.,;:!?-()[]{}"\'/\\@#$%^&*+=<>0123456789':
            continue
        script = _char_to_script(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1

    if not counts:
        return "latin"
    return max(counts, key=counts.get)


def detect_group(text: str) -> str:
    """Detect the coarse group of a text string.

    Returns:
        Group name from GROUPS list.
    """
    script = detect_script(text)
    return SCRIPT_TO_GROUP.get(script, "latin_cyrillic")


def detect_script_id(text: str) -> int:
    """Detect script and return its integer ID."""
    return SCRIPT_TO_ID.get(detect_script(text), 0)


def detect_group_id(text: str) -> int:
    """Detect group and return its integer ID."""
    return GROUP_TO_ID.get(detect_group(text), 0)
