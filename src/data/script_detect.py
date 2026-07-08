"""
Unicode-based script detection and segmentation.

Two related concerns share this file:

  detect_script(text)                — majority-vote a whole string to one
                                       script name (labeling / diagnostics).
  split_by_script(text, parent)      — carve a string into runs by script,
                                       peeling off ASCII to "latin" and
                                       kana/kanji within han/kana parents.

Both operate on Unicode code-point ranges. The tight predicates
(is_kana / is_kanji / is_cjk_punct / is_ascii) are also exported for
callers that need to test one code point at a time.
"""

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
    "han": [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Extension A
        (0x3000, 0x303F),   # CJK Symbols/Punctuation (also valid in kana regions;
                            # majority-vote in detect_script handles mixed segments)
        (0xFF00, 0xFFEF),   # Fullwidth Forms (CJK punctuation)
    ],
    "kana": [
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
        (0x31F0, 0x31FF),   # Katakana Phonetic Extensions (rare Ainu)
    ],
    "korean": [
        (0xAC00, 0xD7AF),  # Hangul Syllables
        (0x1100, 0x11FF),  # Hangul Jamo
        (0x3130, 0x318F),  # Hangul Compatibility Jamo
    ],
    "odia": [(0x0B00, 0x0B7F)],
    "sinhala": [(0x0D80, 0x0DFF)],
    "thai": [(0x0E00, 0x0E7F)],
    "lao": [(0x0E80, 0x0EFF)],
    "burmese": [(0x1000, 0x109F)],
    "khmer": [(0x1780, 0x17FF)],
    "armenian": [(0x0530, 0x058F)],
    "georgian": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
    "ethiopic": [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF)],
    "tibetan": [(0x0F00, 0x0FFF)],
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


def is_kana_cp(cp: int) -> bool:
    """Hiragana + Katakana + Katakana Phonetic Extensions."""
    return 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF


def is_kanji_cp(cp: int) -> bool:
    """CJK Unified Ideographs + CJK Extension A."""
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF


def is_cjk_punct_cp(cp: int) -> bool:
    """CJK Symbols and Punctuation."""
    return 0x3000 <= cp <= 0x303F


def is_ascii_cp(cp: int) -> bool:
    """Printable ASCII (excluding space)."""
    return 0x21 <= cp <= 0x7E


def split_by_script(text: str, parent_script: str) -> list[tuple[str, str]]:
    """Split text into runs of same-script characters, peeling off ASCII.

    ASCII characters (0x21-0x7E) become "latin" segments regardless of
    parent. For han/kana parents, kana and kanji/CJK-punct also split
    into separate runs so the training pipeline can label each segment
    with the correct script for CTC targets. Everything else stays in
    parent_script.

    Returns [(chunk_text, script_name), ...].
    """
    if not text:
        return []

    segments = []
    current: list[str] = []
    current_script: str | None = None

    for ch in text:
        cp = ord(ch)

        if is_ascii_cp(cp):
            script: str | None = "latin"
        elif parent_script in ("han", "kana"):
            if is_kana_cp(cp):
                script = "kana"
            elif is_kanji_cp(cp) or is_cjk_punct_cp(cp):
                script = "han"
            else:
                script = current_script  # non-ASCII, non-CJK → attach
        else:
            script = parent_script

        if script != current_script and current_script is not None and script is not None:
            segments.append(("".join(current), current_script))
            current = []
        if script is not None:
            current_script = script
        current.append(ch)

    if current and current_script is not None:
        segments.append(("".join(current), current_script))

    return segments


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
