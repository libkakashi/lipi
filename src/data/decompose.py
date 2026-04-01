"""
Character decomposition for CJK and Korean.

CJK: decomposes Han characters into structure operators + atomic components.
     Uses CHISE IDS database. ~1300 atomic components represent all 20,992 chars.

Korean: decomposes Hangul syllable blocks into jamo (초성+중성+종성).
        67 jamo represent all 11,172 possible syllables. Mathematical, no lookup.

Other scripts: no decomposition (characters pass through unchanged).

Usage:
    from src.data.decompose import decompose_text, reconstruct_text

    # Encoding (for CTC targets)
    decompose_text("明天好", group="han_kana")  → "⿰日月天⿰女子"
    decompose_text("한국",   group="korean")    → "ㅎㅏㄴㄱㅜㄱ"
    decompose_text("hello",  group="latin")     → "hello"

    # Decoding (from CTC output)
    reconstruct_text("⿰日月天⿰女子", group="han_kana")  → "明天好"
    reconstruct_text("ㅎㅏㄴㄱㅜㄱ",   group="korean")    → "한국"
"""

from pathlib import Path

# =========================================================================
# Korean Jamo decomposition (mathematical — no lookup needed)
# =========================================================================

# 19 initial consonants
INITIALS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")

# 21 medial vowels
MEDIALS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

# 28 final consonants (index 0 = no final)
FINALS = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

_INITIAL_TO_ID = {c: i for i, c in enumerate(INITIALS)}
_MEDIAL_TO_ID = {c: i for i, c in enumerate(MEDIALS)}
_FINAL_TO_ID = {c: i for i, c in enumerate(FINALS) if c}
_FINAL_TO_ID[""] = 0

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3


def decompose_hangul(char: str) -> list[str]:
    """Decompose a single Hangul syllable into jamo components."""
    cp = ord(char)
    if not (HANGUL_BASE <= cp <= HANGUL_END):
        return [char]  # not Hangul, return as-is

    offset = cp - HANGUL_BASE
    initial = offset // (21 * 28)
    medial = (offset % (21 * 28)) // 28
    final = offset % 28

    result = [INITIALS[initial], MEDIALS[medial]]
    if final > 0:
        result.append(FINALS[final])
    return result


def reconstruct_hangul(jamo: list[str]) -> str:
    """Reconstruct Hangul syllables from jamo sequence."""
    result = []
    i = 0
    while i < len(jamo):
        ch = jamo[i]

        # Check if this starts a syllable (initial consonant)
        if ch in _INITIAL_TO_ID and i + 1 < len(jamo) and jamo[i + 1] in _MEDIAL_TO_ID:
            initial = _INITIAL_TO_ID[ch]
            medial = _MEDIAL_TO_ID[jamo[i + 1]]
            final = 0
            consumed = 2

            # Check for final consonant
            if i + 2 < len(jamo) and jamo[i + 2] in _FINAL_TO_ID:
                # But only consume as final if next char is NOT a medial
                # (otherwise it's the initial of the next syllable)
                candidate = jamo[i + 2]
                if candidate in _FINAL_TO_ID:
                    if i + 3 >= len(jamo) or jamo[i + 3] not in _MEDIAL_TO_ID:
                        # Safe to use as final
                        final = _FINAL_TO_ID[candidate]
                        consumed = 3
                    elif candidate in _INITIAL_TO_ID:
                        # Ambiguous: could be final of this or initial of next
                        # If next char after is a medial, this is initial of next syllable
                        pass

            code = HANGUL_BASE + (initial * 21 + medial) * 28 + final
            result.append(chr(code))
            i += consumed
        else:
            result.append(ch)
            i += 1

    return "".join(result)


# =========================================================================
# CJK decomposition (IDS database lookup)
# =========================================================================

STRUCTURE_OPS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

# Lazy-loaded IDS data
_ids_decomp: dict[str, str] | None = None
_ids_reverse: dict[str, str] | None = None


def _load_ids():
    """Load IDS decomposition data from file."""
    global _ids_decomp, _ids_reverse
    if _ids_decomp is not None:
        return

    _ids_decomp = {}
    _ids_reverse = {}

    ids_path = Path(__file__).parent.parent.parent / "training_data" / "ids_decomposition.txt"
    if not ids_path.exists():
        return

    for line in ids_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(";") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            char = parts[1]
            decomp = parts[2].split("[")[0].strip()
            if char and decomp and len(char) == 1 and "&" not in decomp:
                cp = ord(char)
                if 0x4E00 <= cp <= 0x9FFF:
                    _ids_decomp[char] = decomp
                    _ids_reverse[decomp] = char


def decompose_cjk_char(char: str) -> list[str]:
    """Decompose a single CJK character into components (depth 1).

    Returns list of tokens: [structure_op, component1, component2, ...]
    If character can't be decomposed, returns [char] (kept whole).
    """
    _load_ids()
    cp = ord(char)

    # Only decompose CJK Unified Ideographs
    if not (0x4E00 <= cp <= 0x9FFF):
        return [char]

    if _ids_decomp is None or char not in _ids_decomp:
        return [char]

    decomp = _ids_decomp[char]
    tokens = []
    for c in decomp:
        if c.strip() and ord(c) >= 0x80:
            tokens.append(c)

    return tokens if tokens else [char]


def reconstruct_cjk(tokens: list[str]) -> str:
    """Reconstruct CJK text from component token sequence."""
    _load_ids()
    if _ids_reverse is None:
        return "".join(tokens)

    result = []
    i = 0
    while i < len(tokens):
        ch = tokens[i]

        # If it's a structure operator, try to reconstruct a character
        if ch in STRUCTURE_OPS:
            # Determine how many components this operator expects
            if ch in "⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻":
                n_args = 2
            elif ch in "⿲⿳":
                n_args = 3
            else:
                n_args = 2

            if i + n_args < len(tokens):
                # Build the decomposition string to look up
                decomp_str = ch + "".join(tokens[i + 1:i + 1 + n_args])
                if decomp_str in _ids_reverse:
                    result.append(_ids_reverse[decomp_str])
                    i += 1 + n_args
                    continue

            # Couldn't reconstruct — output the operator as-is
            result.append(ch)
            i += 1
        elif 0x4E00 <= ord(ch) <= 0x9FFF:
            # Whole CJK character (atomic, or component used as-is)
            result.append(ch)
            i += 1
        else:
            result.append(ch)
            i += 1

    return "".join(result)


# =========================================================================
# Unified interface
# =========================================================================

def decompose_text(text: str, group: str) -> str:
    """Decompose text into component tokens based on script group.

    Returns a string where each character is a component token.
    For CJK: characters become multi-token sequences.
    For Korean: syllables become 2-3 jamo tokens.
    For others: unchanged.
    """
    if group == "korean":
        tokens = []
        for ch in text:
            tokens.extend(decompose_hangul(ch))
        return "".join(tokens)

    elif group == "han_kana":
        tokens = []
        for ch in text:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF:
                tokens.extend(decompose_cjk_char(ch))
            else:
                tokens.append(ch)  # kana, punctuation — unchanged
        return "".join(tokens)

    else:
        return text  # Latin, Arabic, etc. — no decomposition


def reconstruct_text(text: str, group: str) -> str:
    """Reconstruct original text from decomposed component sequence."""
    tokens = list(text)

    if group == "korean":
        return reconstruct_hangul(tokens)

    elif group == "han_kana":
        return reconstruct_cjk(tokens)

    else:
        return text


def get_vocab_tokens(group: str) -> list[str]:
    """Get the complete set of tokens needed for a group's vocabulary."""
    if group == "korean":
        tokens = set()
        tokens.update(INITIALS)
        tokens.update(MEDIALS)
        tokens.update(f for f in FINALS if f)
        return sorted(tokens)

    elif group == "han_kana":
        _load_ids()
        tokens = set()

        # All atomic components from IDS
        if _ids_decomp:
            for decomp in _ids_decomp.values():
                for c in decomp:
                    if c.strip() and ord(c) >= 0x80:
                        tokens.add(c)

        # Characters without decomposition (atomic CJK)
        all_cjk = set(chr(cp) for cp in range(0x4E00, 0xA000))
        decomposed = set(_ids_decomp.keys()) if _ids_decomp else set()
        atomic = all_cjk - decomposed
        tokens.update(atomic)

        # Structure operators
        tokens.update(STRUCTURE_OPS)

        # Hiragana
        for cp in range(0x3041, 0x3097):
            tokens.add(chr(cp))
        # Katakana
        for cp in range(0x30A1, 0x30FB):
            tokens.add(chr(cp))
        # Marks
        tokens.update("ー々")
        # CJK punctuation
        tokens.update("、。「」『』（）！？・…〜【】《》〔〕")

        return sorted(tokens)

    else:
        return []  # other groups build vocab from their word lists
