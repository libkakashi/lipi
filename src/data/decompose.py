"""
Character decomposition for CJK and Korean.

CJK (han_kana): atom-based decomposition with BPE merges.
    336 leaf atoms + 12 IDS operators + 250 BPE merged tokens.
    Each CJK char maps to a short token sequence via a pre-built table.
    A SEP token separates characters in the decomposed output.
    Kana pass through unchanged.

Korean: hybrid — top-250 common syllables kept whole,
    rare syllables decomposed to jamo (51 unique).
    Total vocab ~301 tokens.

Other scripts: pass through unchanged.
"""

from pathlib import Path

# =========================================================================
# Korean Jamo
# =========================================================================

INITIALS = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")  # 19
MEDIALS = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")  # 21
FINALS = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")  # 28

_INITIAL_SET = set(INITIALS)
_MEDIAL_SET = set(MEDIALS)
_FINAL_MAP = {c: i for i, c in enumerate(FINALS) if c}
_INITIAL_MAP = {c: i for i, c in enumerate(INITIALS)}
_MEDIAL_MAP = {c: i for i, c in enumerate(MEDIALS)}

# Unicode ranges
_HANGUL_BASE = 0xAC00  # First Hangul syllable block
_HANGUL_END = 0xD7A3   # Last Hangul syllable block
_CJK_UNIFIED_START = 0x4E00
_CJK_UNIFIED_END = 0x9FFF
_CJK_EXT_A_START = 0x3400
_CJK_EXT_A_END = 0x4DBF
_HIRAGANA_START = 0x3041
_HIRAGANA_END = 0x3096
_KATAKANA_START = 0x30A1
_KATAKANA_END = 0x30FA

# SEP token: U+2E3B THREE-EM DASH (separates CJK characters in decomposed form)
SEP_CHAR = "\u2E3B"

# PUA range for BPE merged tokens
_PUA_START = 0xE000

# Top-250 common Hangul syllables (kept whole, not decomposed).
# Frozen from frequency analysis of Korean word list. Covers 77.4% of
# syllable occurrences. Remaining syllables decompose to jamo (~67 tokens).
# This is a FIXED constant — does not depend on training data.
_COMMON_HANGUL_250 = (
    "다이하에가의기로리지스고사는서시인자을대도나아한구부수어주정"
    "라전과국은공장으경원트제소를보상성관비학개드미마되무동조치적"
    "일오화선해들계거그와노신연유교게니회우레했세내강위문산었타포"
    "르프카명영만여법발재생감간단진모바실중방금물터남파민식용체통"
    "행군호데안면며크출역요업건작루김코차였음현양당야매러결토각디"
    "분까임합반설판급박래배된입집권히형종피려석네뉴키란심브버테광"
    "두즈후술속할청메표질티난운본등점근불추달초갈력예격페저증병론"
    "승독년령투평직태귀외환약글던울검절너찰누말린돌목료언살록철처"
    "항천베악더날클편골최"
)
_common_hangul: set[str] = set(_COMMON_HANGUL_250)


def _is_hangul(ch: str) -> bool:
    return _HANGUL_BASE <= ord(ch) <= _HANGUL_END


def decompose_hangul_char(char: str) -> list[str]:
    """Decompose one Hangul syllable into jamo."""
    cp = ord(char)
    if not (_HANGUL_BASE <= cp <= _HANGUL_END):
        return [char]
    offset = cp - _HANGUL_BASE
    initial = offset // (21 * 28)
    medial = (offset % (21 * 28)) // 28
    final = offset % 28
    result = [INITIALS[initial], MEDIALS[medial]]
    if final > 0:
        result.append(FINALS[final])
    return result


def decompose_korean(text: str) -> str:
    """Decompose Korean text. Common syllables stay whole, rare -> jamo."""
    parts = []
    for ch in text:
        if _is_hangul(ch) and ch not in _common_hangul:
            parts.extend(decompose_hangul_char(ch))
        else:
            parts.append(ch)
    return "".join(parts)


def reconstruct_korean(tokens: list[str]) -> str:
    """Reconstruct Korean from mixed whole-syllable + jamo sequence."""
    result = []
    i = 0
    n = len(tokens)
    while i < n:
        ch = tokens[i]
        # Already a whole Hangul syllable — pass through
        if _is_hangul(ch):
            result.append(ch)
            i += 1
            continue
        # Try to form a syllable from jamo
        if ch in _INITIAL_SET and i + 1 < n and tokens[i + 1] in _MEDIAL_SET:
            initial = _INITIAL_MAP[ch]
            medial = _MEDIAL_MAP[tokens[i + 1]]
            final = 0
            consumed = 2
            if i + 2 < n and tokens[i + 2] in _FINAL_MAP:
                # Only take as final if next token ISN'T starting a new syllable
                next_starts_syllable = (
                    i + 3 < n
                    and tokens[i + 2] in _INITIAL_SET
                    and tokens[i + 3] in _MEDIAL_SET
                )
                if not next_starts_syllable:
                    final = _FINAL_MAP[tokens[i + 2]]
                    consumed = 3
            code = _HANGUL_BASE + (initial * 21 + medial) * 28 + final
            result.append(chr(code))
            i += consumed
        else:
            result.append(ch)
            i += 1
    return "".join(result)


# =========================================================================
# CJK atom-based decomposition with BPE
# =========================================================================

STRUCTURE_OPS = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

# Loaded lazily at module init
_cjk_char_to_tokens: dict[str, list[str]] | None = None
_cjk_tokens_to_char: dict[tuple[str, ...], str] | None = None
_cjk_vocab_tokens: set[str] | None = None  # All tokens that can appear in CJK decompositions


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (_CJK_UNIFIED_START <= cp <= _CJK_UNIFIED_END or
            _CJK_EXT_A_START <= cp <= _CJK_EXT_A_END)


def _load_cjk_decomposition():
    """Load the pre-built CJK decomposition table and BPE merges (once)."""
    global _cjk_char_to_tokens, _cjk_tokens_to_char, _cjk_vocab_tokens
    if _cjk_char_to_tokens is not None:
        return

    _cjk_char_to_tokens = {}
    _cjk_tokens_to_char = {}
    _cjk_vocab_tokens = set()

    decomp_path = (Path(__file__).parent.parent.parent /
                   "training_data" / "word_lists" / "cjk_decomposition.tsv")
    if not decomp_path.exists():
        return

    for line in decomp_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("character\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        char = parts[0]
        tokens = parts[2].split()
        if char and tokens:
            _cjk_char_to_tokens[char] = tokens
            # Build reverse map: token tuple -> char
            key = tuple(tokens)
            # For unresolved chars, first one wins (deterministic)
            if key not in _cjk_tokens_to_char:
                _cjk_tokens_to_char[key] = char
            _cjk_vocab_tokens.update(tokens)

    # Also include IDS operators as CJK vocab tokens
    _cjk_vocab_tokens.update(STRUCTURE_OPS)


def decompose_han_kana(text: str) -> str:
    """Decompose han_kana text.

    CJK chars -> atom/BPE token sequence + SEP between characters.
    Kana and punctuation pass through unchanged.
    """
    _load_cjk_decomposition()
    if _cjk_char_to_tokens is None:
        return text

    parts: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            tokens = _cjk_char_to_tokens.get(ch)
            if tokens:
                parts.extend(tokens)
            else:
                # Unknown CJK char: emit as-is
                parts.append(ch)
            parts.append(SEP_CHAR)
        else:
            parts.append(ch)
    return "".join(parts)


def reconstruct_han_kana(tokens: list[str]) -> str:
    """Reconstruct CJK text from atom/BPE token sequence.

    Splits on SEP tokens to get per-character groups, then looks up
    each group in the reconstruction table.
    """
    _load_cjk_decomposition()
    if _cjk_tokens_to_char is None:
        return "".join(tokens)

    result: list[str] = []
    current_group: list[str] = []

    for token in tokens:
        if token == SEP_CHAR:
            if current_group:
                key = tuple(current_group)
                char = _cjk_tokens_to_char.get(key)
                if char:
                    result.append(char)
                else:
                    # Fallback: emit tokens as-is
                    result.extend(current_group)
                current_group = []
        elif token in _cjk_vocab_tokens:
            # Part of a CJK decomposition (atom, BPE token, or IDS operator)
            current_group.append(token)
        else:
            # Non-CJK token (kana, punctuation, etc.)
            if current_group:
                # Flush any pending CJK group (missing SEP)
                key = tuple(current_group)
                char = _cjk_tokens_to_char.get(key)
                if char:
                    result.append(char)
                else:
                    result.extend(current_group)
                current_group = []
            result.append(token)

    # Flush final group if any
    if current_group:
        key = tuple(current_group)
        char = _cjk_tokens_to_char.get(key)
        if char:
            result.append(char)
        else:
            result.extend(current_group)

    return "".join(result)


# =========================================================================
# Public API
# =========================================================================

DECOMPOSE_GROUPS = frozenset({"sino_japanese", "korean"})


def decompose_text(text: str, group: str) -> str:
    """Decompose text into component tokens.

    sino_japanese: CJK -> atom/BPE tokens + SEP, kana unchanged.
    korean: common syllables whole, rare -> jamo.
    Others: unchanged.
    """
    if group == "korean":
        return decompose_korean(text)
    if group == "sino_japanese":
        return decompose_han_kana(text)
    return text


def reconstruct_text(text: str, group: str) -> str:
    """Reconstruct original characters from decomposed token string."""
    if group == "korean":
        return reconstruct_korean(list(text))
    if group == "sino_japanese":
        return reconstruct_han_kana(list(text))
    return text


def get_vocab_tokens(group: str) -> list[str]:
    """Get the full set of output tokens for a decomposed group."""
    if group == "korean":
        tokens = set(INITIALS) | set(MEDIALS) | set(f for f in FINALS if f)
        tokens |= _common_hangul or set()
        return sorted(tokens)

    if group == "sino_japanese":
        # Vocab is now defined entirely by the frozen vocab file.
        # This function is only used for reference/testing.
        _load_cjk_decomposition()
        tokens: set[str] = set()

        # All tokens from the decomposition table
        if _cjk_char_to_tokens:
            for char_tokens in _cjk_char_to_tokens.values():
                tokens.update(char_tokens)

        # SEP token
        tokens.add(SEP_CHAR)

        # Hiragana + Katakana
        for cp in range(_HIRAGANA_START, _HIRAGANA_END + 1):
            tokens.add(chr(cp))
        for cp in range(_KATAKANA_START, _KATAKANA_END + 1):
            tokens.add(chr(cp))

        # Marks and punctuation
        tokens.update("ー々、。「」『』（）！？・…〜【】《》〔〕")

        return sorted(tokens)

    return []
