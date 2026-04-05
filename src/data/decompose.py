"""
Character decomposition for CJK and Korean.

CJK (han_kana): atom-based decomposition with SEP-aware BPE.
    Base: 387 leaf atoms + 12 IDS operators.
    SEP appended to prefix-collision chars BEFORE BPE, so BPE naturally
    merges high-frequency (atom, SEP) pairs into single tokens.
    ~2,500 BPE merged tokens. Each CJK char maps to a short token
    sequence via a pre-built table. Kana pass through unchanged.

Korean: hybrid — top-250 common syllables kept whole,
    rare syllables decomposed to jamo (51 unique).

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

# SEP token: U+2E3B THREE-EM DASH
# In decomposition table, SEP appears only after chars with prefix collisions.
# BPE may merge (atom, SEP) pairs into single PUA tokens.
SEP_CHAR = "\u2E3B"

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
# CJK atom-based decomposition with SEP-aware BPE
# =========================================================================

STRUCTURE_OPS = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

# Loaded lazily at module init
_cjk_char_to_tokens: dict[str, list[str]] | None = None
_cjk_tokens_to_char: dict[tuple[str, ...], str] | None = None
_cjk_vocab_tokens: set[str] | None = None


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (_CJK_UNIFIED_START <= cp <= _CJK_UNIFIED_END or
            _CJK_EXT_A_START <= cp <= _CJK_EXT_A_END)


def _load_cjk_decomposition():
    """Load the pre-built CJK decomposition table (once).

    The table already contains SEP tokens where needed (for chars with
    prefix collisions). BPE merges may have fused (atom, SEP) pairs
    into single PUA tokens.
    """
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
            key = tuple(tokens)
            if key not in _cjk_tokens_to_char:
                _cjk_tokens_to_char[key] = char
            _cjk_vocab_tokens.update(tokens)

    # Also include IDS operators and SEP as known CJK vocab tokens
    _cjk_vocab_tokens.update(STRUCTURE_OPS)
    _cjk_vocab_tokens.add(SEP_CHAR)


def decompose_han_kana(text: str) -> str:
    """Decompose han_kana text.

    CJK chars -> pre-built token sequence (SEP included where needed).
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
                parts.append(ch)
        else:
            parts.append(ch)
    return "".join(parts)


def reconstruct_han_kana(tokens: list[str]) -> str:
    """Reconstruct CJK text from token sequence.

    Uses greedy left-to-right matching: after each CJK token is added
    to the current group, check if the group matches a character.
    Non-prefix-collision chars match immediately (no SEP needed).
    Prefix-collision chars have SEP baked into their sequence, which
    forces the match to wait for the full sequence.

    Kana never appear in CJK decompositions (the 3 katakana used as
    IDS shape placeholders are replaced with PUA tokens at build time),
    so kana tokens always act as group boundaries.
    """
    _load_cjk_decomposition()
    if _cjk_tokens_to_char is None:
        return "".join(tokens)

    result: list[str] = []
    current_group: list[str] = []

    for token in tokens:
        if token == SEP_CHAR:
            # SEP is an explicit boundary. Try with SEP in the key first
            # (prefix-collision chars have SEP in their reverse map key).
            if current_group:
                key_with = tuple(current_group) + (SEP_CHAR,)
                key_without = tuple(current_group)
                char = _cjk_tokens_to_char.get(key_with)
                if char is None:
                    char = _cjk_tokens_to_char.get(key_without)
                if char:
                    result.append(char)
                else:
                    result.extend(current_group)
                current_group = []
        elif token in _cjk_vocab_tokens:
            current_group.append(token)
            # Greedy match: try to resolve after every token
            key = tuple(current_group)
            char = _cjk_tokens_to_char.get(key)
            if char:
                result.append(char)
                current_group = []
        else:
            # Non-CJK token (kana, punctuation, etc.)
            if current_group:
                key = tuple(current_group)
                char = _cjk_tokens_to_char.get(key)
                if char:
                    result.append(char)
                else:
                    result.extend(current_group)
                current_group = []
            result.append(token)

    # Flush final group
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

    sino_japanese: CJK -> atom/BPE tokens (SEP included where needed), kana unchanged.
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
        _load_cjk_decomposition()
        tokens: set[str] = set()

        # All tokens from the decomposition table (atoms, BPE, SEP, operators)
        if _cjk_char_to_tokens:
            for char_tokens in _cjk_char_to_tokens.values():
                tokens.update(char_tokens)

        # SEP token (in case no char has bare SEP remaining)
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
