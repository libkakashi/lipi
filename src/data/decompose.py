"""
Character decomposition for CJK and Korean.

CJK (han_kana): depth-2 decomposition using CHISE IDS database.
    ~1,700 components represent all 20,992 CJK chars.
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
_CJK_START = 0x4E00    # CJK Unified Ideographs start
_CJK_END = 0x9FFF      # CJK Unified Ideographs end
_HIRAGANA_START = 0x3041
_HIRAGANA_END = 0x3096
_KATAKANA_START = 0x30A1
_KATAKANA_END = 0x30FA

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


def _load_common_hangul():
    """No-op. Common syllables are now a frozen constant."""
    pass


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
    """Decompose Korean text. Common syllables stay whole, rare → jamo."""
    _load_common_hangul()
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
# CJK IDS decomposition (depth-2)
# =========================================================================

STRUCTURE_OPS = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

_OP_ARITY = {
    "⿰": 2, "⿱": 2, "⿴": 2, "⿵": 2, "⿶": 2,
    "⿷": 2, "⿸": 2, "⿹": 2, "⿺": 2, "⿻": 2,
    "⿲": 3, "⿳": 3,
}

_ids_char_to_seq: dict[str, str] | None = None
_ids_seq_to_char: dict[str, str] | None = None


def _load_ids():
    """Load IDS decomposition database (once)."""
    global _ids_char_to_seq, _ids_seq_to_char
    if _ids_char_to_seq is not None:
        return

    _ids_char_to_seq = {}
    _ids_seq_to_char = {}

    ids_path = Path(__file__).parent.parent.parent / "training_data" / "ids_decomposition.txt"
    if not ids_path.exists():
        return

    for line in ids_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(";") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        char = parts[1]
        decomp = parts[2].split("[")[0].strip()
        if not char or not decomp or len(char) != 1 or "&" in decomp:
            continue
        cp = ord(char)
        if _CJK_START <= cp <= _CJK_END:
            _ids_char_to_seq[char] = decomp
            _ids_seq_to_char[decomp] = char


def _is_cjk(ch: str) -> bool:
    return _CJK_START <= ord(ch) <= _CJK_END


def _decompose_cjk_depth2(char: str) -> list[str]:
    """Decompose one CJK character at depth 2.

    Level 1: char → structure + components
    Level 2: each component that's itself decomposable → structure + sub-components
    Components at level 2 that still have decompositions are kept whole (depth limit).
    """
    _load_ids()
    if not _is_cjk(char) or _ids_char_to_seq is None:
        return [char]
    if char not in _ids_char_to_seq:
        return [char]

    decomp = _ids_char_to_seq[char]
    tokens = []
    for c in decomp:
        if c in STRUCTURE_OPS:
            tokens.append(c)
        elif _is_cjk(c) and c in _ids_char_to_seq:
            # Depth 2: decompose this component one more level
            sub_decomp = _ids_char_to_seq[c]
            for sc in sub_decomp:
                if sc.strip() and ord(sc) >= 0x80:
                    tokens.append(sc)
        elif c.strip() and ord(c) >= 0x80:
            tokens.append(c)
    return tokens if tokens else [char]


def decompose_han_kana(text: str) -> str:
    """Decompose han_kana text. CJK chars → depth-2 components, kana unchanged."""
    _load_ids()
    parts = []
    for ch in text:
        if _is_cjk(ch):
            parts.extend(_decompose_cjk_depth2(ch))
        else:
            parts.append(ch)
    return "".join(parts)


def _consume_component(tokens: list[str], pos: int) -> tuple[str, int]:
    """Consume one component at pos, handling nested structure operators.

    Returns (substring, next_position).
    """
    if pos >= len(tokens):
        return "", pos
    ch = tokens[pos]
    if ch in STRUCTURE_OPS:
        arity = _OP_ARITY.get(ch, 2)
        result = ch
        p = pos + 1
        for _ in range(arity):
            sub, p = _consume_component(tokens, p)
            result += sub
        return result, p
    return ch, pos + 1


def _reconstruct_component(tokens: list[str], pos: int) -> tuple[str, int]:
    """Recursively reconstruct one component, resolving inner structures first.

    For depth-2: inner ⿰木目 → 相, then outer ⿱相心 → 想.
    Returns (reconstructed_char_or_fallback, next_position).
    """
    if pos >= len(tokens):
        return "", pos
    ch = tokens[pos]
    if ch not in STRUCTURE_OPS:
        return ch, pos + 1

    # Parse the structure: operator + N args (each arg may be nested)
    arity = _OP_ARITY.get(ch, 2)
    args = []
    p = pos + 1
    for _ in range(arity):
        arg, p = _reconstruct_component(tokens, p)
        args.append(arg)

    # Try to look up: operator + reconstructed args
    seq = ch + "".join(args)
    reconstructed = _ids_seq_to_char.get(seq) if _ids_seq_to_char else None
    if reconstructed:
        return reconstructed, p

    # Fallback: return the sequence as-is
    return seq, p


def reconstruct_han_kana(tokens: list[str]) -> str:
    """Reconstruct CJK text from depth-2 component sequence.

    Uses recursive bottom-up reconstruction: resolves inner structures
    first (⿰木目 → 相), then outer (⿱相心 → 想).
    """
    _load_ids()
    if _ids_seq_to_char is None:
        return "".join(tokens)

    result = []
    i = 0
    n = len(tokens)
    while i < n:
        ch = tokens[i]
        if ch in STRUCTURE_OPS:
            reconstructed, new_i = _reconstruct_component(tokens, i)
            result.append(reconstructed)
            i = new_i
        else:
            result.append(ch)
            i += 1
    return "".join(result)


# =========================================================================
# Public API
# =========================================================================

DECOMPOSE_GROUPS = frozenset({"han_kana", "korean"})


def decompose_text(text: str, group: str) -> str:
    """Decompose text into component tokens.

    han_kana: CJK → depth-2 components, kana unchanged.
    korean: common syllables whole, rare → jamo.
    Others: unchanged.
    """
    if group == "korean":
        return decompose_korean(text)
    if group == "han_kana":
        return decompose_han_kana(text)
    return text


def reconstruct_text(text: str, group: str) -> str:
    """Reconstruct original characters from decomposed token string."""
    if group == "korean":
        return reconstruct_korean(list(text))
    if group == "han_kana":
        return reconstruct_han_kana(list(text))
    return text


def get_vocab_tokens(group: str) -> list[str]:
    """Get the full set of output tokens for a decomposed group."""
    if group == "korean":
        _load_common_hangul()
        tokens = set(INITIALS) | set(MEDIALS) | set(f for f in FINALS if f)
        tokens |= _common_hangul or set()
        return sorted(tokens)

    if group == "han_kana":
        _load_ids()
        tokens = set()

        # All depth-2 components: decompose every IDS entry one more level
        if _ids_char_to_seq:
            for char, decomp in _ids_char_to_seq.items():
                for c in decomp:
                    if c in STRUCTURE_OPS:
                        tokens.add(c)
                    elif _is_cjk(c) and c in _ids_char_to_seq:
                        # Level 2: add the sub-components
                        for sc in _ids_char_to_seq[c]:
                            if sc.strip() and ord(sc) >= 0x80:
                                tokens.add(sc)
                    elif c.strip() and ord(c) >= 0x80:
                        tokens.add(c)

        # CJK chars not in IDS (atomic)
        decomposed = set(_ids_char_to_seq.keys()) if _ids_char_to_seq else set()
        for cp in range(_CJK_START, _CJK_END + 1):
            ch = chr(cp)
            if ch not in decomposed:
                tokens.add(ch)

        # Structure operators
        tokens.update(STRUCTURE_OPS)

        # Hiragana + Katakana
        for cp in range(_HIRAGANA_START, _HIRAGANA_END + 1):
            tokens.add(chr(cp))
        for cp in range(_KATAKANA_START, _KATAKANA_END + 1):
            tokens.add(chr(cp))

        # Marks and punctuation
        tokens.update("ー々、。「」『』（）！？・…〜【】《》〔〕")

        return sorted(tokens)

    return []
