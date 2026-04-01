"""
Character decomposition for CJK and Korean.

CJK: decomposes Han characters into structure operators + components
     using the CHISE IDS database (depth-1 only).
Korean: decomposes Hangul syllables into jamo. Mathematical, no lookup.
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
_FINAL_MAP = {c: i for i, c in enumerate(FINALS) if c}  # char → index
_INITIAL_MAP = {c: i for i, c in enumerate(INITIALS)}
_MEDIAL_MAP = {c: i for i, c in enumerate(MEDIALS)}

_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3


def _is_hangul(ch: str) -> bool:
    return _HANGUL_BASE <= ord(ch) <= _HANGUL_END


def decompose_hangul(char: str) -> list[str]:
    """Decompose one Hangul syllable → 2 or 3 jamo."""
    if not _is_hangul(char):
        return [char]
    offset = ord(char) - _HANGUL_BASE
    initial = offset // (21 * 28)
    medial = (offset % (21 * 28)) // 28
    final = offset % 28
    result = [INITIALS[initial], MEDIALS[medial]]
    if final > 0:
        result.append(FINALS[final])
    return result


def reconstruct_hangul(jamo: list[str]) -> str:
    """Reconstruct Hangul syllables from jamo list.

    Greedy: consumes initial+medial(+optional final) at each step.
    A consonant is taken as final only if the NEXT token is not a medial
    (i.e., it can't be the initial of a following syllable).
    """
    result = []
    i = 0
    n = len(jamo)
    while i < n:
        ch = jamo[i]
        # Try to start a syllable: need initial + medial
        if ch in _INITIAL_SET and i + 1 < n and jamo[i + 1] in _MEDIAL_SET:
            initial = _INITIAL_MAP[ch]
            medial = _MEDIAL_MAP[jamo[i + 1]]
            final = 0
            consumed = 2
            # Try to consume a final consonant
            if i + 2 < n and jamo[i + 2] in _FINAL_MAP:
                next_is_initial_of_syllable = (
                    i + 3 < n
                    and jamo[i + 2] in _INITIAL_SET
                    and jamo[i + 3] in _MEDIAL_SET
                )
                if not next_is_initial_of_syllable:
                    final = _FINAL_MAP[jamo[i + 2]]
                    consumed = 3
            code = _HANGUL_BASE + (initial * 21 + medial) * 28 + final
            result.append(chr(code))
            i += consumed
        else:
            result.append(ch)
            i += 1
    return "".join(result)


# =========================================================================
# CJK IDS decomposition
# =========================================================================

STRUCTURE_OPS = frozenset("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")

# How many component arguments each structure operator takes
_OP_ARITY = {
    "⿰": 2, "⿱": 2, "⿴": 2, "⿵": 2, "⿶": 2,
    "⿷": 2, "⿸": 2, "⿹": 2, "⿺": 2, "⿻": 2,
    "⿲": 3, "⿳": 3,
}

# Lazy-loaded
_ids_char_to_seq: dict[str, str] | None = None  # char → decomposition string
_ids_seq_to_char: dict[str, str] | None = None  # decomposition string → char


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
        # Skip entries with unresolved entity references
        if not char or not decomp or len(char) != 1 or "&" in decomp:
            continue
        cp = ord(char)
        if 0x4E00 <= cp <= 0x9FFF:
            _ids_char_to_seq[char] = decomp
            _ids_seq_to_char[decomp] = char


def _is_cjk(ch: str) -> bool:
    return 0x4E00 <= ord(ch) <= 0x9FFF


def decompose_cjk_char(char: str) -> list[str]:
    """Decompose one CJK character into tokens (depth 1).

    Returns the IDS decomposition tokens, or [char] if atomic.
    Example: 明 → [⿰, 日, 月]
    """
    _load_ids()
    if not _is_cjk(char):
        return [char]
    if _ids_char_to_seq is None or char not in _ids_char_to_seq:
        return [char]  # atomic or no data
    decomp = _ids_char_to_seq[char]
    tokens = [c for c in decomp if c.strip() and ord(c) >= 0x80]
    return tokens if tokens else [char]


def _consume_component(tokens: list[str], pos: int) -> tuple[str, int]:
    """Consume one component starting at pos, handling nested operators.

    Returns (substring, next_position).
    A component is either a single token or an operator + its args.
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
    else:
        return ch, pos + 1


def reconstruct_cjk(tokens: list[str]) -> str:
    """Reconstruct CJK text from token sequence.

    Handles nested structure operators (e.g., ⿱⿱亠口小 → 京).
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
            # Parse the full nested structure
            seq, new_i = _consume_component(tokens, i)
            reconstructed = _ids_seq_to_char.get(seq)
            if reconstructed:
                result.append(reconstructed)
                i = new_i
            else:
                # Can't reconstruct — emit the raw tokens
                result.append(ch)
                i += 1
        else:
            result.append(ch)
            i += 1
    return "".join(result)


# =========================================================================
# Public API
# =========================================================================

# Groups that use decomposition
DECOMPOSE_GROUPS = frozenset({"han_kana", "korean"})


def decompose_text(text: str, group: str) -> str:
    """Decompose text into component tokens.

    CJK characters → structure operators + components.
    Hangul syllables → jamo.
    Everything else passes through unchanged.
    Returns a string (each character is one token).
    """
    if group == "korean":
        parts = []
        for ch in text:
            parts.extend(decompose_hangul(ch))
        return "".join(parts)

    if group == "han_kana":
        parts = []
        for ch in text:
            if _is_cjk(ch):
                parts.extend(decompose_cjk_char(ch))
            else:
                parts.append(ch)
        return "".join(parts)

    return text


def reconstruct_text(text: str, group: str) -> str:
    """Reconstruct original characters from decomposed token string."""
    if group == "korean":
        return reconstruct_hangul(list(text))
    if group == "han_kana":
        return reconstruct_cjk(list(text))
    return text


def get_vocab_tokens(group: str) -> list[str]:
    """Get the full set of output tokens for a decomposed group.

    Returns sorted list of all tokens the CTC head needs to output.
    For non-decomposed groups, returns empty list (they build vocab from words).
    """
    if group == "korean":
        tokens = set(INITIALS) | set(MEDIALS) | set(f for f in FINALS if f)
        return sorted(tokens)

    if group == "han_kana":
        _load_ids()
        tokens = set()

        # Components that appear in decompositions
        if _ids_char_to_seq:
            for decomp in _ids_char_to_seq.values():
                for c in decomp:
                    if c.strip() and ord(c) >= 0x80:
                        tokens.add(c)

        # CJK characters not in the database (atomic — kept whole)
        if _ids_char_to_seq:
            decomposed = set(_ids_char_to_seq.keys())
        else:
            decomposed = set()
        for cp in range(0x4E00, 0xA000):
            ch = chr(cp)
            if ch not in decomposed:
                tokens.add(ch)

        # Structure operators
        tokens.update(STRUCTURE_OPS)

        # Hiragana (U+3041–U+3096)
        for cp in range(0x3041, 0x3097):
            tokens.add(chr(cp))

        # Katakana (U+30A1–U+30FA)
        for cp in range(0x30A1, 0x30FB):
            tokens.add(chr(cp))

        # Common marks and punctuation
        tokens.update("ー々、。「」『』（）！？・…〜【】《》〔〕")

        return sorted(tokens)

    return []
