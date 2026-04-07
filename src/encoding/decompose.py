"""
Character decomposition for CJK, Korean, and Arabic.

All three use arbitrary N-symbol encoding with word-level BPE:
    Characters in the script's range are assigned fixed codes using N PUA
    symbols, ranked by real-world frequency. Each code ends with SEP (U+2E3B).
    Top chars get shorter codes (1-symbol), rare chars get longer (3-4 symbol).
    Word-level BPE compresses further, crossing character boundaries within
    words.

    CJK: 13 symbols (U+E000-E00C), 2500 BPE merges.
    Korean: 11 symbols (U+EA00-EA0A), 2500 BPE merges.
    Arabic: 9 symbols (U+EB00-EB08), 2500 BPE merges.

Other scripts: pass through unchanged.
"""

from pathlib import Path

# =========================================================================
# Constants
# =========================================================================

# SEP token: U+2E3B THREE-EM DASH
SEP_CHAR = "\u2E3B"
SEP_CODEPOINT = 0x2E3B

# Unicode ranges
_CJK_UNIFIED_START = 0x4E00
_CJK_UNIFIED_END = 0x9FFF
_CJK_EXT_A_START = 0x3400
_CJK_EXT_A_END = 0x4DBF
_HIRAGANA_START = 0x3041
_HIRAGANA_END = 0x3096
_KATAKANA_START = 0x30A1
_KATAKANA_END = 0x30FA
_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3


# =========================================================================
# Generic arbitrary encoding loader
# =========================================================================

# Per-script encoding data, loaded lazily
_encoding_cache: dict[str, dict] = {}


def _load_arbitrary_encoding(script_name: str) -> dict:
    """Load arbitrary encoding tables for a script (once per script).

    Reads:
        training_data/word_lists/{char_codes_file} -- char -> token sequence
        training_data/word_lists/{bpe_merges_file} -- BPE merge table

    Returns dict with:
        char_to_tokens: dict[str, list[str]]
        tokens_to_char: dict[tuple[str, ...], str]  (expanded base codes -> char)
        bpe_merges: list[tuple[str, str, str]]
        pua_to_pair: dict[str, tuple[str, str]]
        base_symbols: frozenset[str]
        vocab_tokens: set[str]
    """
    if script_name in _encoding_cache:
        return _encoding_cache[script_name]

    # Script-specific file naming
    FILE_CONFIG = {
        "han_kana": {
            "char_codes": "cjk_char_codes.tsv",
            "bpe_merges": "bpe_merges.tsv",
            "pua_base": 0xE000,
            "char_ranges": [
                (_CJK_EXT_A_START, _CJK_EXT_A_END),   # CJK Extension A
                (_CJK_UNIFIED_START, _CJK_UNIFIED_END), # CJK Unified
                (0x3040, 0x309F),   # Hiragana
                (0x30A0, 0x30FF),   # Katakana
                (0x3000, 0x303F),   # CJK Symbols and Punctuation
                (0xFF01, 0xFF5E),   # Fullwidth ASCII variants
                (0xFF61, 0xFF9F),   # Halfwidth Katakana
            ],
            "extra_chars": "0123456789(),.!?:;-/'\"% ",
        },
        "korean": {
            "char_codes": "korean_char_codes.tsv",
            "bpe_merges": "korean_bpe_merges.tsv",
            "pua_base": 0xEA00,
            "char_ranges": [
                (_HANGUL_BASE, _HANGUL_END),  # Hangul Syllables
                (0x3131, 0x318E),              # Hangul Compat Jamo
                (0x3000, 0x303F),              # CJK Symbols and Punctuation
                (0x2018, 0x201F),              # Smart quotes
            ],
            "extra_chars": "0123456789(),.!?:;-/'\"% _~",
        },
        "arabic": {
            "char_codes": "arabic_char_codes.tsv",
            "bpe_merges": "arabic_arb_bpe_merges.tsv",
            "pua_base": 0xEB00,
            "char_ranges": [
                (0x0600, 0x06FF),   # Arabic
                (0x0750, 0x077F),   # Arabic Supplement
                (0x0870, 0x089F),   # Arabic Extended-B
                (0x08A0, 0x08FF),   # Arabic Extended-A
                (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
            ],
        },
    }

    if script_name not in FILE_CONFIG:
        data = {
            "char_to_tokens": {},
            "tokens_to_char": {},
            "bpe_merges": [],
            "pua_to_pair": {},
            "base_symbols": frozenset(),
            "vocab_tokens": set(),
            "char_ranges": [],
        }
        _encoding_cache[script_name] = data
        return data

    fcfg = FILE_CONFIG[script_name]
    base_dir = Path(__file__).parent.parent.parent / "training_data" / "word_lists"

    char_to_tokens: dict[str, list[str]] = {}
    tokens_to_char: dict[tuple[str, ...], str] = {}
    bpe_merges: list[tuple[str, str, str]] = []
    pua_to_pair: dict[str, tuple[str, str]] = {}
    vocab_tokens: set[str] = set()

    # Determine N from pua_base and char_codes file
    pua_base = fcfg["pua_base"]

    # Load BPE merges first (needed to expand codes for reverse map)
    merges_path = base_dir / fcfg["bpe_merges"]
    if merges_path.exists():
        for line in merges_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("index\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            a = chr(int(parts[1], 16))
            b = chr(int(parts[2], 16))
            merged = chr(int(parts[3], 16))
            bpe_merges.append((a, b, merged))
            pua_to_pair[merged] = (a, b)

    # Helper to expand BPE tokens
    def expand(token: str) -> list[str]:
        if token in pua_to_pair:
            a, b = pua_to_pair[token]
            return expand(a) + expand(b)
        return [token]

    # Determine base symbols from the char_codes file
    # We'll collect all unique base-level PUA symbols
    base_symbol_set: set[str] = set()

    # Load char codes
    codes_path = base_dir / fcfg["char_codes"]
    if codes_path.exists():
        for line in codes_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("character\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            char = parts[0]
            tokens = [chr(int(h, 16)) for h in parts[1].split()]
            if char and tokens:
                char_to_tokens[char] = tokens
                vocab_tokens.update(tokens)

                # Build reverse map from EXPANDED base codes
                expanded: list[str] = []
                for t in tokens:
                    expanded.extend(expand(t))
                key = tuple(expanded)
                if key not in tokens_to_char:
                    tokens_to_char[key] = char

                # Collect base symbols (non-BPE, non-SEP PUA tokens)
                for t in expanded:
                    cp = ord(t)
                    if cp != SEP_CODEPOINT and t not in pua_to_pair:
                        base_symbol_set.add(t)

    base_symbols = frozenset(base_symbol_set)
    vocab_tokens.update(base_symbols)
    vocab_tokens.add(SEP_CHAR)

    data = {
        "char_to_tokens": char_to_tokens,
        "tokens_to_char": tokens_to_char,
        "bpe_merges": bpe_merges,
        "pua_to_pair": pua_to_pair,
        "base_symbols": base_symbols,
        "vocab_tokens": vocab_tokens,
        "char_ranges": fcfg["char_ranges"],
        "expand_fn": expand,
    }
    _encoding_cache[script_name] = data
    return data


def _is_encoding_token(ch: str, script_name: str) -> bool:
    """Check if a character is an encoding token (base symbol, BPE merge, or SEP)."""
    cp = ord(ch)
    if cp == SEP_CODEPOINT:
        return True
    # PUA range covers both base symbols and BPE merges
    if 0xE000 <= cp <= 0xF8FF:
        return True
    return False


# Per-script BPE merge lookup: {(a, b): (merged, priority)}
_bpe_lookup: dict[str, dict[tuple[str, str], tuple[str, int]]] = {}


def _get_bpe_lookup(script_name: str, bpe_merges: list[tuple[str, str, str]]
                    ) -> dict[tuple[str, str], tuple[str, int]]:
    """Build or return cached merge lookup for priority-based BPE."""
    if script_name not in _bpe_lookup:
        lookup: dict[tuple[str, str], tuple[str, int]] = {}
        for priority, (a, b, merged) in enumerate(bpe_merges):
            pair = (a, b)
            if pair not in lookup:  # first occurrence has highest priority
                lookup[pair] = (merged, priority)
        _bpe_lookup[script_name] = lookup
    return _bpe_lookup[script_name]


def _apply_bpe(parts: list[str], bpe_merges: list[tuple[str, str, str]],
               script_name: str = "") -> list[str]:
    """Apply BPE merges using priority-based pair merging.

    Instead of 2500 sequential full passes, uses a lookup dict to find
    mergeable pairs and processes them in priority order. Much faster
    for short sequences (typical words are 3-15 chars).
    """
    if len(parts) <= 1:
        return parts

    lookup = _get_bpe_lookup(script_name, bpe_merges)

    # Linked-list style: use indices for efficient merge
    # For short sequences, iterative approach with lookup is fast enough
    while True:
        # Find the highest-priority (lowest index) mergeable pair
        best_priority = len(bpe_merges)
        best_pos = -1
        best_merged = ""
        for i in range(len(parts) - 1):
            pair = (parts[i], parts[i + 1])
            entry = lookup.get(pair)
            if entry and entry[1] < best_priority:
                best_merged, best_priority = entry
                best_pos = i

        if best_pos < 0:
            break  # no more merges possible

        # Apply this merge at all occurrences (same priority)
        a, b = parts[best_pos], parts[best_pos + 1]
        new_parts: list[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i] == a and parts[i + 1] == b:
                new_parts.append(best_merged)
                i += 2
            else:
                new_parts.append(parts[i])
                i += 1
        parts = new_parts

    return parts


# Per-script word→tokens cache (populated lazily during decomposition)
_word_cache: dict[str, dict[str, str]] = {}


def _decompose_arbitrary(text: str, script_name: str) -> str:
    """Decompose text using arbitrary encoding.

    1. Script chars -> per-char token codes.
    2. Apply word-level BPE merges (always cross-char).
    Non-script chars pass through unchanged.

    Results are cached per-word for fast repeated lookups.
    """
    enc = _load_arbitrary_encoding(script_name)
    char_to_tokens = enc["char_to_tokens"]
    if not char_to_tokens:
        return text

    if script_name not in _word_cache:
        _word_cache[script_name] = {}
    cache = _word_cache[script_name]

    if text in cache:
        return cache[text]

    # Step 1: per-char lookup
    # Unknown chars are skipped (they'll be dropped by the tokenizer as OOV).
    # This happens legitimately for mixed-script text (Latin in Japanese, etc.)
    parts: list[str] = []
    for ch in text:
        tokens = char_to_tokens.get(ch)
        if tokens:
            parts.extend(tokens)
        # else: foreign char, not in this script's encoding — skip

    # Step 2: always apply word-level BPE merges across character boundaries
    bpe_merges = enc["bpe_merges"]
    if bpe_merges:
        parts = _apply_bpe(parts, bpe_merges, script_name)

    result = "".join(parts)
    cache[text] = result
    return result


def _reconstruct_arbitrary(tokens: list[str], script_name: str) -> str:
    """Reconstruct text from arbitrary encoding tokens.

    1. Expand all BPE merged tokens back to base symbols + SEP.
    2. Parse the symbol stream: SEP marks character boundaries.
    3. Look up each base-symbol sequence + SEP -> original char.

    Non-script tokens pass through unchanged.
    """
    enc = _load_arbitrary_encoding(script_name)
    tokens_to_char = enc["tokens_to_char"]
    if not tokens_to_char:
        return "".join(tokens)

    base_symbols = enc["base_symbols"]
    expand_fn = enc["expand_fn"]

    # Step 1: Expand BPE tokens
    expanded: list[str] = []
    for t in tokens:
        if _is_encoding_token(t, script_name):
            expanded.extend(expand_fn(t))
        else:
            expanded.append(t)

    # Step 2: Parse expanded stream using SEP as char boundary
    result: list[str] = []
    current_group: list[str] = []

    for token in expanded:
        if token == SEP_CHAR:
            if current_group:
                key = tuple(current_group) + (SEP_CHAR,)
                char = tokens_to_char.get(key)
                if char:
                    result.append(char)
                else:
                    result.extend(current_group)
                    result.append(SEP_CHAR)
                current_group = []
        elif token in base_symbols:
            current_group.append(token)
        else:
            # Non-script token (kana, punctuation, etc.)
            if current_group:
                result.extend(current_group)
                current_group = []
            result.append(token)

    if current_group:
        result.extend(current_group)

    return "".join(result)


# =========================================================================
# CJK (han_kana) — uses generic arbitrary encoding
# =========================================================================

def decompose_han_kana(text: str) -> str:
    """Decompose han_kana text.

    CJK chars -> pre-built token sequence (base symbols + SEP + BPE merges).
    Kana and punctuation pass through unchanged.
    """
    return _decompose_arbitrary(text, "han_kana")


def reconstruct_han_kana(tokens: list[str]) -> str:
    """Reconstruct CJK text from token sequence.

    Kana and other non-CJK tokens pass through unchanged.
    """
    return _reconstruct_arbitrary(tokens, "han_kana")


# =========================================================================
# Korean — uses generic arbitrary encoding
# =========================================================================

def decompose_korean(text: str) -> str:
    """Decompose Korean text using arbitrary 11-symbol encoding.

    Hangul syllables -> pre-built token sequence (base symbols + SEP + BPE merges).
    Non-Hangul characters pass through unchanged.
    """
    return _decompose_arbitrary(text, "korean")


def reconstruct_korean(tokens: list[str]) -> str:
    """Reconstruct Korean text from token sequence."""
    return _reconstruct_arbitrary(tokens, "korean")


# =========================================================================
# Public API
# =========================================================================

DECOMPOSE_GROUPS = frozenset({"sino_japanese", "korean", "arabic"})


def decompose_text(text: str, group: str) -> str:
    """Decompose text into component tokens.

    sino_japanese: CJK -> N-symbol codes + BPE merges, kana unchanged.
    korean: Hangul syllables -> N-symbol codes + BPE merges.
    arabic: Arabic chars -> N-symbol codes + BPE merges.
    Others: unchanged.
    """
    if group == "korean":
        return decompose_korean(text)
    if group == "sino_japanese":
        return decompose_han_kana(text)
    if group == "arabic":
        return _decompose_arbitrary(text, "arabic")
    return text


def reconstruct_text(text: str, group: str) -> str:
    """Reconstruct original characters from decomposed token string."""
    if group == "korean":
        return reconstruct_korean(list(text))
    if group == "sino_japanese":
        return reconstruct_han_kana(list(text))
    if group == "arabic":
        return _reconstruct_arbitrary(list(text), "arabic")
    return text


