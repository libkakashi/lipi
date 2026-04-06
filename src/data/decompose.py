"""
Character decomposition for CJK, Korean, and Arabic.

CJK (han_kana) and Korean use arbitrary N-symbol encoding with word-level BPE:
    Characters in the script's range are assigned fixed codes using N PUA
    symbols, ranked by real-world frequency. Each code ends with SEP (U+2E3B).
    Top chars get shorter codes (1-symbol), rare chars get longer (4-symbol).
    Word-level BPE compresses further, crossing character boundaries within
    words.

    CJK: 13 symbols (U+E000-E00C), 2500 BPE merges. All chars encoded
    (CJK + kana + punctuation + digits).
    Korean: 11 symbols (U+F200-F20A), 2500 BPE merges.

Arabic: character-pair BPE (separate system, unchanged).

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
            "char_ranges": [(_HANGUL_BASE, _HANGUL_END)],
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


def _decompose_arbitrary(text: str, script_name: str) -> str:
    """Decompose text using arbitrary encoding.

    Script chars -> pre-built token sequence.
    Non-script chars pass through unchanged.
    """
    enc = _load_arbitrary_encoding(script_name)
    char_to_tokens = enc["char_to_tokens"]
    if not char_to_tokens:
        return text
    parts: list[str] = []
    for ch in text:
        tokens = char_to_tokens.get(ch)
        if tokens:
            parts.extend(tokens)
        else:
            parts.append(ch)
    return "".join(parts)


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

def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (_CJK_UNIFIED_START <= cp <= _CJK_UNIFIED_END or
            _CJK_EXT_A_START <= cp <= _CJK_EXT_A_END)


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

def _is_hangul(ch: str) -> bool:
    return _HANGUL_BASE <= ord(ch) <= _HANGUL_END


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
# Arabic BPE (unchanged — separate system)
# =========================================================================

_ARABIC_PUA_START = 0xF100

_arabic_merges: list[tuple[str, str, str]] | None = None
_arabic_pua_to_chars: dict[str, str] | None = None


def _load_arabic_merges():
    """Load Arabic BPE merge table (once)."""
    global _arabic_merges, _arabic_pua_to_chars
    if _arabic_merges is not None:
        return

    _arabic_merges = []
    _arabic_pua_to_chars = {}

    merges_path = (Path(__file__).parent.parent.parent /
                   "training_data" / "word_lists" / "arabic_bpe_merges.tsv")
    if not merges_path.exists():
        return

    for line in merges_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("index\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        a = chr(int(parts[1], 16))
        b = chr(int(parts[2], 16))
        merged = chr(int(parts[3], 16))
        expanded = parts[4]  # original characters
        _arabic_merges.append((a, b, merged))
        _arabic_pua_to_chars[merged] = expanded


def decompose_arabic(text: str) -> str:
    """Apply BPE merges to Arabic text."""
    _load_arabic_merges()
    if not _arabic_merges:
        return text

    seq = list(text)
    for a, b, merged in _arabic_merges:
        new_seq: list[str] = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq

    return "".join(seq)


def reconstruct_arabic(tokens: list[str]) -> str:
    """Replace PUA tokens back to original Arabic characters."""
    _load_arabic_merges()
    if not _arabic_pua_to_chars:
        return "".join(tokens)

    result: list[str] = []
    for t in tokens:
        if t in _arabic_pua_to_chars:
            result.append(_arabic_pua_to_chars[t])
        else:
            result.append(t)
    return "".join(result)


# =========================================================================
# Public API
# =========================================================================

DECOMPOSE_GROUPS = frozenset({"sino_japanese", "korean", "arabic"})


def decompose_text(text: str, group: str) -> str:
    """Decompose text into component tokens.

    sino_japanese: CJK -> N-symbol codes + BPE merges, kana unchanged.
    korean: Hangul syllables -> N-symbol codes + BPE merges.
    arabic: character pairs -> BPE merged tokens.
    Others: unchanged.
    """
    if group == "korean":
        return decompose_korean(text)
    if group == "sino_japanese":
        return decompose_han_kana(text)
    if group == "arabic":
        return decompose_arabic(text)
    return text


def reconstruct_text(text: str, group: str) -> str:
    """Reconstruct original characters from decomposed token string."""
    if group == "korean":
        return reconstruct_korean(list(text))
    if group == "sino_japanese":
        return reconstruct_han_kana(list(text))
    if group == "arabic":
        return reconstruct_arabic(list(text))
    return text


# =========================================================================
# Backward-compatible aliases for external code that imports private names
# =========================================================================

def _load_cjk_decomposition():
    """Backward-compatible: load han_kana encoding."""
    _load_arbitrary_encoding("han_kana")


# Expose _cjk_char_to_tokens as a module-level property-like accessor.
# Code that does `from src.data.decompose import _cjk_char_to_tokens`
# will get None initially; they must call _load_cjk_decomposition() first,
# which populates the encoding cache. We provide a property-like wrapper.

class _CJKCharToTokensProxy:
    """Proxy that delegates to the encoding cache for han_kana."""
    def __getattr__(self, name):
        enc = _encoding_cache.get("han_kana", {})
        return getattr(enc.get("char_to_tokens", {}), name)
    def __getitem__(self, key):
        enc = _encoding_cache.get("han_kana", {})
        return enc.get("char_to_tokens", {})[key]
    def __contains__(self, key):
        enc = _encoding_cache.get("han_kana", {})
        return key in enc.get("char_to_tokens", {})
    def __iter__(self):
        enc = _encoding_cache.get("han_kana", {})
        return iter(enc.get("char_to_tokens", {}))
    def __len__(self):
        enc = _encoding_cache.get("han_kana", {})
        return len(enc.get("char_to_tokens", {}))
    def __bool__(self):
        enc = _encoding_cache.get("han_kana", {})
        ct = enc.get("char_to_tokens", {})
        return bool(ct)
    def items(self):
        enc = _encoding_cache.get("han_kana", {})
        return enc.get("char_to_tokens", {}).items()
    def values(self):
        enc = _encoding_cache.get("han_kana", {})
        return enc.get("char_to_tokens", {}).values()
    def keys(self):
        enc = _encoding_cache.get("han_kana", {})
        return enc.get("char_to_tokens", {}).keys()
    def get(self, key, default=None):
        enc = _encoding_cache.get("han_kana", {})
        return enc.get("char_to_tokens", {}).get(key, default)


_cjk_char_to_tokens = _CJKCharToTokensProxy()


def _load_korean_merges():
    """Backward-compatible: load korean encoding."""
    _load_arbitrary_encoding("korean")


class _KoreanMergesProxy:
    """Proxy for _korean_merges backward compat."""
    def __bool__(self):
        enc = _encoding_cache.get("korean", {})
        return bool(enc.get("bpe_merges", []))
    def __len__(self):
        enc = _encoding_cache.get("korean", {})
        return len(enc.get("bpe_merges", []))
    def __iter__(self):
        enc = _encoding_cache.get("korean", {})
        return iter(enc.get("bpe_merges", []))
    def __getitem__(self, idx):
        enc = _encoding_cache.get("korean", {})
        return enc.get("bpe_merges", [])[idx]

_korean_merges = _KoreanMergesProxy()


def get_vocab_tokens(group: str) -> list[str]:
    """Get the full set of output tokens for a decomposed group."""
    if group == "korean":
        enc = _load_arbitrary_encoding("korean")
        tokens: set[str] = set()
        # All tokens from char codes (base symbols, BPE merges, SEP)
        if enc["char_to_tokens"]:
            for char_tokens in enc["char_to_tokens"].values():
                tokens.update(char_tokens)
        tokens.add(SEP_CHAR)
        return sorted(tokens)

    if group == "sino_japanese":
        enc = _load_arbitrary_encoding("han_kana")
        tokens = set()
        # All tokens from char codes (base symbols, BPE merges, SEP)
        if enc["char_to_tokens"]:
            for char_tokens in enc["char_to_tokens"].values():
                tokens.update(char_tokens)
        tokens.add(SEP_CHAR)
        return sorted(tokens)

    return []
