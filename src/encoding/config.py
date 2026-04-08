"""
Encoding configuration for all scripts.

No-fusion scripts (1 CTC token = 1 character):
  Character set derived from Unicode ranges.  Token IDs are plain ints:
    0 = CTC BLANK
    1..vocab_size-1 = characters in sorted codepoint order

BPE-encoded scripts (decomposition + BPE merging):
  Token IDs are plain integers, remapped to contiguous range after build.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────

def _build_char_list(*range_groups: list[tuple[int, int]]) -> list[str]:
    """Build sorted, deduplicated char list from Unicode ranges.

    Filters out unassigned (Cn) and control (Cc) codepoints,
    but keeps U+0020 SPACE.
    """
    codepoints: set[int] = set()
    for ranges in range_groups:
        for start, end in ranges:
            codepoints.update(range(start, end + 1))
    chars = []
    for cp in sorted(codepoints):
        c = chr(cp)
        cat = unicodedata.category(c)
        if cat == "Cn":
            continue
        if cat == "Cc" and cp != 0x20:
            continue
        chars.append(c)
    return chars


# ── Common character sets ────────────────────────────────────────────

# ASCII symbols, digits, and punctuation (no Latin letters)
_ASCII_COMMON: list[tuple[int, int]] = [
    (0x0020, 0x0040),   # space !"#$%&'()*+,-./0-9:;<=>?@
    (0x005B, 0x0060),   # [\]^_`
    (0x007B, 0x007E),   # {|}~
]

# Typographic characters shared across scripts
_TYPOGRAPHIC_COMMON: list[tuple[int, int]] = [
    (0x00AB, 0x00AB),   # «
    (0x00BB, 0x00BB),   # »
    (0x2010, 0x2010),   # ‐ hyphen
    (0x2013, 0x2014),   # – —
    (0x2018, 0x2019),   # ' '
    (0x201C, 0x201E),   # " " „
    (0x2026, 0x2026),   # …
]


# ── No-fusion codec ─────────────────────────────────────────────────

class NoFusionCodec:
    """Codec for scripts where 1 CTC token = 1 Unicode character.

    Token IDs: 0 = BLANK, 1..vocab_size-1 = chars (sorted by codepoint).
    """

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.vocab_size = len(chars) + 1          # +1 for BLANK at 0
        self._char_to_id = {c: i + 1 for i, c in enumerate(chars)}

    def decode(self, token_id: int) -> str:
        """Convert CTC token ID → character.  0 (BLANK) → ''."""
        if 0 < token_id <= len(self.chars):
            return self.chars[token_id - 1]
        return ""

    def encode(self, char: str) -> int:
        """Convert character → CTC token ID.  Unknown → 0."""
        return self._char_to_id.get(char, 0)

    def encode_text(self, text: str) -> list[int]:
        """Encode a string to a list of CTC token IDs."""
        text = unicodedata.normalize("NFC", text)
        return [self._char_to_id[c] for c in text if c in self._char_to_id]

    def decode_ids(self, ids: list[int]) -> str:
        """Decode a list of CTC token IDs to a string."""
        return "".join(self.decode(i) for i in ids)

    def __repr__(self) -> str:
        return f"NoFusionCodec(vocab_size={self.vocab_size})"


# ── No-fusion script definitions ────────────────────────────────────

NO_FUSION_SCRIPTS: dict[str, NoFusionCodec] = {
    "latin": NoFusionCodec(_build_char_list(
        [(0x0020, 0x007E)],     # Full printable ASCII
        [(0x00A1, 0x00FF)],     # Latin-1 Supplement
        [(0x0100, 0x024F)],     # Latin Extended-A & B
        [(0x1E00, 0x1EFF)],     # Latin Extended Additional
        [   # Typographic
            (0x2010, 0x2011), (0x2013, 0x2014),
            (0x2018, 0x201A), (0x201C, 0x201E),
            (0x2020, 0x2020), (0x2026, 0x2026),
            (0x2032, 0x2033),
        ],
    )),

    "cyrillic": NoFusionCodec(_build_char_list(
        _ASCII_COMMON,
        [(0x0400, 0x052F)],     # Cyrillic + Supplement
        _TYPOGRAPHIC_COMMON,
        [(0x2011, 0x2011), (0x2020, 0x2020)],  # extra typo
    )),

    "greek": NoFusionCodec(_build_char_list(
        _ASCII_COMMON,
        [(0x0370, 0x03FF)],     # Greek and Coptic
        [(0x1F00, 0x1FFF)],     # Greek Extended
        _TYPOGRAPHIC_COMMON,
    )),

    "hebrew": NoFusionCodec(_build_char_list(
        _ASCII_COMMON,
        [(0x0591, 0x05C7)],     # Hebrew points & accents
        [(0x05D0, 0x05F4)],     # Hebrew letters & ligatures
        [(0xFB1D, 0xFB4F)],     # Hebrew Presentation Forms
        _TYPOGRAPHIC_COMMON,
    )),

    "georgian": NoFusionCodec(_build_char_list(
        _ASCII_COMMON,
        [(0x10A0, 0x10FF)],     # Georgian
        [(0x2D00, 0x2D2F)],     # Georgian Supplement
        _TYPOGRAPHIC_COMMON,
    )),

    "armenian": NoFusionCodec(_build_char_list(
        _ASCII_COMMON,
        [(0x0531, 0x058F)],     # Armenian
        _TYPOGRAPHIC_COMMON,
        [(0x2024, 0x2024)],     # one dot leader
    )),

    "ethiopic": NoFusionCodec(_build_char_list(
        _ASCII_COMMON,
        [(0x1200, 0x1399)],     # Ethiopic
        [(0x2D80, 0x2DDF)],     # Ethiopic Extended
        _TYPOGRAPHIC_COMMON,
    )),

    "emoji": NoFusionCodec(_build_char_list(
        [(0x0020, 0x007E)],     # Full printable ASCII
        [(0x00AB, 0x00AB)],     # «
        [(0x00BB, 0x00BB)],     # »
        _TYPOGRAPHIC_COMMON,
    )),
}


# ── Fusion codec ────────────────────────────────────────────────────

FUSION_VOCAB: dict[str, int] = {
    "arabic": 500,
    "bengali": 750,
    "burmese": 500,
    "devanagari": 750,
    "gujarati": 750,
    "gurmukhi": 500,
    "kannada": 500,
    "khmer": 750,
    "lao": 500,
    "malayalam": 750,
    "odia": 750,
    "sinhala": 500,
    "tamil": 500,
    "telugu": 750,
    "thai": 500,
    "tibetan": 500,
}


class FusionCodec:
    """Codec for scripts where multiple codepoints fuse into one visual char.

    Token IDs:  0 = BLANK
                1..N_base = single-codepoint base characters
                N_base+1..vocab_size-1 = multi-codepoint fusion clusters

    Encode: segment text into grapheme clusters (Unicode \\X), look up each.
            Unknown clusters fall back to individual codepoint tokens.
    Decode: token ID → string (1+ codepoints), concatenate.
    """

    def __init__(self, base_chars: list[str], fusions: list[str]):
        self.base_chars = base_chars
        self.fusions = fusions
        self.tokens = base_chars + fusions
        self.vocab_size = len(self.tokens) + 1      # +1 for BLANK at 0
        self._token_to_id = {t: i + 1 for i, t in enumerate(self.tokens)}
        # Single-char fallback for unknown fusions
        self._char_to_id = {c: i + 1 for i, c in enumerate(base_chars)}

    def decode(self, token_id: int) -> str:
        """Convert CTC token ID → string.  0 (BLANK) → ''."""
        if 0 < token_id <= len(self.tokens):
            return self.tokens[token_id - 1]
        return ""

    def encode(self, cluster: str) -> int:
        """Convert grapheme cluster → CTC token ID.  Unknown → 0."""
        return self._token_to_id.get(cluster, 0)

    def encode_text(self, text: str) -> list[int]:
        """Encode string → list of CTC token IDs.

        Segments text into grapheme clusters.  Known clusters get a single
        token; unknown ones fall back to per-codepoint encoding.
        """
        import regex
        text = unicodedata.normalize("NFC", text)
        clusters = regex.findall(r'\X', text)
        ids = []
        for cluster in clusters:
            tid = self._token_to_id.get(cluster)
            if tid is not None:
                ids.append(tid)
            else:
                for c in cluster:
                    cid = self._char_to_id.get(c)
                    if cid is not None:
                        ids.append(cid)
        return ids

    def decode_ids(self, ids: list[int]) -> str:
        """Decode list of CTC token IDs → string."""
        return "".join(self.decode(i) for i in ids)

    def __repr__(self) -> str:
        return (f"FusionCodec(vocab_size={self.vocab_size}, "
                f"base={len(self.base_chars)}, fusions={len(self.fusions)})")


def build_fusion_codec(
    base_chars: list[str],
    fusions_tsv: Path,
    max_vocab: int,
    guaranteed_fusions: list[str] | None = None,
) -> FusionCodec:
    """Build a FusionCodec from base chars + a fusions frequency TSV.

    Args:
        base_chars: single-codepoint characters for this script.
        fusions_tsv: path to TSV with columns: cluster, count, codepoint_len.
                     Must be sorted by count descending.
        max_vocab: maximum total vocab size (including BLANK).
        guaranteed_fusions: multi-codepoint tokens always included before
                           frequency-ranked fusions (e.g. virama+consonant).

    Returns:
        FusionCodec with guaranteed fusions + top-N fusions that fit.
    """
    avail_slots = max_vocab - 1 - len(base_chars)  # -1 for BLANK
    base_set = set(base_chars)

    fusions: list[str] = []
    seen: set[str] = set()

    # Add guaranteed fusions first
    if guaranteed_fusions:
        for f in guaranteed_fusions:
            if len(fusions) >= avail_slots:
                break
            fusions.append(f)
            seen.add(f)

    # Fill remaining with frequency-ranked fusions from TSV
    if fusions_tsv.exists():
        for line in fusions_tsv.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            cluster = parts[0]
            cp_len = int(parts[2])
            if cp_len <= 1:
                continue
            if cluster in seen:
                continue
            if all(c in base_set for c in cluster):
                fusions.append(cluster)
                seen.add(cluster)
            if len(fusions) >= avail_slots:
                break

    return FusionCodec(base_chars, fusions)


# Virama+consonant pairs for Indic scripts (guaranteed fusions).
# These ensure consonant conjuncts encode as 2 tokens (base + virama+cons)
# instead of 3 (base + virama + cons), matching CTC's horizontal alignment.
_VIRAMA_PAIRS: dict[str, list[str]] = {}

def _build_virama_pairs():
    """Build virama+consonant token lists for all Indic scripts."""
    _scripts = {
        "devanagari": (0x094D, 0x0915, 0x093A),
        "bengali":    (0x09CD, 0x0995, 0x09B0),
        "gurmukhi":   (0x0A4D, 0x0A15, 0x0A39),
        "gujarati":   (0x0ACD, 0x0A95, 0x0AB0),
        "odia":       (0x0B4D, 0x0B15, 0x0B39),
        "kannada":    (0x0CCD, 0x0C95, 0x0CB9),
        "telugu":     (0x0C4D, 0x0C15, 0x0C39),
        "malayalam":  (0x0D4D, 0x0D15, 0x0D39),
        "tamil":      (0x0BCD, 0x0B95, 0x0BB9),
        "sinhala":    (0x0DCA, 0x0D9A, 0x0DC6),
    }
    for script, (virama_cp, con_start, con_end) in _scripts.items():
        v = chr(virama_cp)
        _VIRAMA_PAIRS[script] = [
            v + chr(cp) for cp in range(con_start, con_end + 1)
            if unicodedata.category(chr(cp)) != "Cn"
        ]

_build_virama_pairs()


# ── Fusion scripts ──────────────────────────────────────────────────
# Base character ranges for scripts where multiple codepoints fuse
# into a single visual character (e.g. consonant + matra in Devanagari).
# Fusion clusters are built separately from corpus data.

FUSION_BASE_RANGES: dict[str, list[list[tuple[int, int]]]] = {
    "arabic": [
        _ASCII_COMMON,
        [(0x0600, 0x06FF)],     # Arabic (covers Arabic, Urdu, Persian, Pashto)
        _TYPOGRAPHIC_COMMON,
    ],
    "devanagari": [
        _ASCII_COMMON,
        [(0x0900, 0x094D)],     # Devanagari signs + vowels + consonants + matras + virama
        [(0x0950, 0x0956)],     # OM + vowel signs (skip Kashmiri 094E-094F)
        [(0x0958, 0x0972)],     # Nukta consonants + digits + dandas (skip 0957)
        [(0x0979, 0x097F)],     # Extended consonants (skip Sindhi/Marwari 0973-0978)
        _TYPOGRAPHIC_COMMON,
        [(0x2015, 0x2015)],     # horizontal bar
    ],
    "gurmukhi": [
        _ASCII_COMMON,
        [(0x0A00, 0x0A7F)],     # Gurmukhi
        [(0x0964, 0x0964)],     # Devanagari danda
        _TYPOGRAPHIC_COMMON,
    ],
    "gujarati": [
        _ASCII_COMMON,
        [(0x0A80, 0x0AFF)],     # Gujarati
        _TYPOGRAPHIC_COMMON,
    ],
    "bengali": [
        _ASCII_COMMON,
        [(0x0981, 0x09FB)],     # Bengali (skip 0980 Anji, 09FC-09FE Vedic)
        [(0x0964, 0x0964)],     # Devanagari danda
        [(0x0970, 0x0970)],     # Devanagari abbreviation sign
        _TYPOGRAPHIC_COMMON,
        [(0x2032, 0x2032)],     # prime
    ],
    "odia": [
        _ASCII_COMMON,
        [(0x0B00, 0x0B73)],     # Odia (skip 0B74-0B77 fractions)
        [(0x0964, 0x0964)],     # Devanagari danda
        [(0x0970, 0x0970)],     # Devanagari abbreviation sign
        _TYPOGRAPHIC_COMMON,
    ],
    "kannada": [
        _ASCII_COMMON,
        [(0x0C80, 0x0CFF)],     # Kannada
        _TYPOGRAPHIC_COMMON,
    ],
    "telugu": [
        _ASCII_COMMON,
        [(0x0C00, 0x0C7F)],     # Telugu
        _TYPOGRAPHIC_COMMON,
    ],
    "malayalam": [
        _ASCII_COMMON,
        [(0x0D01, 0x0D57)],     # Malayalam (skip 0D00 Vedic, 0D58-0D5E fractions)
        [(0x0D5F, 0x0D7F)],     # Malayalam letters + chillu + au length mark
        _TYPOGRAPHIC_COMMON,
    ],
    "tamil": [
        _ASCII_COMMON,
        [(0x0B80, 0x0BF9)],     # Tamil (skip 0BFA number sign)
        _TYPOGRAPHIC_COMMON,
    ],
    "sinhala": [
        _ASCII_COMMON,
        [(0x0D82, 0x0DE5)],     # Sinhala (skip 0D81 candrabindu, 0DE6-0DEF Lith digits)
        [(0x00A0, 0x00A0)],     # no-break space
        _TYPOGRAPHIC_COMMON,
    ],
    "thai": [
        _ASCII_COMMON,
        [(0x0E00, 0x0E7F)],     # Thai
        _TYPOGRAPHIC_COMMON,
    ],
    "lao": [
        _ASCII_COMMON,
        [(0x0E80, 0x0EFF)],     # Lao
        [(0x3001, 0x3001)],     # ideographic comma
        [(0xFF08, 0xFF09)],     # fullwidth parentheses
        _TYPOGRAPHIC_COMMON,
    ],
    "burmese": [
        _ASCII_COMMON,
        [(0x1000, 0x109F)],     # Myanmar
        _TYPOGRAPHIC_COMMON,
    ],
    "khmer": [
        _ASCII_COMMON,
        [(0x1780, 0x17FF)],     # Khmer
        _TYPOGRAPHIC_COMMON,
    ],
    "tibetan": [
        _ASCII_COMMON,
        [(0x0F00, 0x0FFF)],     # Tibetan
        [(0xFF08, 0xFF08)],     # fullwidth left paren
        _TYPOGRAPHIC_COMMON,
    ],
}

# Build base char lists for each fusion script
FUSION_BASE_CHARS: dict[str, list[str]] = {
    script: _build_char_list(*ranges)
    for script, ranges in FUSION_BASE_RANGES.items()
}

# Lazy-loaded fusion codecs (loaded on first access)
_fusion_codec_cache: dict[str, FusionCodec] = {}

_FUSIONS_DIR = Path(__file__).parent.parent.parent / "training_data" / "corpora"


def get_fusion_codec(script: str) -> FusionCodec:
    """Get or build the FusionCodec for a script (cached)."""
    if script in _fusion_codec_cache:
        return _fusion_codec_cache[script]

    if script not in FUSION_BASE_CHARS:
        raise ValueError(f"Unknown fusion script: {script}")

    fusions_tsv = _FUSIONS_DIR / f"{script}_fusions.tsv"
    max_vocab = FUSION_VOCAB[script]
    codec = build_fusion_codec(
        FUSION_BASE_CHARS[script],
        fusions_tsv,
        max_vocab=max_vocab,
        guaranteed_fusions=_VIRAMA_PAIRS.get(script),
    )
    _fusion_codec_cache[script] = codec
    return codec


# ── Korean codec ─────────────────────────────────────────────────────

# Hangul decomposition constants
_HANGUL_BASE = 0xAC00
_LEAD_COUNT = 19
_VOWEL_COUNT = 21
_TAIL_COUNT = 28   # includes 0 = no tail

# 27 tail Jamo (index 1-27, 0 = no tail)
_TAIL_JAMO = [chr(0x11A8 + i) for i in range(27)]

# All 399 two-Jamo syllables (lead + vowel, no tail)
_TWO_JAMO = [chr(_HANGUL_BASE + lead * 588 + vowel * 28)
             for lead in range(_LEAD_COUNT) for vowel in range(_VOWEL_COUNT)]


def _decompose_syllable(char: str) -> tuple[int, int, int] | None:
    """Decompose Hangul syllable → (lead, vowel, tail) indices. None if not Hangul."""
    cp = ord(char)
    if not (_HANGUL_BASE <= cp < _HANGUL_BASE + 11172):
        return None
    idx = cp - _HANGUL_BASE
    return idx // 588, (idx % 588) // 28, idx % 28


def _compose_syllable(lead: int, vowel: int, tail: int) -> str:
    """Compose Hangul syllable from (lead, vowel, tail) indices."""
    return chr(_HANGUL_BASE + lead * 588 + vowel * 28 + tail)


class KoreanCodec:
    """Codec for Korean Hangul.

    Token layout:
      0 = BLANK
      1..N_common = common ASCII/punct chars
      N_common+1..+27 = tail Jamo tokens
      +28..+426 = 399 two-Jamo syllables (all lead+vowel combos)
      +427..vocab_size-1 = top three-Jamo syllables by frequency

    Encode: Hangul syllable →
      - If in vocab (2-jamo or frequent 3-jamo): 1 token
      - Else (rare 3-jamo): 2 tokens = 2-jamo token + tail token

    Decode: scan left to right,
      - If 2-jamo token followed by tail token: combine into 3-jamo syllable
      - Otherwise: standalone token
    """

    def __init__(self, common_chars: list[str], top_three_jamo: list[str]):
        # Build token list: common + tails + two-jamo + three-jamo
        self.tokens = common_chars + _TAIL_JAMO + _TWO_JAMO + top_three_jamo
        self.vocab_size = len(self.tokens) + 1   # +1 for BLANK

        self._token_to_id = {t: i + 1 for i, t in enumerate(self.tokens)}

        # Quick lookups
        self._two_jamo_ids = {t: self._token_to_id[t] for t in _TWO_JAMO}
        self._tail_ids = {t: self._token_to_id[t] for t in _TAIL_JAMO}
        self._tail_id_set = set(self._tail_ids.values())
        self._two_jamo_id_set = set(self._two_jamo_ids.values())

        # For decode: map (two_jamo_id, tail_id) → 3-jamo char
        self._combine = {}
        for lead in range(_LEAD_COUNT):
            for vowel in range(_VOWEL_COUNT):
                two_j = chr(_HANGUL_BASE + lead * 588 + vowel * 28)
                two_id = self._two_jamo_ids[two_j]
                for tail_idx in range(27):
                    tail_char = _TAIL_JAMO[tail_idx]
                    tail_id = self._tail_ids[tail_char]
                    three_j = chr(_HANGUL_BASE + lead * 588 + vowel * 28 + tail_idx + 1)
                    self._combine[(two_id, tail_id)] = three_j

    def decode(self, token_id: int) -> str:
        """Convert single token ID → string.  0 (BLANK) → ''."""
        if 0 < token_id <= len(self.tokens):
            return self.tokens[token_id - 1]
        return ""

    def encode_text(self, text: str) -> list[int]:
        """Encode string → list of CTC token IDs."""
        text = unicodedata.normalize("NFC", text)
        ids = []
        for char in text:
            tid = self._token_to_id.get(char)
            if tid is not None:
                ids.append(tid)
            else:
                # Rare 3-jamo: decompose into 2-jamo + tail
                decomp = _decompose_syllable(char)
                if decomp:
                    lead, vowel, tail = decomp
                    two_j = chr(_HANGUL_BASE + lead * 588 + vowel * 28)
                    ids.append(self._two_jamo_ids[two_j])
                    if tail > 0:
                        ids.append(self._tail_ids[_TAIL_JAMO[tail - 1]])
        return ids

    def decode_ids(self, ids: list[int]) -> str:
        """Decode list of CTC token IDs → string."""
        result = []
        i = 0
        while i < len(ids):
            tid = ids[i]
            # Check if this is a 2-jamo followed by a tail → combine
            if tid in self._two_jamo_id_set and i + 1 < len(ids):
                next_tid = ids[i + 1]
                combined = self._combine.get((tid, next_tid))
                if combined is not None:
                    result.append(combined)
                    i += 2
                    continue
            result.append(self.decode(tid))
            i += 1
        return "".join(result)

    def __repr__(self) -> str:
        return f"KoreanCodec(vocab_size={self.vocab_size})"


# Build Korean common chars
_KOREAN_COMMON = _build_char_list(
    _ASCII_COMMON,
    [(0x3000, 0x303F)],     # CJK Symbols and Punctuation
    [(0x2018, 0x201F)],     # Smart quotes
    _TYPOGRAPHIC_COMMON,
)

_KOREAN_VOCAB_SIZE = 1500


def _build_korean_codec() -> KoreanCodec:
    """Build KoreanCodec with top 3-jamo syllables from corpus."""
    from collections import Counter

    # Count 3-jamo frequencies from word list
    wl_path = Path(__file__).parent.parent.parent / "training_data" / "word_lists" / "korean.txt"
    three_freq = Counter()
    if wl_path.exists():
        for line in wl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            for c in line.strip():
                decomp = _decompose_syllable(c)
                if decomp and decomp[2] > 0:  # has tail = 3-jamo
                    three_freq[c] += 1

    # How many slots for 3-jamo?
    n_base = len(_KOREAN_COMMON) + 27 + 399  # common + tails + two-jamo
    n_three = _KOREAN_VOCAB_SIZE - 1 - n_base  # -1 for BLANK
    top_three = [c for c, _ in three_freq.most_common(n_three)]

    return KoreanCodec(_KOREAN_COMMON, top_three)


# Lazy-loaded
_korean_codec: KoreanCodec | None = None


def get_korean_codec() -> KoreanCodec:
    global _korean_codec
    if _korean_codec is None:
        _korean_codec = _build_korean_codec()
    return _korean_codec


# ── CJK codec ───────────────────────────────────────────────────────

_CJK_VOCAB_SIZE = 4000
_CJK_N_ALT = 24


class CJKCodec:
    """Codec for CJK (Chinese + Japanese).

    Token layout:
      0 = BLANK
      1..N_base = base chars (kana, ASCII, CJK punct, fullwidth)
      N_base+1..N_base+N_freq = frequent CJK chars (single token)
      N_base+N_freq+1..+24 = ALT00-ALT23 tokens

    Encode: char →
      - If in vocab: 1 token
      - Else: LEAF + ALT_XX (2 tokens, from visual similarity mapping)

    Decode: scan left to right,
      - Token not followed by ALT: standalone char
      - Token followed by ALT_XX: look up (token_id, XX) → mapped char
    """

    def __init__(
        self,
        base_chars: list[str],
        freq_chars: list[str],
        alt_mapping: dict[str, tuple[int, str]],  # char → (alt_slot, leaf_char)
    ):
        n_alt = _CJK_N_ALT
        self.tokens = base_chars + freq_chars
        self._alt_names = [f"ALT{i:02d}" for i in range(n_alt)]
        self.vocab_size = len(self.tokens) + n_alt + 1  # +1 BLANK, +N ALT

        self._token_to_id = {t: i + 1 for i, t in enumerate(self.tokens)}

        # ALT token IDs come after all regular tokens
        self._alt_base_id = len(self.tokens) + 1
        self._alt_id_set = set(range(self._alt_base_id, self._alt_base_id + n_alt))

        # Encode: char → (leaf_id, alt_id)
        self._char_to_pair: dict[str, tuple[int, int]] = {}
        for char, (slot, leaf_char) in alt_mapping.items():
            leaf_id = self._token_to_id.get(leaf_char)
            if leaf_id is not None:
                self._char_to_pair[char] = (leaf_id, self._alt_base_id + slot)

        # Decode: (leaf_id, alt_id) → char
        self._pair_to_char: dict[tuple[int, int], str] = {}
        for char, (leaf_id, alt_id) in self._char_to_pair.items():
            self._pair_to_char[(leaf_id, alt_id)] = char

    def decode(self, token_id: int) -> str:
        """Convert single token ID → string. 0 (BLANK) → ''."""
        if 0 < token_id <= len(self.tokens):
            return self.tokens[token_id - 1]
        return ""

    def encode_text(self, text: str) -> list[int]:
        """Encode string → list of CTC token IDs."""
        text = unicodedata.normalize("NFKC", text)
        ids = []
        for char in text:
            tid = self._token_to_id.get(char)
            if tid is not None:
                ids.append(tid)
            else:
                pair = self._char_to_pair.get(char)
                if pair:
                    ids.append(pair[0])   # LEAF first
                    ids.append(pair[1])   # ALT_XX second
        return ids

    def decode_ids(self, ids: list[int]) -> str:
        """Decode list of CTC token IDs → string."""
        result = []
        i = 0
        while i < len(ids):
            tid = ids[i]
            # Check if next token is ALT → this is a 2-token char
            if i + 1 < len(ids) and ids[i + 1] in self._alt_id_set:
                mapped = self._pair_to_char.get((tid, ids[i + 1]))
                if mapped:
                    result.append(mapped)
                    i += 2
                    continue
            result.append(self.decode(tid))
            i += 1
        return "".join(result)

    def __repr__(self) -> str:
        return (f"CJKCodec(vocab_size={self.vocab_size}, "
                f"single={len(self.tokens)}, alt_mapped={len(self._char_to_pair)})")


# 214 CJK radicals (the CJK unified codepoints corresponding to Kangxi radicals)
_CJK_RADICALS: list[tuple[int, int]] = [
    (ord(unicodedata.normalize("NFKC", chr(cp))),
     ord(unicodedata.normalize("NFKC", chr(cp))))
    for cp in range(0x2F00, 0x2FD6)
]

# CJK base chars: kana + ASCII + CJK punctuation + 214 radicals
_CJK_BASE = _build_char_list(
    _ASCII_COMMON,
    [(0x3000, 0x303F)],     # CJK Symbols and Punctuation
    [(0x3040, 0x309F)],     # Hiragana
    [(0x30A0, 0x30FF)],     # Katakana
    _CJK_RADICALS,          # 214 radicals
    _TYPOGRAPHIC_COMMON,
)


def _build_cjk_codec() -> CJKCodec:
    """Build CJKCodec from saved vocab list and visual similarity mapping.

    Both files are produced by scripts/cjk_visual_similarity.py:
      cjk_vocab.txt — the exact freq-ranked vocab (single-token chars)
      cjk_visual_mapping.tsv — ALT assignments for non-vocab chars
    """
    # Load exact vocab list (produced by the similarity script)
    vocab_path = _FUSIONS_DIR / "cjk_vocab.txt"
    base_set = set(_CJK_BASE)
    freq_chars: list[str] = []
    if vocab_path.exists():
        for line in vocab_path.read_text(encoding="utf-8").splitlines():
            char = line.strip()
            if char and char not in base_set:
                freq_chars.append(char)

    # Load visual similarity mapping
    mapping_path = _FUSIONS_DIR / "cjk_visual_mapping.tsv"
    alt_mapping: dict[str, tuple[int, str]] = {}
    if mapping_path.exists():
        for line in mapping_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 3:
                char, slot, match = parts[0], int(parts[1]), parts[2]
                alt_mapping[char] = (slot, match)

    return CJKCodec(_CJK_BASE, freq_chars, alt_mapping)


# Lazy-loaded
_cjk_codec: CJKCodec | None = None


def get_cjk_codec() -> CJKCodec:
    global _cjk_codec
    if _cjk_codec is None:
        _cjk_codec = _build_cjk_codec()
    return _cjk_codec
