"""
Renderable-character enumeration.

`get_renderable_chars(script)` returns the characters (or fusion clusters)
that the training pipeline can draw as standalone glyphs for that script.

Semantics per script:
  han_sparse/dense     — assigned CJK Unified + Extension A codepoints routed
                         by deterministic IDS complexity
  kana                 — hiragana + katakana + prolonged-sound-mark
  fusion scripts       — base chars + fusion clusters from the codec
  korean               — displayable tokens from the jamo codec
  no-fusion scripts    — codec.chars, minus whitespace and combining marks

Used to generate per-character training images and to sanity-check codecs
against font coverage. Lives in `encoding/` because it introspects codec
data — not a generic data-loader helper.
"""

import unicodedata

from src.encoding.config import (
    NO_FUSION_SCRIPTS, FUSION_BASE_CHARS, get_fusion_codec,
    get_korean_codec,
)
from src.encoding.han_split import HAN_SCRIPTS, han_script_for_char


def get_renderable_chars(script: str) -> list[str]:
    if script in HAN_SCRIPTS:
        chars = []
        for cp in range(0x3400, 0x4DC0):      # CJK Ext A
            c = chr(cp)
            if (unicodedata.category(c) != 'Cn'
                    and han_script_for_char(c) == script):
                chars.append(c)
        for cp in range(0x4E00, 0xA000):      # CJK Unified
            c = chr(cp)
            if (unicodedata.category(c) != 'Cn'
                    and han_script_for_char(c) == script):
                chars.append(c)
        return chars

    if script == "kana":
        chars = []
        for cp in range(0x3041, 0x3097):      # Hiragana
            chars.append(chr(cp))
        for cp in range(0x30A1, 0x30FB):      # Katakana
            chars.append(chr(cp))
        chars.append('ー')                # Prolonged sound mark
        return chars

    if script in FUSION_BASE_CHARS:
        codec = get_fusion_codec(script)
        # Base chars (single codepoints) + fusion clusters (multi-codepoint)
        return codec.base_chars + codec.fusions

    if script == "korean":
        codec = get_korean_codec()
        return [t for t in codec.tokens
                if t.strip() and ord(t[0]) > 32]

    if script in NO_FUSION_SCRIPTS:
        codec = NO_FUSION_SCRIPTS[script]
        return [c for c in codec.chars
                if c.strip() and ord(c) > 32
                and not unicodedata.category(c).startswith('M')]

    return []
