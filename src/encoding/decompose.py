"""
Character encoding/decoding for all scripts.

Unified interface:
  encode_text(text, script) → list[int]
  decode_ids(ids, script) → str
  script_vocab_size(script) → int

Script types:
  No-fusion (8): 1 token = 1 Unicode char
  Fusion (16): 1 token = 1 grapheme cluster (base + fusions)
  Korean: Jamo-based (2-jamo syllables + tails + frequent 3-jamo)
  CJK: freq-ranked chars + ALT visual similarity variants
"""

from __future__ import annotations

from src.encoding.config import (
    NO_FUSION_SCRIPTS, FUSION_BASE_CHARS, get_fusion_codec,
    get_korean_codec, get_cjk_codec,
)


def encode_text(text: str, script: str) -> list[int]:
    """Encode text → CTC token IDs.  Works for all script types."""
    if script in NO_FUSION_SCRIPTS:
        return NO_FUSION_SCRIPTS[script].encode_text(text)
    if script in FUSION_BASE_CHARS:
        return get_fusion_codec(script).encode_text(text)
    if script == "korean":
        return get_korean_codec().encode_text(text)
    if script == "han_kana":
        return get_cjk_codec().encode_text(text)
    return []


def decode_ids(ids: list[int], script: str) -> str:
    """Decode CTC token IDs → text.  Works for all script types."""
    if script in NO_FUSION_SCRIPTS:
        return NO_FUSION_SCRIPTS[script].decode_ids(ids)
    if script in FUSION_BASE_CHARS:
        return get_fusion_codec(script).decode_ids(ids)
    if script == "korean":
        return get_korean_codec().decode_ids(ids)
    if script == "han_kana":
        return get_cjk_codec().decode_ids(ids)
    return ""


def script_vocab_size(script: str) -> int:
    """Get vocab size for any script (including BLANK at 0)."""
    if script in NO_FUSION_SCRIPTS:
        return NO_FUSION_SCRIPTS[script].vocab_size
    if script in FUSION_BASE_CHARS:
        return get_fusion_codec(script).vocab_size
    if script == "korean":
        return get_korean_codec().vocab_size
    if script == "han_kana":
        return get_cjk_codec().vocab_size
    return 0
