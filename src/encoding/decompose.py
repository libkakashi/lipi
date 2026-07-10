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
    get_korean_codec, get_han_codec,
)


def _codec_for(script: str):
    """Return the codec for a script, or None if unknown.

    All codecs share one interface: encode_text(text) -> list[int],
    decode_ids(ids) -> str, and a vocab_size attribute (BLANK at index 0).
    """
    if script in NO_FUSION_SCRIPTS:
        return NO_FUSION_SCRIPTS[script]
    if script in FUSION_BASE_CHARS:
        return get_fusion_codec(script)
    if script == "korean":
        return get_korean_codec()
    if script == "han":
        return get_han_codec()
    return None


def encode_text(text: str, script: str) -> list[int]:
    """Encode text → CTC token IDs. Works for all script types."""
    codec = _codec_for(script)
    return codec.encode_text(text) if codec is not None else []


def decode_ids(ids: list[int], script: str) -> str:
    """Decode CTC token IDs → text. Works for all script types."""
    codec = _codec_for(script)
    return codec.decode_ids(ids) if codec is not None else ""


def script_vocab_size(script: str) -> int:
    """Vocab size for any script (includes BLANK at index 0)."""
    codec = _codec_for(script)
    return codec.vocab_size if codec is not None else 0
