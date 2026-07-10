"""
Vocabulary loading for all scripts.

All scripts derive their vocab from config.py:
  - No-fusion scripts: NoFusionCodec (latin, cyrillic, greek, hebrew,
    armenian, georgian, ethiopic, kana)
  - Fusion scripts: FusionCodec (arabic, brahmic scripts, thai, lao, etc.)
  - Korean: KoreanCodec
  - Han: CJKCodec (kanji/hanzi, Chinese + Japanese kanji)
"""

from src.encoding.decompose import script_vocab_size


BLANK_TOKEN = "∅"  # ∅ — CTC blank, always token index 0


def build_script_vocab(script: str) -> list[str]:
    """Load the vocab for a script.

    Returns:
        [BLANK_TOKEN, token_1, token_2, ...] — token list.
    """
    from src.encoding.config import (
        NO_FUSION_SCRIPTS, FUSION_BASE_CHARS, get_fusion_codec,
        get_korean_codec, get_han_codec,
    )

    if script in NO_FUSION_SCRIPTS:
        return [BLANK_TOKEN] + NO_FUSION_SCRIPTS[script].chars
    if script in FUSION_BASE_CHARS:
        return [BLANK_TOKEN] + get_fusion_codec(script).tokens
    if script == "korean":
        return [BLANK_TOKEN] + get_korean_codec().tokens
    if script == "han":
        return [BLANK_TOKEN] + get_han_codec().tokens
    raise ValueError(f"Unknown script: {script}")
