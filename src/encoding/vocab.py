"""
Vocabulary loading for all scripts.

All scripts derive their vocab from config.py:
  - No-fusion scripts: NoFusionCodec
  - Fusion scripts: FusionCodec
  - Korean: KoreanCodec
  - CJK: CJKCodec
"""

from src.encoding.tokenizer import BLANK_TOKEN
from src.encoding.decompose import script_vocab_size


def build_script_vocab(script: str) -> list[str]:
    """Load the vocab for a script.

    Returns:
        [BLANK_TOKEN, token_1, token_2, ...] — token list.
    """
    from src.encoding.config import (
        NO_FUSION_SCRIPTS, FUSION_BASE_CHARS, get_fusion_codec,
        get_korean_codec, get_cjk_codec,
    )

    if script in NO_FUSION_SCRIPTS:
        return [BLANK_TOKEN] + NO_FUSION_SCRIPTS[script].chars
    if script in FUSION_BASE_CHARS:
        return [BLANK_TOKEN] + get_fusion_codec(script).tokens
    if script == "korean":
        return [BLANK_TOKEN] + get_korean_codec().tokens
    if script == "han_kana":
        return [BLANK_TOKEN] + get_cjk_codec().tokens
    raise ValueError(f"Unknown script: {script}")
