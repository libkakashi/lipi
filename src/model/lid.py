"""
Script Identification (LID).

Single-stage routing based on visual character set similarity:

  LID-1 (after shared SWA): 14 group classification.
    Routes to group-specific expert attention + expert MLP + group CTC heads.

Groups (14):
  1. Latin (~800 chars, 36 languages, ~3B speakers)
  2. Cyrillic + Greek (~760 chars, ~300M speakers)
  3. Arabic (~490 chars incl. Persian/Urdu, ~500M)
  4. Hebrew (~190 chars, ~9M)
  5. Han (kanji/hanzi, ~2.8K encoding tokens, Chinese/Japanese kanji, ~1.4B)
  6. Kana (hiragana + katakana, ~220 tokens, Japanese syllabaries, ~125M)
  7. Korean (~370 decomposed tokens, Hangul, ~80M)
  8. N+E Indian Brahmic (~850 chars, Devanagari/Gurmukhi/Gujarati/Bengali/Odia, ~1B+)
  9. South Indian Brahmic (~750 chars, Kannada/Telugu/Malayalam/Tamil/Sinhala, ~300M)
  10. SE Asian (~770 chars, Thai/Lao/Burmese/Khmer, ~150M)
  11. Emoji (universal)
  12. Caucasus (~330 chars, Armenian/Georgian, ~10M)
  13. Ethiopic (~520 chars, Amharic, ~57M)
  14. Tibetan (~270 chars, ~6M)
"""

import torch
import torch.nn as nn
from torch import Tensor


# --- Scripts (individual writing systems) ---

SCRIPTS = [
    # Group 1: Latin
    "latin",       # 0
    # Group 2: Cyrillic + Greek
    "cyrillic",    # 1
    "greek",       # 2
    # Group 3: Arabic (incl. Urdu, Persian)
    "arabic",      # 3
    # Group 4: Hebrew
    "hebrew",      # 4
    # Group 5: Han (kanji/hanzi, Chinese + Japanese)
    "han",         # 5
    # Group 6: Kana (hiragana + katakana, Japanese syllabaries)
    "kana",        # 6
    # Group 7: Korean
    "korean",      # 7
    # Group 8: N+E Indian Brahmic
    "devanagari",  # 8
    "gurmukhi",    # 9
    "gujarati",    # 10
    "bengali",     # 11
    "odia",        # 12
    # Group 9: South Indian Brahmic
    "kannada",     # 13
    "telugu",      # 14
    "malayalam",   # 15
    "tamil",       # 16
    "sinhala",     # 17
    # Group 10: SE Asian
    "thai",        # 18
    "lao",         # 19
    "burmese",     # 20
    "khmer",       # 21
    # Group 11: Emoji
    "emoji",       # 22
    # Group 12: Caucasus
    "armenian",    # 23
    "georgian",    # 24
    # Group 13: Ethiopic
    "ethiopic",    # 25
    # Group 14: Tibetan
    "tibetan",     # 26
]

SCRIPT_TO_ID = {name: i for i, name in enumerate(SCRIPTS)}
NUM_SCRIPTS = len(SCRIPTS)


# --- Groups (13 families) ---

GROUPS = [
    "latin",             # 0
    "cyrillic_greek",    # 1
    "arabic",            # 2
    "hebrew",            # 3
    "han",               # 4  (was sino_japanese; kana split off)
    "kana",              # 5  (NEW: hiragana + katakana)
    "korean",            # 6
    "ne_indic",          # 7
    "south_indic",       # 8
    "se_asian",          # 9
    "emoji",             # 10
    "caucasus",          # 11
    "ethiopic",          # 12
    "tibetan",           # 13
]

GROUP_TO_ID = {name: i for i, name in enumerate(GROUPS)}
NUM_GROUPS = len(GROUPS)

# Map each script to its group
SCRIPT_TO_GROUP = {
    "latin": "latin",
    "cyrillic": "cyrillic_greek",
    "greek": "cyrillic_greek",
    "arabic": "arabic",
    "hebrew": "hebrew",
    "han": "han",
    "kana": "kana",
    "korean": "korean",
    "devanagari": "ne_indic",
    "gurmukhi": "ne_indic",
    "gujarati": "ne_indic",
    "bengali": "ne_indic",
    "odia": "ne_indic",
    "kannada": "south_indic",
    "telugu": "south_indic",
    "malayalam": "south_indic",
    "tamil": "south_indic",
    "sinhala": "south_indic",
    "thai": "se_asian",
    "lao": "se_asian",
    "burmese": "se_asian",
    "khmer": "se_asian",
    "emoji": "emoji",
    "armenian": "caucasus",
    "georgian": "caucasus",
    "ethiopic": "ethiopic",
    "tibetan": "tibetan",
}

GROUP_SCRIPTS = {
    "latin": ["latin"],
    "cyrillic_greek": ["cyrillic", "greek"],
    "arabic": ["arabic"],
    "hebrew": ["hebrew"],
    "han": ["han"],
    "kana": ["kana"],
    "korean": ["korean"],
    "ne_indic": ["devanagari", "gurmukhi", "gujarati", "bengali", "odia"],
    "south_indic": ["kannada", "telugu", "malayalam", "tamil", "sinhala"],
    "se_asian": ["thai", "lao", "burmese", "khmer"],
    "emoji": ["emoji"],
    "caucasus": ["armenian", "georgian"],
    "ethiopic": ["ethiopic"],
    "tibetan": ["tibetan"],
}


class LIDCoarse(nn.Module):
    """LID-1: Coarse group classifier on backbone spatial features.

    Takes (B, C, H, W) from backbone, pools to (B, C), classifies
    into one of 14 script groups.
    """

    def __init__(self, in_channels: int, num_groups: int = NUM_GROUPS):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_groups),
        )

    def forward(self, features: Tensor) -> Tensor:
        """(B, C, H, W) → (B, num_groups)"""
        return self.classifier(self.pool(features).flatten(1))
