"""
Script Identification (LID).

Single-stage routing based on visual character set similarity:

  LID-1 (after shared SWA): 13 group classification.
    Routes to group-specific expert attention + expert MLP + group CTC heads.

Groups (13):
  1. Latin (~800 chars, 36 languages, ~3B speakers)
  2. Cyrillic + Greek (~760 chars, ~300M speakers)
  3. Arabic (~490 chars incl. Persian/Urdu, ~500M)
  4. Hebrew (~190 chars, ~9M)
  5. Sino-Japanese (~2K decomposed tokens, Chinese/Japanese, ~1.4B)
  6. Korean (~370 decomposed tokens, Hangul, ~80M)
  7. N+E Indian Brahmic (~850 chars, Devanagari/Gurmukhi/Gujarati/Bengali/Odia, ~1B+)
  8. South Indian Brahmic (~750 chars, Kannada/Telugu/Malayalam/Tamil/Sinhala, ~300M)
  9. SE Asian (~770 chars, Thai/Lao/Burmese/Khmer, ~150M)
  10. Emoji (universal)
  11. Caucasus (~330 chars, Armenian/Georgian, ~10M)
  12. Ethiopic (~520 chars, Amharic, ~57M)
  13. Tibetan (~270 chars, ~6M)
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
    # Group 5: Han + Kana (Chinese/Japanese)
    "han_kana",    # 5
    # Group 6: Korean
    "korean",      # 6
    # Group 7: N+E Indian Brahmic
    "devanagari",  # 7
    "gurmukhi",    # 8
    "gujarati",    # 9
    "bengali",     # 10
    "odia",        # 11
    # Group 8: South Indian Brahmic
    "kannada",     # 12
    "telugu",      # 13
    "malayalam",   # 14
    "tamil",       # 15
    "sinhala",     # 16
    # Group 9: SE Asian
    "thai",        # 17
    "lao",         # 18
    "burmese",     # 19
    "khmer",       # 20
    # Group 10: Emoji
    "emoji",       # 21
    # Group 11: Caucasus
    "armenian",    # 22
    "georgian",    # 23
    # Group 12: Ethiopic
    "ethiopic",    # 24
    # Group 13: Tibetan
    "tibetan",     # 25
]

SCRIPT_TO_ID = {name: i for i, name in enumerate(SCRIPTS)}
NUM_SCRIPTS = len(SCRIPTS)


# --- Groups (13 families) ---

GROUPS = [
    "latin",             # 0
    "cyrillic_greek",    # 1
    "arabic",            # 2
    "hebrew",            # 3
    "sino_japanese",     # 4
    "korean",            # 5
    "ne_indic",          # 6
    "south_indic",       # 7
    "se_asian",          # 8
    "emoji",             # 9
    "caucasus",          # 10
    "ethiopic",          # 11
    "tibetan",           # 12
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
    "han_kana": "sino_japanese",
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
    "sino_japanese": ["han_kana"],
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
    """LID-1: Coarse group classifier with learned attention pooling.

    Learns which token positions are most informative for script
    identification. A single distinctive character (like Ж or ψ)
    can dominate the classification — concentrates gradient on
    informative positions rather than diluting across all T positions.
    """

    def __init__(self, in_channels: int = 288, num_groups: int = NUM_GROUPS):
        super().__init__()
        # Stride-4 pool reduces T from ~384 to ~96 before attention.
        # This strengthens the attention gradient 4× (softmax jacobian ∝ 1/T).
        self.pre_pool = nn.AdaptiveAvgPool1d(96)
        self.pool_attn = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.pool_attn[2].weight)
        nn.init.zeros_(self.pool_attn[2].bias)
        hidden = in_channels * 2
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_groups),
        )

    def forward_seq(self, x: Tensor) -> Tensor:
        """Classify from sequence features (B, T, C) — used by moe_encoder."""
        # Reduce sequence length for stronger attention gradient
        x_pooled = self.pre_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 96, C)
        attn_scores = self.pool_attn(x_pooled)              # (B, 96, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)    # (B, 96, 1)
        pooled = (x_pooled * attn_weights).sum(dim=1)       # (B, C)
        return self.classifier(pooled)

    def forward(self, features: Tensor) -> Tensor:
        """Classify from spatial features (B, C, H, W) — used by train_lid."""
        B, C, H, W = features.shape
        x = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return self.forward_seq(x)

    def predict(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Predict group with confidence."""
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=-1)
        confidences, group_ids = probs.max(dim=-1)
        return group_ids, confidences
