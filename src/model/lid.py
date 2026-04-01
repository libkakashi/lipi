"""
Script Identification (LID).

Single-stage routing based on visual character set similarity:

  LID-1 (after shared SWA): 10 group classification.
    Routes to group-specific expert attention + expert MLP + group CTC heads.

Groups (10):
  1. Latin (~819 chars, 22 languages, ~3B speakers)
  2. Cyrillic + Greek (~1160 chars, ~300M speakers)
  3. Arabic (~720 chars incl. Persian/Urdu, ~500M)
  4. Hebrew (~293 chars, ~9M)
  5. Han + Kana (~21K chars, Chinese/Japanese, ~1.4B)
  6. Korean (~5.7K chars, Hangul, ~80M)
  7. N+E Indian Brahmic (~710 chars, Devanagari/Gurmukhi/Gujarati/Bengali, ~1B+)
  8. South Indian Brahmic (~536 chars, Kannada/Telugu/Malayalam/Tamil, ~285M)
  9. SE Asian (~533 chars, Thai/Lao, ~90M)
  10. Emoji (universal)
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
    # Group 3: Arabic (incl. Urdu, Persian extensions)
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
    # Group 8: South Indian Brahmic (incl. Tamil)
    "kannada",     # 11
    "telugu",      # 12
    "malayalam",   # 13
    "tamil",       # 14
    # Group 9: SE Asian Brahmic
    "thai",        # 15
    "lao",         # 16
    # Group 10: Emoji
    "emoji",       # 17
]

SCRIPT_TO_ID = {name: i for i, name in enumerate(SCRIPTS)}
NUM_SCRIPTS = len(SCRIPTS)


# --- Groups (9 families) ---

GROUPS = [
    "latin",             # 0  ~819 chars (22 languages)
    "cyrillic_greek",    # 1  ~668+492 chars
    "arabic",            # 2  ~720 chars
    "hebrew",            # 3  ~293 chars
    "han_kana",          # 4  ~21K chars (Chinese ideographs + Japanese kana)
    "korean",            # 5  ~5.7K chars (Hangul syllables)
    "ne_indic",          # 6  ~710 chars (Devanagari, Gurmukhi, Gujarati, Bengali)
    "south_indic",       # 7  ~536 chars (Kannada, Telugu, Malayalam, Tamil)
    "southeast_asian",   # 8  ~533 chars (Thai, Lao)
    "emoji",             # 9
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
    "han_kana": "han_kana",
    "korean": "korean",
    "devanagari": "ne_indic",
    "gurmukhi": "ne_indic",
    "gujarati": "ne_indic",
    "bengali": "ne_indic",
    "kannada": "south_indic",
    "telugu": "south_indic",
    "malayalam": "south_indic",
    "tamil": "south_indic",
    "thai": "southeast_asian",
    "lao": "southeast_asian",
    "emoji": "emoji",
}

GROUP_SCRIPTS = {
    "latin": ["latin"],
    "cyrillic_greek": ["cyrillic", "greek"],
    "arabic": ["arabic"],
    "hebrew": ["hebrew"],
    "han_kana": ["han_kana"],
    "korean": ["korean"],
    "ne_indic": ["devanagari", "gurmukhi", "gujarati", "bengali"],
    "south_indic": ["kannada", "telugu", "malayalam", "tamil"],
    "southeast_asian": ["thai", "lao"],
    "emoji": ["emoji"],
}


def script_to_group_id(script: str) -> int:
    """Get coarse group ID for a script."""
    return GROUP_TO_ID[SCRIPT_TO_GROUP[script]]


class LIDCoarse(nn.Module):
    """LID-1: Coarse group classifier with learned attention pooling.

    Instead of mean-pooling, learns which token positions are most
    informative for script identification. A single distinctive character
    (like Ж or ψ) can dominate the classification.
    """

    def __init__(self, in_channels: int = 288, num_groups: int = NUM_GROUPS):
        super().__init__()
        # Learned attention pooling: which tokens matter for classification?
        self.pool_attn = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Classifier MLP
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
        attn_scores = self.pool_attn(x)                    # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (B, T, 1)
        pooled = (x * attn_weights).sum(dim=1)             # (B, C)
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


class LIDFine(nn.Module):
    """LID-2: Fine script classifier on Stage 1 features.

    Global average pool + MLP. Distinguishes scripts within groups
    (e.g., Tamil vs Malayalam, Devanagari vs Bengali).

    MLP needed because Stage 1 output carries rich visual features —
    script identity is one signal among many. Hidden layer disentangles it.
    """

    def __init__(self, in_dim: int = 288, num_scripts: int = NUM_SCRIPTS):
        super().__init__()
        hidden = max(64, in_dim // 3)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_scripts),
        )

    def forward(self, stage1_features: Tensor) -> Tensor:
        """
        Args:
            stage1_features: (B, H*W, C) from Stage 1 output.
        Returns:
            logits: (B, num_scripts)
        """
        pooled = stage1_features.mean(dim=1)  # (B, C)
        return self.classifier(pooled)

    def predict(self, stage1_features: Tensor) -> tuple[Tensor, Tensor]:
        """Predict script with confidence."""
        logits = self.forward(stage1_features)
        probs = torch.softmax(logits, dim=-1)
        confidences, script_ids = probs.max(dim=-1)
        return script_ids, confidences


