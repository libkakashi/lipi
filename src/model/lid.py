"""
Script Identification (LID).

Single-stage routing based on visual character set similarity:

  LID-1 (after stem): 9 group classification.
    Routes to group-specific expert MLPs + group-specific CTC heads.
    Each group has a unified charset covering all scripts in it.

Groups (9):
  1. Latin + Cyrillic (~150 forms, ~3.5B speakers)
  2. Arabic (~120-150 with positional forms, ~500M)
  3. Hebrew (~30-40, ~9M)
  4. CJK (~5000-8000, ~1.5B)
  5. N+E Indian Brahmic (~400-500, Devanagari/Gurmukhi/Gujarati/Bengali/Odia, ~1B+)
  6. South Indian Brahmic (~250-300, Kannada/Telugu/Malayalam, ~200M)
  7. Tamil (~70-80, ~85M)
  8. SE Asian Brahmic (~270-320, Thai/Lao/Khmer/Burmese, ~150M)
  9. Emoji (~4000-5000, universal)
"""

import torch
import torch.nn as nn
from torch import Tensor


# --- Scripts (individual writing systems) ---

SCRIPTS = [
    # Group 1: Latin + Cyrillic
    "latin",       # 0
    "cyrillic",    # 1
    "greek",       # 2
    # Group 2: Arabic
    "arabic",      # 3
    # Group 3: Hebrew
    "hebrew",      # 4
    # Group 4: CJK
    "cjk",         # 5
    "korean",      # 6
    # Group 5: N+E Indian Brahmic
    "devanagari",  # 7
    "gurmukhi",    # 8
    "gujarati",    # 9
    "bengali",     # 10
    "odia",        # 11
    # Group 6: South Indian Brahmic
    "kannada",     # 12
    "telugu",      # 13
    "malayalam",   # 14
    # Group 7: Tamil
    "tamil",       # 15
    # Group 8: SE Asian Brahmic
    "thai",        # 16
    "lao",         # 17
    "khmer",       # 18
    "burmese",     # 19
    # Group 9: Emoji
    "emoji",       # 20
]

SCRIPT_TO_ID = {name: i for i, name in enumerate(SCRIPTS)}
NUM_SCRIPTS = len(SCRIPTS)


# --- Groups (9 families) ---

GROUPS = [
    "latin_cyrillic",    # 0  ~150 forms
    "arabic",            # 1  ~120-150
    "hebrew",            # 2  ~30-40
    "cjk",               # 3  ~5000-8000
    "ne_indic",          # 4  ~400-500 (Devanagari, Gurmukhi, Gujarati, Bengali, Odia)
    "south_indic",       # 5  ~250-300 (Kannada, Telugu, Malayalam)
    "tamil",             # 6  ~70-80
    "southeast_asian",   # 7  ~270-320 (Thai, Lao, Khmer, Burmese)
    "emoji",             # 8  ~4000-5000
]

GROUP_TO_ID = {name: i for i, name in enumerate(GROUPS)}
NUM_GROUPS = len(GROUPS)

# Map each script to its group
SCRIPT_TO_GROUP = {
    "latin": "latin_cyrillic",
    "cyrillic": "latin_cyrillic",
    "greek": "latin_cyrillic",
    "arabic": "arabic",
    "hebrew": "hebrew",
    "cjk": "cjk",
    "korean": "cjk",
    "devanagari": "ne_indic",
    "gurmukhi": "ne_indic",
    "gujarati": "ne_indic",
    "bengali": "ne_indic",
    "odia": "ne_indic",
    "kannada": "south_indic",
    "telugu": "south_indic",
    "malayalam": "south_indic",
    "tamil": "tamil",
    "thai": "southeast_asian",
    "lao": "southeast_asian",
    "khmer": "southeast_asian",
    "burmese": "southeast_asian",
    "emoji": "emoji",
}

# Which scripts belong to each group
GROUP_SCRIPTS = {
    "latin_cyrillic": ["latin", "cyrillic", "greek"],
    "arabic": ["arabic"],
    "hebrew": ["hebrew"],
    "cjk": ["cjk", "korean"],
    "ne_indic": ["devanagari", "gurmukhi", "gujarati", "bengali", "odia"],
    "south_indic": ["kannada", "telugu", "malayalam"],
    "tamil": ["tamil"],
    "southeast_asian": ["thai", "lao", "khmer", "burmese"],
    "emoji": ["emoji"],
}


def script_to_group_id(script: str) -> int:
    """Get coarse group ID for a script."""
    return GROUP_TO_ID[SCRIPT_TO_GROUP[script]]


class LIDCoarse(nn.Module):
    """LID-1: Coarse group classifier on stem features.

    Global average pool + MLP. Separates 6 visually distinct families.
    Hidden dim scales with input — enough capacity to disentangle script
    identity from the rich visual features in stem output.
    """

    def __init__(self, in_channels: int = 64, num_groups: int = NUM_GROUPS):
        super().__init__()
        hidden = in_channels
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_groups),
        )

    def forward(self, stem_features: Tensor) -> Tensor:
        """
        Args:
            stem_features: (B, C, H, W) from stem output.
        Returns:
            logits: (B, num_groups)
        """
        pooled = stem_features.mean(dim=[2, 3])
        return self.classifier(pooled)

    def predict(self, stem_features: Tensor) -> tuple[Tensor, Tensor]:
        """Predict group with confidence."""
        logits = self.forward(stem_features)
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
