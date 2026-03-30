"""
Hierarchical Script Identification (LID).

Two-stage routing for script-specific expert selection:

  LID-1 (after stem): Coarse group classification (6 groups).
    Visually maximally distinct families. Stem conv features handle this.
    Routes to group-specific Stage 1 expert MLPs.

  LID-2 (after Stage 1): Fine script classification (15 scripts).
    Distinguishes within-group scripts (e.g., Tamil vs Malayalam).
    SWA features capture full character shapes needed for this.
    Routes to script-specific Stage 2 expert MLPs + BiLSTM heads.
"""

import torch
import torch.nn as nn
from torch import Tensor


# --- Fine-grained scripts (15 classes) ---

SCRIPTS = [
    # latin_like
    "latin",       # 0  English, French, Spanish, German, etc.
    "cyrillic",    # 1  Russian, Ukrainian, etc.
    "greek",       # 2  Greek (also covers math symbols)
    # indic
    "devanagari",  # 3  Hindi, Marathi, Sanskrit, Nepali
    "bengali",     # 4  Bengali, Assamese
    "tamil",       # 5
    "telugu",      # 6
    "kannada",     # 7
    "malayalam",   # 8
    "gujarati",    # 9
    "gurmukhi",    # 10 Punjabi
    "odia",        # 11
    # arabic
    "arabic",      # 12 Arabic, Urdu, Persian, Hebrew
    # east_asian
    "cjk",         # 13 Chinese, Japanese Kanji
    "korean",      # 14 Hangul
    # southeast_asian
    "thai",        # 15
    # emoji
    "emoji",       # 16 Emoji, pictographs
]

SCRIPT_TO_ID = {name: i for i, name in enumerate(SCRIPTS)}
NUM_SCRIPTS = len(SCRIPTS)


# --- Coarse groups (6 families) ---

GROUPS = [
    "latin_like",       # 0  Latin, Cyrillic, Greek, math symbols
    "indic",            # 1  ALL Indian scripts (Brahmi-derived)
    "arabic",           # 2  Arabic, Hebrew, Perso-Arabic (RTL cursive)
    "east_asian",       # 3  CJK, Korean (dense strokes, boxy)
    "southeast_asian",  # 4  Thai, Lao, Khmer, Myanmar
    "emoji",            # 5  Emoji, pictographs (colorful blobs)
]

GROUP_TO_ID = {name: i for i, name in enumerate(GROUPS)}
NUM_GROUPS = len(GROUPS)

# Map each script to its coarse group
SCRIPT_TO_GROUP = {
    "latin": "latin_like",
    "cyrillic": "latin_like",
    "greek": "latin_like",
    "devanagari": "indic",
    "bengali": "indic",
    "tamil": "indic",
    "telugu": "indic",
    "kannada": "indic",
    "malayalam": "indic",
    "gujarati": "indic",
    "gurmukhi": "indic",
    "odia": "indic",
    "arabic": "arabic",
    "cjk": "east_asian",
    "korean": "east_asian",
    "thai": "southeast_asian",
    "emoji": "emoji",
}

# Which scripts belong to each group
GROUP_SCRIPTS = {
    "latin_like": ["latin", "cyrillic", "greek"],
    "indic": ["devanagari", "bengali", "tamil", "telugu", "kannada",
              "malayalam", "gujarati", "gurmukhi", "odia"],
    "arabic": ["arabic"],
    "east_asian": ["cjk", "korean"],
    "southeast_asian": ["thai"],
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
