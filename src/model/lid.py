"""
Micro-LID Script Classifier.

Tiny CNN (~1MB) that classifies word crop images into script families.
Runs in ~0.2ms — negligible overhead.

Input:  (B, 3, 32, W) — same word crop as recognition model
Output: (B, num_scripts) — logits over script families

Supported script families (11 classes):
  0: Latin (English)
  1: Devanagari (Hindi, Marathi, Sanskrit, Nepali)
  2: Tamil
  3: Telugu
  4: Kannada
  5: Eastern Nagari (Bengali, Assamese)
  6: Odia
  7: Gujarati
  8: Gurmukhi (Punjabi)
  9: Malayalam
  10: Urdu (Nastaliq Perso-Arabic)
"""

import torch.nn as nn
from torch import Tensor


SCRIPT_NAMES = [
    "en",     # 0: Latin
    "hi",     # 1: Devanagari
    "ta",     # 2: Tamil
    "te",     # 3: Telugu
    "kn",     # 4: Kannada
    "bn_as",  # 5: Eastern Nagari
    "or",     # 6: Odia
    "gu",     # 7: Gujarati
    "pa",     # 8: Gurmukhi
    "ml",     # 9: Malayalam
    "ur",     # 10: Urdu
]

NUM_SCRIPTS = len(SCRIPT_NAMES)


class MicroLID(nn.Module):
    """Tiny CNN for script family classification.

    MobileNet-V4-Tiny inspired architecture. Uses depthwise separable
    convolutions for efficiency. AdaptiveAvgPool handles variable width.
    """

    def __init__(self, num_scripts: int = NUM_SCRIPTS):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 16, stride 2
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Block 2: depthwise separable 16 -> 32, stride 2
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 3: depthwise separable 32 -> 64, stride 2
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Global pool
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_scripts)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, 32, W) — word crop images.

        Returns:
            (B, num_scripts) — logits over script families.
        """
        x = self.features(x).flatten(1)
        return self.classifier(x)
