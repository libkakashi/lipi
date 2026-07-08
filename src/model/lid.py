"""
LID-1 model head.

The script/group taxonomy that used to live here has moved to
`src.taxonomy` — a data-only module that data / encoding / training can
import without pulling torch in. This file only holds the nn.Module.
"""

import torch.nn as nn
from torch import Tensor

from src.taxonomy import NUM_GROUPS


class LIDCoarse(nn.Module):
    """LID-1: Coarse group classifier on backbone spatial features.

    Takes (B, C, H, W) from backbone, pools to (B, C), classifies
    into one of NUM_GROUPS script groups.
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
