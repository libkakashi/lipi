"""
RNN-T Joint Network.

Combines encoder frame features with prediction network context.
Uses additive combination (not concatenation) — standard in RNN-T.

Per-language module (~0.61M params).
"""

import torch.nn as nn
from torch import Tensor


class JointNetwork(nn.Module):
    """RNN-T joint network.

    Takes one encoder frame and one prediction output, combines them,
    and produces a distribution over the vocabulary (including blank).

    The additive approach (project each to joint_dim, then add) is
    standard in RNN-T and works better than concatenation because it
    naturally separates the "what the image shows" signal from the
    "what the language model expects" signal.
    """

    def __init__(
        self,
        enc_dim: int = 384,
        pred_dim: int = 128,
        joint_dim: int = 256,
        vocab_size: int = 401,
    ):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Linear(joint_dim, vocab_size),
        )

    def forward(self, enc_out: Tensor, pred_out: Tensor) -> Tensor:
        """
        Args:
            enc_out:  (B, T, 1, enc_dim)  or (B, 1, 1, enc_dim) for single-frame
            pred_out: (B, 1, U, pred_dim) or (B, 1, U, pred_dim) for single-step

        For training, the full lattice:
            enc_out:  (B, T, 1, enc_dim)  — broadcast over U
            pred_out: (B, 1, U, pred_dim) — broadcast over T
            output:   (B, T, U, vocab_size)

        For greedy decode (single frame, single step):
            enc_out:  (B, 1, 1, enc_dim)
            pred_out: (B, 1, 1, pred_dim)
            output:   (B, 1, 1, vocab_size)

        Returns:
            logits: (B, T, U, vocab_size)
        """
        # Project to joint_dim and add (broadcasting handles T vs U)
        joint = self.enc_proj(enc_out) + self.pred_proj(pred_out)
        return self.output(joint)
