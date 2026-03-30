"""
Full RNN-T Model: Encoder + Prediction Network + Joint Network.

Assembles all components into a single module for training.
For inference, the components are used separately (encoder runs once,
then prediction + joint run in a decode loop).
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork


class LipiRNNT(nn.Module):
    """Full Lipi RNN-T model for training.

    Computes the full (B, T, U, vocab_size) logit lattice needed
    for the RNN-T loss function.

    For inference, use the encoder, prediction_net, and joint_net
    separately with greedy_decode or beam_search.
    """

    def __init__(
        self,
        encoder: LipiEncoder | None = None,
        vocab_size: int = 171,
        pred_embed_dim: int = 128,
        pred_hidden_dim: int = 128,
        joint_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else LipiEncoder()
        self.prediction_net = PredictionNetwork(
            vocab_size=vocab_size,
            embed_dim=pred_embed_dim,
            hidden_dim=pred_hidden_dim,
        )
        self.joint_net = JointNetwork(
            enc_dim=self.encoder.output_dim,
            pred_dim=pred_hidden_dim,
            joint_dim=joint_dim,
            vocab_size=vocab_size,
        )
        self.vocab_size = vocab_size

    def forward(
        self,
        images: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            images: (B, 3, 32, W) — batch of word crop images.
            targets: (B, U_max) — padded target token sequences.
                     Token 0 is blank (used as BOS for prediction net).

        Returns:
            logits: (B, T, U+1, vocab_size) — full RNN-T lattice.
            enc_lengths: (B,) — encoder output lengths (T per sample).
            target_lengths: not computed here — caller must track.
        """
        # Encode images
        enc_out, enc_lengths = self.encoder(images)  # (B, T, enc_dim)

        # Prepend blank token to targets for prediction network input
        # targets: (B, U) -> pred_input: (B, U+1) starting with blank
        B = targets.shape[0]
        blank = torch.zeros(B, 1, dtype=targets.dtype, device=targets.device)
        pred_input = torch.cat([blank, targets], dim=1)  # (B, U+1)

        # Run prediction network on full target sequence
        pred_out, _ = self.prediction_net(pred_input)  # (B, U+1, pred_dim)

        # Expand dims for broadcasting in joint network
        # enc_out:  (B, T, enc_dim) -> (B, T, 1, enc_dim)
        # pred_out: (B, U+1, pred_dim) -> (B, 1, U+1, pred_dim)
        enc_expanded = enc_out.unsqueeze(2)
        pred_expanded = pred_out.unsqueeze(1)

        # Joint network: produces full lattice
        logits = self.joint_net(enc_expanded, pred_expanded)  # (B, T, U+1, vocab)

        return logits, enc_lengths
