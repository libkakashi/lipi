"""
RNN-T Prediction Network.

1-layer GRU, 128-dim. Provides language-specific autoregressive priors.
Fully swapped per language (not LoRA adapted) — only 0.36M params per language.

QUANTIZATION: Stays at FP8 (E4M3), NOT NVFP4.
GRU gating mechanisms (sigmoid gates) become too coarse at 4-bit.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PredictionNetwork(nn.Module):
    """RNN-T prediction network (language model component).

    Given the previously emitted token, produces a context vector that
    encodes what token is likely next. Combined with the encoder output
    in the Joint Network to produce the next token distribution.
    """

    def __init__(
        self,
        vocab_size: int = 171,
        embed_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)

    def forward(
        self, tokens: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            tokens: (B, U) — token IDs (previous tokens in the sequence).
                    For single-step inference: (B, 1).
            hidden: (1, B, hidden_dim) — GRU hidden state.
                    None for first step (will be zero-initialized).

        Returns:
            output: (B, U, hidden_dim) — prediction context vectors.
            hidden: (1, B, hidden_dim) — updated GRU hidden state.
        """
        embedded = self.embedding(tokens)  # (B, U, embed_dim)
        output, hidden = self.gru(embedded, hidden)  # (B, U, hidden_dim)
        return output, hidden
