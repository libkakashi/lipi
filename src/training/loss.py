"""
Training loss functions.

RNN-T Loss: Primary training objective for Phase 2 adapter training.
CTC Loss: Used in Phase 1 foundation training (simpler, faster convergence).
Distillation Loss: KL divergence for soft targets from Qwen3-VL (Phase 2 supplement).
"""

import warnings
warnings.filterwarnings("ignore", message=".*rnnt_loss.*deprecated.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchaudio


def rnnt_loss(
    logits: Tensor,
    targets: Tensor,
    logit_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
) -> Tensor:
    """RNN-T loss wrapper.

    Uses torchaudio.functional.rnnt_loss:
      - CUDA-optimized forward-backward algorithm
      - fused_log_softmax=True for numerical stability

    torchaudio 2.11: deprecation was reversed, function is preserved.

    Args:
        logits: (B, T, U+1, vocab_size) — full lattice from joint network.
        targets: (B, U_max) — target token IDs (no blank).
        logit_lengths: (B,) — encoder output lengths.
        target_lengths: (B,) — actual target lengths (no padding).
        blank: Blank token ID (default 0).

    Returns:
        Scalar loss (mean over batch).
    """
    # Try warp-rnnt first (CUDA-native, much faster)
    # Fall back to torchaudio (CPU kernel, slower)
    # Use warp_rnnt (GPU CUDA kernel) if available, else torchaudio (CPU)
    try:
        import warp_rnnt
        log_probs = logits.log_softmax(-1)
        return warp_rnnt.rnnt_loss(
            log_probs=log_probs,
            labels=targets.int().contiguous(),
            frames_lengths=logit_lengths.int().contiguous(),
            labels_lengths=target_lengths.int().contiguous(),
            blank=blank,
            reduction="mean",
        )
    except ImportError:
        return torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=targets.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=blank,
            reduction="mean",
            fused_log_softmax=True,
        )


def ctc_loss(
    logits: Tensor,
    targets: Tensor,
    logit_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = 0,
) -> Tensor:
    """CTC loss for Phase 1 foundation training.

    Args:
        logits: (B, T, vocab_size) — encoder output projected to vocab.
        targets: (B, U_max) — target token IDs (padded).
        logit_lengths: (B,) — valid encoder output lengths.
        target_lengths: (B,) — actual target lengths.
        blank: Blank token ID (default 0).

    Returns:
        Scalar loss (mean over batch).
    """
    # CTC expects (T, B, vocab_size) log probabilities
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
    return F.ctc_loss(
        log_probs,
        targets,
        logit_lengths,
        target_lengths,
        blank=blank,
        reduction="mean",
        zero_infinity=True,
    )


class DistillationLoss(nn.Module):
    """KL divergence loss for knowledge distillation from VLM teacher.

    Only applied to real scanned crops where teacher confidence > gate threshold.
    Not applied to PDF-extracted or synthetic data (those have perfect labels).
    """

    def __init__(self, temperature: float = 2.0, confidence_gate: float = 0.9):
        super().__init__()
        self.temperature = temperature
        self.confidence_gate = confidence_gate

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        teacher_confidence: Tensor,
    ) -> Tensor:
        """Compute gated KL divergence loss.

        Args:
            student_logits: (B, vocab_size) student output logits.
            teacher_logits: (B, vocab_size) teacher soft targets.
            teacher_confidence: (B,) teacher confidence scores.

        Returns:
            Scalar loss (mean over confident samples only).
        """
        # Gate: only use samples where teacher is confident
        mask = teacher_confidence >= self.confidence_gate

        if not mask.any():
            return torch.tensor(0.0, device=student_logits.device)

        # Temperature-scaled softmax
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits[mask] / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits[mask] / T, dim=-1)

        # KL divergence (scaled by T^2 per Hinton et al.)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return kl * (T * T)
