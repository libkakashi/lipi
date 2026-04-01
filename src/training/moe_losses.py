"""
MoE training loss computation.

Separates loss logic from training loop for clarity and testability.
Each loss function takes explicit inputs and returns a scalar tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_lid1_loss(
    group_logits: Tensor,
    true_group_ids: Tensor,
    ce_loss_fn: nn.CrossEntropyLoss,
) -> Tensor:
    """LID-1 group classification loss. Computed on ALL samples."""
    return ce_loss_fn(group_logits, true_group_ids)


def compute_lid2_loss(
    script_logits_per_group: list[tuple],
    true_script_ids: Tensor,
    lid1_ok: Tensor,
    ce_loss_fn: nn.CrossEntropyLoss,
) -> Tensor:
    """LID-2 script classification loss.

    Computed on correctly LID-1-routed samples only (lid1_ok mask).
    Averaged across groups to prevent multi-group sum from dominating.
    """
    lid2_loss = torch.zeros(1, device=true_script_ids.device)
    lid2_count = 0
    for _g, script_logits, group_mask in script_logits_per_group:
        routed_ok = lid1_ok[group_mask]
        sl = script_logits[routed_ok]
        if sl.shape[0] > 0:
            lid2_loss = lid2_loss + ce_loss_fn(
                sl, true_script_ids[group_mask][routed_ok])
            lid2_count += 1
    if lid2_count > 0:
        lid2_loss = lid2_loss / lid2_count
    return lid2_loss


def compute_ctc_loss(
    logits: Tensor,
    targets: Tensor,
    enc_lengths: Tensor,
    tgt_lens: Tensor,
    ctc_ok: Tensor,
    true_group_ids: Tensor,
    true_script_ids: Tensor,
    group_script_vocabs: list[list[int]],
) -> Tensor:
    """Per-script CTC loss with exact vocab slicing.

    Each script's CTC head outputs its own vocab size. We slice logits
    to that exact size before log_softmax, preventing zero-padded positions
    from stealing softmax probability mass.

    Only computed on correctly-routed samples (both LID-1 and LID-2 correct).
    """
    device = logits.device
    ctc_loss = torch.zeros(1, device=device)
    ctc_samples = 0

    for g, script_vocabs in enumerate(group_script_vocabs):
        for s, vs in enumerate(script_vocabs):
            s_mask = ctc_ok & (true_group_ids == g) & (true_script_ids == s)
            s_logits = logits[s_mask]
            if s_logits.shape[0] == 0:
                continue
            # Slice to this script's exact vocab size before log_softmax
            s_log_probs = (s_logits[:, :, :vs].float()
                           .log_softmax(dim=-1).permute(1, 0, 2))
            s_ctc = F.ctc_loss(
                s_log_probs, targets[s_mask],
                enc_lengths[s_mask], tgt_lens[s_mask],
                blank=0, reduction="sum", zero_infinity=True,
            )
            ctc_loss = ctc_loss + s_ctc
            ctc_samples += tgt_lens[s_mask].sum()

    if ctc_samples > 0:
        ctc_loss = torch.clamp(ctc_loss / ctc_samples, min=0.0, max=100.0)
    return ctc_loss
