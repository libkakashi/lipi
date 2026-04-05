"""
MoE training loss computation.

Separates loss logic from training loop for clarity and testability.
Each loss function takes explicit inputs and returns a scalar tensor.

Losses:
  - LID-1: group classification (cross-entropy)
  - LID-2: script classification within multi-script groups (cross-entropy)
  - CTC: sequence-level character recognition
  - Alignment CE: per-frame cross-entropy using CTC's forced alignment,
    giving partial credit for multi-token characters
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


def compute_alignment_ce_loss(
    logits: Tensor,
    targets: Tensor,
    enc_lengths: Tensor,
    tgt_lens: Tensor,
    ctc_ok: Tensor,
    true_group_ids: Tensor,
    true_script_ids: Tensor,
    group_script_vocabs: list[list[int]],
) -> Tensor:
    """Per-frame cross-entropy using CTC's forced alignment.

    For each sample, finds the best CTC alignment (argmax path), then
    computes cross-entropy at every non-blank frame against its aligned
    target token. This gives partial credit for multi-token characters:
    if 4/5 tokens are correct, 4 frames get low CE loss.

    Uses the same per-script vocab slicing as compute_ctc_loss.
    """
    device = logits.device
    ce_loss = torch.zeros(1, device=device)
    ce_frames = 0

    for g, script_vocabs in enumerate(group_script_vocabs):
        for s, vs in enumerate(script_vocabs):
            s_mask = ctc_ok & (true_group_ids == g) & (true_script_ids == s)
            s_logits = logits[s_mask]
            if s_logits.shape[0] == 0:
                continue

            s_targets = targets[s_mask]          # (N, U_max)
            s_tgt_lens = tgt_lens[s_mask]        # (N,)
            s_enc_lens = enc_lengths[s_mask]      # (N,)

            # Slice to exact vocab size
            s_logits_vs = s_logits[:, :, :vs].float()  # (N, T, vs)

            # Get best alignment: argmax over vocab at each frame
            best_path = s_logits_vs.argmax(dim=-1)  # (N, T)

            # For each sample, build frame-level targets from the alignment.
            # Non-blank frames get the corresponding target token.
            # Blank frames are ignored (masked out of the CE computation).
            n, t_max = best_path.shape

            # Expand targets into frame-level labels via the alignment
            frame_targets = torch.zeros(n, t_max, dtype=torch.long, device=device)
            frame_mask = torch.zeros(n, t_max, dtype=torch.bool, device=device)

            for i in range(n):
                path = best_path[i, :s_enc_lens[i]]
                tgt = s_targets[i, :s_tgt_lens[i]]

                # Walk the alignment: skip blanks and repeats to map
                # each non-blank frame to a target token position
                tgt_idx = 0
                prev = -1
                for f in range(path.shape[0]):
                    token_id = path[f].item()
                    if token_id == 0:  # blank
                        prev = 0
                        continue
                    if token_id == prev:  # repeat
                        # Same target as previous non-blank frame
                        if tgt_idx > 0:
                            frame_targets[i, f] = tgt[tgt_idx - 1]
                            frame_mask[i, f] = True
                        continue
                    # New non-blank token
                    prev = token_id
                    if tgt_idx < tgt.shape[0]:
                        frame_targets[i, f] = tgt[tgt_idx]
                        frame_mask[i, f] = True
                        tgt_idx += 1

            # Compute CE only on valid (non-blank, aligned) frames
            if frame_mask.any():
                masked_logits = s_logits_vs[frame_mask]    # (K, vs)
                masked_targets = frame_targets[frame_mask]  # (K,)
                ce_loss = ce_loss + F.cross_entropy(
                    masked_logits, masked_targets, reduction="sum")
                ce_frames += frame_mask.sum().item()

    if ce_frames > 0:
        ce_loss = torch.clamp(ce_loss / ce_frames, min=0.0, max=100.0)
    return ce_loss
