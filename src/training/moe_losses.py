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
        ctc_loss = ctc_loss / ctc_samples
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
    """Per-frame cross-entropy giving partial credit for multi-token chars.

    Spreads target tokens uniformly across encoder frames, then computes
    cross-entropy at each frame. This works from epoch 1 (unlike forced
    alignment which requires the model to already produce non-blank output).

    For a target of length U and encoder length T, each target token gets
    assigned to T/U consecutive frames. The model learns "around frame 10
    you should be producing token X" — a strong per-token learning signal
    that complements CTC's sequence-level loss.

    Processes samples in mini-chunks to avoid OOM from materializing the
    full (N*T, vocab) tensor. Uses the same per-script vocab slicing as
    compute_ctc_loss.
    """
    device = logits.device
    ce_loss = torch.zeros(1, device=device)
    ce_frames = 0
    chunk_size = 64  # Process this many samples at a time

    for g, script_vocabs in enumerate(group_script_vocabs):
        for s, vs in enumerate(script_vocabs):
            s_mask = ctc_ok & (true_group_ids == g) & (true_script_ids == s)
            s_logits = logits[s_mask]
            if s_logits.shape[0] == 0:
                continue

            s_targets = targets[s_mask]      # (N, U_max)
            s_tgt_lens = tgt_lens[s_mask]    # (N,)
            s_enc_lens = enc_lengths[s_mask]  # (N,)
            n = s_logits.shape[0]

            for chunk_start in range(0, n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n)
                c_logits = s_logits[chunk_start:chunk_end, :, :vs].float()
                c_targets = s_targets[chunk_start:chunk_end]
                c_tgt_lens = s_tgt_lens[chunk_start:chunk_end]
                c_enc_lens = s_enc_lens[chunk_start:chunk_end]
                cn, t_max, _ = c_logits.shape

                # Build frame-level targets via uniform spread (vectorized)
                # frame_idx[i, f] = f * U_i / T_i, clamped to [0, U_i-1]
                frames = torch.arange(t_max, device=device).unsqueeze(0)  # (1, T)
                u = c_tgt_lens.unsqueeze(1).float()  # (cn, 1)
                t = c_enc_lens.unsqueeze(1).float()   # (cn, 1)
                # Avoid division by zero
                t = t.clamp(min=1)
                tgt_indices = (frames * u / t).long().clamp(
                    min=0, max=c_targets.shape[1] - 1)  # (cn, T)

                # Gather target tokens for each frame
                frame_targets = c_targets.gather(1, tgt_indices)  # (cn, T)

                # Mask: only frames within encoder length
                frame_mask = frames < c_enc_lens.unsqueeze(1)  # (cn, T)

                if frame_mask.any():
                    masked_logits = c_logits[frame_mask]      # (K, vs)
                    masked_targets = frame_targets[frame_mask]  # (K,)
                    ce_loss = ce_loss + F.cross_entropy(
                        masked_logits, masked_targets, reduction="sum")
                    ce_frames += frame_mask.sum().item()

    if ce_frames > 0:
        ce_loss = torch.clamp(ce_loss / ce_frames, min=0.0, max=100.0)
    return ce_loss
