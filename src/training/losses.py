"""
MoE training loss computation.

Separates loss logic from training loop for clarity and testability.
Each loss function takes explicit inputs and returns a scalar tensor.

Losses:
  - LID-1: group classification (cross-entropy)
  - LID-2: script classification within multi-script groups (cross-entropy)
  - CTC: sequence-level character recognition
  - Regional token: spatial partial credit for multi-token characters
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
            # Concatenate targets (1-D form) so CTC allocates DP tables
            # proportional to actual target lengths, not the padded max.
            s_tgt_lens = tgt_lens[s_mask]
            s_targets = torch.cat([
                targets[i, :s_tgt_lens[j]]
                for j, i in enumerate(s_mask.nonzero(as_tuple=False).squeeze(1))
            ])
            s_ctc = F.ctc_loss(
                s_log_probs, s_targets,
                enc_lengths[s_mask], s_tgt_lens,
                blank=0, reduction="sum", zero_infinity=True,
            )
            ctc_loss = ctc_loss + s_ctc
            ctc_samples += tgt_lens[s_mask].sum()

    if ctc_samples > 0:
        ctc_loss = ctc_loss / ctc_samples
    return ctc_loss


def _regional_token_loss_single(
    logits: Tensor,
    targets: Tensor,
    tgt_len: int,
    enc_len: int,
) -> tuple[Tensor, int]:
    """Compute regional token loss for a single sample.

    Divides the encoder frames into regions (one per target token),
    max-pools each region's logits, and computes CE against the
    target token for that region.

    Args:
        logits: (T, vocab) — encoder output for this sample
        targets: (U_max,) — target token IDs (padded)
        tgt_len: number of valid target tokens
        enc_len: number of valid encoder frames

    Returns:
        (loss_sum, n_regions) — unnormalized loss and count
    """
    if tgt_len == 0 or enc_len == 0:
        return torch.zeros(1, device=logits.device), 0

    T = enc_len
    U = tgt_len
    loss = torch.zeros(1, device=logits.device)

    for u in range(U):
        # Assign each target token to a region of frames
        frame_start = u * T // U
        frame_end = (u + 1) * T // U
        if frame_end <= frame_start:
            frame_end = frame_start + 1
        frame_end = min(frame_end, T)

        # Max-pool this region: best logit per vocab token across frames
        # Shape: (region_len, vocab) → (vocab,)
        region_logits = logits[frame_start:frame_end]
        pooled = region_logits.max(dim=0).values  # (vocab,)

        # CE: this region should contain target token u
        target = targets[u].unsqueeze(0)  # (1,)
        loss = loss + F.cross_entropy(pooled.unsqueeze(0), target, reduction="sum")

    return loss, U


def compute_regional_token_loss(
    logits: Tensor,
    targets: Tensor,
    enc_lengths: Tensor,
    tgt_lens: Tensor,
    ctc_ok: Tensor,
    true_group_ids: Tensor,
    true_script_ids: Tensor,
    group_script_vocabs: list[list[int]],
) -> Tensor:
    """Regional token loss: spatial partial credit for multi-token characters.

    Divides encoder frames into regions (one per target token) and checks
    whether each region contains the correct token via max-pooled CE.

    Unlike CTC (all-or-nothing at sequence level), this gives partial credit:
    getting 3/5 tokens in the right regions = 60% credit. Complements CTC
    by providing per-token gradient signal.

    Unlike uniform-spread alignment CE, this uses max-pooling instead of
    forcing each frame to predict one token — compatible with CTC's blank
    output pattern.

    Same per-script vocab slicing as compute_ctc_loss.
    """
    device = logits.device
    total_loss = torch.zeros(1, device=device)
    total_regions = 0

    for g, script_vocabs in enumerate(group_script_vocabs):
        for s, vs in enumerate(script_vocabs):
            s_mask = ctc_ok & (true_group_ids == g) & (true_script_ids == s)
            s_logits = logits[s_mask]
            if s_logits.shape[0] == 0:
                continue

            s_logits = s_logits[:, :, :vs].float()
            s_targets = targets[s_mask]
            s_tgt_lens = tgt_lens[s_mask]
            s_enc_lens = enc_lengths[s_mask]

            for i in range(s_logits.shape[0]):
                loss_i, n_i = _regional_token_loss_single(
                    s_logits[i],
                    s_targets[i],
                    s_tgt_lens[i].item(),
                    s_enc_lens[i].item(),
                )
                total_loss = total_loss + loss_i
                total_regions += n_i

    if total_regions > 0:
        total_loss = torch.clamp(total_loss / total_regions, min=0.0, max=100.0)
    return total_loss
