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

from src.encoding.decompose import encode_text, script_vocab_size


def compute_lid1_loss(
    group_logits: Tensor,
    true_group_ids: Tensor,
    ce_loss_fn: nn.CrossEntropyLoss,
) -> Tensor:
    """LID-1 per-frame group cross-entropy.

    group_logits: (B, T, num_groups) — per-frame predictions.
    true_group_ids: (B,) broadcast to all frames, or (B, T) per-frame.
    """
    B, T, G = group_logits.shape
    if true_group_ids.dim() == 1:
        frame_labels = true_group_ids.unsqueeze(1).expand(B, T)
    else:
        frame_labels = true_group_ids
    return ce_loss_fn(group_logits.reshape(B * T, G), frame_labels.reshape(B * T))


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
            # 1-D concatenated targets so CTC allocates DP tables
            # proportional to actual target lengths, not the padded max.
            s_tgt_lens = tgt_lens[s_mask]
            s_targets_2d = targets[s_mask]
            col_idx = torch.arange(s_targets_2d.shape[1], device=device)
            s_targets = s_targets_2d[col_idx < s_tgt_lens.unsqueeze(1)]
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


def compute_ctc_loss_segments(
    logits: Tensor,
    segments_batch: list[list[dict]],
    enc_lengths: Tensor,
    group_script_names: list[list[str]],
    group_script_vocabs: list[list[int]],
) -> Tensor:
    """Per-segment CTC loss for mixed-script support.

    Each image has segments: [{group_id, script_id, text, width, offset}, ...]
    Each segment's text is encoded with its script, and CTC loss is computed
    on the corresponding frame slice using the correct group's vocab.
    """
    device = logits.device
    B, T, _ = logits.shape
    ctc_loss = torch.zeros(1, device=device)
    ctc_chars = 0

    skipped_no_script = 0
    skipped_no_ids = 0
    skipped_too_long = 0
    total_segs = 0

    for b in range(B):
        segs = segments_batch[b]

        for seg in segs:
            total_segs += 1
            text = seg["text"]
            g = seg["group_id"]
            s = seg["script_id"]
            offset_px = seg["offset"]
            width_px = seg["width"]

            if not text or width_px == 0:
                continue

            # Pixel range → frame range (W → W/2 downsampling)
            frame_start = offset_px // 2
            frame_end = min((offset_px + width_px + 1) // 2, T)
            seg_len = frame_end - frame_start
            if seg_len < 1:
                continue

            # Encode text with correct script
            if g >= len(group_script_names) or s >= len(group_script_names[g]):
                skipped_no_script += 1
                continue
            script_name = group_script_names[g][s]
            if not script_name:
                skipped_no_script += 1
                continue
            ids = encode_text(text, script_name)
            if not ids:
                skipped_no_ids += 1
                continue
            if len(ids) > seg_len:
                skipped_too_long += 1
                continue

            vs = group_script_vocabs[g][s] if g < len(group_script_vocabs) and s < len(group_script_vocabs[g]) else 0
            if vs == 0:
                continue

            # CTC on this segment's frames
            seg_logits = logits[b, frame_start:frame_end, :vs]  # (seg_len, vs)
            seg_log_probs = seg_logits.float().log_softmax(dim=-1).unsqueeze(1)  # (seg_len, 1, vs)
            seg_targets = torch.tensor(ids, dtype=torch.long, device=device)
            seg_input_len = torch.tensor([seg_len], dtype=torch.long, device=device)
            seg_target_len = torch.tensor([len(ids)], dtype=torch.long, device=device)

            seg_ctc = F.ctc_loss(
                seg_log_probs, seg_targets, seg_input_len, seg_target_len,
                blank=0, reduction="sum", zero_infinity=True)
            ctc_loss = ctc_loss + seg_ctc
            ctc_chars += len(ids)

    skip_total = skipped_no_script + skipped_no_ids + skipped_too_long
    if skip_total > total_segs * 0.05:  # only warn if >5% skipped
        print(f"    [CTC segments] {ctc_chars} chars from {total_segs} segs | "
              f"skipped: {skipped_no_script} no_script, {skipped_no_ids} no_ids, "
              f"{skipped_too_long} too_long", flush=True)

    if ctc_chars > 0:
        ctc_loss = ctc_loss / ctc_chars
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
