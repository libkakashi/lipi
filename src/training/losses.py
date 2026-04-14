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


def _ctc_loss_pure(
    log_probs: Tensor,  # (T, 1, V)
    targets: Tensor,    # (L,)
    blank: int = 0,
) -> Tensor:
    """Pure-PyTorch CTC loss (forward algorithm in log space).

    Used as a fallback when torch.nn.functional.ctc_loss isn't available
    on the device (e.g., MPS as of PyTorch 2.5).

    Returns scalar negative log probability of the target sequence.
    """
    T = log_probs.shape[0]
    L = targets.shape[0]
    NEG_INF = torch.finfo(log_probs.dtype).min

    # Extended target: [blank, t0, blank, t1, ..., t_{L-1}, blank]
    # Length: 2L + 1
    S = 2 * L + 1
    extended = torch.full((S,), blank, dtype=torch.long, device=targets.device)
    extended[1::2] = targets

    log_probs_2d = log_probs.squeeze(1)  # (T, V)

    # alpha[s] = log P(in state s at frame t)
    alpha = torch.full((S,), NEG_INF, device=log_probs.device, dtype=log_probs.dtype)
    # Init at t=0: can start from blank (s=0) or first symbol (s=1)
    alpha[0] = log_probs_2d[0, blank]
    if S > 1:
        alpha[1] = log_probs_2d[0, extended[1]]

    # Transition allowed from s-2 if extended[s] != blank and != extended[s-2]
    can_skip = torch.zeros(S, dtype=torch.bool, device=targets.device)
    if S > 2:
        can_skip[2:] = (extended[2:] != blank) & (extended[2:] != extended[:-2])

    for t in range(1, T):
        # alpha_new[s] = (alpha[s] + alpha[s-1] [+ alpha[s-2] if can_skip[s]]) * P(extended[s]|t)
        # Stack candidates: stay, advance, skip
        prev = alpha
        prev_shifted1 = torch.cat([
            torch.full((1,), NEG_INF, device=alpha.device, dtype=alpha.dtype),
            prev[:-1]
        ])
        prev_shifted2 = torch.cat([
            torch.full((2,), NEG_INF, device=alpha.device, dtype=alpha.dtype),
            prev[:-2]
        ])
        prev_shifted2 = torch.where(can_skip, prev_shifted2,
                                     torch.full_like(prev_shifted2, NEG_INF))

        candidates = torch.stack([prev, prev_shifted1, prev_shifted2], dim=0)
        alpha = torch.logsumexp(candidates, dim=0) + log_probs_2d[t, extended]

    # Final: end at last blank (S-1) or last symbol (S-2)
    if S >= 2:
        return -torch.logsumexp(torch.stack([alpha[-1], alpha[-2]]), dim=0)
    return -alpha[-1]


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

            # PyTorch CTC isn't on MPS — use pure-PyTorch fallback there
            if seg_log_probs.device.type == "mps":
                seg_ctc = _ctc_loss_pure(seg_log_probs, seg_targets, blank=0)
            else:
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


