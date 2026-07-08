"""
MoE training loss computation.

Separates loss logic from training loop for clarity and testability.
Each loss function takes explicit inputs and returns a scalar tensor.

Provides:
  - compute_lid1_loss: per-frame group classification (cross-entropy)
  - compute_lid2_loss: per-frame script classification within multi-script groups
  - compute_ctc_loss_segments: per-segment CTC, batched by (group, script)
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.encoding.decompose import encode_text as _encode_text, script_vocab_size


# Cache encoded token sequences — segments repeat the same (text, script)
# pairs across batches (e.g. char-level data uses same 1-char texts), and
# encode_text is pure-Python with unicode normalization and codec lookups.
# Cache size 100K fits typical dedup factor of 10-100x.
@functools.lru_cache(maxsize=100_000)
def _encode_text_cached(text: str, script: str) -> tuple:
    return tuple(_encode_text(text, script))


def encode_text(text: str, script: str) -> list:
    return list(_encode_text_cached(text, script))


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


def compute_lid2_loss(
    lid2_logits_per_group: dict,
    frame_group_ids: Tensor,
    frame_script_ids: Tensor,
    label_smoothing: float = 0.1,
) -> Tensor:
    """LID-2 per-frame script cross-entropy within multi-script groups.

    Averages cross-entropy across the multi-script groups that have any
    frames present in the batch. Returns a zero scalar if none do.

    lid2_logits_per_group: {group_id_str: (B, T, num_scripts_in_group)}.
    frame_group_ids / frame_script_ids: (B, T) — pre-truncated to match T.
    """
    device = frame_group_ids.device
    total = torch.zeros(1, device=device)
    n = 0
    for g_str, lid2_logits in lid2_logits_per_group.items():
        g = int(g_str)
        mask = (frame_group_ids == g)
        if not mask.any():
            continue
        total = total + F.cross_entropy(
            lid2_logits[mask], frame_script_ids[mask],
            label_smoothing=label_smoothing)
        n += 1
    if n > 0:
        total = total / n
    return total


def compute_ctc_loss_segments(
    logits: Tensor,
    segments_batch: list[list[dict]],
    enc_lengths: Tensor,
    group_script_names: list[list[str]],
    group_script_vocabs: list[list[int]],
) -> Tensor:
    """Per-segment CTC loss, batched by (group, script) for speed.

    Each image has segments: [{group_id, script_id, text, width, offset}, ...]
    Each segment's text is encoded with its script, and CTC loss is computed
    on the corresponding frame slice using the correct group's vocab.

    Segments with the same (group, script) are padded to max length and
    processed in a single F.ctc_loss call instead of one call per segment.
    """
    device = logits.device
    B, T, _ = logits.shape

    # Group valid segments by (group, script) for batched CTC
    buckets: dict[tuple[int, int], list[dict]] = {}
    skipped_no_script = 0
    skipped_no_ids = 0
    skipped_too_long = 0
    total_segs = 0

    for b in range(B):
        for seg in segments_batch[b]:
            total_segs += 1
            text = seg["text"]
            g = seg["group_id"]
            s = seg["script_id"]
            offset_px = seg["offset"]
            width_px = seg["width"]

            if not text or width_px == 0:
                continue

            frame_start = offset_px // 2
            frame_end = min((offset_px + width_px + 1) // 2, T)
            seg_len = frame_end - frame_start
            if seg_len < 1:
                continue

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
            n_repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
            if seg_len < len(ids) + n_repeats:
                skipped_too_long += 1
                continue
            vs = group_script_vocabs[g][s] if g < len(group_script_vocabs) and s < len(group_script_vocabs[g]) else 0
            if vs == 0:
                continue

            buckets.setdefault((g, s), []).append({
                "b": b, "frame_start": frame_start, "frame_end": frame_end,
                "seg_len": seg_len, "ids": ids, "vs": vs,
            })

    ctc_loss = torch.zeros(1, device=device)
    ctc_chars = 0

    # Cap per-call tensor size to limit peak memory for big-vocab buckets
    # (e.g. han vs~3800 with many segments at long max_T).
    MAX_BUCKET_ELEMS = 32 * 1024 * 1024  # 32M fp32 = 128MB per padded tensor

    def _run_chunk(chunk_segs, max_T, vs):
        """Run batched CTC on one chunk, return (loss, n_chars)."""
        N = len(chunk_segs)
        batched_logits = torch.zeros(max_T, N, vs, device=device, dtype=logits.dtype)
        input_lens = torch.zeros(N, dtype=torch.long, device=device)
        target_lens = torch.zeros(N, dtype=torch.long, device=device)
        concat_targets = []
        for i, sg in enumerate(chunk_segs):
            b, fs, fe = sg["b"], sg["frame_start"], sg["frame_end"]
            batched_logits[:sg["seg_len"], i, :] = logits[b, fs:fe, :vs]
            input_lens[i] = sg["seg_len"]
            target_lens[i] = len(sg["ids"])
            concat_targets.extend(sg["ids"])

        log_probs = batched_logits.float().log_softmax(dim=-1)
        targets = torch.tensor(concat_targets, dtype=torch.long, device=device)

        if device.type == "mps":
            offset = 0
            loss = torch.zeros(1, device=device)
            for i in range(N):
                U = target_lens[i].item()
                T_i = input_lens[i].item()
                lp_i = log_probs[:T_i, i:i+1, :]
                tgt_i = targets[offset:offset + U]
                offset += U
                loss = loss + _ctc_loss_pure(lp_i, tgt_i, blank=0)
            return loss, len(concat_targets)
        else:
            loss = F.ctc_loss(log_probs, targets, input_lens, target_lens,
                              blank=0, reduction="sum", zero_infinity=True)
            return loss, len(concat_targets)

    for (g, s), segs in buckets.items():
        vs = segs[0]["vs"]
        # Sort by seg_len so chunks have similar padding waste
        segs = sorted(segs, key=lambda sg: sg["seg_len"])
        chunk: list[dict] = []
        chunk_max_T = 0
        for sg in segs:
            new_max_T = max(chunk_max_T, sg["seg_len"])
            new_elems = new_max_T * (len(chunk) + 1) * vs
            if chunk and new_elems > MAX_BUCKET_ELEMS:
                loss, n = _run_chunk(chunk, chunk_max_T, vs)
                ctc_loss = ctc_loss + loss
                ctc_chars += n
                chunk = []
                chunk_max_T = 0
            chunk.append(sg)
            chunk_max_T = max(chunk_max_T, sg["seg_len"])
        if chunk:
            loss, n = _run_chunk(chunk, chunk_max_T, vs)
            ctc_loss = ctc_loss + loss
            ctc_chars += n

    skip_total = skipped_no_script + skipped_no_ids + skipped_too_long
    if skip_total > total_segs * 0.05:
        print(f"    [CTC segments] {ctc_chars} chars from {total_segs} segs | "
              f"skipped: {skipped_no_script} no_script, {skipped_no_ids} no_ids, "
              f"{skipped_too_long} too_long", flush=True)

    if ctc_chars > 0:
        ctc_loss = ctc_loss / ctc_chars
    return ctc_loss


