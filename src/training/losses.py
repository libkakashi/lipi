"""
MoE training loss computation.

Separates loss logic from training loop for clarity and testability.
Each loss function takes explicit inputs and returns a scalar tensor.

Provides:
  - compute_lid1_loss: per-frame group classification (cross-entropy)
  - compute_lid2_loss: per-frame script classification within multi-script groups
  - compute_ctc_loss_segments: per-segment CTC, batched by (group, script)
  - compute_consistency_loss: symmetric KL between two augmented views
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.encoding.decompose import encode_text as _encode_text, script_vocab_size
from src.encoding.direction import segment_is_rtl


# Cache encoded token sequences — segments repeat the same (text, script)
# pairs across batches (e.g. char-level data uses same 1-char texts), and
# encode_text is pure-Python with unicode normalization and codec lookups.
# Cache size 100K fits typical dedup factor of 10-100x. Returns a tuple
# so the lru_cache can hold it; callers do list(...) if they need mutation.
@functools.lru_cache(maxsize=100_000)
def _encode_text_cached(text: str, script: str) -> tuple:
    return tuple(_encode_text(text, script))


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


def _sym_kl(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    """Symmetric KL between two batches of logits over the last dim."""
    lp_a = F.log_softmax(logits_a.float(), dim=-1)
    lp_b = F.log_softmax(logits_b.float(), dim=-1)
    return 0.5 * (
        F.kl_div(lp_a, lp_b, log_target=True, reduction="batchmean")
        + F.kl_div(lp_b, lp_a, log_target=True, reduction="batchmean"))


def compute_consistency_loss(
    logits1: Tensor | None,     # (B, T, max_vocab) view 1, or None to skip CTC term
    logits2: Tensor | None,     # (B, T, max_vocab) view 2
    group_logits1: Tensor,      # (B, T, num_groups+1) view 1
    group_logits2: Tensor,      # (B, T, num_groups+1) view 2
    flat_scripts: Tensor,       # (B, T) GT flat script routing (-1 blank)
    gl_frames: Tensor,          # (B, T) LID-1 CE targets (-100 padding)
    aligned: Tensor,            # (B,) bool — views geometrically aligned
    group_script_vocabs: list[list[int]],
) -> Tensor:
    """Two-view consistency: symmetric KL between augmented views.

    Optimizes the robustness objective directly — the same clean render
    under two different degradations must produce the same posteriors,
    not merely the correct label on each. Applied to per-frame LID-1
    distributions (non-padding frames) and per-frame CTC distributions
    (segment frames, per-script vocab slice). Samples whose view-1
    x-geometry was shifted by augmentation (aligned=False) are skipped —
    their frames don't correspond.

    Averages the LID term and per-script CTC terms.
    """
    device = group_logits1.device
    total = torch.zeros(1, device=device)
    n_terms = 0

    a_mask = aligned[:, None]  # (B, 1) broadcast over T

    # LID-1 consistency over real (non-padding) frames
    T = group_logits1.shape[1]
    lid_mask = (gl_frames[:, :T] >= 0) & a_mask
    if lid_mask.any():
        total = total + _sym_kl(group_logits1[lid_mask],
                                group_logits2[lid_mask])
        n_terms += 1

    # CTC consistency per present script, over that script's vocab slice
    # (logits are zero-padded past each script's vocab — comparing the
    # full max_vocab width would let the padding dilute the KL).
    if logits1 is not None and logits2 is not None:
        flat_vocabs = [vs for vs_list in group_script_vocabs for vs in vs_list]
        fs = torch.where(a_mask, flat_scripts, torch.full_like(flat_scripts, -1))
        # One sync for all per-script counts
        counts = torch.zeros(len(flat_vocabs), dtype=torch.long, device=device)
        valid = fs >= 0
        counts.scatter_add_(0, fs[valid].reshape(-1),
                            torch.ones_like(fs[valid].reshape(-1)))
        counts = counts.tolist()
        for fid, n in enumerate(counts):
            if n == 0:
                continue
            m = (fs == fid)
            vs = flat_vocabs[fid]
            total = total + _sym_kl(logits1[m][:, :vs], logits2[m][:, :vs])
            n_terms += 1

    if n_terms > 0:
        total = total / n_terms
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
    buckets: dict[tuple[int, int, bool], list[dict]] = {}
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

            frame_start = offset_px // 4
            frame_end = min((offset_px + width_px + 3) // 4, T)
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
            ids = list(_encode_text_cached(text, script_name))
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

            # Digit segments of RTL scripts read left-to-right, so buckets
            # are keyed by direction too — every chunk stays uniform.
            seg_rtl = segment_is_rtl(script_name, text)
            buckets.setdefault((g, s, seg_rtl), []).append({
                "b": b, "frame_start": frame_start, "frame_end": frame_end,
                "seg_len": seg_len, "ids": ids, "vs": vs,
                "rtl": seg_rtl,
            })

    ctc_loss = torch.zeros(1, device=device)
    ctc_chars = 0

    # Cap per-call tensor size to limit peak memory for big-vocab buckets
    # (e.g. Han heads at ~2000 with many segments at long max_T).
    MAX_BUCKET_ELEMS = 32 * 1024 * 1024  # 32M fp32 = 128MB per padded tensor

    def _run_chunk(chunk_segs, max_T, vs):
        """Run batched CTC on one chunk, return (loss, n_chars)."""
        N = len(chunk_segs)
        # One advanced-index gather for the whole chunk. The old per-segment
        # slice-copy loop (`batched[:len, i] = logits[b, fs:fe, :vs]`) made
        # autograd allocate + accumulate a full-size (B, T, max_vocab)
        # gradient buffer per segment in backward — hundreds per step, which
        # dominated the entire training step. One indexing op has one
        # scatter-add backward. Steps past a segment's end re-read its own
        # last frame: always in-bounds, ignored by CTC (t >= input_len),
        # and their grad is exactly zero, so the result is identical.
        b_idx = torch.tensor([sg["b"] for sg in chunk_segs], device=device)
        f_start = torch.tensor([sg["frame_start"] for sg in chunk_segs],
                               device=device)
        input_lens = torch.tensor([sg["seg_len"] for sg in chunk_segs],
                                  dtype=torch.long, device=device)
        steps = torch.arange(max_T, device=device)
        if chunk_segs[0]["rtl"]:
            frame_last = f_start + input_lens - 1
            t_idx = torch.maximum(frame_last[:, None] - steps[None, :],
                                  f_start[:, None])
        else:
            t_idx = torch.minimum(f_start[:, None] + steps[None, :],
                                  (f_start + input_lens - 1)[:, None])
        gathered = logits[b_idx[:, None], t_idx, :vs]  # (N, max_T, vs)

        log_probs = gathered.permute(1, 0, 2).float().log_softmax(dim=-1)
        target_lens = torch.tensor([len(sg["ids"]) for sg in chunk_segs],
                                   dtype=torch.long, device=device)
        concat_targets = [i for sg in chunk_segs for i in sg["ids"]]
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

    for (g, s, _), segs in buckets.items():
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
