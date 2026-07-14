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

import numpy as np
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
    present_groups: set[int] | None = None,
) -> Tensor:
    """LID-2 per-frame script cross-entropy within multi-script groups.

    Averages cross-entropy across the multi-script groups that have any
    frames present in the batch. Returns a zero scalar if none do.

    lid2_logits_per_group: {group_id_str: (B, T, num_scripts_in_group)}.
    frame_group_ids / frame_script_ids: (B, T) — pre-truncated to match T.
    present_groups: CPU-side set of group ids present in the batch (from
        build_frame_labels_from_segments). When given, absent groups skip
        with zero GPU work; masking goes through ignore_index instead of
        boolean indexing, so the whole loss is sync-free.
    """
    device = frame_group_ids.device
    total = torch.zeros(1, device=device)
    n = 0
    for g_str, lid2_logits in lid2_logits_per_group.items():
        g = int(g_str)
        if present_groups is not None:
            if g not in present_groups:
                continue
        else:
            mask = frame_group_ids == g
            if not mask.any():
                continue
        # Frames of other groups become ignore_index — same mean-over-
        # valid-frames semantics as boolean masking, no nonzero() sync.
        # sum/clamp(count) instead of reduction="mean" so a group whose
        # frames were all truncated away contributes 0, not NaN.
        in_group = frame_group_ids == g
        targets = torch.where(in_group, frame_script_ids,
                              frame_script_ids.new_full((), -100))
        ce_sum = F.cross_entropy(
            lid2_logits.reshape(-1, lid2_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100, label_smoothing=label_smoothing,
            reduction="sum")
        total = total + ce_sum / in_group.sum().clamp(min=1)
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
        # CTC logits run at emit_per_frame slots per frame (2T for the v6
        # encoder) while flat_scripts stays at frame granularity — expand
        # the mask so both of a frame's emission slots are compared.
        E = logits1.shape[1] // flat_scripts.shape[1]
        fs_slots = fs.repeat_interleave(E, dim=1) if E > 1 else fs
        # One sync for all per-script counts
        counts = torch.zeros(len(flat_vocabs), dtype=torch.long, device=device)
        valid = fs >= 0
        counts.scatter_add_(0, fs[valid].reshape(-1),
                            torch.ones_like(fs[valid].reshape(-1)))
        counts = counts.tolist()
        for fid, n in enumerate(counts):
            if n == 0:
                continue
            m = (fs_slots == fid)
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
    emit_per_frame: int = 1,
) -> Tensor:
    """Per-segment CTC loss, batched by (group, script) for speed.

    Each image has segments: [{group_id, script_id, text, width, offset}, ...]
    Each segment's text is encoded with its script, and CTC loss is computed
    on the corresponding frame slice using the correct group's vocab.

    Segments with the same (group, script) are padded to max length and
    processed in a single F.ctc_loss call instead of one call per segment.

    Convenience wrapper over build_ctc_loss_plan + apply_ctc_loss_plan;
    when the same segments feed several logits tensors (final + interCTC),
    build the plan once and apply it to each.
    """
    device = logits.device
    plan = build_ctc_loss_plan(
        segments_batch, logits.shape[1], group_script_names,
        group_script_vocabs, device, emit_per_frame=emit_per_frame)
    return apply_ctc_loss_plan(logits, plan, device)


def _build_buckets(segments_batch, T, group_script_names,
                   group_script_vocabs, emit_per_frame=1):
    """Group valid segments by (group, script, direction) for batched CTC.

    T and the produced frame ranges are in *emission slots*: with
    emit_per_frame=E the model emits E CTC tokens per W/4 frame, so a
    segment spanning pixels [off, off+w) owns slots
    [off·E/4, ceil((off+w)·E/4)). At E=2 this doubles every segment's
    emission room — dense segments (Arabic tooth runs, jamo fallbacks)
    that failed the len(ids)+repeats feasibility check at E=1 and were
    silently dropped from the loss become trainable.
    """
    buckets: dict[tuple[int, int, bool], list[dict]] = {}
    skipped_no_script = 0
    skipped_no_ids = 0
    skipped_too_long = 0
    total_segs = 0
    B = len(segments_batch)

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

            frame_start = offset_px * emit_per_frame // 4
            frame_end = min(
                ((offset_px + width_px) * emit_per_frame + 3) // 4, T)
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

    return buckets, (total_segs, skipped_no_script, skipped_no_ids,
                     skipped_too_long)


# Cap per-call tensor size to limit peak memory for big-vocab buckets
# (e.g. the Han head at ~3800 with many segments at long max_T).
MAX_BUCKET_ELEMS = 32 * 1024 * 1024  # 32M fp32 = 128MB per padded tensor


def _plan_from_buckets(buckets, device, stats):
    """Assemble the logits-independent CTC work plan.

    All indexing math (chunk membership, gather indices including the RTL
    reversal, targets) happens in NumPy; the result crosses to the GPU in
    three batched pinned uploads instead of ~5 syncing copies per chunk.
    Length tensors stay on CPU — F.ctc_loss round-trips CUDA length
    tensors back to CPU internally, one D2H sync each per call.
    """
    total_segs, skipped_no_script, skipped_no_ids, skipped_too_long = stats
    chunks = []           # (n_start, N, max_T, vs, lens_cpu, tlens_cpu, t_off)
    b_parts, t_parts, target_parts = [], [], []
    n_rows = 0            # rows accumulated across chunks
    t_elems = 0           # flat t_idx elements accumulated
    tgt_count = 0
    ctc_chars = 0

    def _add_chunk(chunk_segs, max_T, vs):
        nonlocal n_rows, t_elems, tgt_count, ctc_chars
        N = len(chunk_segs)
        f_start = np.array([sg["frame_start"] for sg in chunk_segs],
                           dtype=np.int64)
        lens = np.array([sg["seg_len"] for sg in chunk_segs], dtype=np.int64)
        steps = np.arange(max_T, dtype=np.int64)
        last = f_start + lens - 1
        # Steps past a segment's end re-read its own last frame: always
        # in-bounds, ignored by CTC (t >= input_len), grad exactly zero.
        if chunk_segs[0]["rtl"]:
            t_idx = np.maximum(last[:, None] - steps[None, :],
                               f_start[:, None])
        else:
            t_idx = np.minimum(f_start[:, None] + steps[None, :],
                               last[:, None])
        b_parts.append(np.array([sg["b"] for sg in chunk_segs],
                                dtype=np.int64))
        t_parts.append(t_idx.reshape(-1))
        targets = [i for sg in chunk_segs for i in sg["ids"]]
        target_parts.append(np.array(targets, dtype=np.int64))
        chunks.append((
            n_rows, N, max_T, vs,
            torch.from_numpy(lens),
            torch.tensor([len(sg["ids"]) for sg in chunk_segs],
                         dtype=torch.long),
            t_elems, tgt_count,
        ))
        n_rows += N
        t_elems += N * max_T
        tgt_count += len(targets)
        ctc_chars += len(targets)

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
                _add_chunk(chunk, chunk_max_T, vs)
                chunk = []
                chunk_max_T = 0
            chunk.append(sg)
            chunk_max_T = max(chunk_max_T, sg["seg_len"])
        if chunk:
            _add_chunk(chunk, chunk_max_T, vs)

    skip_total = skipped_no_script + skipped_no_ids + skipped_too_long
    if skip_total > total_segs * 0.05:
        print(f"    [CTC segments] {ctc_chars} chars from {total_segs} segs | "
              f"skipped: {skipped_no_script} no_script, {skipped_no_ids} no_ids, "
              f"{skipped_too_long} too_long", flush=True)

    if not chunks:
        return None

    def _upload(parts):
        t = torch.from_numpy(np.concatenate(parts))
        if device.type == "cuda":
            t = t.pin_memory()
        return t.to(device, non_blocking=True)

    return {
        "chunks": chunks,
        "b_all": _upload(b_parts),
        "t_all": _upload(t_parts),
        "targets_all": _upload(target_parts),
        "chars": ctc_chars,
    }


def build_ctc_loss_plan(
    segments_batch: list[list[dict]],
    T: int,
    group_script_names: list[list[str]],
    group_script_vocabs: list[list[int]],
    device: torch.device,
    emit_per_frame: int = 1,
):
    """Build the CTC plan once per step; apply it to any number of logits
    tensors (final + intermediate CTC share it — the ~3000-segment Python
    pass depends only on the segments, not the logits)."""
    buckets, stats = _build_buckets(
        segments_batch, T, group_script_names, group_script_vocabs,
        emit_per_frame)
    return _plan_from_buckets(buckets, device, stats)


def apply_ctc_loss_plan(logits, plan, device=None) -> Tensor:
    """Gather + CTC for a prebuilt plan. Returns the per-char mean loss."""
    device = logits.device if device is None else device
    if plan is None:
        return torch.zeros(1, device=device)

    ctc_loss = torch.zeros(1, device=device)
    b_all, t_all = plan["b_all"], plan["t_all"]
    targets_all = plan["targets_all"]
    for (row0, N, max_T, vs, input_lens, target_lens,
         t_off, tgt_off) in plan["chunks"]:
        b_idx = b_all[row0:row0 + N]
        t_idx = t_all[t_off:t_off + N * max_T].view(N, max_T)
        gathered = logits[b_idx[:, None], t_idx, :vs]  # (N, max_T, vs)
        log_probs = gathered.permute(1, 0, 2).float().log_softmax(dim=-1)
        targets = targets_all[tgt_off:tgt_off + int(target_lens.sum())]

        if device.type == "mps":
            offset = 0
            for i in range(N):
                U = int(target_lens[i])
                T_i = int(input_lens[i])
                lp_i = log_probs[:T_i, i:i + 1, :]
                tgt_i = targets[offset:offset + U]
                offset += U
                ctc_loss = ctc_loss + _ctc_loss_pure(lp_i, tgt_i, blank=0)
        else:
            ctc_loss = ctc_loss + F.ctc_loss(
                log_probs, targets, input_lens, target_lens,
                blank=0, reduction="sum", zero_infinity=True)

    if plan["chars"] > 0:
        ctc_loss = ctc_loss / plan["chars"]
    return ctc_loss
