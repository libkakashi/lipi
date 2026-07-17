"""
MoE model evaluation.

Computes LID-1, LID-2, word accuracy, and character accuracy
per-group and per-script. Single inference forward pass (no GT routing).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from src.encoding.decompose import decode_ids, encode_text, script_vocab_size
from src.encoding.direction import ctc_segment_order
from src.training.losses import compute_lid1_loss
from src.training.routing import build_frame_labels_from_segments


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,      # deletion
                curr[j] + 1,           # insertion
                prev[j] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def _batched_ctc_val_loss(logits, segments_batch, group_script_names,
                          group_script_vocab_sizes, device,
                          emit_per_frame=1):
    """Compute batched CTC val loss without per-segment kernel launches.

    Frame ranges are in emission slots (see losses._build_buckets)."""
    B, T, _ = logits.shape
    buckets: dict[tuple[int, int], list] = {}

    for b in range(B):
        for seg in segments_batch[b]:
            text = seg["text"]
            g = seg["group_id"]
            s = seg.get("script_id", 0)
            if not text or seg["width"] == 0:
                continue
            if g >= len(group_script_names) or s >= len(group_script_names[g]):
                continue
            script_name = group_script_names[g][s]
            if not script_name:
                continue

            frame_start = seg["offset"] * emit_per_frame // 4
            frame_end = min(
                ((seg["offset"] + seg["width"]) * emit_per_frame + 3) // 4, T)
            seg_len = frame_end - frame_start
            if seg_len < 1:
                continue

            ids = encode_text(text, script_name)
            if not ids:
                continue
            n_repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
            if seg_len < len(ids) + n_repeats:
                continue

            vs = group_script_vocab_sizes[g][s]
            buckets.setdefault((g, s), []).append({
                "b": b, "frame_start": frame_start, "frame_end": frame_end,
                "seg_len": seg_len, "ids": ids, "vs": vs, "text": text,
            })

    total_loss = 0.0
    total_chars = 0

    for (g, s), segs in buckets.items():
        segs.sort(key=lambda x: x["seg_len"])
        vs = segs[0]["vs"]
        max_T = segs[-1]["seg_len"]
        N = len(segs)

        batched_logits = torch.zeros(max_T, N, vs, device=device, dtype=logits.dtype)
        input_lens = torch.zeros(N, dtype=torch.long, device=device)
        target_lens = torch.zeros(N, dtype=torch.long, device=device)
        concat_targets = []

        for i, sg in enumerate(segs):
            seg_logits = logits[
                sg["b"], sg["frame_start"]:sg["frame_end"], :vs]
            script_name = group_script_names[g][s]
            batched_logits[:sg["seg_len"], i, :] = ctc_segment_order(
                seg_logits, script_name, sg["text"])
            input_lens[i] = sg["seg_len"]
            target_lens[i] = len(sg["ids"])
            concat_targets.extend(sg["ids"])

        log_probs = batched_logits.float().log_softmax(dim=-1)
        targets = torch.tensor(concat_targets, dtype=torch.long, device=device)

        loss = F.ctc_loss(log_probs, targets, input_lens, target_lens,
                          blank=0, reduction="sum", zero_infinity=True)
        total_loss += loss.item()
        total_chars += len(concat_targets)

    return total_loss, total_chars


@torch.no_grad()
def evaluate(model, val_loader, group_tokenizers, group_script_names,
             active_groups, device, device_type, use_amp, amp_dtype,
             group_script_vocab_sizes=None, max_batches=50,
             oracle_routing=False, route_smooth=False):
    """oracle_routing=True feeds ground-truth frame group/script ids to the
    model instead of letting LID route. The word/char delta between the two
    modes IS the routing tax — the accuracy lost to LID misroutes alone.
    route_smooth=True applies inference-time routing diffusion (absorb
    short blank/foreign islands into matching flanks) to the predicted
    routing; no effect under oracle_routing."""
    model.eval()
    n_groups = len(group_tokenizers)
    # Emission slots per frame (2 for the v6 encoder; compiled wrappers
    # proxy class attributes). All logits-axis segment math scales by it.
    emit_pf = getattr(model, "emit_per_frame", 1)

    # Val loss
    val_ctc_loss = 0.0
    val_ctc_chars = 0
    val_lid1_loss = 0.0
    val_lid1_frames = 0

    # Global stats
    lid1_frame_correct = lid1_frame_total = 0
    lid2_correct = lid2_total = 0
    ctc_correct = ctc_total = total_chars = correct_chars = 0
    ctc_correct_cs = 0                    # case-sensitive exact matches
    line_correct = line_total = 0         # single-segment samples (real lines)
    wordseg_correct = wordseg_total = 0   # multi-segment words (synth)

    # Per-group stats
    g_lid_frame_correct = [0] * n_groups
    g_lid_frame_total = [0] * n_groups
    g_word_correct = [0] * n_groups
    g_word_total = [0] * n_groups
    g_char_correct = [0] * n_groups
    g_char_total = [0] * n_groups

    # Per-script stats (keyed by (group_idx, local_script_idx))
    s_word_correct: dict[tuple[int, int], int] = {}
    s_word_total: dict[tuple[int, int], int] = {}
    s_char_correct: dict[tuple[int, int], int] = {}
    s_char_total: dict[tuple[int, int], int] = {}
    s_lid2_correct: dict[tuple[int, int], int] = {}
    s_lid2_total: dict[tuple[int, int], int] = {}

    ce_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break
        imgs, targets, tgt_lens, gids, sids, labels, group_labels, batch_segments = batch
        group_labels = group_labels.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)

        B = imgs.shape[0]

        # GT frame labels (span-based, same convention as training routing
        # and the CTC decode ranges) — needed up front for oracle routing.
        T_est = imgs.shape[3] // 4
        segs = (batch_segments if batch_segments is not None
                else [[] for _ in range(B)])
        gl_gt, sl_gt, _ = build_frame_labels_from_segments(
            segs, T_est, n_groups, device)

        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            if oracle_routing:
                out = model(imgs, group_ids=gl_gt, script_ids=sl_gt)
            else:
                out = model(imgs, group_ids=None,
                            route_smooth=route_smooth)

        T = out["group_logits"].shape[1]
        # LID-1 metric target: span-based (what routing/CTC decode need a
        # frame routed as), with -100 padding passed through from the
        # pixel labels so pad columns stay excluded. The old pixel-strided
        # target under-counted segment-boundary frames.
        gl_pix = group_labels[:, ::4][:, :T]
        gl_frames = torch.where(gl_pix == -100, gl_pix, gl_gt[:, :T])

        # --- LID-1 val loss ---
        lid1_l = compute_lid1_loss(out["group_logits"], gl_frames, ce_fn)
        non_pad = (gl_frames >= 0)
        n_frames = non_pad.sum().item()
        val_lid1_loss += lid1_l.item() * n_frames
        val_lid1_frames += n_frames

        # --- Batched CTC val loss (one call per (group, script) bucket) ---
        if batch_segments is not None and group_script_vocab_sizes:
            ctc_l, ctc_c = _batched_ctc_val_loss(
                out["logits"], batch_segments, group_script_names,
                group_script_vocab_sizes, device, emit_per_frame=emit_pf)
            val_ctc_loss += ctc_l
            val_ctc_chars += ctc_c

        # --- LID-1 frame-level accuracy ---
        frame_preds = out["group_logits"].argmax(dim=-1)
        lid1_frame_correct += (frame_preds[non_pad] == gl_frames[non_pad]).sum().item()
        lid1_frame_total += n_frames

        for g in range(n_groups):
            g_frame_mask = (gl_frames == g)
            if g_frame_mask.any():
                g_lid_frame_total[g] += g_frame_mask.sum().item()
                g_lid_frame_correct[g] += (frame_preds[g_frame_mask] == g).sum().item()

        # --- LID-2 per-frame accuracy ---
        sl_frames_gt = sl_gt[:, :T]

        for g_int, lid2_log in out.get("lid2_logits_per_group", {}).items():
            g_idx = int(g_int)
            g_mask = (gl_frames == g_idx)
            if not g_mask.any():
                continue
            pred_scripts = lid2_log[:, :T].argmax(dim=-1)[g_mask]
            true_scripts = sl_frames_gt[g_mask]
            lid2_correct += (pred_scripts == true_scripts).sum().item()
            lid2_total += true_scripts.numel()
            for ls in range(lid2_log.shape[-1]):
                s_mask = (true_scripts == ls)
                if s_mask.any():
                    key = (g_idx, ls)
                    s_lid2_total[key] = s_lid2_total.get(key, 0) + s_mask.sum().item()
                    s_lid2_correct[key] = s_lid2_correct.get(key, 0) + (
                        pred_scripts[s_mask] == ls).sum().item()

        # --- Per-segment CTC decode ---
        # Keep out["logits"] on GPU; argmax per-segment on GPU, then a
        # single concat + .tolist() to move everything to CPU in one
        # transfer. Previously we moved the full (B, T, max_vocab) float
        # tensor to CPU (~1 GB/batch with max_vocab=3800); the argmax
        # result is ~3800× smaller.
        logits_gpu = out["logits"]
        T_logits = logits_gpu.shape[1]
        gids_cpu = gids.cpu().tolist()
        sids_cpu = sids.cpu().tolist()

        # Phase 1: on GPU, compute per-segment argmaxes into a list.
        # Each entry is (metadata, argmax_tensor_1d_on_gpu).
        seg_gpu_preds = []
        for i, (label, true_g, local_sid) in enumerate(
                zip(labels, gids_cpu, sids_cpu)):

            if batch_segments is not None:
                img_segs = batch_segments[i]
            else:
                img_segs = [{"group_id": true_g, "script_id": local_sid,
                             "text": label,
                             "width": T_logits * 4 // emit_pf, "offset": 0}]

            is_line = len(img_segs) == 1
            for seg in img_segs:
                seg_text = seg["text"]
                seg_g = seg["group_id"]
                seg_s = seg.get("script_id", 0)

                ref_raw = str(seg_text).strip()
                ref_s = ref_raw.lower()
                if not ref_s:
                    continue

                frame_start = seg["offset"] * emit_pf // 4
                frame_end = min(
                    ((seg["offset"] + seg["width"]) * emit_pf + 3) // 4,
                    T_logits)
                if frame_end <= frame_start:
                    continue

                s_idx = (min(seg_s, len(group_script_names[seg_g]) - 1)
                         if seg_g < len(group_script_names) else 0)
                script_name = (group_script_names[seg_g][s_idx]
                               if seg_g < len(group_script_names) else "")
                if not script_name:
                    continue

                if group_script_vocab_sizes:
                    vs = group_script_vocab_sizes[seg_g][s_idx]
                else:
                    vs = script_vocab_size(script_name)

                seg_logits = logits_gpu[i, frame_start:frame_end, :vs]
                seq_gpu = ctc_segment_order(
                    seg_logits, script_name, str(seg_text)).argmax(dim=-1)
                seg_gpu_preds.append(
                    (seg_g, seg_s, script_name, ref_s, ref_raw, is_line,
                     seq_gpu))

        # Phase 2: single GPU→CPU transfer for all segments in this batch.
        if seg_gpu_preds:
            seg_lens = [t.shape[0] for *_, t in seg_gpu_preds]
            big_cpu = torch.cat([t for *_, t in seg_gpu_preds]).tolist()
        else:
            seg_lens = []
            big_cpu = []

        # Phase 3: decode on CPU.
        offset = 0
        for (seg_g, seg_s, script_name, ref_s, ref_raw, is_line, _), sl in zip(
                seg_gpu_preds, seg_lens):
            seq = big_cpu[offset:offset + sl]
            offset += sl

            ids = []
            prev = -1
            for tok in seq:
                if tok != prev and tok != 0:
                    ids.append(tok)
                prev = tok

            dec_raw = decode_ids(ids, script_name).strip() if ids else ""
            dec_s = dec_raw.lower()

            key = (seg_g, seg_s)
            ctc_total += 1
            g_word_total[seg_g] += 1
            s_word_total[key] = s_word_total.get(key, 0) + 1
            if is_line:
                line_total += 1
            else:
                wordseg_total += 1
            if dec_raw == ref_raw:
                ctc_correct_cs += 1
            if dec_s == ref_s:
                ctc_correct += 1
                g_word_correct[seg_g] += 1
                s_word_correct[key] = s_word_correct.get(key, 0) + 1
                if is_line:
                    line_correct += 1
                else:
                    wordseg_correct += 1
            edits = _edit_distance(dec_s, ref_s)
            matched = max(0, len(ref_s) - edits)
            total_chars += len(ref_s)
            correct_chars += matched
            g_char_total[seg_g] += len(ref_s)
            g_char_correct[seg_g] += matched
            s_char_total[key] = s_char_total.get(key, 0) + len(ref_s)
            s_char_correct[key] = s_char_correct.get(key, 0) + matched

    # Print results
    lid1_frame_acc = 100 * lid1_frame_correct / max(lid1_frame_total, 1)
    lid2_acc = 100 * lid2_correct / max(lid2_total, 1)
    ctc_acc = 100 * ctc_correct / max(ctc_total, 1)
    char_acc = 100 * correct_chars / max(total_chars, 1)

    avg_ctc_loss = val_ctc_loss / max(val_ctc_chars, 1)
    avg_lid1_loss = val_lid1_loss / max(val_lid1_frames, 1)

    ctc_acc_cs = 100 * ctc_correct_cs / max(ctc_total, 1)
    line_acc = 100 * line_correct / max(line_total, 1)
    wordseg_acc = 100 * wordseg_correct / max(wordseg_total, 1)
    mode = ("ORACLE GT routing" if oracle_routing
            else "pred+smooth" if route_smooth else "predicted routing")
    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  [{mode:<17s}]                        │")
    print(f"  │  LID-1: {lid1_frame_acc:5.1f}%   LID-2: {lid2_acc:5.1f}%              │")
    print(f"  │  Word:  {ctc_acc:5.1f}%   Char:  {char_acc:5.1f}%              │")
    print(f"  │  Word (case-sens): {ctc_acc_cs:5.1f}%                     │")
    print(f"  │  Line-exact (1-seg): {line_acc:5.1f}% (n={line_total})       ")
    print(f"  │  Word-seg (multi):   {wordseg_acc:5.1f}% (n={wordseg_total})       ")
    print(f"  │  Val loss: ctc={avg_ctc_loss:.4f}  lid1={avg_lid1_loss:.4f}     │")
    print(f"  └──────────────────────────────────────────────┘")

    print(f"\n  {'Group / Script':<20s} {'LID1':>6s} {'Word':>6s} "
          f"{'Char':>6s} {'LID2':>6s}")
    print(f"  {'─' * 50}")

    for g in range(n_groups):
        frame_g = 100 * g_lid_frame_correct[g] / max(g_lid_frame_total[g], 1)
        word_g = 100 * g_word_correct[g] / max(g_word_total[g], 1)
        char_g = 100 * g_char_correct[g] / max(g_char_total[g], 1)
        name = active_groups[g] if g < len(active_groups) else f"group{g}"
        print(f"  {name:<20s} {frame_g:5.1f}% {word_g:5.1f}% {char_g:5.1f}%")

        scripts = group_script_names[g] if g < len(group_script_names) else []
        for ls, sname in enumerate(scripts):
            key = (g, ls)
            sw = s_word_total.get(key, 0)
            if sw == 0:
                continue
            word_s = 100 * s_word_correct.get(key, 0) / max(sw, 1)
            char_s = 100 * s_char_correct.get(key, 0) / max(s_char_total.get(key, 0), 1)
            lid2_str = ""
            if key in s_lid2_total:
                lid2_s = 100 * s_lid2_correct.get(key, 0) / max(s_lid2_total[key], 1)
                lid2_str = f"{lid2_s:5.1f}%"
            print(f"    {sname:<18s} {'':>6s} {word_s:5.1f}% {char_s:5.1f}% {lid2_str}")

    print(f"  {'─' * 56}")

    return {"lid1_acc": lid1_frame_acc, "lid2_acc": lid2_acc,
            "word_acc": ctc_acc, "char_acc": char_acc,
            "word_acc_cs": ctc_acc_cs, "line_acc": line_acc,
            "wordseg_acc": wordseg_acc, "oracle_routing": oracle_routing,
            "val_ctc_loss": avg_ctc_loss, "val_lid1_loss": avg_lid1_loss}
