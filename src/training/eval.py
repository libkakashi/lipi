"""
MoE model evaluation.

Computes LID-1, LID-2, word accuracy, and character accuracy
per-group and per-script.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from src.encoding.decompose import decode_ids, script_vocab_size
from src.training.losses import compute_lid1_loss


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


@torch.no_grad()
def evaluate(model, val_loader, group_tokenizers, group_script_names,
             active_groups, device, device_type, use_amp, amp_dtype,
             group_script_vocab_sizes=None):
    model.eval()
    n_groups = len(group_tokenizers)

    # Val loss accumulators
    val_ctc_loss = 0.0
    val_lid1_loss = 0.0
    val_loss_samples = 0

    # Global stats
    lid1_frame_correct = lid1_frame_total = 0
    lid2_correct = lid2_total = 0
    ctc_correct = ctc_total = total_chars = correct_chars = 0

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

    for batch_idx, batch in enumerate(val_loader):
        imgs, targets, tgt_lens, gids, sids, labels, group_labels, batch_segments = batch
        group_labels = group_labels.to(device, non_blocking=True)
        imgs = imgs.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)
        sids_dev = sids.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)
        tgt_lens = tgt_lens.to(device, non_blocking=True)

        B = imgs.shape[0]

        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(imgs, group_ids=None)
            # Also run with ground truth routing to compute val loss
            T_est = imgs.shape[3] // 2
            gl_frames_gt = group_labels[:, ::2][:, :T_est]
            # Replace padding (-100) with blank for model routing
            gl_for_model = gl_frames_gt.clone()
            gl_for_model[gl_for_model < 0] = n_groups  # n_groups = blank/whitespace
            # Build per-frame script labels from segment metadata
            sl_frames = torch.zeros(B, T_est, dtype=torch.long, device=device)
            for b in range(B):
                if batch_segments is not None:
                    for seg in batch_segments[b]:
                        fs = seg["offset"] // 2
                        fe = min((seg["offset"] + seg["width"] + 1) // 2, T_est)
                        sl_frames[b, fs:fe] = seg.get("script_id", 0)
            out_gt = model(imgs, group_ids=gl_for_model, script_ids=sl_frames)
        ctc_ok = (tgt_lens <= out_gt["lengths"]) & (tgt_lens > 0)
        if group_script_vocab_sizes and ctc_ok.any():
            for g, script_vocabs in enumerate(group_script_vocab_sizes):
                for s, vs in enumerate(script_vocabs):
                    s_mask = ctc_ok & (gids == g) & (sids_dev == s)
                    if not s_mask.any():
                        continue
                    s_logits = out_gt["logits"][s_mask]
                    s_log_probs = (s_logits[:, :, :vs].float()
                                   .log_softmax(dim=-1).permute(1, 0, 2))
                    s_tgt_lens = tgt_lens[s_mask]
                    s_targets_2d = targets[s_mask]
                    col_idx = torch.arange(s_targets_2d.shape[1], device=device)
                    s_targets_flat = s_targets_2d[col_idx < s_tgt_lens.unsqueeze(1)]
                    ctc_l = F.ctc_loss(
                        s_log_probs, s_targets_flat,
                        out_gt["lengths"][s_mask], s_tgt_lens,
                        blank=0, reduction="sum", zero_infinity=True)
                    val_ctc_loss += ctc_l.item()
                    val_loss_samples += s_tgt_lens.sum().item()

        # Val LID-1 loss (per-frame, -100 padding ignored by default)
        ce_fn = torch.nn.CrossEntropyLoss()
        T_gt = out_gt["group_logits"].shape[1]
        gl_frames = group_labels[:, ::2][:, :T_gt]
        lid1_l = compute_lid1_loss(out_gt["group_logits"], gl_frames, ce_fn)
        val_lid1_loss += lid1_l.item() * imgs.shape[0]

        del out_gt

        # LID-1 frame-level accuracy (exclude padding, include whitespace)
        frame_preds = out["group_logits"].argmax(dim=-1)  # (B, T)
        T_lid = frame_preds.shape[1]
        gl_frames = group_labels[:, ::2][:, :T_lid]

        non_pad = (gl_frames >= 0)  # exclude -100 padding
        lid1_frame_correct += (frame_preds[non_pad] == gl_frames[non_pad]).sum().item()
        lid1_frame_total += non_pad.sum().item()

        for g in range(n_groups):
            g_frame_mask = (gl_frames == g)
            if g_frame_mask.any():
                g_lid_frame_total[g] += g_frame_mask.sum().item()
                g_lid_frame_correct[g] += (frame_preds[g_frame_mask] == g).sum().item()

        # LID-2 per multi-script group
        for g_idx, script_logits, group_mask in out["script_logits_per_group"]:
            if script_logits is None:
                continue
            pred_scripts = script_logits.argmax(-1)
            true_scripts = sids_dev[group_mask]
            # All samples in this group were routed here by frame predictions —
            # evaluate LID-2 on all of them
            lid2_correct += (pred_scripts == true_scripts).sum().item()
            lid2_total += true_scripts.shape[0]
            for ls in range(script_logits.shape[-1]):
                s_mask = (true_scripts == ls)
                if s_mask.any():
                    key = (g_idx, ls)
                    s_lid2_total[key] = s_lid2_total.get(key, 0) + s_mask.sum().item()
                    s_lid2_correct[key] = s_lid2_correct.get(key, 0) + (
                        pred_scripts[s_mask] == ls).sum().item()

        # Per-segment CTC decode and eval
        all_logits = out["logits"].float().cpu()
        gl_cpu = out["group_logits"].cpu()
        gids_cpu = gids.cpu().tolist()
        sids_cpu = sids.cpu().tolist()

        # batch_segments already set from batch unpacking above

        for i, (label, true_g, local_sid) in enumerate(
                zip(labels, gids_cpu, sids_cpu)):

            frame_preds = gl_cpu[i].argmax(dim=-1)  # (T,)

            # Get ground truth segments for this image
            if batch_segments is not None:
                img_segs = batch_segments[i]
            else:
                img_segs = [{"group_id": true_g, "script_id": local_sid,
                             "text": label, "width": all_logits.shape[1] * 2, "offset": 0}]

            # Decode and evaluate each segment
            T_img = frame_preds.shape[0]
            for seg in img_segs:
                seg_text = seg["text"]
                seg_g = seg["group_id"]
                seg_s = seg.get("script_id", 0)
                seg_offset = seg["offset"]
                seg_width = seg["width"]

                ref_s = str(seg_text).strip().lower()
                if not ref_s:
                    continue

                key = (seg_g, seg_s)

                # Frame range for this segment
                frame_start = seg_offset // 2
                frame_end = min((seg_offset + seg_width + 1) // 2, T_img)
                if frame_end <= frame_start:
                    continue

                # Decode this segment's frames using its group's vocab
                # (no mode/rounding — per-frame CTC logits already come from
                # the correct group's head, so just decode directly)
                s_idx = min(seg_s, len(group_script_names[seg_g]) - 1) if seg_g < len(group_script_names) else 0
                if group_script_vocab_sizes:
                    vs = group_script_vocab_sizes[seg_g][s_idx]
                else:
                    vs = script_vocab_size(group_script_names[seg_g][s_idx])

                seg_logits = all_logits[i, frame_start:frame_end, :vs]
                seq = seg_logits.argmax(dim=-1).tolist()
                ids = []
                prev = -1
                for tok in seq:
                    if tok != prev and tok != 0:
                        ids.append(tok)
                    prev = tok

                script_name = group_script_names[seg_g][s_idx] if seg_g < len(group_script_names) else ""
                dec_s = decode_ids(ids, script_name).strip().lower() if script_name and ids else ""

                ctc_total += 1
                g_word_total[seg_g] += 1
                s_word_total[key] = s_word_total.get(key, 0) + 1
                if dec_s == ref_s:
                    ctc_correct += 1
                    g_word_correct[seg_g] += 1
                    s_word_correct[key] = s_word_correct.get(key, 0) + 1
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

    avg_ctc_loss = val_ctc_loss / max(val_loss_samples, 1)
    avg_lid1_loss = val_lid1_loss / max(lid1_frame_total, 1)

    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  LID-1: {lid1_frame_acc:5.1f}%   LID-2: {lid2_acc:5.1f}%              │")
    print(f"  │  Word:  {ctc_acc:5.1f}%   Char:  {char_acc:5.1f}%              │")
    print(f"  │  Val loss: ctc={avg_ctc_loss:.4f}  lid1={avg_lid1_loss:.4f}     │")
    print(f"  └──────────────────────────────────────────────┘")

    print(f"\n  {'Group / Script':<20s} {'LID1':>6s} {'Word':>6s} {'Char':>6s} {'LID2':>6s}")
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
            "val_ctc_loss": avg_ctc_loss, "val_lid1_loss": avg_lid1_loss}
