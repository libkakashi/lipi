"""
MoE model evaluation.

Computes LID-1, LID-2, word accuracy, and character accuracy
per-group and per-script.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from src.encoding.decompose import decode_ids, script_vocab_size


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
    lid1_correct = lid1_total = 0
    lid2_correct = lid2_total = 0
    ctc_correct = ctc_total = total_chars = correct_chars = 0

    # Per-group stats
    g_lid_correct = [0] * n_groups
    g_lid_total = [0] * n_groups
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
        imgs, targets, tgt_lens, gids, sids, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)
        sids_dev = sids.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)
        tgt_lens = tgt_lens.to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(imgs, group_ids=None)
            # Also run with ground truth routing to compute val loss
            out_gt = model(imgs, group_ids=gids, script_ids=sids_dev)

        # Val CTC loss (with ground truth routing, same as training)
        B = imgs.shape[0]
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

        # Val LID-1 loss
        val_lid1_loss += F.cross_entropy(
            out_gt["group_logits"], gids, reduction="sum").item()

        del out_gt

        # LID-1
        gl = out["group_logits"]
        if gl.dim() == 3:
            pred_gids = gl[:, :, 1:].argmax(dim=-1).mode(dim=-1).values
        else:
            pred_gids = gl.argmax(-1)
        lid1_correct += (pred_gids == gids).sum().item()
        lid1_total += gids.shape[0]

        for g in range(n_groups):
            mask = (gids == g)
            if mask.any():
                g_lid_total[g] += mask.sum().item()
                g_lid_correct[g] += (pred_gids[mask] == g).sum().item()

        # LID-2 per multi-script group
        for g_idx, script_logits, group_mask in out["script_logits_per_group"]:
            if script_logits is None:
                continue
            pred_scripts = script_logits.argmax(-1)
            true_scripts = sids_dev[group_mask]
            lid1_ok = (pred_gids[group_mask] == gids[group_mask])
            if lid1_ok.any():
                lid2_correct += (pred_scripts[lid1_ok] == true_scripts[lid1_ok]).sum().item()
                lid2_total += lid1_ok.sum().item()
                for ls in range(script_logits.shape[-1]):
                    s_mask = (true_scripts == ls) & lid1_ok
                    if s_mask.any():
                        key = (g_idx, ls)
                        s_lid2_total[key] = s_lid2_total.get(key, 0) + s_mask.sum().item()
                        s_lid2_correct[key] = s_lid2_correct.get(key, 0) + (
                            pred_scripts[s_mask] == ls).sum().item()

        # Build predicted script ID per sample (from LID-2)
        pred_sids = sids.clone()
        for g_idx, script_logits, group_mask in out["script_logits_per_group"]:
            if script_logits is not None:
                pred_sids[group_mask.cpu()] = script_logits.argmax(-1).cpu()

        # CTC decode — slice to per-script vocab size before argmax
        # (positions beyond vocab_size are zero-padded and would win argmax
        # once real logits go negative during training)
        all_logits = out["logits"].float().cpu()
        pred_gids_cpu = pred_gids.cpu().tolist()
        gids_cpu = gids.cpu().tolist()
        sids_cpu = sids.cpu().tolist()
        pred_sids_cpu = pred_sids.cpu().tolist()
        for i, (label, pred_g, true_g, local_sid, pred_sid) in enumerate(
                zip(labels, pred_gids_cpu, gids_cpu, sids_cpu, pred_sids_cpu)):
            if pred_g >= n_groups:
                continue
            key = (true_g, local_sid)
            ref_s = str(label).strip().lower()

            # Only decode when LID-1 correct
            if pred_g != true_g:
                ctc_total += 1
                g_word_total[true_g] += 1
                g_char_total[true_g] += len(ref_s)
                total_chars += len(ref_s)
                s_word_total[key] = s_word_total.get(key, 0) + 1
                s_char_total[key] = s_char_total.get(key, 0) + len(ref_s)
                continue

            # CTC greedy decode: argmax → collapse repeats → remove blanks
            pred_sid_safe = min(pred_sid, len(group_script_names[pred_g]) - 1)
            vs = group_script_vocab_sizes[pred_g][pred_sid_safe] if group_script_vocab_sizes else script_vocab_size(group_script_names[pred_g][pred_sid_safe])
            seq = all_logits[i, :, :vs].argmax(dim=-1).tolist()
            ids = []
            prev = -1
            for t in seq:
                if t != prev and t != 0:
                    ids.append(t)
                prev = t

            script_name = group_script_names[pred_g][pred_sid_safe] if pred_g < len(group_script_names) else ""
            raw_decoded = decode_ids(ids, script_name) if script_name else ""
            dec_s = raw_decoded.strip().lower()

            ctc_total += 1
            g_word_total[true_g] += 1
            s_word_total[key] = s_word_total.get(key, 0) + 1
            if dec_s == ref_s:
                ctc_correct += 1
                g_word_correct[true_g] += 1
                s_word_correct[key] = s_word_correct.get(key, 0) + 1
            # CER via edit distance: correct_chars = ref_len - edit_dist
            edits = _edit_distance(dec_s, ref_s)
            matched = max(0, len(ref_s) - edits)
            total_chars += len(ref_s)
            correct_chars += matched
            g_char_total[true_g] += len(ref_s)
            g_char_correct[true_g] += matched
            s_char_total[key] = s_char_total.get(key, 0) + len(ref_s)
            s_char_correct[key] = s_char_correct.get(key, 0) + matched

    # Print results
    lid1_acc = 100 * lid1_correct / max(lid1_total, 1)
    lid2_acc = 100 * lid2_correct / max(lid2_total, 1)
    ctc_acc = 100 * ctc_correct / max(ctc_total, 1)
    char_acc = 100 * correct_chars / max(total_chars, 1)

    avg_ctc_loss = val_ctc_loss / max(val_loss_samples, 1)
    avg_lid1_loss = val_lid1_loss / max(lid1_total, 1)

    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  LID-1: {lid1_acc:5.1f}%   LID-2: {lid2_acc:5.1f}%              │")
    print(f"  │  Word:  {ctc_acc:5.1f}%   Char:  {char_acc:5.1f}%              │")
    print(f"  │  Val loss: ctc={avg_ctc_loss:.4f}  lid1={avg_lid1_loss:.4f}     │")
    print(f"  └──────────────────────────────────────────────┘")

    print(f"\n  {'Group / Script':<20s} {'LID1':>6s} {'Word':>6s} {'Char':>6s} {'LID2':>6s}")
    print(f"  {'─' * 50}")

    for g in range(n_groups):
        lid_g = 100 * g_lid_correct[g] / max(g_lid_total[g], 1)
        word_g = 100 * g_word_correct[g] / max(g_word_total[g], 1)
        char_g = 100 * g_char_correct[g] / max(g_char_total[g], 1)
        name = active_groups[g] if g < len(active_groups) else f"group{g}"
        print(f"  {name:<20s} {lid_g:5.1f}% {word_g:5.1f}% {char_g:5.1f}%")

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

    print(f"  {'─' * 50}")

    return {"lid1_acc": lid1_acc, "lid2_acc": lid2_acc,
            "word_acc": ctc_acc, "char_acc": char_acc,
            "val_ctc_loss": avg_ctc_loss, "val_lid1_loss": avg_lid1_loss}
