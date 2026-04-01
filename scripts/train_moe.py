#!/usr/bin/env python3
"""
Train Lipi MoE Encoder end-to-end.

Losses: CTC (text recognition) + LID-1 (group routing)

Usage:
    # Generate data first:
    python scripts/generate_data.py --scripts latin,cyrillic,greek,devanagari,gurmukhi,gujarati,bengali --out data/shards

    # Train:
    python scripts/train_moe.py --data data/shards --epochs 10 --batch-size 192

    # Resume:
    python scripts/train_moe.py --data data/shards --epochs 20 --resume checkpoints/moe/moe_epoch10.pt
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.moe_encoder import LipiMoEEncoder
from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID, SCRIPT_TO_ID, GROUP_SCRIPTS
from src.data.bigrams import LipiTokenizer, BASE_CHARS, BLANK_TOKEN


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

SCRIPT_TO_LANG = {
    "latin": "en", "cyrillic": "en", "greek": "en",
    "arabic": "ur", "hebrew": "en", "han_kana": "en", "korean": "en",
    "devanagari": "hi", "gurmukhi": "pa", "gujarati": "gu",
    "bengali": "bn_as", "kannada": "kn", "telugu": "te",
    "malayalam": "ml", "tamil": "ta", "thai": "en", "lao": "en",
    "emoji": "en",
}


from src.data.decompose import decompose_text, reconstruct_text, get_vocab_tokens, DECOMPOSE_GROUPS


def build_tokenizer(words: list[str]) -> LipiTokenizer:
    """Build vocab from training words."""
    chars = set(BASE_CHARS)
    for word in words:
        for ch in word:
            chars.add(ch)
    vocab = [BLANK_TOKEN] + sorted(chars)
    return LipiTokenizer(vocab=vocab, bigrams=set())


def build_script_tokenizers(
    labels: list[str], script_ids: torch.Tensor,
    active_scripts: list[str], active_groups: list[str],
    global_to_local_group: dict[int, int],
) -> tuple[list[list[LipiTokenizer]], list[list[int]], list[list[str]]]:
    """Build per-script tokenizers organized by group.

    Returns:
        group_tokenizers: group_tokenizers[g][s] = tokenizer for script s in group g
        group_script_vocab_sizes: vocab sizes per script per group
        group_script_names: script names per group
    """
    # Collect words per script (script_ids are GLOBAL from SCRIPT_TO_ID)
    script_words: dict[str, list[str]] = {s: [] for s in active_scripts}
    active_set = set(active_scripts)
    for label, sid in zip(labels, script_ids.tolist()):
        script = SCRIPTS[sid] if sid < len(SCRIPTS) else None
        if script and script in active_set:
            script_words[script].append(label)

    # Organize scripts by group
    group_tokenizers = []
    group_script_vocab_sizes = []
    group_script_names = []

    for g, group_name in enumerate(active_groups):
        scripts_in_group = [s for s in active_scripts if SCRIPT_TO_GROUP.get(s) == group_name]
        tokenizers = []
        vocab_sizes = []

        for script in scripts_in_group:
            words = script_words.get(script, [])
            if group_name in DECOMPOSE_GROUPS:
                vocab_tokens = get_vocab_tokens(group_name)
                all_tokens = set(BASE_CHARS) | set(vocab_tokens)
                for word in words:
                    for ch in decompose_text(word, group_name):
                        all_tokens.add(ch)
                vocab = [BLANK_TOKEN] + sorted(all_tokens)
                tok = LipiTokenizer(vocab=vocab, bigrams=set())
                tag = "decomposed"
            else:
                tok = build_tokenizer(words)
                tag = "chars"
            tokenizers.append(tok)
            vocab_sizes.append(tok.vocab_size)
            print(f"    Group {g} ({group_name}) / {script}: {tok.vocab_size} {tag}, {len(words)} words")

        group_tokenizers.append(tokenizers)
        group_script_vocab_sizes.append(vocab_sizes)
        group_script_names.append(scripts_in_group)

    return group_tokenizers, group_script_vocab_sizes, group_script_names


# ---------------------------------------------------------------------------
# CTC decode
# ---------------------------------------------------------------------------

def ctc_greedy_decode(logits: torch.Tensor, tokenizer: LipiTokenizer) -> list[str]:
    """Greedy CTC decode: argmax -> collapse repeats -> remove blank."""
    preds = logits.argmax(dim=-1)  # (B, T)
    results = []
    for seq in preds:
        chars = []
        prev = -1
        for t in seq.tolist():
            if t != prev and t != tokenizer.blank_id:
                chars.append(t)
            prev = t
        results.append(tokenizer.decode(chars))
    return results


# ---------------------------------------------------------------------------
# Data loading from shards
# ---------------------------------------------------------------------------

def load_shards(shard_dir: Path):
    """Load all shards in parallel. Returns (images, labels, script_ids, group_ids, meta)."""
    meta = torch.load(shard_dir / "metadata.pt", weights_only=False)
    shard_files = sorted(shard_dir.glob("shard_*.pt"))
    print(f"  {len(shard_files)} shards")

    all_imgs, all_labels, all_sids, all_gids = [], [], [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for shard in pool.map(lambda p: torch.load(p, weights_only=False), shard_files):
            all_imgs.append(shard["images"])
            all_labels.extend(shard["labels"])
            all_sids.append(shard["script_ids"])
            all_gids.append(shard["group_ids"])

    images = torch.cat(all_imgs)
    script_ids = torch.cat(all_sids)
    group_ids = torch.cat(all_gids)
    del all_imgs, all_sids, all_gids
    print(f"  {len(all_labels)} images loaded")
    return images, all_labels, script_ids, group_ids, meta


class MoEDataset(Dataset):
    """Pre-encoded dataset with group IDs, local script IDs, and string labels."""

    def __init__(self, images, targets, target_lens, group_ids, local_script_ids, labels):
        self.images = images
        self.targets = targets
        self.target_lens = target_lens
        self.group_ids = group_ids
        self.local_script_ids = local_script_ids
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return (self.images[idx], self.targets[idx], self.target_lens[idx],
                self.group_ids[idx], self.local_script_ids[idx], self.labels[idx])


def collate_moe(batch):
    """Stack pre-encoded batch."""
    imgs, targets, tgt_lens, gids, sids, labels = zip(*batch)
    return (torch.stack(imgs), torch.stack(targets), torch.stack(tgt_lens),
            torch.stack(gids), torch.stack(sids), list(labels))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class CUDAPrefetcher:
    """Overlap CPU→GPU data transfer with GPU computation using a separate stream."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self.next_batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = tuple(
                x.to(self.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                for x in self.next_batch
            )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                    group_tokenizers, ce_loss_fn, device, device_type, use_amp,
                    amp_dtype, epoch, total_epochs, grad_accum, log_interval,
                    lid1_weight=1.0):
    """Predicted routing with self-paced CTC: skip CTC on misrouted samples."""
    model.train()
    n_batches = 0

    # Accumulate losses on GPU — avoid .item() sync every batch
    ctc_loss_accum = torch.zeros(1, device=device)
    lid1_loss_accum = torch.zeros(1, device=device)
    total_loss_accum = torch.zeros(1, device=device)
    log_ctc_acc = torch.zeros(1, device=device)
    log_lid1_acc = torch.zeros(1, device=device)
    log_lid2_acc = torch.zeros(1, device=device)
    log_total_acc = torch.zeros(1, device=device)
    log_count = 0

    for batch_idx, (imgs, targets, tgt_lens, gids, sids, _labels) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        tgt_lens = tgt_lens.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)
        sids = sids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(imgs, group_ids=None, script_ids=None)
            logits = out["logits"]
            enc_lengths = out["lengths"]
            pred_gids = out["group_ids"]
            lid1_loss = ce_loss_fn(out["group_logits"], gids)

        # CTC only on correctly-routed samples (misrouted → wrong CTC head → skip)
        valid = (pred_gids == gids) & (tgt_lens <= enc_lengths) & (tgt_lens > 0)

        valid_logits = logits[valid]
        ctc_loss = torch.zeros(1, device=device)
        if valid_logits.shape[0] > 0:
            log_probs = valid_logits.float().log_softmax(dim=-1).permute(1, 0, 2)
            ctc_raw = F.ctc_loss(
                log_probs, targets[valid],
                enc_lengths[valid], tgt_lens[valid],
                blank=0, reduction="mean", zero_infinity=True,
            )
            ctc_loss = torch.clamp(ctc_raw, min=0.0, max=100.0)

        # LID-2 loss only on correctly-routed samples
        lid2_loss = torch.zeros(1, device=device)
        for _g, script_logits, group_mask in out["script_logits_per_group"]:
            routed_ok = valid[group_mask]
            sl = script_logits[routed_ok]
            if sl.shape[0] > 0:
                lid2_loss = lid2_loss + ce_loss_fn(sl, sids[group_mask][routed_ok])

        loss = ctc_loss + lid1_weight * lid1_loss.float() + lid2_loss.float()

        if grad_accum > 1:
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:
                scheduler.step()

        # Accumulate on GPU — no sync
        mult = float(grad_accum) if grad_accum > 1 else 1.0
        ctc_loss_accum += ctc_loss.detach()
        lid1_loss_accum += lid1_loss.detach()
        total_loss_accum += loss.detach() * mult
        log_ctc_acc += ctc_loss.detach()
        log_lid1_acc += lid1_loss.detach()
        log_lid2_acc += lid2_loss.detach()
        log_total_acc += loss.detach() * mult
        n_batches += 1
        log_count += 1

        # Only sync to CPU at log intervals
        if n_batches % log_interval == 0:
            avg_ctc = log_ctc_acc.item() / log_count
            avg_lid1 = log_lid1_acc.item() / log_count
            avg_lid2 = log_lid2_acc.item() / log_count
            avg_total = log_total_acc.item() / log_count
            lr = scheduler.get_last_lr()[0]
            steps = len(train_loader)
            print(f"  [{epoch}/{total_epochs}] batch {batch_idx+1}/{steps}  "
                  f"loss={avg_total:.4f} "
                  f"(ctc={avg_ctc:.4f} lid1={avg_lid1:.4f} lid2={avg_lid2:.4f})  "
                  f"lr={lr:.2e}")
            log_ctc_acc.zero_()
            log_lid1_acc.zero_()
            log_lid2_acc.zero_()
            log_total_acc.zero_()
            log_count = 0

    if n_batches == 0:
        print(f"Epoch {epoch}: no valid batches")
        return {}

    return {
        "ctc": ctc_loss_accum.item() / n_batches,
        "lid1": lid1_loss_accum.item() / n_batches,
        "total": total_loss_accum.item() / n_batches,
    }


@torch.no_grad()
def evaluate(model, val_loader, group_tokenizers, group_script_names,
             active_groups, device, device_type, use_amp, amp_dtype, max_batches=50):
    model.eval()
    n_groups = len(group_tokenizers)

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
        if batch_idx >= max_batches:
            break
        imgs, targets, tgt_lens, gids, sids, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)
        sids_dev = sids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(imgs, group_ids=None)

        # LID-1
        pred_gids = out["group_logits"].argmax(-1)
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
            # Only evaluate LID-2 on correctly LID-1-routed samples
            lid1_ok = (pred_gids[group_mask] == gids[group_mask])
            if lid1_ok.any():
                lid2_correct += (pred_scripts[lid1_ok] == true_scripts[lid1_ok]).sum().item()
                lid2_total += lid1_ok.sum().item()
                # Per-script LID-2
                for ls in range(script_logits.shape[-1]):
                    s_mask = (true_scripts == ls) & lid1_ok
                    if s_mask.any():
                        key = (g_idx, ls)
                        s_lid2_total[key] = s_lid2_total.get(key, 0) + s_mask.sum().item()
                        s_lid2_correct[key] = s_lid2_correct.get(key, 0) + (
                            pred_scripts[s_mask] == ls).sum().item()

        # CTC decode
        preds = out["logits"].float().cpu().argmax(dim=-1)
        pred_gids_cpu = pred_gids.cpu().tolist()
        gids_cpu = gids.cpu().tolist()
        sids_cpu = sids.cpu().tolist()
        for i, (label, pred_g, true_g, local_sid) in enumerate(
                zip(labels, pred_gids_cpu, gids_cpu, sids_cpu)):
            if pred_g >= n_groups:
                continue
            key = (true_g, local_sid)
            ref_s = str(label).strip().lower()

            # Only decode when LID-1 correct
            if pred_g != true_g:
                ctc_total += 1
                g_word_total[true_g] += 1
                total_chars += len(ref_s)
                s_word_total[key] = s_word_total.get(key, 0) + 1
                s_char_total[key] = s_char_total.get(key, 0) + len(ref_s)
                continue

            local_sid_safe = min(local_sid, len(group_tokenizers[true_g]) - 1)
            tok = group_tokenizers[true_g][local_sid_safe]
            seq = preds[i].tolist()
            chars = []
            prev = -1
            for t in seq:
                if t != prev and t != 0:
                    chars.append(t)
                prev = t
            raw_decoded = tok.decode(chars)
            group_name = active_groups[pred_g] if pred_g < len(active_groups) else ""
            if group_name in DECOMPOSE_GROUPS:
                raw_decoded = reconstruct_text(raw_decoded, group_name)
            dec_s = raw_decoded.strip().lower()

            ctc_total += 1
            g_word_total[true_g] += 1
            s_word_total[key] = s_word_total.get(key, 0) + 1
            matched = sum(1 for a, b in zip(dec_s, ref_s) if a == b)
            if dec_s == ref_s:
                ctc_correct += 1
                g_word_correct[true_g] += 1
                s_word_correct[key] = s_word_correct.get(key, 0) + 1
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
    print(f"  LID-1: {lid1_acc:.1f}%  |  LID-2: {lid2_acc:.1f}%  |  Word: {ctc_acc:.1f}%  |  Char: {char_acc:.1f}%")
    print(f"  Per group/script:")
    for g in range(n_groups):
        gl = g_lid_total[g]
        lid_g = 100 * g_lid_correct[g] / max(gl, 1)
        word_g = 100 * g_word_correct[g] / max(g_word_total[g], 1)
        char_g = 100 * g_char_correct[g] / max(g_char_total[g], 1)
        name = active_groups[g] if g < len(active_groups) else f"group{g}"
        print(f"    {name:18s} LID1:{lid_g:5.1f}%  Word:{word_g:5.1f}%  Char:{char_g:5.1f}%")
        # Per-script within group
        scripts = group_script_names[g] if g < len(group_script_names) else []
        for ls, sname in enumerate(scripts):
            key = (g, ls)
            sw = s_word_total.get(key, 0)
            if sw == 0:
                continue
            sw_correct = s_word_correct.get(key, 0)
            sc_correct = s_char_correct.get(key, 0)
            sc_total = s_char_total.get(key, 0)
            word_s = 100 * sw_correct / max(sw, 1)
            char_s = 100 * sc_correct / max(sc_total, 1)
            lid2_str = ""
            if key in s_lid2_total:
                lid2_s = 100 * s_lid2_correct.get(key, 0) / max(s_lid2_total[key], 1)
                lid2_str = f"  LID2:{lid2_s:5.1f}%"
            print(f"      {sname:16s} Word:{word_s:5.1f}%  Char:{char_s:5.1f}%{lid2_str}")
    return {"lid1_acc": lid1_acc, "lid2_acc": lid2_acc, "word_acc": ctc_acc, "char_acc": char_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Lipi MoE Encoder")
    parser.add_argument("--data", type=str, required=True,
                        help="Shard directory from generate_data.py")
    parser.add_argument("--scripts", type=str, default="all",
                        help="Comma-separated scripts, or 'all'")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="checkpoints/moe")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    # Model
    parser.add_argument("--stem-depth", type=int, default=3)
    parser.add_argument("--shared-dim", type=int, default=288)
    parser.add_argument("--shared-blocks-4x4", type=int, default=8)
    parser.add_argument("--shared-blocks-4x16", type=int, default=4)
    parser.add_argument("--stage1-dim", type=int, default=288)
    parser.add_argument("--stage1-blocks", type=int, default=12)
    parser.add_argument("--stage2-dim", type=int, default=576)
    parser.add_argument("--stage2-blocks", type=int, default=8)
    parser.add_argument("--head-hidden", type=int, default=384)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (saves ~5-10GB VRAM)")
    parser.add_argument("--lid1-weight", type=float, default=1.0,
                        help="LID-1 loss weight")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    device_type = device.type
    print(f"Device: {device}")

    # Load shards
    data_path = Path(args.data)
    print(f"\nLoading data from {data_path}/...")
    images, labels, script_ids_global, group_ids_global, meta = load_shards(data_path)
    active_scripts = meta["active_scripts"]

    # Filter scripts if specified
    if args.scripts != "all":
        selected = set(s.strip() for s in args.scripts.split(","))
        active_scripts = [s for s in active_scripts if s in selected]
    print(f"Scripts: {active_scripts}")

    # Determine active groups + remap group IDs to 0..N-1
    active_groups = []
    seen = set()
    for s in active_scripts:
        g = SCRIPT_TO_GROUP.get(s)
        if g and g not in seen:
            active_groups.append(g)
            seen.add(g)

    global_to_local_group = {}
    for local_id, gname in enumerate(active_groups):
        global_to_local_group[GROUP_TO_ID[gname]] = local_id

    n_groups = len(active_groups)
    print(f"Groups: {n_groups} -> {active_groups}")

    # Remap group IDs
    group_ids = group_ids_global.clone()
    for gid, lid in global_to_local_group.items():
        group_ids[group_ids_global == gid] = lid

    # Build per-script local IDs within each group
    # local_script_ids[i] = index of sample i's script within its group (0, 1, 2, ...)
    local_script_ids = torch.zeros_like(group_ids)
    group_script_list = []  # group_script_list[g] = [script_name, ...]
    for g, group_name in enumerate(active_groups):
        scripts_in_group = [s for s in active_scripts if SCRIPT_TO_GROUP.get(s) == group_name]
        group_script_list.append(scripts_in_group)
        for local_s, script in enumerate(scripts_in_group):
            global_sid = SCRIPT_TO_ID[script]
            mask = (script_ids_global == global_sid)
            local_script_ids[mask] = local_s
    for g, scripts in enumerate(group_script_list):
        print(f"  Group {g} ({active_groups[g]}): scripts {scripts}")

    # Build per-script tokenizers
    print("\nBuilding per-script tokenizers...")
    group_tokenizers, group_script_vocab_sizes, group_script_names = build_script_tokenizers(
        labels, script_ids_global, active_scripts, active_groups, global_to_local_group)
    print(f"  Per-group vocab sizes: {group_script_vocab_sizes}")

    # Pre-encode labels using each sample's script tokenizer
    print("Pre-encoding labels...")
    max_len = 0
    encoded = []
    for i, (label, gid, lsid) in enumerate(zip(labels, group_ids.tolist(), local_script_ids.tolist())):
        group_name = active_groups[gid] if gid < len(active_groups) else ""
        if group_name in DECOMPOSE_GROUPS:
            label_tokens = decompose_text(label, group_name)
        else:
            label_tokens = label
        tok = group_tokenizers[gid][lsid]
        ids = tok.encode(label_tokens)
        encoded.append(ids)
        max_len = max(max_len, len(ids))

    if max_len == 0:
        max_len = 1
    target_tensor = torch.zeros(len(encoded), max_len, dtype=torch.long)
    target_len_tensor = torch.zeros(len(encoded), dtype=torch.long)
    for i, ids in enumerate(encoded):
        target_len_tensor[i] = len(ids)
        if ids:
            target_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    print(f"  Max label length: {max_len}")

    # Pre-filter: remove samples with empty labels or labels too long for CTC.
    # This eliminates GPU→CPU sync points in the training loop.
    max_enc_len = images.shape[3] // 4  # width // stride
    valid = (target_len_tensor > 0) & (target_len_tensor <= max_enc_len)
    n_filtered = (~valid).sum().item()
    if n_filtered > 0:
        keep = valid.nonzero(as_tuple=True)[0]
        images = images[keep]
        target_tensor = target_tensor[keep]
        target_len_tensor = target_len_tensor[keep]
        group_ids = group_ids[keep]
        local_script_ids = local_script_ids[keep]
        labels = [labels[i] for i in keep.tolist()]
        print(f"  Filtered {n_filtered} samples (empty or too long for CTC)")

    # Dataset + split
    dataset = MoEDataset(images, target_tensor, target_len_tensor, group_ids, local_script_ids, labels)
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"Train: {n_train}, Val: {n_val}")

    # DataLoaders — data in RAM, no workers needed
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_moe, pin_memory=(device_type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_moe, pin_memory=(device_type == "cuda"))

    # Model
    model = LipiMoEEncoder(
        stem_depth=args.stem_depth,
        shared_dim=args.shared_dim,
        shared_blocks_4x4=args.shared_blocks_4x4,
        shared_blocks_4x16=args.shared_blocks_4x16,
        stage1_dim=args.stage1_dim,
        stage1_blocks=args.stage1_blocks,
        stage2_dim=args.stage2_dim,
        stage2_blocks=args.stage2_blocks,
        num_groups=n_groups,
        group_script_vocab_sizes=group_script_vocab_sizes,
        group_script_names=group_script_names,
        head_hidden=args.head_hidden,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params / 1e6:.1f}M params ({n_groups} groups)")

    # torch.compile — fuses kernels, reduces launch overhead
    # Disabled by default: uses ~5-10GB extra VRAM on 575M model
    if device_type == "cuda" and not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("  Done.")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # AMP
    use_amp = device_type in ("cuda", "mps")
    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_scaler = amp_dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        print(f"AMP: {amp_dtype}")
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False)

    # Scheduler
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(steps_per_epoch, total_steps // 10)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # Resume
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed at epoch {start_epoch}")

    # Training
    ce_loss_fn = nn.CrossEntropyLoss()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    eff_batch = args.batch_size * args.grad_accum
    print(f"\n{'=' * 60}")
    print(f"TRAINING: epochs {start_epoch}-{args.epochs}, lr={args.lr}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} = {eff_batch} effective")
    print(f"  Losses: CTC x1.0 + LID1 x{args.lid1_weight}")
    print(f"  Routing: predicted (skip CTC on misroutes)")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            group_tokenizers, ce_loss_fn, device, device_type, use_amp, amp_dtype,
            epoch, args.epochs, args.grad_accum, args.log_interval,
            lid1_weight=args.lid1_weight)

        elapsed = time.time() - t0

        if metrics:
            print(f"\nEpoch {epoch}/{args.epochs}: "
                  f"ctc={metrics['ctc']:.4f} lid1={metrics['lid1']:.4f}  "
                  f"time={elapsed:.0f}s")

        # Save
        ckpt_path = save_dir / f"moe_epoch{epoch}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # Eval
        print(f"\n  Eval epoch {epoch}:")
        evaluate(model, val_loader, group_tokenizers, group_script_names,
                 active_groups, device, device_type, use_amp, amp_dtype)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
