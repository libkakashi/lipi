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
from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID, GROUPS
from src.data.bigrams import LipiTokenizer, SCRIPT_CHARSETS, BASE_CHARS, BLANK_TOKEN


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

SCRIPT_TO_LANG = {
    "latin": "en", "cyrillic": "en", "greek": "en",
    "arabic": "ur", "hebrew": "en", "cjk": "en", "korean": "en",
    "devanagari": "hi", "gurmukhi": "pa", "gujarati": "gu",
    "bengali": "bn_as", "kannada": "kn", "telugu": "te",
    "malayalam": "ml", "tamil": "ta", "thai": "en", "lao": "en",
    "emoji": "en",
}


def build_tokenizer(scripts: list[str], words: list[str]) -> LipiTokenizer:
    """Build vocab from charset tables + actual training words."""
    chars = set(BASE_CHARS)
    for script in scripts:
        lang = SCRIPT_TO_LANG.get(script, "en")
        for ch in SCRIPT_CHARSETS.get(lang, []):
            chars.add(ch)
    for word in words:
        for ch in word:
            chars.add(ch)
    vocab = [BLANK_TOKEN] + sorted(chars)
    return LipiTokenizer(vocab=vocab, bigrams=set())


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

def load_shards(shard_dir: Path) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """Load all shards in parallel. Returns (images, labels, script_ids, group_ids)."""
    meta = torch.load(shard_dir / "metadata.pt", weights_only=False)
    shard_files = sorted(shard_dir.glob("shard_*.pt"))
    print(f"  {len(shard_files)} shards")

    all_imgs, all_labels, all_gids = [], [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for shard in pool.map(lambda p: torch.load(p, weights_only=False), shard_files):
            all_imgs.append(shard["images"])
            all_labels.extend(shard["labels"])
            all_gids.append(shard["group_ids"])

    images = torch.cat(all_imgs)
    group_ids = torch.cat(all_gids)
    del all_imgs, all_gids
    print(f"  {len(all_labels)} images loaded")
    return images, all_labels, group_ids, meta


class MoEDataset(Dataset):
    """Pre-encoded dataset: images + CTC targets + group IDs + string labels."""

    def __init__(self, images, targets, target_lens, group_ids, labels):
        self.images = images
        self.targets = targets
        self.target_lens = target_lens
        self.group_ids = group_ids
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return (self.images[idx], self.targets[idx], self.target_lens[idx],
                self.group_ids[idx], self.labels[idx])


def collate_moe(batch):
    """Stack pre-encoded batch. All images same size from shards."""
    imgs, targets, tgt_lens, gids, labels = zip(*batch)
    return (torch.stack(imgs), torch.stack(targets), torch.stack(tgt_lens),
            torch.stack(gids), list(labels))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                    tokenizer, ce_loss_fn, device, device_type, use_amp,
                    amp_dtype, epoch, total_epochs, grad_accum, log_interval):
    model.train()
    ctc_loss_sum = lid1_loss_sum = total_loss_sum = 0.0
    n_batches = 0

    for batch_idx, (imgs, targets, tgt_lens, gids, _labels) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        tgt_lens = tgt_lens.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)

        # Skip empty labels
        if (tgt_lens == 0).any():
            continue

        # Forward
        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(imgs, group_ids=gids)
            logits = out["logits"]
            enc_lengths = out["lengths"]
            lid1_loss = ce_loss_fn(out["group_logits"], gids)

        # CTC constraint: input_length >= target_length
        if (tgt_lens > enc_lengths).any():
            continue

        # CTC loss in float32
        log_probs = logits.float().log_softmax(dim=-1).permute(1, 0, 2)
        ctc_loss = F.ctc_loss(
            log_probs, targets, enc_lengths, tgt_lens,
            blank=tokenizer.blank_id, reduction="mean", zero_infinity=True,
        )

        if torch.isinf(ctc_loss) or torch.isnan(ctc_loss):
            continue

        loss = ctc_loss + 1.0 * lid1_loss.float()

        if grad_accum > 1:
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scaler.get_scale() >= old_scale:
                scheduler.step()

        ctc_loss_sum += ctc_loss.item()
        lid1_loss_sum += lid1_loss.item()
        total_loss_sum += loss.item() * (grad_accum if grad_accum > 1 else 1)
        n_batches += 1

        if n_batches % log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            steps = len(train_loader)
            print(f"  [{epoch}/{total_epochs}] batch {batch_idx+1}/{steps}  "
                  f"loss={loss.item()*(grad_accum if grad_accum>1 else 1):.4f} "
                  f"(ctc={ctc_loss.item():.4f} lid1={lid1_loss.item():.4f})  "
                  f"lr={lr:.2e}")

    if n_batches == 0:
        print(f"Epoch {epoch}: no valid batches")
        return {}

    return {
        "ctc": ctc_loss_sum / n_batches,
        "lid1": lid1_loss_sum / n_batches,
        "total": total_loss_sum / n_batches,
    }


@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device, device_type, use_amp, amp_dtype, pre_encoded):
    model.eval()
    lid_correct = lid_total = ctc_correct = ctc_total = 0

    for batch in val_loader:
        imgs, targets, tgt_lens, gids, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(imgs, group_ids=None)

        # LID accuracy
        pred_gids = out["group_logits"].argmax(-1)
        lid_correct += (pred_gids == gids).sum().item()
        lid_total += gids.shape[0]

        # CTC accuracy
        decoded = ctc_greedy_decode(out["logits"].float().cpu(), tokenizer)
        for dec, label in zip(decoded, labels):
            ctc_total += 1
            if dec.strip().lower() == str(label).strip().lower():
                ctc_correct += 1

    lid_acc = 100 * lid_correct / max(lid_total, 1)
    ctc_acc = 100 * ctc_correct / max(ctc_total, 1)
    print(f"  LID-1 accuracy: {lid_acc:.1f}%")
    print(f"  CTC  accuracy:  {ctc_acc:.1f}% ({ctc_correct}/{ctc_total})")
    return {"lid1_acc": lid_acc, "ctc_acc": ctc_acc}


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
    parser.add_argument("--shared-blocks", type=int, default=3)
    parser.add_argument("--shared-dim", type=int, default=288)
    parser.add_argument("--stage1-dim", type=int, default=288)
    parser.add_argument("--stage1-blocks", type=int, default=5)
    parser.add_argument("--stage2-dim", type=int, default=576)
    parser.add_argument("--stage2-blocks", type=int, default=9)
    parser.add_argument("--head-hidden", type=int, default=246)
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
    images, labels, group_ids, meta = load_shards(data_path)
    active_scripts = meta["active_scripts"]

    # Filter scripts if specified
    if args.scripts != "all":
        selected = set(s.strip() for s in args.scripts.split(","))
        active_scripts = [s for s in active_scripts if s in selected]
    print(f"Scripts: {active_scripts}")

    # Determine active groups + remap IDs to 0..N-1
    active_groups = []
    seen = set()
    for s in active_scripts:
        g = SCRIPT_TO_GROUP.get(s)
        if g and g not in seen:
            active_groups.append(g)
            seen.add(g)

    global_to_local = {}
    for local_id, gname in enumerate(active_groups):
        global_to_local[GROUP_TO_ID[gname]] = local_id

    n_groups = len(active_groups)
    print(f"Groups: {n_groups} -> {active_groups}")
    print(f"  Remap: {global_to_local}")

    # Remap group IDs
    remapped = group_ids.clone()
    for gid, lid in global_to_local.items():
        remapped[group_ids == gid] = lid
    group_ids = remapped

    # Build tokenizer from actual training words
    tokenizer = build_tokenizer(active_scripts, labels)
    print(f"Vocab: {tokenizer.vocab_size}")

    # Pre-encode labels
    print("Pre-encoding labels...")
    encoded = [tokenizer.encode(w) for w in labels]
    max_len = max(len(e) for e in encoded) if encoded else 1
    target_tensor = torch.zeros(len(encoded), max_len, dtype=torch.long)
    target_len_tensor = torch.zeros(len(encoded), dtype=torch.long)
    for i, ids in enumerate(encoded):
        target_len_tensor[i] = len(ids)
        if ids:
            target_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    print(f"  Max label length: {max_len}")

    # Dataset + split
    dataset = MoEDataset(images, target_tensor, target_len_tensor, group_ids, labels)
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"Train: {n_train}, Val: {n_val}")

    # DataLoaders (data in RAM — no workers needed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_moe, pin_memory=(device_type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_moe, pin_memory=(device_type == "cuda"))

    # Model
    model = LipiMoEEncoder(
        stem_depth=args.stem_depth,
        shared_dim=args.shared_dim,
        shared_blocks=args.shared_blocks,
        stage1_dim=args.stage1_dim,
        stage1_blocks=args.stage1_blocks,
        stage2_dim=args.stage2_dim,
        stage2_blocks=args.stage2_blocks,
        num_groups=n_groups,
        vocab_size=tokenizer.vocab_size,
        head_hidden=args.head_hidden,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params / 1e6:.1f}M params ({n_groups} groups)")

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
    print(f"  Losses: CTC x1.0 + LID1 x1.0")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            tokenizer, ce_loss_fn, device, device_type, use_amp, amp_dtype,
            epoch, args.epochs, args.grad_accum, args.log_interval)

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
        evaluate(model, val_loader, tokenizer, device, device_type, use_amp, amp_dtype, True)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
