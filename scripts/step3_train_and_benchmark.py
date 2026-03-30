#!/usr/bin/env python3
"""
Step 3: Train 12.5M backbone on MJSynth, evaluate on STR benchmarks.

Trains Phase 1 CTC on a subset of MJSynth (default 1M crops, 3 epochs).
Evaluates on IIIT5K, SVT, IC13, IC15, SVTP, CUTE80.

Pass criteria:
  IIIT5K: >82% (good: >86%)
  IC13:   >88% (good: >92%)
  IC15:   >65% (good: >72%)

Usage:
    python scripts/step3_train_and_benchmark.py \
        --train_dir training_data/external/train/synth/MJ/MJ_train \
        --test_dir training_data/external/test \
        --epochs 3 --max_samples 1000000
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiEncoder
from src.training.foundation_trainer import CTCHead
from src.training.loss import ctc_loss
from src.data.bigrams import LipiTokenizer
from src.data.parseq_lmdb import PARSeqLMDB, discover_parseq_structure
from src.data.dataset import collate_ocr


def ctc_greedy_decode(logits, tokenizer):
    preds = logits.argmax(dim=-1)
    results = []
    for i in range(preds.shape[0]):
        p = preds[i].tolist()
        collapsed = [p[0]] + [p[j] for j in range(1, len(p)) if p[j] != p[j-1]]
        collapsed = [x for x in collapsed if x != 0]
        results.append(tokenizer.decode(collapsed))
    return results


def evaluate(encoder, ctc_head, tokenizer, test_dir, device, batch_size=64):
    """Evaluate on all available benchmarks in test_dir."""
    encoder.eval()
    ctc_head.eval()

    benchmarks = discover_parseq_structure(test_dir)
    if not benchmarks:
        print(f"  No benchmarks found in {test_dir}")
        return {}

    results = {}
    total_correct = 0
    total_total = 0

    print(f"\n  {'Benchmark':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*55}")

    for name, path in sorted(benchmarks.items()):
        try:
            dataset = PARSeqLMDB(path)
        except Exception as e:
            print(f"  {name:<25} {'ERROR':>8} — {e}")
            continue

        if len(dataset) == 0:
            continue

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ocr,
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_imgs, batch_labels, widths in loader:
                batch_imgs = batch_imgs.to(device)
                features, enc_lengths = encoder(batch_imgs)
                logits = ctc_head(features)
                decoded = ctc_greedy_decode(logits, tokenizer)

                for d, l in zip(decoded, batch_labels):
                    # Case-insensitive comparison (standard for STR benchmarks)
                    if d.lower() == l.lower():
                        correct += 1
                    total += 1

        acc = correct / max(total, 1) * 100
        results[name] = acc
        total_correct += correct
        total_total += total
        print(f"  {name:<25} {correct:>8} {total:>8} {acc:>9.1f}%")

        dataset.close()

    if total_total > 0:
        avg = total_correct / total_total * 100
        print(f"  {'-'*55}")
        print(f"  {'AVERAGE':<25} {total_correct:>8} {total_total:>8} {avg:>9.1f}%")
        results["AVERAGE"] = avg

    return results


def main():
    parser = argparse.ArgumentParser(description="Step 3: Train and benchmark")
    parser.add_argument("--train_dir", type=str, required=True, help="MJSynth LMDB path")
    parser.add_argument("--test_dir", type=str, required=True, help="Test benchmarks directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=1000000, help="Limit training samples")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="checkpoints/step3")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load training data
    print(f"Loading training data from {args.train_dir}...")
    train_dataset = PARSeqLMDB(args.train_dir)
    print(f"  Full dataset: {len(train_dataset)} samples")

    # Load raw bytes into RAM (fast, no decoding), decode per-batch during training
    max_load = args.max_samples if args.max_samples and args.max_samples < len(train_dataset) else None
    train_dataset.preload_raw(max_samples=max_load)

    # Build tokenizer
    tokenizer = LipiTokenizer.build_character_level("en")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Build model
    encoder = LipiEncoder().to(device)
    ctc_head = CTCHead(encoder.output_dim, tokenizer.vocab_size).to(device)

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder params: {total_params/1e6:.2f}M")

    params = list(encoder.parameters()) + list(ctc_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ocr,
        drop_last=True, num_workers=16, pin_memory=True, persistent_workers=True,
    )

    # LR scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1,
    )

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        ctc_head.load_state_dict(ckpt["ctc_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed at epoch {start_epoch}, loss was {ckpt.get('loss', '?')}")

    # Training
    print(f"\n{'='*60}")
    print(f"TRAINING: epochs {start_epoch}-{args.epochs}, lr={args.lr}, batch={args.batch_size}")
    print(f"{'='*60}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        ctc_head.train()

        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_imgs, batch_labels, widths in pbar:
            batch_imgs = batch_imgs.to(device, non_blocking=True)

            target_ids = [tokenizer.encode(l) for l in batch_labels]
            target_lengths = torch.tensor(
                [len(ids) for ids in target_ids], device=device
            )

            # Skip batches where any label is empty
            if (target_lengths == 0).any():
                continue

            max_tgt = target_lengths.max().item()
            targets = torch.zeros(len(batch_labels), max_tgt, dtype=torch.long, device=device)
            for i, ids in enumerate(target_ids):
                if len(ids) > 0:
                    targets[i, :len(ids)] = torch.tensor(ids, device=device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                features, enc_lengths = encoder(batch_imgs)
                logits = ctc_head(features)

            loss = ctc_loss(logits.float(), targets, enc_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint every 10% of epoch
            total_batches = len(train_loader)
            save_every = max(total_batches // 10, 1)
            if n_batches % save_every == 0:
                pct = n_batches * 100 // total_batches
                ckpt_path = save_dir / f"step3_e{epoch}_p{pct}.pt"
                torch.save({
                    "encoder": encoder.state_dict(),
                    "ctc_head": ctc_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "step": n_batches,
                    "loss": loss.item(),
                }, ckpt_path)
                print(f"\n  Checkpoint saved: {ckpt_path}")

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - start
        print(f"\nEpoch {epoch}: avg_loss={avg_loss:.4f}, time={elapsed:.0f}s")

        # Save end-of-epoch checkpoint
        ckpt_path = save_dir / f"step3_epoch{epoch}.pt"
        torch.save({
            "encoder": encoder.state_dict(),
            "ctc_head": ctc_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": n_batches,
            "loss": avg_loss,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # Evaluate after each epoch
        print(f"\n  Evaluation after epoch {epoch}:")
        results = evaluate(encoder, ctc_head, tokenizer, args.test_dir, device)

    # Final summary
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Final loss: {avg_loss:.4f}")
    print(f"  Checkpoints: {save_dir}")

    # Check pass criteria
    if results:
        iiit = results.get("IIIT5k_3000", results.get("IIIT5K", 0))
        ic13 = results.get("IC13_1015", results.get("IC13", 0))
        ic15 = results.get("IC15_2077", results.get("IC15", 0))

        print(f"\n  Checkpoint 2 Criteria:")
        print(f"    IIIT5K: {iiit:.1f}% {'PASS' if iiit > 82 else 'FAIL'} (min: 82%, good: 86%)")
        print(f"    IC13:   {ic13:.1f}% {'PASS' if ic13 > 88 else 'FAIL'} (min: 88%, good: 92%)")
        print(f"    IC15:   {ic15:.1f}% {'PASS' if ic15 > 65 else 'FAIL'} (min: 65%, good: 72%)")


if __name__ == "__main__":
    main()
