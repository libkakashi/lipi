#!/usr/bin/env python3
"""
Step 4: Bigram vs Character-Level Comparison (Checkpoint 3).

Using the backbone from Step 3, train TWO RNN-T heads on the same data:
  Head A: pure character vocab (~83 tokens for English)
  Head B: character + bigrams (~133-233 tokens)

Compare accuracy on all benchmarks. This decides whether bigrams help.

Pass criteria: bigram accuracy >= character accuracy.
Even if equal, bigrams win on speed (fewer decode steps).

Usage:
    python scripts/step4_bigram_vs_char.py \
        --backbone checkpoints/step3/step3_epoch3.pt \
        --train_dir training_data/external/train/synth/MJ/train \
        --test_dir training_data/external/test \
        --epochs 2 --max_samples 500000
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import time
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.decode import greedy_decode
from src.training.loss import rnnt_loss
from src.training.foundation_trainer import FoundationTrainer
from src.data.bigrams import LipiTokenizer
from src.data.parseq_lmdb import PARSeqLMDB, discover_parseq_structure
from src.data.dataset import collate_ocr


def evaluate_rnnt(encoder, pred_net, joint_net, tokenizer, test_dir, device, batch_size=64):
    """Evaluate RNN-T model on benchmarks."""
    encoder.eval()
    pred_net.eval()
    joint_net.eval()

    benchmarks = discover_parseq_structure(test_dir)
    results = {}

    # Only eval on the standard 6 benchmarks
    standard = ["IIIT5k", "SVT", "IC13_1015", "IC15_2077", "SVTP", "CUTE80"]

    print(f"    {'Benchmark':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"    {'-'*50}")

    total_correct = 0
    total_total = 0

    for name in standard:
        path = benchmarks.get(name)
        if not path:
            continue

        dataset = PARSeqLMDB(path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_ocr)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_imgs, batch_labels, widths in loader:
                batch_imgs = batch_imgs.to(device)
                features, _ = encoder(batch_imgs)
                decoded_ids = greedy_decode(features, pred_net, joint_net, max_tokens=25)

                for pred_ids, label in zip(decoded_ids, batch_labels):
                    pred_text = tokenizer.decode(pred_ids)
                    if pred_text.lower() == label.lower():
                        correct += 1
                    total += 1

        acc = correct / max(total, 1) * 100
        results[name] = acc
        total_correct += correct
        total_total += total
        print(f"    {name:<20} {correct:>8} {total:>8} {acc:>9.1f}%")

        dataset.close()

    if total_total > 0:
        avg = total_correct / total_total * 100
        print(f"    {'-'*50}")
        print(f"    {'AVERAGE':<20} {total_correct:>8} {total_total:>8} {avg:>9.1f}%")
        results["AVERAGE"] = avg

    return results


def train_rnnt_head(
    encoder, tokenizer, train_dataset, test_dir,
    device, epochs, batch_size, lr, label,
):
    """Train an RNN-T head on a frozen encoder and evaluate."""
    print(f"\n  Training {label} head (vocab_size={tokenizer.vocab_size})...")

    pred_net = PredictionNetwork(
        vocab_size=tokenizer.vocab_size, embed_dim=128, hidden_dim=128
    ).to(device)
    joint_net = JointNetwork(
        enc_dim=encoder.output_dim, pred_dim=128, joint_dim=256,
        vocab_size=tokenizer.vocab_size,
    ).to(device)

    head_params = list(pred_net.parameters()) + list(joint_net.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=0.01)

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ocr,
        drop_last=True, num_workers=16, pin_memory=True, persistent_workers=True,
    )

    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    encoder.eval()  # Frozen
    pred_net.train()
    joint_net.train()

    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(loader, desc=f"  {label} Epoch {epoch}/{epochs}")
        for batch_imgs, batch_labels, widths in pbar:
            batch_imgs = batch_imgs.to(device, non_blocking=True)

            target_ids = [tokenizer.encode(l) for l in batch_labels]
            target_lengths = torch.tensor([len(ids) for ids in target_ids], device=device)

            if (target_lengths == 0).any():
                continue

            max_tgt = target_lengths.max().item()
            targets = torch.zeros(len(batch_labels), max_tgt, dtype=torch.long, device=device)
            for i, ids in enumerate(target_ids):
                if len(ids) > 0:
                    targets[i, :len(ids)] = torch.tensor(ids, device=device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                with torch.no_grad():
                    features, _ = encoder(batch_imgs)

                B = targets.shape[0]
                T = features.shape[1]
                enc_lengths = torch.full((B,), T, dtype=torch.long, device=device)

                blank = torch.zeros(B, 1, dtype=torch.long, device=device)
                pred_input = torch.cat([blank, targets], dim=1)
                pred_out, _ = pred_net(pred_input)

                logits = joint_net(features.unsqueeze(2), pred_out.unsqueeze(1))

            loss = rnnt_loss(logits.float(), targets, enc_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only step scheduler if optimizer actually stepped
            if scaler.get_scale() >= old_scale:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  {label} Epoch {epoch}: avg_loss={avg_loss:.4f}")

    elapsed = time.time() - start
    print(f"  {label} training time: {elapsed:.0f}s")

    # Evaluate
    print(f"\n  {label} Evaluation:")
    results = evaluate_rnnt(encoder, pred_net, joint_net, tokenizer, test_dir, device)

    return results, pred_net, joint_net


def main():
    parser = argparse.ArgumentParser(description="Step 4: Bigram vs Character comparison")
    parser.add_argument("--backbone", type=str, required=True, help="Step 3 backbone checkpoint")
    parser.add_argument("--train_dir", type=str, required=True, help="MJSynth LMDB path")
    parser.add_argument("--test_dir", type=str, required=True, help="Test benchmarks directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load frozen backbone
    print(f"Loading backbone from {args.backbone}...")
    encoder = FoundationTrainer.load_backbone(args.backbone, device="cpu").to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"  Backbone: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params (frozen)")

    # Load training data with raw byte preload
    train_full = PARSeqLMDB(args.train_dir)
    max_load = args.max_samples if args.max_samples < len(train_full) else None
    train_full.preload_raw(max_samples=max_load)
    train_dataset = train_full
    print(f"  Training data: {len(train_dataset)} samples")

    # Build tokenizers
    # Character: 95 ASCII + blank = 96 tokens
    # Bigram: 95 ASCII + 75 curated bigrams + blank = 171 tokens
    char_tokenizer = LipiTokenizer.build_character_level("en")
    bigram_tokenizer = LipiTokenizer.build_with_curated_bigrams("en")

    print(f"  Character vocab: {char_tokenizer.vocab_size} tokens")
    print(f"  Bigram vocab: {bigram_tokenizer.vocab_size} tokens")

    # Compare average tokens per word
    test_words = ["Section", "Court", "Judgment", "the", "CUSTOMERS", "12345"]
    print(f"\n  Token comparison:")
    for word in test_words:
        char_ids = char_tokenizer.encode(word)
        bigram_ids = bigram_tokenizer.encode(word)
        reduction = (1 - len(bigram_ids) / len(char_ids)) * 100 if char_ids else 0
        print(f"    {word:15s}: char={len(char_ids)} tokens, bigram={len(bigram_ids)} tokens ({reduction:+.0f}%)")

    # Train Head A: character-level
    print(f"\n{'='*60}")
    print("HEAD A: CHARACTER-LEVEL")
    print(f"{'='*60}")
    char_results, _, _ = train_rnnt_head(
        encoder, char_tokenizer, train_dataset, args.test_dir,
        device, args.epochs, args.batch_size, args.lr, "CHAR",
    )

    # Train Head B: character + bigrams
    print(f"\n{'='*60}")
    print("HEAD B: CHARACTER + BIGRAMS")
    print(f"{'='*60}")
    bigram_results, _, _ = train_rnnt_head(
        encoder, bigram_tokenizer, train_dataset, args.test_dir,
        device, args.epochs, args.batch_size, args.lr, "BIGRAM",
    )

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: BIGRAM vs CHARACTER")
    print(f"{'='*60}")

    print(f"\n  {'Benchmark':<20} {'Char':>10} {'Bigram':>10} {'Delta':>10}")
    print(f"  {'-'*55}")

    for name in ["IIIT5k", "SVT", "IC13_1015", "IC15_2077", "SVTP", "CUTE80", "AVERAGE"]:
        c = char_results.get(name, 0)
        b = bigram_results.get(name, 0)
        delta = b - c
        marker = "+" if delta > 0 else ""
        print(f"  {name:<20} {c:>9.1f}% {b:>9.1f}% {marker}{delta:>8.1f}%")

    # Decision
    char_avg = char_results.get("AVERAGE", 0)
    bigram_avg = bigram_results.get("AVERAGE", 0)

    print(f"\n  Decision: ", end="")
    if bigram_avg >= char_avg:
        print(f"USE BIGRAMS (bigram {bigram_avg:.1f}% >= char {char_avg:.1f}%)")
        print(f"  Bigrams give equal or better accuracy with fewer decode steps.")
    else:
        gap = char_avg - bigram_avg
        if gap < 0.5:
            print(f"USE BIGRAMS (gap is only {gap:.1f}%, bigrams win on speed)")
        else:
            print(f"USE CHARACTERS (char {char_avg:.1f}% > bigram {bigram_avg:.1f}% by {gap:.1f}%)")
            print(f"  Consider dropping bigrams and using pure character-level.")


if __name__ == "__main__":
    main()
