#!/usr/bin/env python3
"""
Step 4b: CTC vs RNN-T × Characters vs Bigrams.

Tests 4 decoder/vocab combinations on a frozen backbone:
  1. CTC + characters (baseline, fastest)
  2. CTC + bigrams (fast + potential accuracy boost)
  3. RNN-T + characters (proven accurate)
  4. RNN-T + bigrams (proven worse for RNN-T)

Usage:
    python scripts/step4b_ctc_vs_rnnt.py \
        --backbone checkpoints/step5/step5_epoch3.pt \
        --train_dir training_data/external/train/synth/MJ/train \
        --test_dir training_data/external/test \
        --epochs 3 --max_samples 1000000 --batch_size 600
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.decode import greedy_decode
from src.training.foundation_trainer import FoundationTrainer, CTCHead
from src.training.loss import ctc_loss, rnnt_loss
from src.data.bigrams import LipiTokenizer
from src.data.parseq_lmdb import PARSeqLMDB, discover_parseq_structure
from src.data.dataset import collate_ocr
from src.data.width_sampler import WidthBucketSampler, get_image_widths


def ctc_greedy_decode(logits, tokenizer):
    """CTC greedy: argmax, collapse repeats, remove blanks."""
    preds = logits.argmax(dim=-1)
    results = []
    for i in range(preds.shape[0]):
        p = preds[i].tolist()
        collapsed = [p[0]] + [p[j] for j in range(1, len(p)) if p[j] != p[j - 1]]
        collapsed = [x for x in collapsed if x != 0]
        results.append(tokenizer.decode(collapsed))
    return results


def evaluate_ctc(encoder, ctc_head, tokenizer, test_dir, device, batch_size=64):
    """Evaluate CTC head on benchmarks with latency measurement."""
    encoder.eval()
    ctc_head.eval()

    benchmarks = discover_parseq_structure(test_dir)
    standard = ["IIIT5k", "SVT", "IC13_1015", "IC15_2077", "SVTP", "CUTE80"]
    results = {}
    total_correct = 0
    total_total = 0
    total_time = 0.0
    total_tokens = 0

    print(f"    {'Benchmark':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"    {'-' * 50}")

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

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                features, enc_lengths = encoder(batch_imgs)
                logits = ctc_head(features)
                decoded = ctc_greedy_decode(logits, tokenizer)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_time += time.perf_counter() - t0

                for d, l in zip(decoded, batch_labels):
                    total_tokens += len(d)
                    if d.lower() == l.lower():
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
        ms = total_time / total_total * 1000
        avg_tok = total_tokens / total_total
        print(f"    {'-' * 50}")
        print(f"    {'AVERAGE':<20} {total_correct:>8} {total_total:>8} {avg:>9.1f}%")
        print(f"    Decode: {ms:.2f}ms/word, {avg_tok:.1f} chars/word")
        results["AVERAGE"] = avg
        results["ms_per_word"] = ms
        results["tokens_per_word"] = avg_tok

    return results


def evaluate_rnnt(encoder, pred_net, joint_net, tokenizer, test_dir, device, batch_size=64):
    """Evaluate RNN-T head on benchmarks with latency measurement."""
    encoder.eval()
    pred_net.eval()
    joint_net.eval()

    benchmarks = discover_parseq_structure(test_dir)
    standard = ["IIIT5k", "SVT", "IC13_1015", "IC15_2077", "SVTP", "CUTE80"]
    results = {}
    total_correct = 0
    total_total = 0
    total_time = 0.0
    total_tokens = 0

    print(f"    {'Benchmark':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"    {'-' * 50}")

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

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                decoded_ids = greedy_decode(features, pred_net, joint_net, max_tokens=25)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_time += time.perf_counter() - t0

                for pred_ids, label in zip(decoded_ids, batch_labels):
                    pred_text = tokenizer.decode(pred_ids)
                    total_tokens += len(pred_ids)
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
        ms = total_time / total_total * 1000
        avg_tok = total_tokens / total_total
        print(f"    {'-' * 50}")
        print(f"    {'AVERAGE':<20} {total_correct:>8} {total_total:>8} {avg:>9.1f}%")
        print(f"    Decode: {ms:.2f}ms/word, {avg_tok:.1f} tokens/word")
        results["AVERAGE"] = avg
        results["ms_per_word"] = ms
        results["tokens_per_word"] = avg_tok

    return results


def train_ctc_head(encoder, tokenizer, train_dataset, test_dir,
                   device, epochs, batch_size, lr, label):
    """Train a CTC head on frozen encoder."""
    print(f"\n  Training {label} CTC head (vocab_size={tokenizer.vocab_size})...")

    ctc_head = CTCHead(encoder.output_dim, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(ctc_head.parameters(), lr=lr, weight_decay=0.01)

    widths = get_image_widths(train_dataset, target_height=32, max_width=192)
    sampler = WidthBucketSampler(widths, batch_size=batch_size)
    loader = DataLoader(
        train_dataset, batch_sampler=sampler, collate_fn=collate_ocr,
        num_workers=4, pin_memory=True,
    )

    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    encoder.eval()
    ctc_head.train()

    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(loader, desc=f"  {label} CTC Epoch {epoch}/{epochs}")
        for batch_imgs, batch_labels, widths_batch in pbar:
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
                    features, enc_lengths = encoder(batch_imgs)
                logits = ctc_head(features)

            loss = ctc_loss(logits.float(), targets, enc_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ctc_head.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  {label} CTC Epoch {epoch}: avg_loss={avg_loss:.4f}")

    elapsed = time.time() - start
    print(f"  {label} CTC training time: {elapsed:.0f}s")

    print(f"\n  {label} CTC Evaluation:")
    results = evaluate_ctc(encoder, ctc_head, tokenizer, test_dir, device)
    return results


def train_rnnt_head(encoder, tokenizer, train_dataset, test_dir,
                    device, epochs, batch_size, lr, label):
    """Train an RNN-T head on frozen encoder."""
    print(f"\n  Training {label} RNN-T head (vocab_size={tokenizer.vocab_size})...")

    pred_net = PredictionNetwork(
        vocab_size=tokenizer.vocab_size, embed_dim=128, hidden_dim=128
    ).to(device)
    joint_net = JointNetwork(
        enc_dim=encoder.output_dim, pred_dim=128, joint_dim=256,
        vocab_size=tokenizer.vocab_size,
    ).to(device)

    head_params = list(pred_net.parameters()) + list(joint_net.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=0.01)

    widths = get_image_widths(train_dataset, target_height=32, max_width=192)
    sampler = WidthBucketSampler(widths, batch_size=batch_size)
    loader = DataLoader(
        train_dataset, batch_sampler=sampler, collate_fn=collate_ocr,
        num_workers=4, pin_memory=True,
    )

    total_steps = len(loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    encoder.eval()
    pred_net.train()
    joint_net.train()

    start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(loader, desc=f"  {label} RNN-T Epoch {epoch}/{epochs}")
        for batch_imgs, batch_labels, widths_batch in pbar:
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
            if scaler.get_scale() >= old_scale:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  {label} RNN-T Epoch {epoch}: avg_loss={avg_loss:.4f}")

    elapsed = time.time() - start
    print(f"  {label} RNN-T training time: {elapsed:.0f}s")

    print(f"\n  {label} RNN-T Evaluation:")
    results = evaluate_rnnt(encoder, pred_net, joint_net, tokenizer, test_dir, device)
    return results


def main():
    parser = argparse.ArgumentParser(description="Step 4b: CTC vs RNN-T × Chars vs Bigrams")
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--only", type=str, default=None,
                        choices=["ctc_char", "ctc_bigram", "rnnt_char", "rnnt_bigram"],
                        help="Run only one combination")
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
    print(f"  Backbone: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M params (frozen)")

    # Load training data
    train_dataset = PARSeqLMDB(args.train_dir, max_width=192)
    print(f"  Full dataset: {len(train_dataset)} samples")
    train_dataset.preload_raw(max_samples=args.max_samples)
    print(f"  Training data: {len(train_dataset)} samples")

    # Build tokenizers
    char_tok = LipiTokenizer.build_character_level("en")
    bigram_tok = LipiTokenizer.build_with_curated_bigrams("en")
    print(f"  Character vocab: {char_tok.vocab_size} tokens")
    print(f"  Bigram vocab: {bigram_tok.vocab_size} tokens")

    # Run combinations
    all_results = {}
    combos = [
        ("ctc_char", "CTC+Char", "ctc", char_tok),
        ("ctc_bigram", "CTC+Bigram", "ctc", bigram_tok),
        ("rnnt_char", "RNNT+Char", "rnnt", char_tok),
        ("rnnt_bigram", "RNNT+Bigram", "rnnt", bigram_tok),
    ]

    for key, label, decoder, tokenizer in combos:
        if args.only and args.only != key:
            continue

        print(f"\n{'=' * 60}")
        print(f"  {label} (vocab={tokenizer.vocab_size})")
        print(f"{'=' * 60}")

        if decoder == "ctc":
            results = train_ctc_head(
                encoder, tokenizer, train_dataset, args.test_dir,
                device, args.epochs, args.batch_size, args.lr, label,
            )
        else:
            results = train_rnnt_head(
                encoder, tokenizer, train_dataset, args.test_dir,
                device, args.epochs, args.batch_size, args.lr, label,
            )

        all_results[key] = results

    if args.only:
        print("\nDone (single combination mode).")
        return

    # Full comparison
    print(f"\n{'=' * 60}")
    print("FULL COMPARISON")
    print(f"{'=' * 60}")

    benchmarks = ["IIIT5k", "SVT", "IC13_1015", "IC15_2077", "SVTP", "CUTE80", "AVERAGE"]

    print(f"\n  {'Benchmark':<20}", end="")
    for key, label, _, _ in combos:
        if key in all_results:
            print(f" {label:>12}", end="")
    print()
    print(f"  {'-' * 72}")

    for bench in benchmarks:
        print(f"  {bench:<20}", end="")
        for key, label, _, _ in combos:
            if key in all_results:
                val = all_results[key].get(bench, 0)
                print(f" {val:>11.1f}%", end="")
        print()

    # Speed comparison
    print(f"\n  {'Metric':<20}", end="")
    for key, label, _, _ in combos:
        if key in all_results:
            print(f" {label:>12}", end="")
    print()
    print(f"  {'-' * 72}")

    print(f"  {'ms/word':<20}", end="")
    for key, _, _, _ in combos:
        if key in all_results:
            ms = all_results[key].get("ms_per_word", 0)
            print(f" {ms:>11.2f}", end="")
    print()

    print(f"  {'tokens/word':<20}", end="")
    for key, _, _, _ in combos:
        if key in all_results:
            tok = all_results[key].get("tokens_per_word", 0)
            print(f" {tok:>11.1f}", end="")
    print()

    # Decision
    print(f"\n  Best accuracy: ", end="")
    best_key = max(all_results, key=lambda k: all_results[k].get("AVERAGE", 0))
    best_acc = all_results[best_key].get("AVERAGE", 0)
    best_label = dict((k, l) for k, l, _, _ in combos)[best_key]
    print(f"{best_label} ({best_acc:.1f}%)")

    print(f"  Fastest: ", end="")
    fastest_key = min(all_results, key=lambda k: all_results[k].get("ms_per_word", 999))
    fastest_ms = all_results[fastest_key].get("ms_per_word", 0)
    fastest_label = dict((k, l) for k, l, _, _ in combos)[fastest_key]
    print(f"{fastest_label} ({fastest_ms:.2f}ms/word)")


if __name__ == "__main__":
    main()
