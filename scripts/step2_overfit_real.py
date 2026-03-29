#!/usr/bin/env python3
"""
Step 2: Overfit test with real crops from IIIT5K.

Pick 100 visually diverse crops, train for 1000 steps with high LR.
The model should memorize them to ~100% accuracy.

This proves the full training pipeline works end-to-end on GPU with real data.

Usage:
    python scripts/step2_overfit_real.py --data_dir training_data/external/test/IIIT5k_3000
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiEncoder
from src.training.foundation_trainer import CTCHead
from src.training.loss import ctc_loss
from src.data.bigrams import LipiTokenizer
from src.data.parseq_lmdb import PARSeqLMDB
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


def main():
    parser = argparse.ArgumentParser(description="Step 2: Overfit on real crops")
    parser.add_argument("--data_dir", type=str, required=True, help="IIIT5K LMDB path")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")

    # Load dataset
    dataset = PARSeqLMDB(args.data_dir)
    print(f"Dataset size: {len(dataset)}")

    # Sample N diverse indices
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(args.n_samples, len(dataset)))
    subset = Subset(dataset, indices)

    # Check a few samples
    print(f"\nSample labels:")
    for i in range(min(5, len(subset))):
        img, label = subset[i]
        print(f"  [{i}] shape={img.shape}, label=\"{label}\"")

    # Build tokenizer from the labels we'll train on
    tokenizer = LipiTokenizer.build_character_level("en")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Verify all labels encode correctly
    encode_failures = 0
    for i in range(len(subset)):
        _, label = subset[i]
        ids = tokenizer.encode(label)
        decoded = tokenizer.decode(ids)
        if decoded != label:
            encode_failures += 1
            if encode_failures <= 3:
                print(f"  Encode failure: \"{label}\" -> \"{decoded}\"")
    if encode_failures:
        print(f"  WARNING: {encode_failures} labels don't roundtrip (chars outside vocab)")

    # Build model
    encoder = LipiEncoder().to(device)
    ctc_head = CTCHead(encoder.output_dim, tokenizer.vocab_size).to(device)

    params = list(encoder.parameters()) + list(ctc_head.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ocr,
        drop_last=True, num_workers=2, pin_memory=(device.type == "cuda"),
    )

    print(f"\nTraining: {args.n_steps} steps, lr={args.lr}, batch={args.batch_size}")
    print("=" * 60)

    encoder.train()
    ctc_head.train()

    step = 0
    start = time.time()
    loader_iter = iter(loader)

    while step < args.n_steps:
        try:
            batch_imgs, batch_labels, widths = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_imgs, batch_labels, widths = next(loader_iter)

        batch_imgs = batch_imgs.to(device, non_blocking=True)

        target_ids = [tokenizer.encode(l) for l in batch_labels]
        target_lengths = torch.tensor([len(ids) for ids in target_ids], device=device)
        max_tgt = max(len(ids) for ids in target_ids) if target_ids else 1
        targets = torch.zeros(len(batch_labels), max_tgt, dtype=torch.long, device=device)
        for i, ids in enumerate(target_ids):
            targets[i, :len(ids)] = torch.tensor(ids)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            features, enc_lengths = encoder(batch_imgs)
            logits = ctc_head(features)

        loss = ctc_loss(logits.float(), targets, enc_lengths, target_lengths)

        if torch.isinf(loss) or torch.isnan(loss):
            step += 1
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % 100 == 0 or step == 1:
            encoder.eval()
            ctc_head.eval()
            with torch.no_grad():
                test_logits = ctc_head(encoder(batch_imgs)[0])
                decoded = ctc_greedy_decode(test_logits, tokenizer)
                batch_correct = sum(1 for d, l in zip(decoded, batch_labels) if d == l)
            encoder.train()
            ctc_head.train()
            elapsed = time.time() - start
            print(f"  Step {step:5d}: loss={loss.item():.4f}, "
                  f"batch_acc={batch_correct}/{len(batch_labels)}, "
                  f"time={elapsed:.0f}s, "
                  f"sample: \"{batch_labels[0]}\" -> \"{decoded[0]}\"")

    # Final evaluation on all training samples
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    encoder.eval()
    ctc_head.eval()

    eval_loader = DataLoader(
        subset, batch_size=64, shuffle=False, collate_fn=collate_ocr,
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_imgs, batch_labels, widths in eval_loader:
            batch_imgs = batch_imgs.to(device)
            features, _ = encoder(batch_imgs)
            logits = ctc_head(features)
            decoded = ctc_greedy_decode(logits, tokenizer)

            for d, l in zip(decoded, batch_labels):
                if d == l:
                    correct += 1
                elif total < 200:  # Print first few misses
                    pass  # silent
                total += 1

    accuracy = correct / max(total, 1) * 100
    elapsed = time.time() - start
    print(f"\n  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Total time: {elapsed:.0f}s")

    # Pass criteria
    print(f"\n  {'PASS' if accuracy > 90 else 'FAIL'}: ", end="")
    if accuracy > 95:
        print(f"Excellent — model memorized real crops ({accuracy:.1f}%)")
    elif accuracy > 80:
        print(f"Good — model mostly memorized ({accuracy:.1f}%), may need more steps")
    elif accuracy > 50:
        print(f"Partial — model learning but not memorized ({accuracy:.1f}%), increase steps/lr")
    else:
        print(f"Failed — model not memorizing ({accuracy:.1f}%), debug training pipeline")


if __name__ == "__main__":
    main()
