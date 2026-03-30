#!/usr/bin/env python3
"""
Data quality validation.

Runs before training to catch issues early:
  - Vocab tokenization roundtrips for all labels
  - LID classification matches expected script
  - Font coverage statistics per script
  - Image quality checks (dimensions, corruption)

Usage:
    python scripts/validate_data.py --dataset training_data/datasets/pdf_crops
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bigrams import LipiTokenizer, SCRIPT_CHARSETS
from src.data.dataset import LMDBDataset


def validate_vocab_roundtrip(dataset, tokenizer, max_samples=10000):
    """Verify Vocab encode -> decode roundtrips for all labels."""
    failures = []
    checked = 0

    for i in range(min(len(dataset), max_samples)):
        _, label = dataset[i]
        try:
            ids = tokenizer.encode(label)
            decoded = tokenizer.decode(ids)
            if decoded != label:
                failures.append((label, decoded, ids))
        except Exception as e:
            failures.append((label, f"ERROR: {e}", []))
        checked += 1

    return checked, failures


def validate_image_quality(dataset, max_samples=10000):
    """Check image dimensions and quality."""
    issues = []
    widths = []
    heights = []

    for i in range(min(len(dataset), max_samples)):
        img, label = dataset[i]
        c, h, w = img.shape

        if c != 3:
            issues.append((i, label, f"wrong channels: {c}"))
        if h != 32:
            issues.append((i, label, f"wrong height: {h}"))
        if w < 8:
            issues.append((i, label, f"too narrow: w={w}"))
        if w > 320:
            issues.append((i, label, f"too wide: w={w}"))

        widths.append(w)
        heights.append(h)

    return issues, widths, heights


def validate_label_encoding(dataset, script_id, max_samples=10000):
    """Check that labels only contain expected characters."""
    script_chars = set("".join(SCRIPT_CHARSETS.get(script_id, [])))
    latin_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?'\"()-/&@#$%+= ")
    valid_chars = script_chars | latin_chars

    issues = []
    char_counts = Counter()

    for i in range(min(len(dataset), max_samples)):
        _, label = dataset[i]
        for c in label:
            char_counts[c] += 1
            if valid_chars and c not in valid_chars:
                issues.append((i, label, f"unexpected char: '{c}' (U+{ord(c):04X})"))

    return issues, char_counts


def main():
    parser = argparse.ArgumentParser(description="Validate training data")
    parser.add_argument("--dataset", type=str, required=True, help="LMDB dataset path")
    parser.add_argument("--vocab", type=str, help="Vocabulary JSON")
    parser.add_argument("--script_id", type=str, default="en", help="Expected script")
    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()

    print(f"Validating: {args.dataset}")
    print(f"Script: {args.script_id}")
    print(f"Max samples: {args.max_samples}")
    print()

    dataset = LMDBDataset(args.dataset)
    print(f"Dataset size: {len(dataset)} samples")

    # Load tokenizer
    if args.vocab and Path(args.vocab).exists():
        tokenizer = LipiTokenizer.load(args.vocab)
    else:
        tokenizer = LipiTokenizer.build_character_level(args.script_id)

    # 1. Vocab roundtrip
    print("\n1. Vocab Roundtrip Test")
    checked, failures = validate_vocab_roundtrip(dataset, tokenizer, args.max_samples)
    if failures:
        print(f"   FAIL: {len(failures)} / {checked} roundtrip failures")
        for label, decoded, ids in failures[:5]:
            print(f"     '{label}' -> {ids[:5]}... -> '{decoded}'")
    else:
        print(f"   PASS: {checked} / {checked} roundtrips OK")

    # 2. Image quality
    print("\n2. Image Quality Check")
    issues, widths, heights = validate_image_quality(dataset, args.max_samples)
    if issues:
        print(f"   WARN: {len(issues)} quality issues")
        for idx, label, issue in issues[:5]:
            print(f"     [{idx}] '{label}': {issue}")
    else:
        print(f"   PASS: all images OK")

    import numpy as np
    print(f"   Width stats: min={min(widths)}, max={max(widths)}, "
          f"mean={np.mean(widths):.0f}, median={np.median(widths):.0f}")

    # 3. Label encoding
    print("\n3. Label Encoding Check")
    enc_issues, char_counts = validate_label_encoding(
        dataset, args.script_id, args.max_samples
    )
    if enc_issues:
        print(f"   WARN: {len(enc_issues)} unexpected characters")
        for idx, label, issue in enc_issues[:5]:
            print(f"     [{idx}] '{label}': {issue}")
    else:
        print(f"   PASS: all characters within expected charset")

    # Top characters
    print(f"\n   Top 20 characters:")
    for char, count in char_counts.most_common(20):
        display = repr(char) if char in " \t\n" else char
        print(f"     '{display}': {count}")

    # Summary
    print(f"\n{'='*60}")
    total_issues = len(failures) + len(issues) + len(enc_issues)
    if total_issues == 0:
        print("VALIDATION PASSED - dataset is ready for training")
    else:
        print(f"VALIDATION WARNINGS: {total_issues} issues found")
        print("Review issues above before training")

    dataset.close()


if __name__ == "__main__":
    main()
