#!/usr/bin/env python3
"""
Generate synthetic training data.

Creates word crop images using the synthetic renderer with diverse fonts.
Outputs to LMDB for fast training data loading.

Usage:
    python scripts/generate_synth.py \
        --word_list training_data/word_lists/english_100k.txt \
        --output training_data/datasets/en_synth \
        --n_per_word 50 --n_words 10000
"""

import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synth import render_word
from src.data.degradation import apply_degradation
from src.data.dataset import create_lmdb


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OCR data")
    parser.add_argument("--word_list", type=str, required=True, help="Word list file")
    parser.add_argument("--output", type=str, required=True, help="LMDB output path")
    parser.add_argument("--n_per_word", type=int, default=50, help="Variants per word")
    parser.add_argument("--n_words", type=int, default=None, help="Limit to N words")
    parser.add_argument("--degrade", type=str, default="medium", choices=["none", "light", "medium", "heavy"])
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load word list
    word_list_path = Path(args.word_list)
    if not word_list_path.exists():
        print(f"ERROR: Word list not found: {word_list_path}")
        sys.exit(1)

    words = [line.strip() for line in word_list_path.read_text().splitlines() if line.strip()]
    if args.n_words:
        words = words[:args.n_words]

    total = len(words) * args.n_per_word
    print(f"Generating {total} images ({len(words)} words × {args.n_per_word} variants)")
    print(f"Degradation: {args.degrade}")

    images = []
    labels = []

    for word in tqdm(words, desc="Rendering"):
        for _ in range(args.n_per_word):
            img = render_word(word, height=args.height)
            if args.degrade != "none":
                img = apply_degradation(img, preset=args.degrade)
            images.append(img)
            labels.append(word)

    # Shuffle
    pairs = list(zip(images, labels))
    random.shuffle(pairs)
    images = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]

    # Write to LMDB
    print(f"\nWriting {len(images)} samples to LMDB: {args.output}")
    map_size = max(len(images) * 50000, 1 << 28)  # ~50KB per image estimate
    create_lmdb(args.output, images, labels, map_size=map_size)

    print(f"Done! {len(images)} samples written.")


if __name__ == "__main__":
    main()
