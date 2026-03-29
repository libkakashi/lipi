#!/usr/bin/env python3
"""
Apply synthetic degradation to clean PDF crops.

Takes clean crops from PDF extraction and applies realistic
scanning artifacts (blur, noise, warp, shadow, compression).

Usage:
    python scripts/apply_degradation.py \
        --input training_data/datasets/en_pdf_crops \
        --output training_data/datasets/en_pdf_crops_degraded \
        --preset medium
"""

import argparse
import io
import sys
from pathlib import Path

import lmdb
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.degradation import apply_degradation
from src.data.dataset import create_lmdb


def main():
    parser = argparse.ArgumentParser(description="Apply degradation to clean crops")
    parser.add_argument("--input", type=str, required=True, help="Input LMDB path")
    parser.add_argument("--output", type=str, required=True, help="Output LMDB path")
    parser.add_argument("--preset", type=str, default="medium",
                        choices=["light", "medium", "heavy"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # Read input LMDB
    env = lmdb.open(args.input, readonly=True, lock=False)
    with env.begin() as txn:
        num_samples = int(txn.get(b"num-samples").decode())

    print(f"Input: {num_samples} samples from {args.input}")
    print(f"Preset: {args.preset}")

    images = []
    labels = []

    with env.begin() as txn:
        for i in tqdm(range(num_samples), desc="Degrading"):
            img_bytes = txn.get(f"image-{i:09d}".encode())
            label = txn.get(f"label-{i:09d}".encode()).decode("utf-8")

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            degraded = apply_degradation(img, preset=args.preset)

            images.append(degraded)
            labels.append(label)

    env.close()

    print(f"\nWriting {len(images)} degraded samples to: {args.output}")
    create_lmdb(args.output, images, labels)
    print("Done!")


if __name__ == "__main__":
    main()
