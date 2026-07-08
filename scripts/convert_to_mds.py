#!/usr/bin/env python3
"""
Convert existing .pt shard data to MDS (Mosaic Data Shard) format.

Usage:
    python scripts/convert_to_mds.py --input data/all_shards --output data/mds

Reads all shard_*.pt, char_shard_*.pt, real_*.pt, mlt50m_*.pt files,
splits 90/10 into train/val, and writes MDS format with a widths.npy
sidecar for fast batch sampling.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from streaming import MDSWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.taxonomy import SCRIPTS

# MDS column schema
MDS_COLUMNS = {
    "image": "ndarray:uint8",       # (3, 32, W) — variable width
    "label": "str",                  # text label
    "script_id": "int",             # global script ID (0-25)
    "group_id": "int",              # global group ID (0-12)
    "target_ids": "ndarray:int64",  # (L,) — variable length, unpadded
    "target_len": "int",            # actual target length
    "width": "int",                 # image width (for batch sampling)
}


def deterministic_split(label: str, val_ratio: float = 0.1) -> str:
    """Assign sample to train/val split deterministically based on label hash."""
    h = int(hashlib.md5(label.encode()).hexdigest(), 16)
    return "val" if (h % 1000) < int(val_ratio * 1000) else "train"


def find_shard_files(shard_dir: Path) -> list[Path]:
    """Find all shard files in the directory."""
    patterns = ["shard_*.pt", "char_shard_*.pt", "real_*.pt", "mlt50m_*.pt"]
    files = []
    for pattern in patterns:
        files.extend(sorted(shard_dir.glob(pattern)))
    return files


def convert(input_dir: Path, output_dir: Path, val_ratio: float = 0.1,
            compression: str = "zstd:7"):
    shard_files = find_shard_files(input_dir)
    if not shard_files:
        print(f"No shard files found in {input_dir}")
        return

    print(f"Found {len(shard_files)} shard files in {input_dir}")

    # Load metadata
    meta_path = input_dir / "metadata.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, weights_only=False)
        print(f"Scripts: {meta.get('active_scripts', 'unknown')}")
    else:
        meta = {}

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_writer = MDSWriter(out=str(train_dir), columns=MDS_COLUMNS,
                             compression=compression, size_limit=1 << 26)  # 64MB shards
    val_writer = MDSWriter(out=str(val_dir), columns=MDS_COLUMNS,
                           compression=compression, size_limit=1 << 26)

    train_widths = []
    val_widths = []
    total = 0
    n_train = 0
    n_val = 0

    for si, shard_path in enumerate(shard_files):
        shard = torch.load(shard_path, weights_only=False)
        images = shard["images"]          # (N, C, H, W)
        labels = shard["labels"]          # list[str]
        script_ids = shard["script_ids"]  # (N,)
        group_ids = shard["group_ids"]    # (N,)
        has_targets = "target_ids" in shard and "target_lens" in shard
        n = len(labels)

        for i in range(n):
            img = images[i].numpy()  # (C, H, W) uint8

            label = labels[i]
            sid = int(script_ids[i].item())
            gid = int(group_ids[i].item())

            if has_targets:
                tlen = int(shard["target_lens"][i].item())
                tids = shard["target_ids"][i, :tlen].numpy().astype(np.int64)
            else:
                tlen = 0
            # MDS can't encode 0-length arrays
            if tlen == 0:
                tids = np.zeros(1, dtype=np.int64)

            width = img.shape[2]

            sample = {
                "image": img,
                "label": label,
                "script_id": sid,
                "group_id": gid,
                "target_ids": tids,
                "target_len": tlen,
                "width": width,
            }

            # Deterministic split
            split = deterministic_split(f"{si}_{i}_{label}", val_ratio)
            if split == "val":
                val_writer.write(sample)
                val_widths.append(width)
                n_val += 1
            else:
                train_writer.write(sample)
                train_widths.append(width)
                n_train += 1

            total += 1

        if (si + 1) % 50 == 0 or si == len(shard_files) - 1:
            print(f"  {si+1}/{len(shard_files)} shards, {total} samples "
                  f"(train={n_train}, val={n_val})", flush=True)

        del shard

    train_writer.finish()
    val_writer.finish()

    # Save widths sidecar for fast batch sampling
    np.save(str(train_dir / "widths.npy"), np.array(train_widths, dtype=np.int32))
    np.save(str(val_dir / "widths.npy"), np.array(val_widths, dtype=np.int32))

    # Copy metadata
    if meta:
        torch.save(meta, output_dir / "metadata.pt")

    print(f"\nDone: {total} samples → {n_train} train + {n_val} val")
    print(f"Output: {output_dir}")
    print(f"  train: {n_train} samples, widths.npy ({len(train_widths)} entries)")
    print(f"  val:   {n_val} samples, widths.npy ({len(val_widths)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Convert .pt shards to MDS format")
    parser.add_argument("--input", type=str, required=True, help="Input shard directory")
    parser.add_argument("--output", type=str, required=True, help="Output MDS directory")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--compression", type=str, default="zstd:7")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output),
            val_ratio=args.val_ratio, compression=args.compression)


if __name__ == "__main__":
    main()
