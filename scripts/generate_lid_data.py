#!/usr/bin/env python3
"""
Generate synthetic LID training data and save to disk.

Run once, then train multiple times with different hyperparams:
    python scripts/generate_lid_data.py --samples-per-script 30000 --out data/lid_30k.pt
    python scripts/train_lid.py --data data/lid_30k.pt --epochs 15 --batch-size 512
    python scripts/train_lid.py --data data/lid_30k.pt --epochs 30 --lr 1e-3
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, GROUP_SCRIPTS, GROUPS
from src.data.color import rgb_to_input
from src.data.augmentation import RandAugmentOCR

# Reuse rendering + font discovery from train_lid
from scripts.train_lid import (
    find_fonts_for_script, font_can_render, render_word, render_emoji,
    random_ink_color, random_bg_color,
    SCRIPT_SAMPLES, load_all_script_samples,
    _SCRIPT_EXTRA_FILES, WORD_LIST_DIR, _LOCAL_FONT_MAP,
)


def _generate_batch(args_tuple):
    """Generate all images for one script."""
    script, count, fonts, words, h, mw, do_augment = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None
    results = []
    attempts = 0
    t0 = time.time()
    while len(results) < count and attempts < count * 5:
        attempts += 1
        if script == "emoji":
            img = render_emoji(h, mw)
        else:
            img = render_word(random.choice(words), random.choice(fonts), h)
        if img is None:
            continue
        if img.width > mw:
            img = img.resize((mw, h), Image.BILINEAR)
        elif img.width < mw:
            padded = Image.new("RGB", (mw, h), (240, 240, 240))
            padded.paste(img, (0, 0))
            img = padded
        if aug is not None:
            img = aug(img)
        results.append(rgb_to_input(img))
        if len(results) % 1000 == 0:
            elapsed = time.time() - t0
            rate = len(results) / elapsed if elapsed > 0 else 0
            print(f"    [{script}] {len(results)}/{count} ({rate:.0f} img/s)")
    elapsed = time.time() - t0
    rate = len(results) / elapsed if elapsed > 0 else 0
    print(f"  {script:<15} {len(results):>5} images in {elapsed:.0f}s ({rate:.0f} img/s)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate LID training data")
    parser.add_argument("--samples-per-script", type=int, default=30000)
    parser.add_argument("--balance-groups", action="store_true", default=True)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=256)
    parser.add_argument("--out", type=str, default="data/lid_shards",
                        help="Output directory for shards")
    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    print("Loading word lists...")
    load_all_script_samples()

    print("Discovering fonts...")
    script_fonts = {}
    active_scripts = []
    for script in SCRIPTS:
        if script == "emoji":
            script_fonts[script] = ["__emoji__"]
            active_scripts.append(script)
            print(f"  {'emoji':<15}   - (synthetic)")
            continue
        if script not in SCRIPT_SAMPLES:
            continue
        fonts = find_fonts_for_script(script)
        sample_text = SCRIPT_SAMPLES[script][0]
        valid_fonts = [f for f in fonts if font_can_render(f, sample_text)]
        if valid_fonts:
            # Weight fonts: 70% clean, 20% handwriting, 10% display
            weighted = []
            for f in valid_fonts:
                name = Path(f).name.lower()
                if any(k in name for k in ["caveat", "dancing", "indie", "patrick",
                        "shadow", "kalam", "nanum_pen", "nanumpen", "chilanka",
                        "handwrit", "cursive", "script"]):
                    weighted.append((f, 2))
                elif any(k in name for k in ["permanent", "amatic", "lobster",
                        "pacifico", "special", "display"]):
                    weighted.append((f, 1))
                else:
                    weighted.append((f, 7))
            expanded = []
            for font, weight in weighted:
                expanded.extend([font] * weight)
            script_fonts[script] = expanded
            active_scripts.append(script)
            print(f"  {script:<15} {len(valid_fonts):>3} fonts")
        else:
            print(f"  {script:<15}   0 fonts — SKIPPED")

    if len(active_scripts) < 2:
        print("ERROR: Need at least 2 scripts with fonts")
        sys.exit(1)

    # Build tasks
    script_to_idx = {s: i for i, s in enumerate(active_scripts)}
    active_groups = []
    seen = set()
    for s in active_scripts:
        g = SCRIPT_TO_GROUP[s]
        if g not in seen:
            active_groups.append(g)
            seen.add(g)
    group_to_idx = {g: i for i, g in enumerate(active_groups)}

    tasks = []
    for script in active_scripts:
        group = SCRIPT_TO_GROUP[script]
        if args.balance_groups:
            scripts_in_group = [s for s in active_scripts if SCRIPT_TO_GROUP[s] == group]
            target = args.samples_per_script // len(scripts_in_group)
        else:
            target = args.samples_per_script
        tasks.append((script, target))

    total_est = sum(t[1] for t in tasks)
    print(f"\nGenerating {total_est} images across {len(active_scripts)} scripts, {len(active_groups)} groups...")
    start = time.time()

    # Parallel generation — split each script into chunks across 48 workers
    n_workers = 48
    all_images = []
    all_script_labels = []
    all_group_labels = []

    # Build chunks: split large scripts across multiple workers
    chunks = []
    for script, target in tasks:
        fonts = script_fonts[script]
        words = SCRIPT_SAMPLES.get(script, ["placeholder"])
        # Split into chunks of ~2000 images each
        chunk_size = max(500, target // max(1, n_workers // len(tasks)))
        remaining = target
        while remaining > 0:
            batch = min(chunk_size, remaining)
            chunks.append((script, batch, fonts, words, args.height, args.max_width, args.augment))
            remaining -= batch

    print(f"  {len(chunks)} chunks across {n_workers} workers\n")

    # Save incrementally — each chunk appends to a shard dir, merge at end
    shard_dir = Path(args.out)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    done_count = 0
    total_chunks = len(chunks)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_generate_batch, chunk): chunk[0] for chunk in chunks}

        for future in as_completed(futures):
            script = futures[future]
            group = SCRIPT_TO_GROUP[script]
            batch = future.result()

            # Save shard immediately
            imgs = torch.stack(batch)
            s_labels = torch.full((len(batch),), script_to_idx[script], dtype=torch.long)
            g_labels = torch.full((len(batch),), group_to_idx[group], dtype=torch.long)
            torch.save({"images": imgs, "script_labels": s_labels, "group_labels": g_labels},
                       shard_dir / f"shard_{shard_idx:04d}.pt")
            shard_idx += 1
            done_count += len(batch)

            elapsed = time.time() - start
            rate = done_count / elapsed if elapsed > 0 else 0
            print(f"  [{shard_idx}/{total_chunks}] +{len(batch)} {script:<12s} | total: {done_count}/{total_est} ({rate:.0f} img/s)")

    elapsed = time.time() - start
    print(f"\nGenerated {done_count} images in {elapsed:.0f}s")

    # Save metadata alongside shards
    torch.save({
        "active_scripts": active_scripts,
        "active_groups": active_groups,
        "script_to_idx": script_to_idx,
        "group_to_idx": group_to_idx,
        "height": args.height,
        "max_width": args.max_width,
        "augmented": args.augment,
        "samples_per_script": args.samples_per_script,
        "total_images": done_count,
        "num_shards": shard_idx,
    }, shard_dir / "metadata.pt")

    total_mb = sum(f.stat().st_size for f in shard_dir.glob("*.pt")) / 1e6
    print(f"\nSaved {shard_idx} shards to {shard_dir}/ ({total_mb:.0f} MB)")
    print(f"  {done_count} images, {len(active_scripts)} scripts, {len(active_groups)} groups")
    print(f"\nTrain with: python scripts/train_lid.py --data {shard_dir}")


if __name__ == "__main__":
    main()
