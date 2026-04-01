#!/usr/bin/env python3
"""
Generate synthetic training data for LID and MoE training.

Outputs per shard: images, text labels, script IDs, group IDs.
Works for both LID training (uses group labels) and MoE training (uses text labels + CTC).

Usage:
    python scripts/generate_data.py --samples-per-script 10000 --out data/shards
    python scripts/generate_data.py --samples-per-script 30000 --balance-groups --out data/lid_shards
    python scripts/generate_data.py --scripts latin,cyrillic,greek,devanagari --out data/moe_2group

Then train:
    python scripts/train_lid.py --data data/lid_shards ...
    python scripts/train_moe.py --data data/moe_2group ...
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID, SCRIPT_TO_ID
from src.data.color import rgb_to_input
from src.data.augmentation import RandAugmentOCR
from scripts.train_lid import (
    find_fonts_for_script, font_can_render, render_word, render_emoji,
    SCRIPT_SAMPLES, load_all_script_samples,
)


def _generate_batch(args_tuple):
    """Generate images for one chunk. Saves shard to disk directly."""
    script, count, fonts, words, h, mw, do_augment, shard_path = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None

    images = []
    labels = []
    attempts = 0
    t0 = time.time()

    while len(images) < count and attempts < count * 5:
        attempts += 1
        if script == "emoji":
            img = render_emoji(h, mw)
            label = "emoji"
        else:
            word = random.choice(words)
            img = render_word(word, random.choice(fonts), h)
            label = word

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

        images.append(rgb_to_input(img))
        labels.append(label)

        if len(images) % 1000 == 0:
            elapsed = time.time() - t0
            rate = len(images) / elapsed if elapsed > 0 else 0
            print(f"    [{script}] {len(images)}/{count} ({rate:.0f} img/s)", flush=True)

    n = len(images)
    script_id = SCRIPT_TO_ID[script]
    group_id = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]

    torch.save({
        "images": torch.stack(images),
        "labels": labels,
        "script_ids": torch.full((n,), script_id, dtype=torch.long),
        "group_ids": torch.full((n,), group_id, dtype=torch.long),
    }, shard_path)
    del images, labels

    elapsed = time.time() - t0
    rate = n / elapsed if elapsed > 0 else 0
    print(f"  {script:<15} {n:>5} images in {elapsed:.0f}s ({rate:.0f} img/s)", flush=True)
    return shard_path, script, n


def main():
    parser = argparse.ArgumentParser(description="Generate training data (LID + MoE)")
    parser.add_argument("--samples-per-script", type=int, default=10000)
    parser.add_argument("--scripts", type=str, default="all",
                        help="Comma-separated scripts, or 'all'")
    parser.add_argument("--balance-groups", action="store_true",
                        help="Balance samples per group (for LID training)")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=192)
    parser.add_argument("--out", type=str, default="data/shards")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    if args.scripts == "all":
        active_scripts = list(SCRIPTS)
    else:
        active_scripts = [s.strip() for s in args.scripts.split(",")]

    print("Loading word lists...")
    load_all_script_samples()

    print("Discovering fonts...")
    script_fonts = {}
    valid_scripts = []
    for script in active_scripts:
        if script == "emoji":
            script_fonts[script] = ["__emoji__"]
            valid_scripts.append(script)
            print(f"  {'emoji':<15}   - (synthetic)")
            continue
        if script not in SCRIPT_SAMPLES:
            print(f"  {script:<15}   no word list — SKIPPED")
            continue
        fonts = find_fonts_for_script(script)
        sample = SCRIPT_SAMPLES[script][0]
        valid = [f for f in fonts if font_can_render(f, sample)]
        if valid:
            weighted = []
            for f in valid:
                name = Path(f).name.lower()
                if any(k in name for k in ["caveat", "dancing", "indie", "patrick",
                        "shadow", "kalam", "nanumpen", "chilanka", "handwrit", "cursive", "script"]):
                    weighted.extend([f] * 2)
                elif any(k in name for k in ["permanent", "amatic", "lobster", "pacifico", "special", "display"]):
                    weighted.extend([f] * 1)
                else:
                    weighted.extend([f] * 7)
            script_fonts[script] = weighted
            valid_scripts.append(script)
            print(f"  {script:<15} {len(valid):>3} fonts")
        else:
            print(f"  {script:<15}   0 fonts — SKIPPED")

    if len(valid_scripts) < 2:
        print("ERROR: Need at least 2 scripts with fonts")
        sys.exit(1)

    # Build targets per script
    tasks = []
    for script in valid_scripts:
        group = SCRIPT_TO_GROUP[script]
        if args.balance_groups:
            scripts_in_group = [s for s in valid_scripts if SCRIPT_TO_GROUP[s] == group]
            target = args.samples_per_script // len(scripts_in_group)
        else:
            target = args.samples_per_script
        tasks.append((script, target))

    shard_dir = Path(args.out)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Build chunks, skip existing shards (resume support)
    chunks = []
    shard_idx = 0
    skipped = 0
    for script, target in tasks:
        fonts = script_fonts[script]
        words = SCRIPT_SAMPLES.get(script, ["placeholder"])
        chunk_size = max(500, target // max(1, args.workers // len(tasks)))
        remaining = target
        while remaining > 0:
            batch = min(chunk_size, remaining)
            shard_path = str(shard_dir / f"shard_{shard_idx:04d}.pt")
            if Path(shard_path).exists() and Path(shard_path).stat().st_size > 100:
                skipped += batch
            else:
                chunks.append((script, batch, fonts, words, args.height, args.max_width,
                              args.augment, shard_path))
            shard_idx += 1
            remaining -= batch

    total_est = sum(c[1] for c in chunks)
    if skipped > 0:
        print(f"\nResuming: {skipped} images in existing shards, {total_est} remaining")
    if not chunks:
        print("All shards exist. Done.")
    else:
        print(f"\nGenerating {total_est} images, {len(chunks)} chunks, {min(args.workers, len(chunks))} workers\n")
        start = time.time()
        done = 0
        with Pool(processes=min(args.workers, len(chunks)), maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_generate_batch, chunks):
                _, script, n = result
                done += n
                elapsed = time.time() - start
                print(f"    total: {done}/{total_est} ({done/elapsed:.0f} img/s)", flush=True)

    # Save metadata
    active_groups = []
    seen = set()
    for s in valid_scripts:
        g = SCRIPT_TO_GROUP[s]
        if g not in seen:
            active_groups.append(g)
            seen.add(g)

    torch.save({
        "active_scripts": valid_scripts,
        "active_groups": active_groups,
        "script_to_idx": {s: i for i, s in enumerate(valid_scripts)},
        "group_to_idx": {g: i for i, g in enumerate(active_groups)},
        "height": args.height,
        "max_width": args.max_width,
        "augmented": args.augment,
        "samples_per_script": args.samples_per_script,
        "has_labels": True,
    }, shard_dir / "metadata.pt")

    total_shards = len(list(shard_dir.glob("shard_*.pt")))
    total_mb = sum(f.stat().st_size for f in shard_dir.glob("*.pt")) / 1e6
    print(f"\nDone. {total_shards} shards ({total_mb:.0f} MB) in {shard_dir}/")


if __name__ == "__main__":
    main()
