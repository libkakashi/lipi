#!/usr/bin/env python3
"""
Generate synthetic training data for LID and MoE training.

Usage:
    python scripts/generate_data.py --samples-per-script 10000 --out data/shards
    python scripts/generate_data.py --samples-per-script 30000 --balance-groups --include-chars --out data/shards
"""

import argparse
import random
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID, SCRIPT_TO_ID
from src.data.color import rgb_to_input
from src.data.augmentation import RandAugmentOCR
from src.data.vocab import build_script_vocab
from src.data.rendering import (
    render_word, render_emoji, image_has_ink,
    resize_or_pad, filter_fonts_by_cmap,
)
from src.data.fonts import find_fonts_for_script, build_weighted_font_list
from src.data.word_lists import load_all_word_lists


# ---------------------------------------------------------------------------
# Renderable character list (from frozen vocab)
# ---------------------------------------------------------------------------

def get_renderable_chars(script: str) -> list[str]:
    """Get renderable characters from the frozen vocab for a script.

    Excludes blank token and non-printable chars (space, control chars).
    """
    from src.data.bigrams import BLANK_TOKEN
    group = SCRIPT_TO_GROUP[script]
    vocab = build_script_vocab(script, group)
    return [ch for ch in vocab if ch.strip() and ord(ch) > 32 and ch != BLANK_TOKEN]


# ---------------------------------------------------------------------------
# Shard saving
# ---------------------------------------------------------------------------

def save_shard(images, labels, script, shard_path):
    """Save a shard to disk."""
    if not images:
        return 0
    n = len(images)
    script_id = SCRIPT_TO_ID[script]
    group_id = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]
    torch.save({
        "images": torch.stack(images),
        "labels": labels,
        "script_ids": torch.full((n,), script_id, dtype=torch.long),
        "group_ids": torch.full((n,), group_id, dtype=torch.long),
    }, shard_path)
    return n


# ---------------------------------------------------------------------------
# Worker functions (called in multiprocessing pool)
# ---------------------------------------------------------------------------

def _generate_word_batch(args_tuple):
    """Generate word images for one chunk."""
    script, count, fonts, words, h, mw, do_augment, shard_path = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None
    t0 = time.time()

    images, labels = [], []
    attempts = 0
    # First 30% clean, rest augmented — guarantees clean examples
    clean_target = int(count * 0.3)

    while len(images) < count and attempts < count * 5:
        attempts += 1
        if script == "emoji":
            img = render_emoji(h, mw)
            label = "emoji"
        else:
            word = random.choice(words)
            img = render_word(word, random.choice(fonts), h)
            label = word

        if img is None or not image_has_ink(img):
            continue

        img = resize_or_pad(img, h, mw)

        # First 30% of images are clean (no augmentation)
        if aug is not None and len(images) >= clean_target:
            img = aug(img)
            if not image_has_ink(img, min_ink_pixels=5):
                continue

        images.append(rgb_to_input(img))
        labels.append(label)

        if len(images) % 1000 == 0:
            elapsed = time.time() - t0
            rate = len(images) / elapsed if elapsed > 0 else 0
            print(f"    [{script}] {len(images)}/{count} ({rate:.0f} img/s)", flush=True)

    n = save_shard(images, labels, script, shard_path)
    del images, labels

    elapsed = time.time() - t0
    rate = n / elapsed if elapsed > 0 else 0
    print(f"  {script:<15} {n:>5} images in {elapsed:.0f}s ({rate:.0f} img/s)", flush=True)
    return shard_path, script, n


def _generate_char_batch(args_tuple):
    """Generate single-character images with cmap validation."""
    script, chars, reps_per_char, fonts, h, mw, do_augment, shard_path = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None
    t0 = time.time()

    images, labels = [], []
    skipped_chars = 0

    # Pre-filter fonts per char using cmap
    char_fonts = filter_fonts_by_cmap(fonts, chars)
    skipped_chars += len(chars) - len(char_fonts)

    for ch in char_fonts:
        got_any = False
        for rep in range(reps_per_char):
            font = random.choice(char_fonts[ch])
            img = render_word(ch, font, h)
            if img is None or not image_has_ink(img):
                continue
            got_any = True

            img = resize_or_pad(img, h, mw)

            # First rep clean, rest augmented
            if aug is not None and rep > 0:
                img = aug(img)
                if not image_has_ink(img, min_ink_pixels=5):
                    continue

            images.append(rgb_to_input(img))
            labels.append(ch)
        if not got_any:
            skipped_chars += 1

    n = save_shard(images, labels, script, shard_path)
    del images, labels

    elapsed = time.time() - t0
    skip_str = f", {skipped_chars} chars skipped (bad render)" if skipped_chars else ""
    print(f"  {script:<15} {n:>5} char images "
          f"({len(chars)} unique × {reps_per_char} reps{skip_str}) in {elapsed:.0f}s", flush=True)
    return shard_path, script, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--samples-per-script", type=int, default=10000)
    parser.add_argument("--scripts", type=str, default="all")
    parser.add_argument("--balance-groups", action="store_true")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=192)
    parser.add_argument("--include-chars", action="store_true")
    parser.add_argument("--char-reps", type=int, default=3)
    parser.add_argument("--out", type=str, default="data/shards")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    active_scripts = list(SCRIPTS) if args.scripts == "all" else [
        s.strip() for s in args.scripts.split(",")]

    # Load word lists
    print("Loading word lists...")
    word_lists = load_all_word_lists(active_scripts)

    # Discover fonts
    print("Discovering fonts...")
    script_fonts = {}
    valid_scripts = []
    for script in active_scripts:
        if script == "emoji":
            script_fonts[script] = ["__emoji__"]
            valid_scripts.append(script)
            print(f"  {'emoji':<15}   - (synthetic)")
            continue
        if not word_lists.get(script):
            print(f"  {script:<15}   no word list — SKIPPED")
            continue
        fonts = find_fonts_for_script(script)
        sample = word_lists[script][0]
        weighted = build_weighted_font_list(fonts, sample)
        if weighted:
            script_fonts[script] = weighted
            valid_scripts.append(script)
            n_unique = len(set(weighted))
            print(f"  {script:<15} {n_unique:>3} fonts")
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

    # Build word image chunks (skip existing shards for resume)
    chunks = []
    shard_idx = 0
    skipped = 0
    for script, target in tasks:
        fonts = script_fonts[script]
        words = word_lists.get(script, ["placeholder"])
        chunk_size = max(500, target // max(1, args.workers // len(tasks)))
        remaining = target
        while remaining > 0:
            batch = min(chunk_size, remaining)
            shard_path = str(shard_dir / f"shard_{shard_idx:04d}.pt")
            if Path(shard_path).exists() and Path(shard_path).stat().st_size > 100:
                skipped += batch
            else:
                chunks.append((script, batch, fonts, words,
                              args.height, args.max_width, args.augment, shard_path))
            shard_idx += 1
            remaining -= batch

    total_est = sum(c[1] for c in chunks)
    if skipped > 0:
        print(f"\nResuming: {skipped} images in existing shards, {total_est} remaining")
    if not chunks:
        print("All shards exist. Done.")
    else:
        print(f"\nGenerating {total_est} images, {len(chunks)} chunks, "
              f"{min(args.workers, len(chunks))} workers\n")
        start = time.time()
        done = 0
        with Pool(processes=min(args.workers, len(chunks)), maxtasksperchild=1) as pool:
            for result in pool.imap_unordered(_generate_word_batch, chunks):
                _, script, n = result
                done += n
                elapsed = time.time() - start
                print(f"    total: {done}/{total_est} ({done/elapsed:.0f} img/s)", flush=True)

    # Single-character images
    if args.include_chars:
        script_word_target = {s: t for s, t in tasks}
        char_budget_ratio = 0.25

        print(f"\n{'='*60}")
        print(f"Generating single-character images (max {char_budget_ratio:.0%} of word budget)")
        print(f"{'='*60}")

        char_chunks = []
        for script in valid_scripts:
            if script == "emoji":
                continue
            fonts = script_fonts[script]
            chars = get_renderable_chars(script)
            if not chars:
                continue

            word_target = script_word_target.get(script, args.samples_per_script)
            char_budget = int(word_target * char_budget_ratio)
            reps = max(1, min(args.char_reps, char_budget // len(chars)))
            est = len(chars) * reps
            print(f"  {script:<15} {len(chars):>5} unique chars × {reps} reps = ~{est} images "
                  f"(word budget: {word_target})")

            chunk_size = max(200, len(chars) // max(1, args.workers // len(valid_scripts)))
            for ci in range(0, len(chars), chunk_size):
                char_subset = chars[ci:ci + chunk_size]
                shard_path = str(shard_dir / f"char_shard_{shard_idx:04d}.pt")
                if Path(shard_path).exists() and Path(shard_path).stat().st_size > 100:
                    shard_idx += 1
                    continue
                char_chunks.append((script, char_subset, reps, fonts,
                                    args.height, args.max_width, args.augment, shard_path))
                shard_idx += 1

        if char_chunks:
            print(f"\n  {len(char_chunks)} char chunks across "
                  f"{min(args.workers, len(char_chunks))} workers\n")
            char_start = time.time()
            char_done = 0
            with Pool(processes=min(args.workers, len(char_chunks)), maxtasksperchild=1) as pool:
                for result in pool.imap_unordered(_generate_char_batch, char_chunks):
                    _, script, n = result
                    char_done += n
            print(f"\n  Total: {char_done} char images in {time.time()-char_start:.0f}s")
        else:
            print("  All char shards exist.")

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
