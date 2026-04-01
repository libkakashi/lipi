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

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID, SCRIPT_TO_ID
from src.data.color import rgb_to_input
from src.data.augmentation import RandAugmentOCR
from src.data.renderer import font_has_codepoint
from scripts.train_lid import (
    find_fonts_for_script, font_can_render, render_word, render_emoji,
    SCRIPT_SAMPLES, load_all_script_samples,
)


def _get_script_chars(script: str, words: list[str]) -> list[str]:
    """Get all unique characters from a script's word list."""
    chars = set()
    for word in words:
        for ch in word:
            if ch.strip() and ord(ch) > 32:
                chars.add(ch)
    return sorted(chars)


def _image_has_ink(img: Image.Image, min_ink_pixels: int = 10) -> bool:
    """Check if a rendered image has visible content (not blank/faint).

    Only checks for blank or nearly-invisible renders. Tofu detection is
    handled upstream via font_has_codepoint (cmap check), which is definitive
    and avoids false positives on box-shaped characters like 口, ㅁ, ם, O.
    """
    arr = np.array(img)
    if arr.ndim == 3:
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    else:
        gray = arr.astype(float)

    # Background from corner pixels
    corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
    bg = np.median(corners)

    # Count pixels that differ significantly from background
    ink_pixels = (np.abs(gray - bg) > 30).sum()
    return ink_pixels >= min_ink_pixels


def _generate_char_batch(args_tuple):
    """Generate single-character images with validation.

    Skips characters that don't render properly (blank, tofu boxes, invisible).
    Tries multiple fonts per character before giving up.
    """
    script, chars, reps_per_char, fonts, h, mw, do_augment, shard_path = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None

    images = []
    labels = []
    skipped_chars = 0
    t0 = time.time()

    # Pre-filter: which unique fonts have each char in their cmap?
    # This prevents tofu without false-positiving on box-shaped chars (口, ㅁ, ם, O).
    # Then rebuild weighted list from the originals so font diversity is preserved.
    unique_fonts = list(dict.fromkeys(fonts))  # dedupe for cmap check
    cmap_ok = {}  # font_path -> set of chars it can render
    for f in unique_fonts:
        cmap_ok[f] = set()
        for ch in chars:
            if font_has_codepoint(f, ch):
                cmap_ok[f].add(ch)

    char_fonts = {}
    for ch in chars:
        # Keep original weighted list, just filter to fonts whose cmap has this char
        valid = [f for f in fonts if ch in cmap_ok.get(f, set())]
        if valid:
            char_fonts[ch] = valid
        else:
            skipped_chars += 1

    for ch in char_fonts:
        got_any = False
        for rep in range(reps_per_char):
            font = random.choice(char_fonts[ch])
            img = render_word(ch, font, h)
            # cmap presence doesn't guarantee a good render (some fonts have
            # empty/zero-width glyphs), so still check for visible ink
            if img is None or not _image_has_ink(img):
                continue
            got_any = True

            if img.width > mw:
                img = img.resize((mw, h), Image.BILINEAR)
            elif img.width < mw:
                padded = Image.new("RGB", (mw, h), (240, 240, 240))
                padded.paste(img, (0, 0))
                img = padded
            if aug is not None:
                img = aug(img)
            images.append(rgb_to_input(img))
            labels.append(ch)
        if not got_any:
            skipped_chars += 1

    if not images:
        return shard_path, script, 0

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
    skip_str = f", {skipped_chars} chars skipped (bad render)" if skipped_chars else ""
    print(f"  {script:<15} {n:>5} char images ({len(chars)} unique × {reps_per_char} reps{skip_str}) in {elapsed:.0f}s", flush=True)
    return shard_path, script, n


def _generate_batch(args_tuple):
    """Generate word images for one chunk. Saves shard to disk directly."""
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
    parser.add_argument("--include-chars", action="store_true",
                        help="Also generate single-character images for all unique chars")
    parser.add_argument("--char-reps", type=int, default=3,
                        help="Augmentation repetitions per character (default: 3)")
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

    # === Single-character images ===
    if args.include_chars:
        # Cap char images at 25% of word target per script so they supplement
        # word context rather than dominating. Adaptive reps: scripts with huge
        # charsets (han_kana ~21K) get 1 rep, small charsets (latin ~100) get many.
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
            words = SCRIPT_SAMPLES.get(script, [])
            chars = _get_script_chars(script, words)
            if not chars:
                continue

            # Adaptive reps: cap total char images at char_budget_ratio * word target
            word_target = script_word_target.get(script, args.samples_per_script)
            char_budget = int(word_target * char_budget_ratio)
            reps = max(1, min(args.char_reps, char_budget // len(chars)))
            est = len(chars) * reps
            print(f"  {script:<15} {len(chars):>5} unique chars × {reps} reps = ~{est} images "
                  f"(word budget: {word_target})")

            # Split into chunks of ~500 chars for parallelism
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
            print(f"\n  {len(char_chunks)} char chunks across {min(args.workers, len(char_chunks))} workers\n")
            char_start = time.time()
            char_done = 0
            with Pool(processes=min(args.workers, len(char_chunks)), maxtasksperchild=1) as pool:
                for result in pool.imap_unordered(_generate_char_batch, char_chunks):
                    _, script, n = result
                    char_done += n
            char_elapsed = time.time() - char_start
            print(f"\n  Total: {char_done} char images in {char_elapsed:.0f}s")
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
