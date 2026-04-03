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
from src.data.augmentation import (
    RandAugmentOCR,
    jpeg_compress, blur, low_resolution, photocopy,
    exposure_jitter, uneven_lighting, glare, striped_shadow,
    rotation, perspective_warp, wave_distortion,
    bleed_through, fold_crease, aged_document, scanner_edge, water_stain,
    stroke_variation, smudge,
    noise, color_jitter, to_grayscale,
    occlusion, weather_damage,
)
from src.data.vocab import build_script_vocab
from src.data.rendering import (
    render_word, render_emoji, image_has_ink,
    resize_or_pad, filter_fonts_by_cmap, font_covers_text,
)
from src.data.fonts import find_fonts_for_script, build_weighted_font_list
from src.data.word_lists import load_all_word_lists

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLEAN_RATIO = 0.3
CHAR_BUDGET_RATIO = 0.25
MIN_SHARD_BYTES = 100

# ---------------------------------------------------------------------------
# Data styles — font selection + augmentation ops per style
# ---------------------------------------------------------------------------

_HANDWRITING_KEYWORDS = {"caveat", "dancing", "indie", "patrick", "kalam",
                         "nanumpen", "chilanka", "handwrit", "cursive", "script"}
_DISPLAY_KEYWORDS = {"permanent", "amatic", "lobster", "pacifico", "special",
                     "display", "bold", "condensed", "black"}

STYLES = {
    "clean": {
        "proportion": 0.40,
        "ops": [],
        "font_filter": "regular",
    },
    "document": {
        "proportion": 0.20,
        "ops": [jpeg_compress, blur, photocopy, uneven_lighting,
                fold_crease, bleed_through, aged_document, scanner_edge],
        "font_filter": "regular",
    },
    "handwriting": {
        "proportion": 0.15,
        "ops": [stroke_variation, smudge, noise, exposure_jitter, rotation],
        "font_filter": "handwriting",
    },
    "signage": {
        "proportion": 0.15,
        "ops": [perspective_warp, rotation, exposure_jitter, glare,
                weather_damage, color_jitter, uneven_lighting],
        "font_filter": "display",
    },
    "degraded": {
        "proportion": 0.10,
        "ops": [blur, jpeg_compress, noise, exposure_jitter,
                low_resolution, rotation, color_jitter],
        "font_filter": "all",
    },
}


def filter_fonts_by_style(fonts: list[str], style: str) -> list[str]:
    """Filter font list by style. Falls back to full list if no matches."""
    font_filter = STYLES[style]["font_filter"]
    if font_filter == "all":
        return fonts

    filtered = []
    for f in fonts:
        name = Path(f).name.lower()
        is_handwriting = any(k in name for k in _HANDWRITING_KEYWORDS)
        is_display = any(k in name for k in _DISPLAY_KEYWORDS)

        if font_filter == "regular" and not is_handwriting and not is_display:
            filtered.append(f)
        elif font_filter == "handwriting" and is_handwriting:
            filtered.append(f)
        elif font_filter == "display" and (is_display or is_handwriting):
            filtered.append(f)

    return filtered if filtered else fonts  # fallback to all if no matches


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
# Helpers
# ---------------------------------------------------------------------------

def shard_exists(path: str) -> bool:
    """Check if a shard file exists and has meaningful content."""
    p = Path(path)
    return p.exists() and p.stat().st_size > MIN_SHARD_BYTES


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
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--samples-per-script", type=int, default=10000)
    parser.add_argument("--scripts", type=str, default="all")
    parser.add_argument("--balance-groups", action="store_true")
    parser.add_argument("--vocab-proportional", action="store_true",
                        help="Scale samples per script by sqrt(vocab_size). "
                             "Scripts with more characters get more training data.")
    parser.add_argument("--style", type=str, default="all",
                        choices=list(STYLES.keys()) + ["all"],
                        help="Data style: clean, document, handwriting, signage, degraded, or all")
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=192)
    parser.add_argument("--include-chars", action="store_true")
    parser.add_argument("--char-reps", type=int, default=3)
    parser.add_argument("--out", type=str, default="data/shards")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    # Validate numeric args
    if args.height <= 0:
        parser.error("--height must be > 0")
    if args.max_width <= 0:
        parser.error("--max-width must be > 0")
    if args.samples_per_script <= 0:
        parser.error("--samples-per-script must be > 0")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    # Validate script names
    if args.scripts != "all":
        requested = [s.strip() for s in args.scripts.split(",")]
        unknown = [s for s in requested if s not in SCRIPTS]
        if unknown:
            parser.error(f"Unknown script(s): {', '.join(unknown)}. "
                         f"Valid: {', '.join(sorted(SCRIPTS))}")

    return args


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

def discover_fonts(active_scripts, word_lists):
    """Discover fonts for each script. Returns (script_fonts, valid_scripts)."""
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

    return script_fonts, valid_scripts


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def build_word_chunks(tasks, script_fonts, word_lists, args, shard_dir, styles_to_gen):
    """Build word-image chunks with resume support. Returns (chunks, next_shard_idx, skipped)."""
    chunks = []
    shard_idx = 0
    skipped = 0
    for style in styles_to_gen:
        proportion = STYLES[style]["proportion"] if len(styles_to_gen) > 1 else 1.0
        for script, target in tasks:
            style_target = max(1, int(target * proportion))
            fonts = filter_fonts_by_style(script_fonts[script], style)
            words = word_lists.get(script, ["placeholder"])
            chunk_size = min(5000, max(500, style_target // max(1, args.workers // len(tasks))))
            remaining = style_target
            while remaining > 0:
                batch = min(chunk_size, remaining)
                shard_path = str(shard_dir / f"shard_{shard_idx:04d}.pt")
                if shard_exists(shard_path):
                    skipped += batch
                else:
                    chunks.append((script, batch, fonts, words,
                                  args.height, args.max_width, args.augment,
                                  shard_path, style))
                shard_idx += 1
                remaining -= batch
    return chunks, shard_idx, skipped


def build_char_chunks(valid_scripts, script_fonts, tasks, args, shard_dir, start_shard_idx):
    """Build single-character chunks with resume support. Returns char_chunks."""
    script_word_target = {s: t for s, t in tasks}

    print(f"\n{'='*60}")
    print(f"Generating single-character images (max {CHAR_BUDGET_RATIO:.0%} of word budget)")
    print(f"{'='*60}")

    char_chunks = []
    shard_idx = start_shard_idx
    for script in valid_scripts:
        if script == "emoji":
            continue
        fonts = script_fonts[script]
        chars = get_renderable_chars(script)
        if not chars:
            continue

        word_target = script_word_target.get(script, args.samples_per_script)
        char_budget = int(word_target * CHAR_BUDGET_RATIO)
        reps = max(1, min(args.char_reps, char_budget // len(chars)))
        est = len(chars) * reps
        print(f"  {script:<15} {len(chars):>5} unique chars × {reps} reps = ~{est} images "
              f"(word budget: {word_target})")

        chunk_size = max(200, len(chars) // max(1, args.workers // len(valid_scripts)))
        for ci in range(0, len(chars), chunk_size):
            char_subset = chars[ci:ci + chunk_size]
            shard_path = str(shard_dir / f"char_shard_{shard_idx:04d}.pt")
            if shard_exists(shard_path):
                shard_idx += 1
                continue
            char_chunks.append((script, char_subset, reps, fonts,
                                args.height, args.max_width, args.augment, shard_path))
            shard_idx += 1

    return char_chunks


# ---------------------------------------------------------------------------
# Pool runner
# ---------------------------------------------------------------------------

def run_generation_pool(chunks, worker_fn, n_workers, label):
    """Run a generation function over chunks using a multiprocessing pool."""
    total_est = sum(c[1] if isinstance(c[1], int) else len(c[1]) * c[2] for c in chunks)
    print(f"\nGenerating {total_est} {label} images, {len(chunks)} chunks, "
          f"{min(n_workers, len(chunks))} workers\n")

    start = time.time()
    done = 0
    with Pool(processes=min(n_workers, len(chunks)), maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(worker_fn, chunks):
            _, script, n = result
            done += n
            elapsed = time.time() - start
            print(f"    total: {done}/{total_est} ({done/elapsed:.0f} img/s)", flush=True)
    return done


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_metadata(valid_scripts, args, shard_dir):
    """Save metadata and print summary. Warns if existing metadata has different dimensions."""
    meta_path = shard_dir / "metadata.pt"
    if meta_path.exists():
        try:
            old = torch.load(meta_path, weights_only=True)
            if old.get("height") != args.height or old.get("max_width") != args.max_width:
                print(f"WARNING: Existing metadata has height={old.get('height')}, "
                      f"max_width={old.get('max_width')} but current run uses "
                      f"height={args.height}, max_width={args.max_width}. "
                      f"Overwriting metadata.")
        except Exception:
            pass

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
    }, meta_path)

    total_shards = len(list(shard_dir.glob("shard_*.pt")))
    total_mb = sum(f.stat().st_size for f in shard_dir.glob("*.pt")) / 1e6
    print(f"\nDone. {total_shards} shards ({total_mb:.0f} MB) in {shard_dir}/")


# ---------------------------------------------------------------------------
# Worker functions (called in multiprocessing pool)
# ---------------------------------------------------------------------------

def _generate_word_batch(args_tuple):
    """Generate word images for one chunk."""
    script, count, fonts, words, h, mw, do_augment, shard_path, style = args_tuple
    # Style-specific augmentation
    if not do_augment or style == "clean":
        aug = None
    else:
        style_ops = STYLES.get(style, {}).get("ops", None)
        aug = RandAugmentOCR(n_ops=2, p=0.5, ops=style_ops if style_ops else None)
    t0 = time.time()

    images, labels = [], []
    attempts = 0
    # Clean styles get no augmentation; others get first 30% clean
    clean_target = 0 if style == "clean" else int(count * CLEAN_RATIO)

    while len(images) < count and attempts < count * 5:
        attempts += 1
        if script == "emoji":
            img = render_emoji(h, mw)
            label = "emoji"
        else:
            word = random.choice(words)
            font = random.choice(fonts)
            # Verify font can render ALL chars in the word (prevents partial renders)
            if not font_covers_text(font, word):
                continue
            img = render_word(word, font, h)
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
    args = parse_args()

    active_scripts = (list(SCRIPTS) if args.scripts == "all"
                      else [s.strip() for s in args.scripts.split(",")])

    print("Loading word lists...")
    word_lists = load_all_word_lists(active_scripts)

    script_fonts, valid_scripts = discover_fonts(active_scripts, word_lists)

    # Build per-script targets
    if args.vocab_proportional:
        import math
        script_vocabs = {}
        for script in valid_scripts:
            if script == "emoji":
                script_vocabs[script] = 10  # minimal vocab, fixed budget
            else:
                group = SCRIPT_TO_GROUP[script]
                vocab = build_script_vocab(script, group)
                script_vocabs[script] = len(vocab)
        min_vocab = min(script_vocabs.values())
        print(f"\nVocab-proportional scaling (base={args.samples_per_script}):")
        tasks = []
        for script in valid_scripts:
            scale = math.sqrt(script_vocabs[script] / min_vocab)
            target = int(args.samples_per_script * scale)
            tasks.append((script, target))
            print(f"  {script:<15s} vocab={script_vocabs[script]:>5d}  "
                  f"scale={scale:.2f}x  samples={target}")
    else:
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

    # Resolve styles
    if args.style == "all":
        styles_to_gen = list(STYLES.keys())
        style_desc = ', '.join(f'{s} ({STYLES[s]["proportion"]:.0%})' for s in styles_to_gen)
        print(f"\nGenerating all styles: {style_desc}")
    else:
        styles_to_gen = [args.style]
        print(f"\nGenerating style: {args.style}")

    # Word images
    chunks, next_shard_idx, skipped = build_word_chunks(
        tasks, script_fonts, word_lists, args, shard_dir, styles_to_gen)
    if skipped > 0:
        print(f"\nResuming: {skipped} images in existing shards, "
              f"{sum(c[1] for c in chunks)} remaining")
    if chunks:
        run_generation_pool(chunks, _generate_word_batch, args.workers, "word")
    else:
        print("All shards exist. Done.")

    # Character images
    if args.include_chars:
        char_chunks = build_char_chunks(valid_scripts, script_fonts, tasks, args,
                                        shard_dir, next_shard_idx)
        if char_chunks:
            print(f"\n  {len(char_chunks)} char chunks across "
                  f"{min(args.workers, len(char_chunks))} workers\n")
            run_generation_pool(char_chunks, _generate_char_batch, args.workers, "char")
        else:
            print("  All char shards exist.")

    save_metadata(valid_scripts, args, shard_dir)


if __name__ == "__main__":
    main()
