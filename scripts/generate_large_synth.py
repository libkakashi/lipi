#!/usr/bin/env python3
"""
Large-scale synthetic OCR data generator.

Generates millions of word crop images with diverse fonts, backgrounds,
colors, and augmentations. Output: LMDB ready for training.

Features:
  - Auto-discovers all system fonts
  - Random backgrounds (solid, gradient, textured noise)
  - Random text/background color combinations
  - Font size variation
  - Integrated augmentation (18 transforms)
  - Multi-process generation for speed
  - LMDB output for fast training

Usage:
    python scripts/generate_large_synth.py \
        --words training_data/word_lists/english_common.txt \
        --output training_data/datasets/en_synth_large \
        --n-per-word 50 \
        --augment \
        --workers 8
"""

import argparse
import io
import os
import random
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_fonts(font_dirs=None, min_size=10000):
    """Find all usable TrueType/OpenType fonts on the system.

    Args:
        font_dirs: List of directories to scan. None = auto-detect OS defaults.
        min_size: Minimum font file size in bytes (skip tiny/broken fonts).

    Returns:
        List of font file paths.
    """
    if font_dirs is None:
        font_dirs = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
            os.path.expanduser("~/.fonts"),
            "C:\\Windows\\Fonts",
        ]

    extensions = {".ttf", ".ttc", ".otf"}
    fonts = []

    for d in font_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.suffix.lower() in extensions and f.stat().st_size >= min_size:
                fonts.append(str(f))

    return fonts


def random_background(w, h):
    """Generate a random background image."""
    bg_type = random.choice(["solid", "solid", "gradient", "noise"])

    if bg_type == "solid":
        # Random light color (paper-like)
        r = random.randint(200, 255)
        g = random.randint(200, 255)
        b = random.randint(200, 255)
        return Image.new("RGB", (w, h), (r, g, b))

    elif bg_type == "gradient":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        c1 = np.array([random.randint(210, 255) for _ in range(3)])
        c2 = np.array([random.randint(210, 255) for _ in range(3)])
        for x in range(w):
            t = x / max(w - 1, 1)
            arr[:, x] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        return Image.fromarray(arr)

    elif bg_type == "noise":
        base = random.randint(220, 250)
        noise = np.random.randint(-15, 15, (h, w, 3))
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def random_text_color():
    """Random dark color for text."""
    style = random.choice(["black", "dark", "colored"])

    if style == "black":
        v = random.randint(0, 40)
        return (v, v, v)
    elif style == "dark":
        return tuple(random.randint(0, 80) for _ in range(3))
    elif style == "colored":
        # Dark but colored (blue ink, red stamp, etc.)
        return (
            random.randint(0, 120),
            random.randint(0, 80),
            random.randint(0, 120),
        )


def render_word_advanced(
    text: str,
    fonts: list[str],
    height: int = 32,
    augmentor=None,
) -> Image.Image | None:
    """Render a word with random font, size, color, and background.

    Returns None if rendering fails (bad font for this text).
    """
    # Pick random font and size
    font_path = random.choice(fonts)
    font_size = random.randint(18, 28)

    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except (IOError, OSError):
        return None

    # Measure text
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if text_w <= 0 or text_h <= 0:
        return None

    # Image dimensions with padding
    pad_h = random.randint(2, 8)
    pad_v = random.randint(2, 6)
    img_w = text_w + 2 * pad_h
    img_h = height

    # Create background
    img = random_background(img_w, img_h)
    draw = ImageDraw.Draw(img)

    # Center text vertically
    y_offset = max(0, (img_h - text_h) // 2 - bbox[1])
    x_offset = pad_h + random.randint(-2, 2)

    # Draw text
    color = random_text_color()
    draw.text((x_offset, y_offset), text, fill=color, font=font)

    # Apply augmentation
    if augmentor is not None:
        img = augmentor(img)

    return img


def generate_batch(args):
    """Generate a batch of images for a list of words. Called by each worker."""
    words, fonts, n_per_word, height, augment, seed = args

    random.seed(seed)
    np.random.seed(seed)

    augmentor = None
    if augment:
        from src.data.augmentation import RandAugmentOCR
        augmentor = RandAugmentOCR(n_ops=2, p=0.5)

    results = []  # list of (image_bytes, label)

    for word in words:
        for _ in range(n_per_word):
            img = render_word_advanced(text=word, fonts=fonts, height=height, augmentor=augmentor)
            if img is None:
                # Retry with different font
                img = render_word_advanced(text=word, fonts=fonts, height=height, augmentor=augmentor)
            if img is None:
                continue

            # Encode as JPEG
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(80, 95))
            results.append((buf.getvalue(), word))

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate large-scale synthetic OCR data")
    parser.add_argument("--words", type=str, required=True, help="Word list file (one per line)")
    parser.add_argument("--output", type=str, required=True, help="Output LMDB path")
    parser.add_argument("--n-per-word", type=int, default=50, help="Variants per word")
    parser.add_argument("--max-words", type=int, default=None, help="Limit number of unique words")
    parser.add_argument("--height", type=int, default=32, help="Image height")
    parser.add_argument("--augment", action="store_true", help="Apply augmentation during generation")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--font-dirs", type=str, nargs="+", default=None, help="Font directories")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load word list
    words = [line.strip() for line in Path(args.words).read_text().splitlines() if line.strip()]
    if args.max_words:
        words = words[:args.max_words]
    words = list(set(words))  # deduplicate
    print(f"Words: {len(words)} unique")

    # Discover fonts
    fonts = discover_fonts(args.font_dirs)
    print(f"Fonts: {len(fonts)} found")

    if not fonts:
        print("ERROR: No fonts found. Install fonts or specify --font-dirs")
        sys.exit(1)

    # Verify some fonts work
    working_fonts = []
    for f in fonts:
        try:
            ImageFont.truetype(f, size=20)
            working_fonts.append(f)
        except:
            pass
    fonts = working_fonts
    print(f"Fonts (working): {len(fonts)}")

    total = len(words) * args.n_per_word
    print(f"Generating {total:,} images ({len(words)} words × {args.n_per_word} variants)")

    # Split work across workers
    n_workers = args.workers or min(cpu_count(), 16)
    chunk_size = max(1, len(words) // n_workers)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        seed = args.seed + i
        chunks.append((chunk_words, fonts, args.n_per_word, args.height, args.augment, seed))

    print(f"Workers: {n_workers}, chunks: {len(chunks)}")

    # Generate in parallel
    t0 = time.time()

    all_results = []
    with Pool(n_workers) as pool:
        for i, batch_results in enumerate(pool.imap_unordered(generate_batch, chunks)):
            all_results.extend(batch_results)
            elapsed = time.time() - t0
            rate = len(all_results) / elapsed
            print(f"  {len(all_results):,}/{total:,} generated ({rate:.0f}/sec, "
                  f"{elapsed:.0f}s elapsed)", end="\r")

    print(f"\n  Generated {len(all_results):,} images in {time.time()-t0:.0f}s")

    # Shuffle
    random.shuffle(all_results)

    # Write to LMDB
    print(f"Writing to LMDB: {args.output}")
    import lmdb

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    map_size = max(len(all_results) * 20000, 1 << 30)  # ~20KB per image estimate, min 1GB
    env = lmdb.open(str(output_path), map_size=map_size)

    with env.begin(write=True) as txn:
        for i, (img_bytes, label) in enumerate(all_results):
            txn.put(f"image-{i+1:09d}".encode(), img_bytes)
            txn.put(f"label-{i+1:09d}".encode(), label.encode("utf-8"))

            if (i + 1) % 100000 == 0:
                print(f"  {i+1:,}/{len(all_results):,} written")

        txn.put(b"num-samples", str(len(all_results)).encode())

    env.close()

    elapsed = time.time() - t0
    size_gb = sum(os.path.getsize(f) for f in output_path.glob("*")) / 1e9
    print(f"\nDone!")
    print(f"  Samples: {len(all_results):,}")
    print(f"  LMDB size: {size_gb:.1f} GB")
    print(f"  Total time: {elapsed:.0f}s")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
