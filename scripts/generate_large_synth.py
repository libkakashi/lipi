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
    """Find all usable fonts, categorize them, and return a weighted list.

    Returns a list where common font categories (sans, serif) appear more
    often than rare ones (handwriting), matching real-world text distribution.

    Distribution target:
      40% sans-serif (modern docs, signs, forms, UI)
      25% serif (books, legal, newspapers)
      15% monospace (code, typewriter, receipts)
      10% handwriting (notes, casual, signatures)
      10% other (misc fonts that render Latin)
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
    raw_fonts = []

    for d in font_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.suffix.lower() in extensions and f.stat().st_size >= min_size:
                raw_fonts.append(str(f))

    # Categorize by name
    sans_kw = ['helvetica', 'arial', 'verdana', 'futura', 'avenir', 'gill', 'gothic',
                'grotesk', 'tahoma', 'trebuchet', 'calibri', 'geneva', 'lucida', 'sans',
                'roboto', 'open', 'lato', 'inter', 'noto sans', 'source sans']
    serif_kw = ['times', 'georgia', 'garamond', 'baskerville', 'palatino', 'bodoni',
                'cambria', 'charter', 'book', 'roman', 'cochin', 'didot', 'iowan',
                'noto serif', 'source serif', 'libre', 'serif']
    mono_kw = ['courier', 'menlo', 'consolas', 'monaco', 'mono', 'code', 'terminal']
    hand_kw = ['brush', 'script', 'hand', 'cursive', 'comic', 'marker', 'zapfino',
               'snell', 'bradley', 'chalkboard', 'noteworthy', 'papyrus']

    categories = {'sans': [], 'serif': [], 'mono': [], 'hand': [], 'other': []}

    for f in raw_fonts:
        name = Path(f).stem.lower()
        if any(k in name for k in mono_kw):
            categories['mono'].append(f)
        elif any(k in name for k in hand_kw):
            categories['hand'].append(f)
        elif any(k in name for k in serif_kw):
            categories['serif'].append(f)
        elif any(k in name for k in sans_kw):
            categories['sans'].append(f)
        else:
            categories['other'].append(f)

    # Build weighted font list
    # Target: sans 40%, serif 25%, mono 15%, hand 10%, other 10%
    target_total = 200  # target list size
    weights = {'sans': 0.40, 'serif': 0.25, 'mono': 0.15, 'hand': 0.10, 'other': 0.10}

    weighted = []
    for cat, weight in weights.items():
        cat_fonts = categories[cat]
        if not cat_fonts:
            continue
        n_slots = max(1, int(target_total * weight))
        # Repeat fonts to fill slots (small categories get repeated more)
        for i in range(n_slots):
            weighted.append(cat_fonts[i % len(cat_fonts)])

    print(f"  Font categories: " + ", ".join(f"{k}={len(v)}" for k, v in categories.items()))
    print(f"  Weighted font list: {len(weighted)} entries")

    return weighted


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


def _font_has_glyphs(font, text="Hello"):
    """Check if a font renders real text, not boxes/bars/symbols.

    Tests:
    1. Has ink (not blank)
    2. Different characters render differently (not all same glyph)
    3. Characters have reasonable aspect ratio (not bars)
    """
    try:
        # Test 1: renders something
        img = Image.new("L", (200, 50), 255)
        ImageDraw.Draw(img).text((5, 5), text, fill=0, font=font)
        arr = np.array(img)
        ink = (arr < 200).sum()
        if ink < len(text) * 5:
            return False

        # Test 2: different characters produce different images
        renders = {}
        for ch in set(text):
            ch_img = Image.new("L", (40, 40), 255)
            ImageDraw.Draw(ch_img).text((5, 5), ch, fill=0, font=font)
            renders[ch] = np.array(ch_img)

        # Compare pairs — at least some should differ
        chars = list(renders.keys())
        if len(chars) >= 2:
            diffs = 0
            for i in range(min(3, len(chars))):
                for j in range(i + 1, min(4, len(chars))):
                    diff = np.abs(renders[chars[i]].astype(int) - renders[chars[j]].astype(int)).sum()
                    if diff > 50:
                        diffs += 1
            if diffs == 0:
                return False  # all characters look the same — symbol font

        # Test 3: characters aren't too tall/narrow (bars) or too wide (blocks)
        bbox = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), "H", font=font)
        h_w = bbox[2] - bbox[0]
        h_h = bbox[3] - bbox[1]
        if h_h > 0 and (h_w / h_h > 3 or h_w / h_h < 0.15):
            return False  # weird aspect ratio

        return True
    except Exception:
        return False


def render_word_advanced(
    text: str,
    fonts: list[str],
    height: int = 32,
    augmentor=None,
) -> Image.Image | None:
    """Render a word with random font, size, color, and background.

    Validates that:
    - Font can render the characters (no replacement boxes)
    - Text fits within the image height
    - Reasonable width (not too wide or narrow)

    Returns None if rendering fails.
    """
    # Try up to 3 fonts
    for _ in range(3):
        font_path = random.choice(fonts)

        # Start with a size that fits the height, then randomize slightly
        base_size = max(12, height - 8)
        font_size = random.randint(max(10, base_size - 6), base_size + 2)

        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except (IOError, OSError):
            continue

        # Check font can render this text
        if not _font_has_glyphs(font, text):
            continue

        # Measure text
        dummy = Image.new("RGB", (1, 1))
        bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= 0 or text_h <= 0:
            continue

        # Skip if text is too tall (would be cropped)
        if text_h > height - 2:
            # Reduce font size and retry
            font_size = int(font_size * (height - 4) / text_h)
            font_size = max(8, font_size)
            try:
                font = ImageFont.truetype(font_path, size=font_size)
                bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if text_w <= 0 or text_h <= 0 or text_h > height - 2:
                    continue
            except:
                continue

        # Image dimensions with padding
        pad_h = random.randint(2, 6)
        img_w = text_w + 2 * pad_h
        img_h = height

        # Create background
        img = random_background(img_w, img_h)
        draw = ImageDraw.Draw(img)

        # Center text vertically
        y_offset = max(0, (img_h - text_h) // 2 - bbox[1])
        x_offset = pad_h + random.randint(-1, 1)

        # Draw text
        color = random_text_color()
        draw.text((x_offset, y_offset), text, fill=color, font=font)

        # Apply augmentation
        if augmentor is not None:
            img = augmentor(img)

        return img

    return None


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

    # Filter: each font must actually render Latin text
    print("  Filtering fonts...")
    working_fonts = []
    seen = set()
    for f in fonts:
        if f in seen:
            working_fonts.append(f)  # keep duplicates from weighting
            continue
        seen.add(f)
        try:
            font = ImageFont.truetype(f, size=20)
            if _font_has_glyphs(font, "Hello"):
                working_fonts.append(f)
        except:
            pass
    fonts = working_fonts
    print(f"Fonts (filtered): {len(fonts)}")

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
