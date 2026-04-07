#!/usr/bin/env python3
"""
Dump augmentation examples for visual inspection.

Renders words in multiple scripts, applies each augmentation op,
and saves before/after images to a directory.

Usage:
    python scripts/dump_augmentation_examples.py
    python scripts/dump_augmentation_examples.py --out-dir /tmp/aug_examples
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentation import AUGMENT_OPS, RandAugmentOCR


# Sample words per script with language codes for font lookup
SCRIPT_SAMPLES = {
    "latin": ("en", ["Hello", "Justice", "München", "café", "résumé"]),
    "cyrillic": ("ru", ["Привет", "Москва", "Закон", "Правда"]),
    "greek": ("el", ["Αθήνα", "δικαιοσύνη", "κόσμος"]),
    "arabic": ("ar", ["مرحبا", "عدالة", "محكمة"]),
    "hebrew": ("he", ["שלום", "משפט", "ירושלים"]),
    "devanagari": ("hi", ["नमस्ते", "न्याय", "अदालत"]),
    "bengali": ("bn", ["নমস্কার", "বিচার", "কলকাতা"]),
    "tamil": ("ta", ["வணக்கம்", "நீதி", "சென்னை"]),
    "thai": ("th", ["สวัสดี", "กฎหมาย", "กรุงเทพ"]),
    "korean": ("ko", ["안녕하세요", "법원", "서울"]),
    "han_kana": ("zh", ["你好", "法院", "北京"]),
}


def find_font(lang: str) -> str | None:
    """Find a system font for a language."""
    try:
        result = subprocess.run(
            ["fc-list", f":lang={lang}", "file"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            path = line.strip().rstrip(":")
            if path and Path(path).exists():
                return path
    except Exception:
        pass
    return None


sys.path.insert(0, str(Path(__file__).parent))
import random\ndef random_ink_color(): return tuple(random.randint(0,80) for _ in range(3))\ndef random_bg_color(): return tuple(random.randint(180,255) for _ in range(3))


def render_word(text: str, font_path: str, height: int = 32) -> Image.Image | None:
    """Render a word with random ink/paper colors from train_lid."""
    try:
        font = ImageFont.truetype(font_path, size=22)
        dummy = Image.new("RGB", (1, 1))
        bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= 0 or text_h <= 0:
            return None

        pad_x, pad_y = 6, 4
        img_w = text_w + 2 * pad_x
        img_h = text_h + 2 * pad_y

        # Pick colors with contrast check (same as training)
        for _ in range(5):
            bg = random_bg_color()
            ink = random_ink_color()
            bg_lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            ink_lum = 0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2]
            if abs(bg_lum - ink_lum) > 60:
                break

        img = Image.new("RGB", (img_w, img_h), bg)
        ImageDraw.Draw(img).text(
            (pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink, font=font,
        )

        scale = height / img_h
        new_w = max(4, int(img_w * scale))
        return img.resize((new_w, height), Image.BILINEAR)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Dump augmentation examples")
    parser.add_argument("--out-dir", type=str, default="debug/augmentation_examples")
    parser.add_argument("--pipeline-samples", type=int, default=10,
                        help="Number of pipeline (multi-op) examples per script")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    print(f"Saving augmentation examples to {out_dir}/\n")

    # Find fonts
    fonts = {}
    for script, (lang, words) in SCRIPT_SAMPLES.items():
        font = find_font(lang)
        if font:
            fonts[script] = font
            print(f"  {script}: {Path(font).name}")
        else:
            print(f"  {script}: NO FONT FOUND — skipping")

    # 1. Per-op examples: show each augmentation on each script
    per_op_dir = out_dir / "per_op"
    per_op_dir.mkdir(exist_ok=True)

    for script, (lang, words) in SCRIPT_SAMPLES.items():
        if script not in fonts:
            continue

        word = words[0]
        orig = render_word(word, fonts[script])
        if orig is None:
            continue

        # Save original
        orig.save(per_op_dir / f"{script}_00_original.png")

        # Apply each op
        for i, op in enumerate(AUGMENT_OPS):
            try:
                random.seed(42 + i)
                result = op(orig.copy())
                result.save(per_op_dir / f"{script}_{i+1:02d}_{op.__name__}.png")
            except Exception as e:
                print(f"    {script}/{op.__name__}: FAILED ({e})")

    print(f"\n  Saved {len(AUGMENT_OPS)} per-op examples per script to {per_op_dir}/")

    # 2. Pipeline examples: show RandAugmentOCR with 2 ops stacked
    pipeline_dir = out_dir / "pipeline"
    pipeline_dir.mkdir(exist_ok=True)

    aug = RandAugmentOCR(n_ops=2, p=1.0)

    for script, (lang, words) in SCRIPT_SAMPLES.items():
        if script not in fonts:
            continue

        for j in range(args.pipeline_samples):
            word = random.choice(words)
            orig = render_word(word, fonts[script])
            if orig is None:
                continue

            random.seed(j * 100 + hash(script))
            result = aug(orig.copy())

            # Save side-by-side
            combined_w = orig.width + result.width + 4
            combined = Image.new("RGB", (combined_w, 32), (200, 200, 200))
            combined.paste(orig, (0, 0))
            combined.paste(result, (orig.width + 4, 0))
            combined.save(pipeline_dir / f"{script}_{j:02d}.png")

    print(f"  Saved {args.pipeline_samples} pipeline examples per script to {pipeline_dir}/")

    # 3. Heavy augmentation: 3-4 ops stacked (stress test)
    heavy_dir = out_dir / "heavy"
    heavy_dir.mkdir(exist_ok=True)

    heavy_aug = RandAugmentOCR(n_ops=4, p=1.0)

    for script, (lang, words) in SCRIPT_SAMPLES.items():
        if script not in fonts:
            continue

        for j in range(5):
            word = random.choice(words)
            orig = render_word(word, fonts[script])
            if orig is None:
                continue

            random.seed(j * 200 + hash(script))
            result = heavy_aug(orig.copy())

            combined_w = orig.width + result.width + 4
            combined = Image.new("RGB", (combined_w, 32), (200, 200, 200))
            combined.paste(orig, (0, 0))
            combined.paste(result, (orig.width + 4, 0))
            combined.save(heavy_dir / f"{script}_{j:02d}.png")

    print(f"  Saved 5 heavy (4-op) examples per script to {heavy_dir}/")
    print(f"\nDone. Open {out_dir}/ to inspect.")


if __name__ == "__main__":
    main()
