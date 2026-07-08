#!/usr/bin/env python3
"""Render sample words for each script and save as images for visual inspection."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from src.taxonomy import SCRIPTS, SCRIPT_TO_GROUP
from src.data.word_lists import load_word_list
from src.data.fonts import find_fonts_for_script, build_weighted_font_list
from src.data.rendering import render_word, resize_or_pad, font_covers_text

OUT_DIR = Path("render_check")
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)
SAMPLES_PER_SCRIPT = 10

for script in SCRIPTS:
    if script == "emoji":
        continue
    words = load_word_list(script)
    if not words:
        print(f"{script:<15s} NO WORDS")
        continue
    fonts = find_fonts_for_script(script)
    weighted = build_weighted_font_list(fonts, words[0])
    if not weighted:
        print(f"{script:<15s} NO FONTS")
        continue

    unique = list(set(weighted))
    images = []
    used_fonts = set()

    for w in random.sample(words, min(SAMPLES_PER_SCRIPT * 3, len(words))):
        if len(images) >= SAMPLES_PER_SCRIPT:
            break
        f = random.choice(unique)
        if not font_covers_text(f, w):
            continue
        img = render_word(w, f, 32, clean=True)
        if img is None:
            continue
        img = resize_or_pad(img, 32, 192)
        images.append((img, w, Path(f).name))
        used_fonts.add(Path(f).name)

    if not images:
        print(f"{script:<15s} RENDER FAILED")
        continue

    # Stitch into one tall image
    combined = Image.new("RGB", (192, 32 * len(images)), (255, 255, 255))
    for i, (img, w, fname) in enumerate(images):
        combined.paste(img, (0, i * 32))

    combined.save(OUT_DIR / f"{script}.png")
    print(f"{script:<15s} {len(images):>2d} images, {len(used_fonts)} unique fonts: {sorted(used_fonts)[:3]}")

print(f"\nSaved to {OUT_DIR}/")
