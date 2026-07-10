#!/usr/bin/env python3
"""Check which fonts work for each script."""
from pathlib import Path
from PIL import ImageFont, Image, ImageDraw
import numpy as np

FONT_DIR = Path(__file__).parent.parent / "training_data" / "fonts"

print(f"Font dir: {FONT_DIR}")
print(f"Fonts found: {list(FONT_DIR.glob('*.*'))}\n")

TEST_WORDS = {
    "latin": "Hello",
    "arabic": "مرحبا",
    "hebrew": "שלום",
    "han_sparse": "你好",
    "han_dense": "語學",
    "kana": "あいう",
    "korean": "안녕",
    "devanagari": "नमस्ते",
    "bengali": "নমস্কার",
    "tamil": "வணக்கம்",
    "thai": "สวัสดี",
}

for f in sorted(FONT_DIR.glob("*.*")):
    try:
        font = ImageFont.truetype(str(f), size=24)
        name = font.getname()
        renders = []
        for script, word in TEST_WORDS.items():
            img = Image.new("L", (300, 50), 255)
            ImageDraw.Draw(img).text((5, 5), word, fill=0, font=font)
            ink = (np.array(img) < 200).sum()
            if ink > len(word) * 3:
                renders.append(script)
        print(f"  OK: {f.name:40s} renders: {', '.join(renders) if renders else 'NOTHING'}")
    except Exception as e:
        print(f"  ERR: {f.name:40s} {e}")
