#!/usr/bin/env python3
"""Generate the stable dense-Han manifest from a reference CJK font."""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FONT = ROOT / "training_data" / "fonts" / "NotoSansCJKsc-Regular.otf"
VOCAB = ROOT / "training_data" / "corpora" / "cjk_vocab.txt"
OUTPUT = ROOT / "training_data" / "corpora" / "han_dense_chars.txt"
OUTLINE_OP_THRESHOLD = 60


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    font = TTFont(args.font, lazy=True)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    dense = []
    missing = []
    chars = {line.strip() for line in VOCAB.read_text(encoding="utf-8").splitlines()}
    chars.update(unicodedata.normalize("NFKC", chr(cp))
                 for cp in range(0x2F00, 0x2FD6))
    for char in sorted(chars, key=ord):
        if len(char) != 1 or not (0x3400 <= ord(char) <= 0x9FFF):
            continue
        glyph_name = cmap.get(ord(char))
        if glyph_name is None:
            missing.append(char)
            continue
        pen = DecomposingRecordingPen(glyphs)
        glyphs[glyph_name].draw(pen)
        if len(pen.value) > OUTLINE_OP_THRESHOLD:
            dense.append(char)

    args.output.write_text("".join(dense) + "\n", encoding="utf-8")
    print(f"dense={len(dense)} missing={len(missing)} output={args.output}")


if __name__ == "__main__":
    main()
