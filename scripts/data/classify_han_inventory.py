#!/usr/bin/env python3
"""Classify the full Han inventory (URO + Ext A) by rendered visual density.

Stage 1 of the vocab rebuild: every one of the 27,584 BMP Han codepoints
gets deterministic visual-density statistics measured the way LID-2 sees
characters — rendered at training resolution across the text-font pool —
instead of lexical proxies such as outline ops or IDS component count.

Fonts equalize apparent ink ("typographic gray": dense glyphs get thinner
strokes), so raw ink ratio barely separates complexity classes. What does:

  strokes  — mean black/white crossings per row+column at a 112 px body,
             where every stroke still resolves: a direct, font-honest
             stroke-count proxy, monotonic in complexity
  merge    — crossings at the 28 px training body divided by crossings at
             112 px: ~1.0 while strokes survive downscaling, collapsing
             once they fuse into texture. Low merge IS the dense regime —
             the property LID-2 can actually see.

Per char: median across covering fonts. Characters no text font covers
are flagged with n_fonts=0; they cannot be rendered in training anyway.

Writes training_data/han_inventory_stats.tsv:
  char  cp  n_fonts  ink  strokes  merge  in_vocab
and prints the distribution summary for threshold/band selection.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FONT_DIR = ROOT / "training_data" / "fonts"
# Representative printed-text CJK fonts only — handwriting styles
# (MaShanZheng, Yomogi, ...) would blur the density metric.
CLASSIFIER_FONTS = [
    "NotoSansCJKsc-Regular.otf",
    "NotoSansCJKjp-Regular.otf",
    "NotoSansCJKkr-Regular.otf",
    "NotoSerifCJKsc-Regular.otf",
    "NotoSansSC[wght].ttf",
]
GLYPH_PX = 28          # glyph body inside a 32 px training line
GLYPH_PX_HI = 112      # 4x: strokes never merge at this size
INK_THRESHOLD = 128

VOCAB_PATH = ROOT / "training_data" / "corpora" / "cjk_vocab.txt"
OUTPUT = ROOT / "training_data" / "han_inventory_stats.tsv"


def inventory() -> list[int]:
    return [*range(0x4E00, 0xA000), *range(0x3400, 0x4DC0)]


def load_fonts():
    from fontTools.ttLib import TTFont

    fonts = []
    for name in CLASSIFIER_FONTS:
        path = FONT_DIR / name
        if not path.exists():
            print(f"  [skip] {name} not found")
            continue
        pil_lo = ImageFont.truetype(str(path), GLYPH_PX)
        pil_hi = ImageFont.truetype(str(path), GLYPH_PX_HI)
        cmap = TTFont(str(path), lazy=True, fontNumber=0).getBestCmap()
        fonts.append((name, pil_lo, pil_hi, cmap))
        print(f"  [font] {name}: {len(cmap)} mapped glyphs")
    if not fonts:
        raise SystemExit("no classifier fonts found under training_data/fonts")
    return fonts


def _ink_box(char: str, pil_font, size_px: int) -> np.ndarray | None:
    canvas = size_px + 12
    img = Image.new("L", (canvas, canvas), 255)
    ImageDraw.Draw(img).text((canvas // 2, canvas // 2), char,
                             font=pil_font, fill=0, anchor="mm")
    mask = np.asarray(img) < INK_THRESHOLD
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    box = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return box if box.shape[0] >= 3 and box.shape[1] >= 3 else None


def _crossings(box: np.ndarray) -> float:
    """Mean ink-run crossings per row plus per column (absolute counts —
    size-invariant while strokes resolve, dropping once they merge)."""
    as_i8 = box.astype(np.int8)
    row = np.abs(np.diff(as_i8, axis=1)).sum() / (2.0 * box.shape[0])
    col = np.abs(np.diff(as_i8, axis=0)).sum() / (2.0 * box.shape[1])
    return float(row + col)


def glyph_stats(char: str, pil_lo, pil_hi) -> tuple[float, float, float] | None:
    box_lo = _ink_box(char, pil_lo, GLYPH_PX)
    box_hi = _ink_box(char, pil_hi, GLYPH_PX_HI)
    if box_lo is None or box_hi is None:
        return None
    strokes = _crossings(box_hi)
    if strokes <= 0:
        return None
    merge = min(_crossings(box_lo) / strokes, 1.5)
    return float(box_lo.mean()), strokes, merge


def main() -> None:
    print("Loading fonts...")
    fonts = load_fonts()
    vocab = {line.strip() for line in
             VOCAB_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}

    rows = []
    uncovered = []
    cps = inventory()
    print(f"Rendering {len(cps)} codepoints x {len(fonts)} fonts "
          f"at {GLYPH_PX}px + {GLYPH_PX_HI}px...")
    for i, cp in enumerate(cps):
        char = chr(cp)
        inks, strokes_l, merges = [], [], []
        for _, pil_lo, pil_hi, cmap in fonts:
            if cp not in cmap:
                continue
            stats = glyph_stats(char, pil_lo, pil_hi)
            if stats is not None:
                inks.append(stats[0])
                strokes_l.append(stats[1])
                merges.append(stats[2])
        if inks:
            rows.append((char, cp, len(inks), float(np.median(inks)),
                         float(np.median(strokes_l)), float(np.median(merges)),
                         char in vocab))
        else:
            uncovered.append((char, cp))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(cps)} ({len(uncovered)} uncovered so far)")

    with OUTPUT.open("w", encoding="utf-8") as f:
        f.write("char\tcp\tn_fonts\tink\tstrokes\tmerge\tin_vocab\n")
        for char, cp, n, ink, strokes, merge, in_v in rows:
            f.write(f"{char}\t{cp:04X}\t{n}\t{ink:.4f}\t{strokes:.3f}"
                    f"\t{merge:.4f}\t{int(in_v)}\n")
        for char, cp in uncovered:
            f.write(f"{char}\t{cp:04X}\t0\t\t\t\t{int(char in vocab)}\n")

    ink = np.array([r[3] for r in rows])
    strokes = np.array([r[4] for r in rows])
    merge = np.array([r[5] for r in rows])
    print(f"\nCovered: {len(rows)}  uncovered: {len(uncovered)}")
    print(f"Wrote {OUTPUT}")

    def q(a, p):
        return float(np.percentile(a, p))

    print("\nDistribution (covered chars):")
    for name, a in (("ink", ink), ("strokes", strokes), ("merge", merge)):
        print(f"  {name:<8} p5={q(a,5):.3f} p25={q(a,25):.3f} "
              f"p50={q(a,50):.3f} p75={q(a,75):.3f} p95={q(a,95):.3f}")
    print(f"  corr(strokes, merge) = "
          f"{float(np.corrcoef(strokes, merge)[0, 1]):+.3f}")

    # Merge-vs-strokes profile: where does downscaling start destroying
    # strokes? The dense regime starts where the merge curve leaves ~1.0.
    print("\nMerge ratio by stroke-crossing decile:")
    edges = np.percentile(strokes, np.arange(0, 101, 10))
    for d in range(10):
        m = (strokes >= edges[d]) & (strokes <= edges[d + 1])
        print(f"  strokes {edges[d]:5.2f}-{edges[d+1]:5.2f}: "
              f"merge p50={float(np.median(merge[m])):.3f} "
              f"p25={q(merge[m],25):.3f}  n={int(m.sum())}")


if __name__ == "__main__":
    main()
