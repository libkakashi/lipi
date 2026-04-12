#!/usr/bin/env python3
"""
Generate synthetic training data for LID and MoE training.

Usage:
    python scripts/generate_data.py --samples-per-script 10000 --out data/shards
    python scripts/generate_data.py --samples-per-script 30000 --balance-groups --include-chars --out data/shards
"""

import argparse
import json
import random
import sys
import time

import numpy as np
from pathlib import Path
from multiprocessing import Pool

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID, SCRIPT_TO_ID, GROUPS
from src.data.color import rgb_to_input
from src.data.augmentation import (
    RandAugmentOCR,
    jpeg_compress, blur, low_resolution, photocopy,
    exposure_jitter, uneven_lighting, glare, striped_shadow,
    rotation, perspective_warp, wave_distortion,
    bleed_through, fold_crease, aged_document, scanner_edge, water_stain,
    stroke_variation, smudge,
    variable_baseline, slant, ink_fade, variable_stroke, lined_paper,
    noise, color_jitter, to_grayscale,
    occlusion, weather_damage,
)
from src.encoding.vocab import build_script_vocab
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
# Label encoding (tokenizers built once per worker process)
# ---------------------------------------------------------------------------

# Module-level cache for tokenizers (populated lazily in each worker)
_worker_group_tokenizers = None
_worker_active_groups = None
_worker_all_scripts = None
_worker_group_script_names = None


def _ensure_tokenizers():
    """Build tokenizers once per worker process (lazy init)."""
    global _worker_group_tokenizers, _worker_active_groups
    global _worker_all_scripts, _worker_group_script_names
    if _worker_group_tokenizers is not None:
        return

    import io
    from src.training.dataloader import build_script_tokenizers

    _worker_active_groups = list(GROUPS)
    _worker_all_scripts = list(SCRIPT_TO_GROUP.keys())
    # Suppress per-script vocab prints in worker processes
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        _worker_group_tokenizers, _, _worker_group_script_names = build_script_tokenizers(
            _worker_all_scripts, _worker_active_groups)
    finally:
        sys.stdout = old_stdout


def _get_local_group_and_script(script: str):
    """Get (local_group_id, local_script_id) for a script."""
    _ensure_tokenizers()
    group_name = SCRIPT_TO_GROUP[script]
    local_gid = _worker_active_groups.index(group_name)
    local_sid = _worker_group_script_names[local_gid].index(script)
    return local_gid, local_sid


def encode_labels_for_shard(labels: list[str], script: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode labels into token IDs for a single script.

    Returns:
        target_ids: (N, max_len) zero-padded token IDs, dtype long
        target_lens: (N,) actual lengths, dtype long
    """
    from src.encoding.decompose import encode_text

    encoded = []
    max_len = 0
    for label in labels:
        ids = encode_text(label, script)
        encoded.append(ids)
        if len(ids) > max_len:
            max_len = len(ids)

    if max_len == 0:
        max_len = 1
    target_ids = torch.zeros(len(encoded), max_len, dtype=torch.long)
    target_lens = torch.zeros(len(encoded), dtype=torch.long)
    for i, ids in enumerate(encoded):
        target_lens[i] = len(ids)
        if ids:
            target_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    return target_ids, target_lens

# ---------------------------------------------------------------------------
# Data styles — font selection + augmentation ops per style
# ---------------------------------------------------------------------------

_HANDWRITING_KEYWORDS = {"caveat", "dancing", "indie", "patrick", "kalam",
                         "nanumpen", "chilanka", "handwrit", "cursive"}
_DISPLAY_KEYWORDS = {"permanent", "amatic", "lobster", "pacifico", "special",
                     "display", "condensed"}

STYLES = {
    "clean": {
        "proportion": 0.10,
        "ops": [],
        "font_filter": "regular",
        "clean_render": True,
    },
    "printed": {
        "proportion": 0.35,
        "ops": [jpeg_compress, blur, photocopy, uneven_lighting,
                fold_crease, bleed_through, aged_document, scanner_edge,
                water_stain, exposure_jitter, noise, to_grayscale],
        "font_filter": "regular",
        "clean_render": False,
    },
    "handwritten": {
        "proportion": 0.25,
        "ops": [stroke_variation, smudge, variable_stroke,
                variable_baseline, slant, ink_fade, lined_paper,
                noise, exposure_jitter, rotation, wave_distortion,
                bleed_through],
        "font_filter": "handwriting",
        "clean_render": False,
    },
    "signage": {
        "proportion": 0.20,
        "ops": [perspective_warp, rotation, exposure_jitter, glare,
                weather_damage, color_jitter, uneven_lighting, occlusion],
        "font_filter": "display",
        "clean_render": False,
    },
    "degraded": {
        "proportion": 0.10,
        "ops": [blur, jpeg_compress, noise, exposure_jitter,
                low_resolution, rotation, color_jitter,
                wave_distortion, striped_shadow],
        "font_filter": "all",
        "clean_render": False,
    },
}


def filter_fonts_by_style(fonts: list[str], style: str) -> list[str]:
    """Filter font list by style. Falls back to full list if too few matches.

    For non-Latin scripts, handwriting/display fonts rarely exist.
    The augmentation pipeline handles style simulation in those cases.
    Threshold of 5 unique fonts prevents degenerate font selection.
    """
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
        elif font_filter == "display" and is_display:
            filtered.append(f)

    # Fall back to regular fonts (not all) if too few style-specific fonts.
    # For non-Latin scripts, handwriting/display fonts usually can't render
    # the text anyway — augmentation handles the style simulation.
    if len(set(filtered)) < 5:
        regular = [f for f in fonts
                   if not any(k in Path(f).name.lower() for k in _HANDWRITING_KEYWORDS | _DISPLAY_KEYWORDS)]
        return regular if regular else fonts

    return filtered


# ---------------------------------------------------------------------------
# Renderable character list (from frozen vocab)
# ---------------------------------------------------------------------------

def get_renderable_chars(script: str) -> list[str]:
    """Get standalone-renderable characters/clusters for training image generation.

    For han_kana: returns all CJK Unified + Ext-A characters plus kana.
    For fusion scripts: returns base chars + all fusion clusters.
    For no-fusion scripts: returns chars from the codec.
    For korean: returns chars from the codec.
    """
    import unicodedata
    from src.encoding.config import (
        NO_FUSION_SCRIPTS, FUSION_BASE_CHARS, get_fusion_codec,
        get_korean_codec,
    )

    if script == "han_kana":
        chars = []
        for cp in range(0x3400, 0x4DC0):
            c = chr(cp)
            if unicodedata.category(c) != 'Cn':
                chars.append(c)
        for cp in range(0x4E00, 0xA000):
            c = chr(cp)
            if unicodedata.category(c) != 'Cn':
                chars.append(c)
        for cp in range(0x3041, 0x3097):
            chars.append(chr(cp))
        for cp in range(0x30A1, 0x30FB):
            chars.append(chr(cp))
        chars.append('\u30FC')
        return chars

    if script in FUSION_BASE_CHARS:
        codec = get_fusion_codec(script)
        # Base chars (single codepoints) + fusion clusters (multi-codepoint)
        return codec.base_chars + codec.fusions

    if script == "korean":
        codec = get_korean_codec()
        return [t for t in codec.tokens
                if t.strip() and ord(t[0]) > 32]

    if script in NO_FUSION_SCRIPTS:
        codec = NO_FUSION_SCRIPTS[script]
        return [c for c in codec.chars
                if c.strip() and ord(c) > 32
                and not unicodedata.category(c).startswith('M')]

    return []


# ---------------------------------------------------------------------------
# Punctuation / number mixing
# ---------------------------------------------------------------------------

# Common punctuation and number patterns seen in real documents
_PUNCT_CHARS = list(".,;:!?-()\"'/@#&*+=$%")
_TYPO_PUNCT = ["\u2018", "\u2019", "\u201C", "\u201D", "\u201E",  # ' ' " " „
               "\u2013", "\u2014", "\u2010",                       # – — ‐
               "\u2026",                                            # …
               "\u00AB", "\u00BB"]                                  # « »
_DIGITS = list("0123456789")
_CURRENCY = list("$")

def _typo_quote(w: str) -> str:
    """Wrap word in typographic quotes."""
    style = random.choice(["single", "double", "guillemet", "german"])
    if style == "single":
        return "\u2018" + w + "\u2019"       # 'word'
    elif style == "double":
        return "\u201C" + w + "\u201D"       # "word"
    elif style == "guillemet":
        return "\u00AB" + w + "\u00BB"       # «word»
    else:
        return "\u201E" + w + "\u201C"       # „word"


# Patterns safe for ALL scripts — original word is always preserved,
# so LID-1 can still identify the script from the non-ASCII characters.
_MIX_PATTERNS_UNIVERSAL = [
    # Trailing punctuation: word. word, word; word! word?
    (25, lambda w: w + random.choice(".,;:!?")),
    # Trailing typographic: word… word–
    (5,  lambda w: w + random.choice(["\u2026", "\u2013", "\u2014"])),
    # Leading/trailing quotes or parens: "word" (word) 'word'
    (5,  lambda w: random.choice('"\'(') + w + random.choice('"\')')),
    # Typographic quotes: "word" 'word' «word» „word"
    (5,  lambda w: _typo_quote(w)),
    # Number prefix: 1. word, 2) word, (3) word
    (10, lambda w: random.choice(_DIGITS) + random.choice(".)") + " " + w),
    # Number suffix: word-1, word/2
    (5,  lambda w: w + random.choice("-/") + random.choice(_DIGITS)),
    # Mixed: word-word, word/word (keeps both halves from word list)
    (3,  lambda w: w + random.choice("-/&") + w[:max(2, len(w) // 2)]),
    # Short number (1-3 digits) — common in all scripts (page nos, counts, refs)
    (5,  lambda w: str(random.randint(1, 999))),
]

# Patterns that produce PURE numbers/ASCII — only safe for Latin,
# since the output has no script-specific chars for LID-1 to route on.
_MIX_PATTERNS_LATIN_ONLY = [
    # Pure number sequences: 12345, 1,234, 12.34
    (8,  lambda w: _random_number()),
    # Date-like: 12/03/2024, 12-03-24
    (5,  lambda w: _random_date()),
    # Currency: $56.78
    (5,  lambda w: random.choice(_CURRENCY) + _random_number()),
    # Phone/ID-like: 123-456-7890
    (3,  lambda w: "-".join("".join(random.choices(_DIGITS, k=random.randint(2, 4)))
                            for _ in range(random.randint(2, 3)))),
    # Section/reference: #34, *note
    (3,  lambda w: random.choice("#*") + "".join(random.choices(_DIGITS, k=random.randint(1, 3)))),
]

_UNIVERSAL_WEIGHTS = [p[0] for p in _MIX_PATTERNS_UNIVERSAL]
_UNIVERSAL_BUILDERS = [p[1] for p in _MIX_PATTERNS_UNIVERSAL]
_LATIN_WEIGHTS = [p[0] for p in _MIX_PATTERNS_LATIN_ONLY]
_LATIN_BUILDERS = [p[1] for p in _MIX_PATTERNS_LATIN_ONLY]
_ALL_WEIGHTS = _UNIVERSAL_WEIGHTS + _LATIN_WEIGHTS
_ALL_BUILDERS = _UNIVERSAL_BUILDERS + _LATIN_BUILDERS


def _random_number() -> str:
    """Generate a random number string."""
    r = random.random()
    if r < 0.4:
        # Simple integer: 1-99999
        return str(random.randint(1, 99999))
    elif r < 0.7:
        # Comma-separated: 1,234 or 12,345
        n = random.randint(100, 999999)
        return f"{n:,}"
    else:
        # Decimal: 12.34, 0.5
        return f"{random.uniform(0.1, 9999):.{random.randint(1, 2)}f}"


def _random_date() -> str:
    """Generate a random date-like string."""
    d, m, y = random.randint(1, 31), random.randint(1, 12), random.randint(1950, 2025)
    sep = random.choice("-/.")
    fmt = random.choice(["dmy4", "dmy2", "ymd"])
    if fmt == "dmy4":
        return f"{d:02d}{sep}{m:02d}{sep}{y}"
    elif fmt == "dmy2":
        return f"{d:02d}{sep}{m:02d}{sep}{y % 100:02d}"
    else:
        return f"{y}{sep}{m:02d}{sep}{d:02d}"


def mix_punctuation(word: str, p: float = 0.15, script: str = "latin") -> str:
    """With probability p, mix punctuation/numbers into the word.

    Pure-number patterns (dates, currency, phone numbers) are only used for
    Latin, since they contain no script-specific characters and would confuse
    LID-1 routing if assigned to other scripts.

    Returns the original word unchanged (1-p) of the time.
    """
    if random.random() > p:
        return word
    if script == "latin":
        weights, builders = _ALL_WEIGHTS, _ALL_BUILDERS
    else:
        weights, builders = _UNIVERSAL_WEIGHTS, _UNIVERSAL_BUILDERS
    builder = random.choices(builders, weights=weights, k=1)[0]
    return builder(word)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunk_dir_exists(path: str) -> bool:
    """Check if an MDS chunk directory exists with data."""
    p = Path(path)
    return p.exists() and any(p.glob("shard.*.mds"))


# MDS column schema (shared with convert_to_mds.py)
MDS_COLUMNS = {
    "image": "ndarray:uint8",
    "label": "str",
    "script_id": "int",
    "group_id": "int",
    "target_ids": "ndarray:int64",
    "target_len": "int",
    "width": "int",
    "group_labels": "ndarray:int32",  # per-pixel group IDs (W,)
    "segments": "str",  # JSON: [{"group_id", "script_id", "text", "width", "offset"}]
}


_MAX_VAL_PER_SCRIPT = 500  # Set from args before workers spawn

def save_mds_samples(images, labels, script, train_dir, val_dir, chunk_id):
    """Write samples directly to train/val MDS splits. No merge needed."""
    if not images:
        return 0, [], []

    import hashlib

    from streaming import MDSWriter

    script_id = SCRIPT_TO_ID[script]
    group_id = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]

    from src.encoding.decompose import encode_text
    encoded = [encode_text(label, script) for label in labels]

    t_dir = str(Path(train_dir) / f"chunk_{chunk_id}")
    v_dir = str(Path(val_dir) / f"chunk_{chunk_id}")
    Path(t_dir).mkdir(parents=True, exist_ok=True)
    Path(v_dir).mkdir(parents=True, exist_ok=True)

    train_widths, val_widths = [], []
    val_count = 0
    with MDSWriter(out=t_dir, columns=MDS_COLUMNS, size_limit=1 << 26) as tw, \
         MDSWriter(out=v_dir, columns=MDS_COLUMNS, size_limit=1 << 26) as vw:
        for idx, (img_tensor, label, ids) in enumerate(zip(images, labels, encoded)):
            img_np = img_tensor.numpy()
            tids = np.array(ids, dtype=np.int64) if ids else np.zeros(1, dtype=np.int64)

            w = img_np.shape[2]
            gl = np.full(w, group_id, dtype=np.int32)
            segs = json.dumps([{
                "group_id": group_id, "script_id": script_id,
                "text": label, "width": w, "offset": 0,
            }])
            sample = {
                "image": img_np,
                "label": label,
                "script_id": script_id,
                "group_id": group_id,
                "target_ids": tids,
                "target_len": len(ids),
                "width": w,
                "group_labels": gl,
                "segments": segs,
            }
            h = int(hashlib.md5(f"{chunk_id}_{idx}_{label}".encode()).hexdigest(), 16)
            if h % 1000 < 100 and val_count < _MAX_VAL_PER_SCRIPT:  # 10% val, capped
                vw.write(sample)
                val_widths.append(img_np.shape[2])
                val_count += 1
            else:
                tw.write(sample)
                train_widths.append(img_np.shape[2])

    return len(images), train_widths, val_widths


# ---------------------------------------------------------------------------
# Mixed-script generation
# ---------------------------------------------------------------------------

def _generate_mixed_batch(args_tuple):
    """Generate mixed-script images: two words from different scripts side by side."""
    from PIL import Image
    count, all_script_info, h, mw, do_augment, chunk_id, train_dir, val_dir = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None
    t0 = time.time()

    images, labels, script_ids_list, group_ids_list, group_labels_list, segments_list = [], [], [], [], [], []
    attempts = 0

    # all_script_info: list of (script, fonts, words, script_id, group_id)
    while len(images) < count and attempts < count * 10:
        attempts += 1

        # Pick two different scripts from different groups
        s1_info = random.choice(all_script_info)
        s2_info = random.choice(all_script_info)
        if s1_info[0] == s2_info[0] or s1_info[4] == s2_info[4]:
            continue  # want different groups

        script1, fonts1, words1, sid1, gid1 = s1_info
        script2, fonts2, words2, sid2, gid2 = s2_info

        # Render word 1
        word1 = random.choice(words1)
        font1 = random.choice(fonts1)
        if not font_covers_text(font1, word1):
            continue
        img1 = render_word(word1, font1, h, clean=True)
        if img1 is None or not image_has_ink(img1):
            continue

        # Render word 2 with same style
        word2 = random.choice(words2)
        font2 = random.choice(fonts2)
        if not font_covers_text(font2, word2):
            continue
        img2 = render_word(word2, font2, h, clean=True)
        if img2 is None or not image_has_ink(img2):
            continue

        # Concatenate horizontally (optional small gap)
        gap = random.randint(0, 4)
        w_total = img1.width + gap + img2.width
        if w_total > mw:
            continue

        combined = Image.new("RGB", (w_total, h), (255, 255, 255))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width + gap, 0))

        combined = resize_or_pad(combined, h, mw)

        if aug is not None:
            combined = aug(combined)
            if not image_has_ink(combined, min_ink_pixels=5):
                continue

        # Label: concatenated text
        label = word1 + word2

        # Per-pixel group labels and segments based on render widths
        final_w = combined.width
        boundary = int(img1.width / w_total * final_w)
        gl = np.full(final_w, gid2, dtype=np.int32)
        gl[:boundary] = gid1

        segs = json.dumps([
            {"group_id": gid1, "script_id": sid1, "text": word1,
             "width": boundary, "offset": 0},
            {"group_id": gid2, "script_id": sid2, "text": word2,
             "width": final_w - boundary, "offset": boundary},
        ])

        images.append(rgb_to_input(combined))
        labels.append(label)
        script_ids_list.append(sid1)
        group_ids_list.append(gid1)
        group_labels_list.append(gl)
        segments_list.append(segs)

    # Write to MDS
    if not images:
        return chunk_id, "mixed", 0, [], []

    import hashlib

    from streaming import MDSWriter
    from src.encoding.decompose import encode_text

    t_dir = str(Path(train_dir) / f"chunk_{chunk_id}")
    v_dir = str(Path(val_dir) / f"chunk_{chunk_id}")
    Path(t_dir).mkdir(parents=True, exist_ok=True)
    Path(v_dir).mkdir(parents=True, exist_ok=True)

    train_widths, val_widths = [], []
    val_count = 0
    with MDSWriter(out=t_dir, columns=MDS_COLUMNS, size_limit=1 << 26) as tw, \
         MDSWriter(out=v_dir, columns=MDS_COLUMNS, size_limit=1 << 26) as vw:
        for idx, (img_tensor, label, sid, gid, gl, segs) in enumerate(
                zip(images, labels, script_ids_list, group_ids_list,
                    group_labels_list, segments_list)):
            # Encode using primary script (for legacy target_ids)
            script_name = SCRIPTS[sid]
            ids = encode_text(label, script_name)
            img_np = img_tensor.numpy()
            tids = np.array(ids, dtype=np.int64) if ids else np.zeros(1, dtype=np.int64)
            sample = {
                "image": img_np,
                "label": label,
                "script_id": sid,
                "group_id": gid,
                "target_ids": tids,
                "target_len": len(ids),
                "width": img_np.shape[2],
                "group_labels": gl,
                "segments": segs,
            }
            h_val = int(hashlib.md5(f"{chunk_id}_{idx}_{label}".encode()).hexdigest(), 16)
            if h_val % 1000 < 100 and val_count < _MAX_VAL_PER_SCRIPT:
                vw.write(sample)
                val_widths.append(img_np.shape[2])
                val_count += 1
            else:
                tw.write(sample)
                train_widths.append(img_np.shape[2])

    elapsed = time.time() - t0
    print(f"  {'mixed':<15} {len(images):>5} images in {elapsed:.0f}s", flush=True)
    return chunk_id, "mixed", len(images), train_widths, val_widths


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--samples-per-script", type=int, default=10000)
    parser.add_argument("--val-samples-per-script", type=int, default=500,
                        help="Max val samples per script (default: 500)")
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
    parser.add_argument("--max-width", type=int, default=768)
    parser.add_argument("--punct-prob", type=float, default=0.15,
                        help="Probability of mixing punctuation/numbers into a word (default: 0.15)")
    parser.add_argument("--include-chars", action="store_true")
    parser.add_argument("--char-reps", type=int, default=3)
    parser.add_argument("--mixed-script", type=int, default=0,
                        help="Number of mixed-script (two scripts side by side) samples to generate")
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

    if len(valid_scripts) < 1:
        print("ERROR: No scripts with fonts found")
        sys.exit(1)

    return script_fonts, valid_scripts


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def build_word_chunks(tasks, script_fonts, word_lists, args, shard_dir, styles_to_gen):
    """Build word-image chunks. Returns (chunks, next_chunk_idx, skipped)."""
    train_dir = str(shard_dir / "train")
    val_dir = str(shard_dir / "val")
    chunks = []
    chunk_idx = 0
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
                # Check if this chunk's train subdir already exists
                if chunk_dir_exists(str(Path(train_dir) / f"chunk_{chunk_idx:04d}")):
                    skipped += batch
                else:
                    chunks.append((script, batch, fonts, words,
                                  args.height, args.max_width, args.augment,
                                  chunk_idx, train_dir, val_dir,
                                  style, args.punct_prob))
                chunk_idx += 1
                remaining -= batch
    return chunks, chunk_idx, skipped


def build_char_chunks(valid_scripts, script_fonts, tasks, args, shard_dir, start_chunk_idx):
    """Build single-character chunks. Returns char_chunks."""
    train_dir = str(shard_dir / "train")
    val_dir = str(shard_dir / "val")

    print(f"\n{'='*60}")
    print(f"Generating single-character images")
    print(f"{'='*60}")

    char_chunks = []
    chunk_idx = start_chunk_idx
    for script in valid_scripts:
        if script == "emoji":
            continue
        fonts = script_fonts[script]
        chars = get_renderable_chars(script)
        if not chars:
            continue

        reps = args.char_reps
        est = len(chars) * reps
        print(f"  {script:<15} {len(chars):>5} unique chars × {reps} reps = ~{est} images")

        chunk_size = max(200, len(chars) // max(1, args.workers // len(valid_scripts)))
        for ci in range(0, len(chars), chunk_size):
            char_subset = chars[ci:ci + chunk_size]
            if chunk_dir_exists(str(Path(train_dir) / f"chunk_{chunk_idx:04d}")):
                chunk_idx += 1
                continue
            char_chunks.append((script, char_subset, reps, fonts,
                                args.height, args.max_width, args.augment,
                                chunk_idx, train_dir, val_dir))
            chunk_idx += 1

    return char_chunks


# ---------------------------------------------------------------------------
# Pool runner
# ---------------------------------------------------------------------------

def run_generation_pool(chunks, worker_fn, n_workers, label):
    """Run a generation function over chunks using a multiprocessing pool.

    Returns (total_done, all_train_widths, all_val_widths).
    """
    # Estimate total: word chunks have count at c[1], char chunks at len(c[1])*c[2],
    # mixed chunks have count at c[0]
    def _est(c):
        if isinstance(c[0], int) and not isinstance(c[1], int):
            return c[0]  # mixed: (count, script_info, ...)
        elif isinstance(c[1], int):
            return c[1]  # word: (script, count, ...)
        else:
            return len(c[1]) * c[2]  # char: (script, chars, reps, ...)
    total_est = sum(_est(c) for c in chunks)
    print(f"\nGenerating {total_est} {label} images, {len(chunks)} chunks, "
          f"{min(n_workers, len(chunks))} workers\n")

    start = time.time()
    done = 0
    all_train_widths = []
    all_val_widths = []
    with Pool(processes=min(n_workers, len(chunks)), maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(worker_fn, chunks):
            _, script, n, tw, vw = result
            done += n
            all_train_widths.extend(tw)
            all_val_widths.extend(vw)
            elapsed = time.time() - start
            print(f"    total: {done}/{total_est} ({done/elapsed:.0f} img/s)", flush=True)
    return done, all_train_widths, all_val_widths


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Worker functions (called in multiprocessing pool)
# ---------------------------------------------------------------------------

def _generate_word_batch(args_tuple):
    """Generate word images for one chunk, write to MDS."""
    script, count, fonts, words, h, mw, do_augment, chunk_id, train_dir, val_dir, style, punct_prob = args_tuple
    style_cfg = STYLES.get(style, STYLES["printed"])
    if not do_augment or not style_cfg["ops"]:
        aug = None
    else:
        aug = RandAugmentOCR(n_ops=2, p=0.5, ops=style_cfg["ops"])
    clean_render = style_cfg.get("clean_render", False)
    t0 = time.time()

    images, labels = [], []
    attempts = 0
    clean_target = 0 if clean_render else int(count * CLEAN_RATIO)

    while len(images) < count and attempts < count * 5:
        attempts += 1
        if script == "emoji":
            img = render_emoji(h, mw)
            label = "emoji"
        else:
            word = random.choice(words)
            word = mix_punctuation(word, p=punct_prob, script=script)
            font = random.choice(fonts)
            if not font_covers_text(font, word):
                continue
            img = render_word(word, font, h, clean=clean_render)
            label = word

        if img is None or not image_has_ink(img):
            continue

        img = resize_or_pad(img, h, mw)

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

    n, tw, vw = save_mds_samples(images, labels, script, train_dir, val_dir, chunk_id)
    del images, labels

    elapsed = time.time() - t0
    rate = n / elapsed if elapsed > 0 else 0
    print(f"  {script:<15} {n:>5} images in {elapsed:.0f}s ({rate:.0f} img/s)", flush=True)
    return chunk_id, script, n, tw, vw


def _generate_char_batch(args_tuple):
    """Generate single-character images with cmap validation, write to MDS."""
    script, chars, reps_per_char, fonts, h, mw, do_augment, chunk_id, train_dir, val_dir = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None
    t0 = time.time()

    images, labels = [], []
    skipped_chars = 0

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

            if aug is not None and rep > 0:
                img = aug(img)
                if not image_has_ink(img, min_ink_pixels=5):
                    continue

            images.append(rgb_to_input(img))
            labels.append(ch)
        if not got_any:
            skipped_chars += 1

    n, tw, vw = save_mds_samples(images, labels, script, train_dir, val_dir, chunk_id)
    del images, labels

    elapsed = time.time() - t0
    skip_str = f", {skipped_chars} chars skipped (bad render)" if skipped_chars else ""
    print(f"  {script:<15} {n:>5} char images "
          f"({len(chars)} unique × {reps_per_char} reps{skip_str}) in {elapsed:.0f}s", flush=True)
    return chunk_id, script, n, tw, vw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _MAX_VAL_PER_SCRIPT
    args = parse_args()
    _MAX_VAL_PER_SCRIPT = args.val_samples_per_script

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
                vocab = build_script_vocab(script)
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



    # Create train/val dirs
    (shard_dir / "train").mkdir(parents=True, exist_ok=True)
    (shard_dir / "val").mkdir(parents=True, exist_ok=True)

    all_train_widths = []
    all_val_widths = []

    # Word images
    chunks, next_chunk_idx, skipped = build_word_chunks(
        tasks, script_fonts, word_lists, args, shard_dir, styles_to_gen)
    if skipped > 0:
        print(f"\nResuming: {skipped} images in existing chunks, "
              f"{sum(c[1] for c in chunks)} remaining")
    if chunks:
        _, tw, vw = run_generation_pool(chunks, _generate_word_batch, args.workers, "word")
        all_train_widths.extend(tw)
        all_val_widths.extend(vw)
    else:
        print("All chunks exist. Done.")

    # Character images
    if args.include_chars:
        char_chunks = build_char_chunks(valid_scripts, script_fonts, tasks, args,
                                        shard_dir, next_chunk_idx)
        if char_chunks:
            print(f"\n  {len(char_chunks)} char chunks across "
                  f"{min(args.workers, len(char_chunks))} workers\n")
            _, tw, vw = run_generation_pool(char_chunks, _generate_char_batch, args.workers, "char")
            all_train_widths.extend(tw)
            all_val_widths.extend(vw)
        else:
            print("  All char chunks exist.")

    # Mixed-script images
    if args.mixed_script > 0:
        print(f"\n{'='*60}")
        print(f"Generating {args.mixed_script} mixed-script images")
        print(f"{'='*60}")

        # Build script info for mixed generator
        all_script_info = []
        for script in valid_scripts:
            if script == "emoji":
                continue
            fonts = script_fonts.get(script, [])
            words = word_lists.get(script, [])
            if fonts and words:
                sid = SCRIPT_TO_ID[script]
                gid = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]
                all_script_info.append((script, fonts, words, sid, gid))

        if len(all_script_info) >= 2:
            train_dir = str(shard_dir / "train")
            val_dir = str(shard_dir / "val")
            # Use chunk index after all previous chunks
            mixed_chunk_idx = next_chunk_idx + len(char_chunks) if args.include_chars else next_chunk_idx
            chunk_size = min(2000, args.mixed_script)
            mixed_chunks = []
            remaining = args.mixed_script
            ci = mixed_chunk_idx
            while remaining > 0:
                batch = min(chunk_size, remaining)
                mixed_chunks.append((batch, all_script_info, args.height,
                                     args.max_width, args.augment, ci,
                                     train_dir, val_dir))
                ci += 1
                remaining -= batch

            print(f"  {len(mixed_chunks)} mixed chunks")
            _, tw, vw = run_generation_pool(mixed_chunks, _generate_mixed_batch,
                                            min(args.workers, len(mixed_chunks)), "mixed")
            all_train_widths.extend(tw)
            all_val_widths.extend(vw)

    # Save widths for batch sampling
    np.save(str(shard_dir / "train" / "widths.npy"),
            np.array(all_train_widths, dtype=np.int32))
    np.save(str(shard_dir / "val" / "widths.npy"),
            np.array(all_val_widths, dtype=np.int32))

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
        "height": args.height,
        "max_width": args.max_width,
    }, shard_dir / "metadata.pt")

    n_train = len(all_train_widths)
    n_val = len(all_val_widths)
    print(f"\nDone: {n_train + n_val} samples → {n_train} train + {n_val} val")
    print(f"  {shard_dir}/train  {shard_dir}/val")


if __name__ == "__main__":
    main()
