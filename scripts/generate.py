#!/usr/bin/env python3
"""
Generate synthetic training data for LID and MoE training.

Usage:
    python scripts/generate.py --samples-per-script 10000 --out data/shards
    python scripts/generate.py --samples-per-script 30000 --balance-groups --include-chars --out data/shards
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

from src.taxonomy import SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID, SCRIPT_TO_ID
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

    For han: returns all CJK Unified + Ext-A characters (kanji/hanzi).
    For kana: returns hiragana + katakana characters.
    For fusion scripts: returns base chars + all fusion clusters.
    For no-fusion scripts: returns chars from the codec.
    For korean: returns chars from the codec.
    """
    import unicodedata
    from src.encoding.config import (
        NO_FUSION_SCRIPTS, FUSION_BASE_CHARS, get_fusion_codec,
        get_korean_codec,
    )

    if script == "han":
        chars = []
        for cp in range(0x3400, 0x4DC0):      # CJK Ext A
            c = chr(cp)
            if unicodedata.category(c) != 'Cn':
                chars.append(c)
        for cp in range(0x4E00, 0xA000):      # CJK Unified
            c = chr(cp)
            if unicodedata.category(c) != 'Cn':
                chars.append(c)
        return chars

    if script == "kana":
        chars = []
        for cp in range(0x3041, 0x3097):      # Hiragana
            chars.append(chr(cp))
        for cp in range(0x30A1, 0x30FB):      # Katakana
            chars.append(chr(cp))
        chars.append('\u30FC')                # Prolonged sound mark
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
# Japanese kana/kanji boundary splitting
# ---------------------------------------------------------------------------

def _is_kana_cp(cp):
    return 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF

def _is_kanji_cp(cp):
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF

def _is_cjk_punct_cp(cp):
    return 0x3000 <= cp <= 0x303F

def _is_ascii_cp(cp):
    return 0x21 <= cp <= 0x7E


def split_by_script(text, parent_script):
    """Split text into segments with correct script labels.

    ASCII characters (0x21-0x7E) become "latin" segments.
    For han/kana, also splits at kana/kanji boundaries.
    CJK punctuation (0x3000-0x303F) routes to "han".
    Everything else stays in parent_script.

    Returns [(chunk_text, script_name), ...].
    """
    if not text:
        return []

    segments = []
    current = []
    current_script = None

    for ch in text:
        cp = ord(ch)

        # Determine script for this character
        if _is_ascii_cp(cp):
            script = "latin"
        elif parent_script in ("han", "kana"):
            if _is_kana_cp(cp):
                script = "kana"
            elif _is_kanji_cp(cp) or _is_cjk_punct_cp(cp):
                script = "han"
            else:
                script = current_script  # non-ASCII, non-CJK → attach
        else:
            script = parent_script

        if script != current_script and current_script is not None and script is not None:
            segments.append(("".join(current), current_script))
            current = []
        if script is not None:
            current_script = script
        current.append(ch)

    if current and current_script is not None:
        segments.append(("".join(current), current_script))

    return segments


# ---------------------------------------------------------------------------
# Latin punctuation/number segments for non-latin plans
# ---------------------------------------------------------------------------

from src.encoding.config import ASCII_COMMON, TYPOGRAPHIC_COMMON

# Build flat char lists from the encoding config ranges
_ASCII_COMMON_CHARS = [chr(cp) for start, end in ASCII_COMMON
                       for cp in range(start, end + 1)
                       if chr(cp).strip()]
_TYPO_COMMON_CHARS = [chr(cp) for start, end in TYPOGRAPHIC_COMMON
                      for cp in range(start, end + 1)]


def _random_latin_segment():
    """Generate a random segment from ASCII_COMMON / TYPOGRAPHIC_COMMON chars."""
    r = random.random()
    if r < 0.35:
        # 1-3 random ASCII common chars
        n = random.randint(1, 3)
        return "".join(random.choices(_ASCII_COMMON_CHARS, k=n))
    elif r < 0.50:
        # Typographic chars
        return random.choice(_TYPO_COMMON_CHARS)
    elif r < 0.70:
        # Number: 1-6 digits, possibly with . or ,
        digits = "".join(random.choices("0123456789", k=random.randint(1, 6)))
        if len(digits) >= 4 and random.random() < 0.3:
            digits = f"{int(digits):,}"
        elif len(digits) >= 2 and random.random() < 0.3:
            pos = random.randint(1, len(digits) - 1)
            digits = digits[:pos] + "." + digits[pos:]
        return digits
    elif r < 0.82:
        # Currency + number
        currency = random.choice(["$", "\u00A3", "\u00A5", "\u20AC"])
        return currency + str(random.randint(1, 9999))
    elif r < 0.90:
        # Bracketed expression: (123), [45], {6}
        inner = "".join(random.choices("0123456789", k=random.randint(1, 4)))
        left, right = random.choice([("(", ")"), ("[", "]"), ("{", "}")])
        return left + inner + right
    else:
        # Separated numbers: 12-34, 123.456.789
        sep = random.choice(list("-/."))
        parts = ["".join(random.choices("0123456789", k=random.randint(2, 4)))
                 for _ in range(random.randint(2, 3))]
        return sep.join(parts)


def _maybe_add_latin_segment(plan, p=0.15):
    """With probability p, append a latin punctuation/number segment to the plan."""
    if random.random() > p:
        return
    plan.append({"text": _random_latin_segment(), "script": "latin"})


# ---------------------------------------------------------------------------
# Punctuation / number mixing
# ---------------------------------------------------------------------------

# Common punctuation and number patterns seen in real documents
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


def render_content_plan(plan, fonts_by_script, h, mw, aug=None,
                        word_pools=None):
    """Stage 2: Render a content plan into image + metadata.

    Words are always rendered clean (white bg, black ink). Augmentation
    applies color/style to the composed line so all words share one style.

    Args:
        plan: list of {"text": str, "script": str} — "whitespace" script means gap
        fonts_by_script: {script: [font_paths]}
        h: image height
        mw: max width
        aug: augmentation (applied to final composed image)
        word_pools: {script: [(pil_img, text, width), ...]} — pre-rendered
            word images. If provided, picks from pool instead of rendering.

    Returns:
        (img_tensor, label, group_labels, segments) or None if failed
        segments is a Python list (not JSON) for efficiency.
    """
    from PIL import Image
    from src.taxonomy import NUM_GROUPS as BLANK_ID

    blocks = []  # (img, text, group_id, script_id, width) per rendered block
    total_w = 0

    for item in plan:
        text = item["text"]
        script = item["script"]

        if script == "whitespace":
            ws_w = random.randint(4, 16)
            if total_w + ws_w > mw:
                break
            blocks.append((None, "", BLANK_ID, 0, ws_w))
            total_w += ws_w
            continue

        if script == "emoji":
            img = render_emoji(h, mw - total_w if total_w < mw else 32)
            if img is None:
                continue
            emoji_gid = GROUP_TO_ID.get(SCRIPT_TO_GROUP.get(script, ""), 0)
            emoji_sid = SCRIPT_TO_ID.get(script, 0)
            blocks.append((img, text, emoji_gid, emoji_sid, img.width))
            total_w += img.width
            continue

        # Try word pool first (pre-rendered), fall back to live render
        pool = word_pools.get(script) if word_pools else None
        if pool:
            img, text, _pw = random.choice(pool)
        else:
            fonts = fonts_by_script.get(script, [])
            if not fonts:
                return None
            img = None
            for _ in range(min(3, len(fonts))):
                font = random.choice(fonts)
                if not font_covers_text(font, text):
                    continue
                img = render_word(text, font, h, clean=True)
                if img is not None and image_has_ink(img):
                    break
                img = None
            if img is None:
                return None

        if total_w + img.width > mw:
            img = img.crop((0, 0, min(img.width, mw - total_w), h))
            if img.width < 4:
                break

        gid = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]
        sid = SCRIPT_TO_ID[script]
        blocks.append((img, text, gid, sid, img.width))
        total_w += img.width

    if not blocks or total_w == 0 or not any(b[1] for b in blocks):
        return None

    # Compose: concatenate all blocks
    combined = Image.new("RGB", (total_w, h), (255, 255, 255))
    group_labels = np.full(total_w, BLANK_ID, dtype=np.int32)
    segments = []
    full_label = ""
    offset = 0

    for block_img, text, gid, sid, bw in blocks:
        if block_img is not None:
            combined.paste(block_img, (offset, 0))
            group_labels[offset:offset + bw] = gid
            if text:
                segments.append({
                    "group_id": gid, "script_id": sid,
                    "text": text, "width": bw, "offset": offset,
                })
                full_label += text
        offset += bw

    # Resize/pad to target dimensions
    combined = resize_or_pad(combined, h, mw)
    final_w = combined.width

    # Scale group_labels and segment offsets/widths to match final image width
    if final_w != total_w:
        scale = final_w / total_w
        new_gl = np.full(final_w, BLANK_ID, dtype=np.int32)
        for seg in segments:
            new_offset = round(seg["offset"] * scale)
            new_end = round((seg["offset"] + seg["width"]) * scale)
            new_gl[new_offset:new_end] = seg["group_id"]
            seg["offset"] = new_offset
            seg["width"] = new_end - new_offset
        group_labels = new_gl

    # Apply augmentation
    if aug is not None:
        combined = aug(combined)

    img_tensor = rgb_to_input(combined)
    return img_tensor, full_label, group_labels, segments


def save_rendered_samples(samples, primary_script, train_dir, val_dir, chunk_id):
    """Write rendered samples to MDS.

    Each sample is (img_tensor, label, group_labels_np, segments_list).
    Segments are Python lists (not JSON) — serialized once at write time.

    Args:
        samples: list of (img_tensor, label, group_labels_np, segments_list)
        primary_script: script name (used for script_id/group_id fields)
        train_dir, val_dir: output directories
        chunk_id: chunk index
    """
    if not samples:
        return 0, [], []

    import hashlib
    from streaming import MDSWriter
    from src.encoding.decompose import encode_text

    script_id = SCRIPT_TO_ID[primary_script]
    group_id = GROUP_TO_ID[SCRIPT_TO_GROUP[primary_script]]

    # Reverse lookup: global script_id → script name
    _sid_to_name = {v: k for k, v in SCRIPT_TO_ID.items()}

    t_dir = str(Path(train_dir) / f"chunk_{chunk_id:04d}")
    v_dir = str(Path(val_dir) / f"chunk_{chunk_id:04d}")
    Path(t_dir).mkdir(parents=True, exist_ok=True)
    Path(v_dir).mkdir(parents=True, exist_ok=True)

    train_widths, val_widths = [], []
    val_count = 0
    with MDSWriter(out=t_dir, columns=MDS_COLUMNS, size_limit=1 << 26) as tw, \
         MDSWriter(out=v_dir, columns=MDS_COLUMNS, size_limit=1 << 26) as vw:
        for idx, (img_tensor, label, gl, segs) in enumerate(samples):
            img_np = img_tensor.numpy()

            # Encode target_ids per-segment for correct mixed-script encoding
            all_ids = []
            for seg in segs:
                seg_script = _sid_to_name.get(seg.get("script_id", 0), primary_script)
                all_ids.extend(encode_text(seg["text"], seg_script))

            tids = np.array(all_ids, dtype=np.int64) if all_ids else np.zeros(1, dtype=np.int64)
            sample = {
                "image": img_np,
                "label": label,
                "script_id": script_id,
                "group_id": group_id,
                "target_ids": tids,
                "target_len": len(all_ids),
                "width": img_np.shape[2],
                "group_labels": gl,
                "segments": json.dumps(segs),
            }
            h_val = int(hashlib.md5(f"{chunk_id}_{idx}_{label}".encode()).hexdigest(), 16)
            if h_val % 1000 < 100 and val_count < _MAX_VAL_PER_SCRIPT:
                vw.write(sample)
                val_widths.append(img_np.shape[2])
                val_count += 1
            else:
                tw.write(sample)
                train_widths.append(img_np.shape[2])

    return len(samples), train_widths, val_widths


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
                        help="Data style: clean, printed, handwritten, signage, degraded, or all")
    parser.add_argument("--augment", dest="augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--punct-prob", type=float, default=0.15,
                        help="Probability of mixing punctuation/numbers into a word (default: 0.15)")
    parser.add_argument("--mixed-ratio", type=float, default=0.6,
                        help="Fraction of lines that are mixed-script (default: 0.6)")
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

    if len(valid_scripts) < 1:
        print("ERROR: No scripts with fonts found")
        sys.exit(1)

    return script_fonts, valid_scripts


# ---------------------------------------------------------------------------
# Worker shared state (initialized once per worker via Pool initializer)
# ---------------------------------------------------------------------------

_worker_style_configs = None  # {style: (script_info, fonts_by_script)}
_worker_word_pools = None     # {(style, script): [(pil_img, text, width), ...]}
_worker_group_index = None    # {style: {group_id: [script_info, ...]}}
_worker_mixed_ratio = 0.6     # set from CLI via initializer

WORD_POOL_SIZE = 200  # pre-rendered words per script


def _init_line_worker(style_configs, mixed_ratio, pool_height=32):
    """Pool initializer: load shared data and pre-build word pools."""
    global _worker_style_configs, _worker_word_pools
    global _worker_group_index, _worker_mixed_ratio
    _worker_style_configs = style_configs
    _worker_word_pools = {}
    _worker_mixed_ratio = mixed_ratio
    import os
    pid = os.getpid()
    # Pre-build group index for fast mixed-line script selection
    _worker_group_index = {}
    for style, (script_info, _fonts) in style_configs.items():
        by_group = {}
        for info in script_info:
            by_group.setdefault(info[3], []).append(info)
        _worker_group_index[style] = by_group
    # Pre-build word pools for all scripts (one pool per script, shared
    # across styles since all words render clean white bg + black ink)
    t0 = time.time()
    first_style = next(iter(style_configs))
    script_info, _ = style_configs[first_style]
    for script, fonts, words, _gid in script_info:
        _get_word_pool(first_style, script, fonts, words, h=pool_height)
    elapsed = time.time() - t0
    total_words = sum(len(p) for p in _worker_word_pools.values())
    print(f"    [worker {pid}] pools ready: {len(_worker_word_pools)} scripts, "
          f"{total_words} words, {elapsed:.1f}s", flush=True)


def _get_word_pool(style, script, fonts, words, h=32, punct_prob=0.15):
    """Get or build a pre-rendered word pool for a script.

    Each pool entry is (pil_image, text, width). Built once per worker,
    shared across all styles (words render clean — augmentation applies
    color/style to the composed line).
    """
    key = script  # style-independent since all words render clean
    if key in _worker_word_pools:
        return _worker_word_pools[key]

    pool = []
    attempts = 0
    target = min(WORD_POOL_SIZE, len(words) * 2)
    while len(pool) < target and attempts < target * 3:
        attempts += 1
        word = random.choice(words)
        # Skip any word that isn't STRICTLY the requested script. The pool
        # labels every entry as `script`, so admitting e.g. a pure-katakana
        # word to the "han" pool would label visual kana as han — exactly
        # the mislabel that hurt LID-1 han accuracy previously. Using
        # split_by_script catches mixed AND pure-mismatched words in one
        # check (japanese.txt is ~52% pure-kana, ~3% pure-kanji).
        # For han/kana pools: skip mixed Japanese words (they'd get
        # the wrong single-script label). split_by_script handles them
        # correctly in the plan builders instead.
        if script in ("han", "kana"):
            segs = split_by_script(word, script)
            if len(segs) != 1 or segs[0][1] != script:
                continue
        # For non-latin pools: skip words with ASCII chars.
        # split_by_script in plan builders will route those chars to
        # latin, but the pool renders whole words with a single label.
        if script != "latin" and any(_is_ascii_cp(ord(c)) for c in word):
            continue
        # Only mix punctuation into latin pool entries.
        if script == "latin":
            word = mix_punctuation(word, p=punct_prob, script=script)
        font = random.choice(fonts)
        if not font_covers_text(font, word):
            continue
        img = render_word(word, font, h, clean=True)
        if img is None or not image_has_ink(img):
            continue
        pool.append((img, word, img.width))

    _worker_word_pools[key] = pool
    return pool


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------

def build_line_chunks(tasks, script_fonts, word_lists, valid_scripts, args,
                      shard_dir, styles_to_gen):
    """Build line-image chunks and per-style shared configs.

    Returns (chunks, next_chunk_idx, skipped, style_configs).
    style_configs is passed to workers via Pool initializer to avoid
    serializing font/word lists per chunk.
    """
    train_dir = str(shard_dir / "train")
    val_dir = str(shard_dir / "val")

    # Build all_script_info for mixed lines: (script, fonts, words, group_id)
    all_script_info = []
    for script in valid_scripts:
        if script == "emoji":
            continue
        fonts = script_fonts.get(script, [])
        words = word_lists.get(script, [])
        if fonts and words:
            gid = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]
            all_script_info.append((script, fonts, words, gid))

    # Pre-build per-style configs (passed to workers once via initializer)
    style_configs = {}
    for style in styles_to_gen:
        style_fonts = {}
        for script in valid_scripts:
            style_fonts[script] = filter_fonts_by_style(
                script_fonts[script], style)

        style_script_info = []
        for script, _fonts, words, gid in all_script_info:
            sf = style_fonts.get(script, [])
            if sf:
                style_script_info.append((script, sf, words, gid))

        style_configs[style] = (style_script_info, style_fonts)

    chunks = []
    chunk_idx = 0
    skipped = 0
    for style in styles_to_gen:
        proportion = STYLES[style]["proportion"] if len(styles_to_gen) > 1 else 1.0
        for script, target in tasks:
            style_target = max(1, int(target * proportion))
            chunk_size = min(5000, max(500,
                style_target // max(1, args.workers // len(tasks))))
            remaining = style_target
            while remaining > 0:
                batch = min(chunk_size, remaining)
                if chunk_dir_exists(str(Path(train_dir) / f"chunk_{chunk_idx:04d}")):
                    skipped += batch
                else:
                    # Lightweight chunk: just script, count, style name, and paths
                    chunks.append((script, batch, args.height, args.max_width,
                                   args.augment, chunk_idx, train_dir, val_dir,
                                   style, args.punct_prob))
                chunk_idx += 1
                remaining -= batch
    return chunks, chunk_idx, skipped, style_configs


def build_char_chunks(valid_scripts, script_fonts, args, shard_dir, start_chunk_idx):
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

def run_generation_pool(chunks, worker_fn, n_workers, label,
                        initializer=None, initargs=()):
    """Run a generation function over chunks using a multiprocessing pool.

    Returns (total_done, all_train_widths, all_val_widths).
    """
    # Estimate total: line chunks have count at c[1], char chunks at len(c[1])*c[2]
    def _est(c):
        if isinstance(c[1], int):
            return c[1]  # line: (script, count, ...)
        else:
            return len(c[1]) * c[2]  # char: (script, chars, reps, ...)
    total_est = sum(_est(c) for c in chunks)
    print(f"\nGenerating {total_est} {label} images, {len(chunks)} chunks, "
          f"{min(n_workers, len(chunks))} workers\n")

    n_procs = min(n_workers, len(chunks))
    if initializer:
        print(f"  Starting {n_procs} workers (building word pools)...", flush=True)

    start = time.time()
    done = 0
    all_train_widths = []
    all_val_widths = []
    with Pool(processes=n_procs,
              initializer=initializer, initargs=initargs) as pool:
        if initializer:
            # Workers are initializing — first result confirms they're ready
            print(f"  Workers initialized, generating...", flush=True)
        for result in pool.imap_unordered(worker_fn, chunks):
            _, script, n, tw, vw = result
            done += n
            all_train_widths.extend(tw)
            all_val_widths.extend(vw)
            elapsed = time.time() - start
            print(f"    total: {done}/{total_est} ({done/elapsed:.0f} img/s)", flush=True)
    elapsed = time.time() - start
    print(f"  Done: {done} {label} images in {elapsed:.0f}s "
          f"({done/max(elapsed,0.1):.0f} img/s)", flush=True)
    return done, all_train_widths, all_val_widths


# ---------------------------------------------------------------------------
# Worker functions (called in multiprocessing pool)
# ---------------------------------------------------------------------------

def _generate_line_batch(args_tuple):
    """Generate line images (mixed or single-script) using two-stage pipeline.

    Each line has 2-8 words. Mixed ratio controlled by _worker_mixed_ratio (--mixed-ratio).
    Shared data (script_info, fonts) loaded from worker globals.
    """
    (primary_script, count, h, mw, do_augment, chunk_id,
     train_dir, val_dir, style, punct_prob) = args_tuple

    # Pull shared data from worker globals (set by Pool initializer)
    all_script_info, fonts_by_script = _worker_style_configs[style]

    style_cfg = STYLES.get(style, STYLES["printed"])
    if not do_augment or not style_cfg["ops"]:
        aug = None
    else:
        aug = RandAugmentOCR(n_ops=2, p=0.5, ops=style_cfg["ops"])
    t0 = time.time()

    # Collect pre-built word pools (built at worker init, shared across styles)
    word_pools = {}
    for script, fonts, words, _gid in all_script_info:
        pool = _get_word_pool(style, script, fonts, words, h=h, punct_prob=punct_prob)
        if pool:
            word_pools[script] = pool

    # Find this script's info for single-script lines
    primary_info = None
    for info in all_script_info:
        if info[0] == primary_script:
            primary_info = info
            break

    # Pre-built group index for mixed line selection
    group_index = _worker_group_index[style]

    samples = []
    attempts = 0
    clean_target = int(count * CLEAN_RATIO)
    can_mix = len(group_index) >= 2

    while len(samples) < count and attempts < count * 5:
        attempts += 1

        # Decide mixed vs single-script
        do_mixed = can_mix and random.random() < _worker_mixed_ratio

        if do_mixed:
            plan = _build_mixed_line_plan(group_index)
        else:
            plan = _build_single_line_plan(primary_info)

        if plan is None:
            continue

        use_aug = aug if len(samples) >= clean_target else None
        result = render_content_plan(plan, fonts_by_script, h, mw,
                                     aug=use_aug, word_pools=word_pools)
        if result is None:
            continue

        samples.append(result)

        if len(samples) % 1000 == 0:
            elapsed = time.time() - t0
            rate = len(samples) / elapsed if elapsed > 0 else 0
            print(f"    [{primary_script}] {len(samples)}/{count} "
                  f"({rate:.0f} img/s)", flush=True)

    render_time = time.time() - t0
    n, tw, vw = save_rendered_samples(samples, primary_script,
                                      train_dir, val_dir, chunk_id)
    del samples

    elapsed = time.time() - t0
    rate = n / elapsed if elapsed > 0 else 0
    save_time = elapsed - render_time
    extra = f" (render {render_time:.0f}s + save {save_time:.0f}s)" if save_time > 1 else ""
    print(f"  {primary_script:<15} {n:>5} lines in {elapsed:.0f}s "
          f"({rate:.0f} img/s){extra}", flush=True)
    return chunk_id, primary_script, n, tw, vw


def _build_single_line_plan(script_info):
    """Build a content plan for a single-script line (2-8 words).

    Text values are placeholders — render_content_plan replaces them
    with pre-rendered pool images (which already include punctuation).
    """
    if script_info is None:
        return None

    script, _fonts, words, _gid = script_info
    n_words = random.choices([2, 3, 4, 5, 6, 7, 8],
                             weights=[0.10, 0.20, 0.25, 0.20, 0.15, 0.05, 0.05],
                             k=1)[0]
    plan = []
    for i in range(n_words):
        if i > 0:
            plan.append({"text": " ", "script": "whitespace"})
        if script != "latin":
            _maybe_add_latin_segment(plan)
        word = random.choice(words)
        for seg_text, seg_script in split_by_script(word, script):
            plan.append({"text": seg_text, "script": seg_script})
        if script != "latin":
            _maybe_add_latin_segment(plan)
    return plan


def _build_mixed_line_plan(group_index):
    """Build a content plan for a mixed-script line (2-8 words, 1-4 groups).

    Args:
        group_index: {group_id: [script_info, ...]} — pre-built index.
    """
    n_groups = random.choices([1, 2, 3, 4],
                              weights=[0.10, 0.45, 0.30, 0.15],
                              k=1)[0]
    n_groups = min(n_groups, len(group_index))

    n_words = random.choices([2, 3, 4, 5, 6, 7, 8],
                             weights=[0.10, 0.20, 0.25, 0.20, 0.15, 0.05, 0.05],
                             k=1)[0]
    n_words = max(n_words, n_groups)

    # Pick n_groups distinct groups
    group_ids = random.sample(list(group_index.keys()),
                              min(n_groups, len(group_index)))

    # Pick one script per selected group
    group_scripts = [random.choice(group_index[gid]) for gid in group_ids]

    # Fill n_words: first ensure one word per group, then random fill
    chosen = list(group_scripts)
    for _ in range(n_words - len(chosen)):
        chosen.append(random.choice(group_scripts))
    random.shuffle(chosen)

    plan = []
    for i, (script, _fonts, words, _gid) in enumerate(chosen):
        if i > 0:
            plan.append({"text": " ", "script": "whitespace"})
        if script != "latin":
            _maybe_add_latin_segment(plan)
            _maybe_add_latin_segment(plan)
        word = random.choice(words)
        for seg_text, seg_script in split_by_script(word, script):
            plan.append({"text": seg_text, "script": seg_script})
        if script != "latin":
            _maybe_add_latin_segment(plan)
    return plan


def _generate_char_batch(args_tuple):
    """Generate single-character images using two-stage pipeline."""
    script, chars, reps_per_char, fonts, h, mw, do_augment, chunk_id, train_dir, val_dir = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None
    t0 = time.time()

    fonts_by_script = {script: fonts}
    samples = []
    skipped_chars = 0

    char_fonts = filter_fonts_by_cmap(fonts, chars)
    skipped_chars += len(chars) - len(char_fonts)

    for ch in char_fonts:
        got_any = False
        for rep in range(reps_per_char):
            # Override fonts_by_script to only use fonts that cover this char
            fonts_by_script[script] = char_fonts[ch]
            plan = [{"text": ch, "script": script}]
            result = render_content_plan(plan, fonts_by_script, h, mw,
                                         aug=aug if rep > 0 else None)
            if result is None:
                continue
            got_any = True
            samples.append(result)
        if not got_any:
            skipped_chars += 1

    render_time = time.time() - t0
    n, tw, vw = save_rendered_samples(samples, script, train_dir, val_dir, chunk_id)
    del samples

    elapsed = time.time() - t0
    skip_str = f", {skipped_chars} skipped" if skipped_chars else ""
    save_time = elapsed - render_time
    extra = f" (render {render_time:.0f}s + save {save_time:.0f}s)" if save_time > 1 else ""
    print(f"  {script:<15} {n:>5} char images "
          f"({len(chars)} unique × {reps_per_char} reps{skip_str}) "
          f"in {elapsed:.0f}s{extra}", flush=True)
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

    # Line images (mixed + single-script)
    mixed_ratio = args.mixed_ratio
    print(f"\nLine generation: {mixed_ratio:.0%} mixed, "
          f"{1 - mixed_ratio:.0%} single-script")
    chunks, next_chunk_idx, skipped, style_configs = build_line_chunks(
        tasks, script_fonts, word_lists, valid_scripts, args,
        shard_dir, styles_to_gen)
    if skipped > 0:
        print(f"\nResuming: {skipped} lines in existing chunks, "
              f"{sum(c[1] for c in chunks)} remaining")
    if chunks:
        _, tw, vw = run_generation_pool(
            chunks, _generate_line_batch, args.workers, "line",
            initializer=_init_line_worker,
            initargs=(style_configs, mixed_ratio, args.height))
        all_train_widths.extend(tw)
        all_val_widths.extend(vw)
    else:
        print("All chunks exist. Done.")

    # Character images
    if args.include_chars:
        char_chunks = build_char_chunks(valid_scripts, script_fonts, args,
                                        shard_dir, next_chunk_idx)
        if char_chunks:
            print(f"\n  {len(char_chunks)} char chunks across "
                  f"{min(args.workers, len(char_chunks))} workers\n")
            _, tw, vw = run_generation_pool(char_chunks, _generate_char_batch, args.workers, "char")
            all_train_widths.extend(tw)
            all_val_widths.extend(vw)
        else:
            print("  All char chunks exist.")

    # Save widths for batch sampling
    print("\nSaving widths and metadata...", flush=True)
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
