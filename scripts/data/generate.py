#!/usr/bin/env python3
"""
Generate synthetic training data for LID and MoE training.

Usage:
    python scripts/generate.py --samples-per-script 10000 --out data/shards
    python scripts/generate.py --samples-per-script 30000 --balance-groups --include-chars --out data/shards
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
from pathlib import Path
from multiprocessing import Pool

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.taxonomy import (
    SCRIPTS,
    SCRIPT_TO_GROUP,
    GROUP_TO_ID,
    SCRIPT_TO_ID,
    TAXONOMY_VERSION,
)
from src.data.color import rgb_to_input
from src.data.augmentation import (
    RandAugmentOCR, CHAINS_BY_NAME,
    jpeg_compress, blur, low_resolution, photocopy, binarize,
    exposure_jitter, uneven_lighting, glare, striped_shadow,
    rotation, perspective_warp, wave_distortion,
    bleed_through, fold_crease, aged_document, scanner_edge, water_stain,
    stroke_variation, smudge,
    variable_baseline, slant, ink_fade, variable_stroke, lined_paper,
    noise, color_jitter, to_grayscale,
    occlusion, weather_damage,
    polarity_invert, adjacent_line_clutter,
    text_decoration, highlighter, table_rules, photo_background,
)
from src.encoding.vocab import build_script_vocab
from src.encoding.direction import is_rtl_script
from src.encoding.han_split import (
    HAN_DENSE,
    HAN_SCRIPTS,
    HAN_SPARSE,
    HAN_SPLIT_VERSION,
)
from src.data.rendering import (
    render_word, image_has_ink,
    resize_or_pad, filter_fonts_by_cmap, font_covers_text,
    compose_line_baseline, visual_order_blocks,
)
from src.data.text_renderer import render_word_baseline, pick_weight
from src.data.fonts import find_fonts_for_script, build_weighted_font_list
from src.data.word_lists import load_all_word_lists, WordSampler

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

# Each style pairs a font filter with augmentation ops (random-op branch)
# and a subset of scenario chains (chain branch). Chains are restricted per
# style — previously RandAugmentOCR applied the global chain list regardless
# of style, so "handwritten" got outdoor-sign textures and "signage" only
# got its colored backgrounds by accident.
STYLES = {
    "clean": {
        "proportion": 0.10,
        "ops": [],
        "chains": [],
        "font_filter": "regular",
    },
    "printed": {
        "proportion": 0.35,
        "ops": [jpeg_compress, blur, photocopy, binarize, uneven_lighting,
                fold_crease, bleed_through, aged_document, scanner_edge,
                water_stain, exposure_jitter, noise, to_grayscale,
                adjacent_line_clutter,
                text_decoration, highlighter, table_rules],
        "chains": ["phone_document", "old_scan", "photocopy_fax",
                   "book_page", "quick_snap", "screenshot",
                   "dense_document", "dark_screen"],
        "font_filter": "regular",
    },
    "handwritten": {
        "proportion": 0.25,
        "ops": [stroke_variation, smudge, variable_stroke,
                variable_baseline, slant, ink_fade, lined_paper,
                noise, exposure_jitter, rotation, wave_distortion,
                bleed_through],
        "chains": ["book_page", "quick_snap", "phone_document",
                   "dense_document", "notebook"],
        "font_filter": "handwriting",
    },
    "signage": {
        "proportion": 0.20,
        "ops": [perspective_warp, rotation, exposure_jitter, glare,
                weather_damage, color_jitter, uneven_lighting, occlusion,
                polarity_invert, photo_background],
        "chains": ["phone_sign", "outdoor_sign", "worn_label", "occluded",
                   "distant_photo", "flash_photo", "dark_sign"],
        "font_filter": "display",
    },
    "degraded": {
        "proportion": 0.10,
        "ops": [blur, jpeg_compress, noise, exposure_jitter,
                low_resolution, rotation, color_jitter,
                wave_distortion, striped_shadow, polarity_invert],
        "chains": list(CHAINS_BY_NAME),
        "font_filter": "all",
    },
}


def style_chains(style: str) -> list:
    """Resolve a style's chain names to (name, ops, weight) tuples."""
    return [(n, *CHAINS_BY_NAME[n]) for n in STYLES[style]["chains"]]


def apply_style_proportions(spec: str):
    """Override STYLES proportions from 'name=frac,...' and renormalize.

    Unlisted styles keep their default weight; everything is scaled to
    sum to 1 afterwards, so partial overrides behave intuitively.
    """
    for part in spec.split(","):
        name, sep, val = part.partition("=")
        name = name.strip()
        if not sep or name not in STYLES:
            raise SystemExit(
                f"--style-proportions: bad entry {part!r} "
                f"(valid styles: {', '.join(STYLES)})")
        try:
            frac = float(val)
        except ValueError:
            raise SystemExit(f"--style-proportions: bad value in {part!r}")
        if frac < 0:
            raise SystemExit(f"--style-proportions: negative value in {part!r}")
        STYLES[name]["proportion"] = frac
    total = sum(s["proportion"] for s in STYLES.values())
    if total <= 0:
        raise SystemExit("--style-proportions: proportions sum to 0")
    for s in STYLES.values():
        s["proportion"] /= total


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


# get_renderable_chars, split_by_script, and the CJK/ASCII cp predicates
# used to live here — moved to src.encoding.renderable and
# src.data.script_detect where the corresponding data lives.
from src.data.script_detect import split_by_script, is_ascii_cp
from src.encoding.renderable import get_renderable_chars


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


_ID_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_TLDS = ["com", "org", "net", "io", "in", "gov", "edu", "co", "de"]
_DOMAIN_WORDS = ["mail", "docs", "example", "acme", "corp", "info", "data",
                 "cloud", "app", "shop", "tech", "web", "portal", "office"]


def _random_alnum_id() -> str:
    """Reference IDs \u2014 invoice/case/serial numbers: INV-2024-0031, AB123."""
    r = random.random()
    if r < 0.4:
        return ("".join(random.choices(_ID_LETTERS, k=random.randint(2, 4)))
                + random.choice("-/")
                + "".join(random.choices("0123456789", k=random.randint(2, 6))))
    if r < 0.7:
        return ("".join(random.choices(_ID_LETTERS, k=random.randint(1, 3)))
                + "".join(random.choices("0123456789", k=random.randint(2, 5))))
    return ("".join(random.choices(_ID_LETTERS, k=random.randint(2, 3)))
            + "-" + str(random.randint(1, 9999))
            + random.choice("-/") + str(random.randint(1, 99)))


def _random_email() -> str:
    user = random.choice(_DOMAIN_WORDS)
    if random.random() < 0.4:
        user += str(random.randint(1, 99))
    return f"{user}@{random.choice(_DOMAIN_WORDS)}.{random.choice(_TLDS)}"


def _random_url() -> str:
    host = f"{random.choice(_DOMAIN_WORDS)}.{random.choice(_TLDS)}"
    r = random.random()
    if r < 0.4:
        return "www." + host
    if r < 0.7:
        return "https://" + host
    return host + "/" + random.choice(_DOMAIN_WORDS)


def _random_latin_segment():
    """Generate a random segment from ASCII_COMMON / TYPOGRAPHIC_COMMON chars.

    Includes the token classes agentic doc processing actually queries
    for \u2014 IDs, emails, URLs \u2014 alongside numbers/dates/currency.
    """
    r = random.random()
    if r < 0.28:
        # 1-3 random ASCII common chars
        n = random.randint(1, 3)
        return "".join(random.choices(_ASCII_COMMON_CHARS, k=n))
    elif r < 0.40:
        # Typographic chars
        return random.choice(_TYPO_COMMON_CHARS)
    elif r < 0.57:
        # Number: 1-6 digits, possibly with . or ,
        digits = "".join(random.choices("0123456789", k=random.randint(1, 6)))
        if len(digits) >= 4 and random.random() < 0.3:
            digits = f"{int(digits):,}"
        elif len(digits) >= 2 and random.random() < 0.3:
            pos = random.randint(1, len(digits) - 1)
            digits = digits[:pos] + "." + digits[pos:]
        return digits
    elif r < 0.67:
        # Currency + number
        currency = random.choice(["$", "\u00A3", "\u00A5", "\u20AC"])
        return currency + str(random.randint(1, 9999))
    elif r < 0.74:
        # Bracketed expression: (123), [45], {6}
        inner = "".join(random.choices("0123456789", k=random.randint(1, 4)))
        left, right = random.choice([("(", ")"), ("[", "]"), ("{", "}")])
        return left + inner + right
    elif r < 0.82:
        # Separated numbers: 12-34, 123.456.789
        sep = random.choice(list("-/."))
        parts = ["".join(random.choices("0123456789", k=random.randint(2, 4)))
                 for _ in range(random.randint(2, 3))]
        return sep.join(parts)
    elif r < 0.90:
        return _random_alnum_id()
    elif r < 0.95:
        return _random_email()
    else:
        return _random_url()


def _maybe_add_latin_segment(plan, p=0.15):
    """With probability p, append a latin punctuation/number segment to the plan.

    Marked "live" so render_content_plan renders the actual text instead of
    substituting a pool word — otherwise dates/currency/numbers never make
    it into the data.
    """
    if random.random() > p:
        return
    plan.append({"text": _random_latin_segment(), "script": "latin", "live": True})


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
    # Number prefix: 1.word, 2)word — no space: inter-word gaps are
    # deliberately unlabeled (CTC blank), so a labeled space inside a pool
    # entry would contradict the visually identical unlabeled gaps.
    (10, lambda w: random.choice(_DIGITS) + random.choice(".)") + w),
    # Number suffix: word-1, word/2
    (5,  lambda w: w + random.choice("-/") + random.choice(_DIGITS)),
    # Mixed: word-word, word/word (keeps both halves from word list)
    (3,  lambda w: w + random.choice("-/&") + w[:max(2, len(w) // 2)]),
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
    # Short number (1-3 digits) — page nos, counts, refs. Latin-only:
    # a pure-digit pool entry labeled as a non-Latin script would put
    # wrong per-pixel group labels on the digits.
    (5,  lambda w: str(random.randint(1, 999))),
]

# Script-native punctuation — real Hindi text has danda (U+0964) everywhere,
# Arabic uses its own comma/question/full stop, CJK uses fullwidth forms.
# All entries are verified encodable by their script. CJK punctuation is
# shared by the two Han codecs and attaches to its neighboring Han run.
_SCRIPT_PUNCT = {
    "devanagari": ["।", "॥"],
    "bengali": ["।"],
    "gurmukhi": ["।"],
    "odia": ["।"],
    "arabic": ["،", "؛", "؟", "۔"],
    HAN_SPARSE: ["。", "、", "，", "！", "？"],
    HAN_DENSE: ["。", "、", "，", "！", "？"],
    "armenian": ["։", "՞"],
    "ethiopic": ["።", "፣"],
    "tibetan": ["།"],
    "khmer": ["។"],
    "burmese": ["။", "၊"],
    "thai": ["ๆ", "ฯ"],
}

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


# ---------------------------------------------------------------------------
# Casing — real documents are full of Title Case and ALL-CAPS headers;
# signage is mostly caps. Word lists are lowercase-dominant, so without
# this the model rarely sees uppercase shapes in context.
# ---------------------------------------------------------------------------

_BICAMERAL = {"latin", "cyrillic", "greek", "armenian"}


def sample_line_casing() -> str:
    """Line-level casing mode, sampled once per line."""
    return random.choices(
        ["none", "sentence", "title", "upper"],
        weights=[0.72, 0.12, 0.08, 0.08], k=1)[0]


def apply_casing(word: str, script: str, mode: str, word_idx: int = 0) -> str:
    """Apply a line casing mode to one word of a bicameral script.

    sentence: first word capitalized; title: every word capitalized
    (headline); upper: ALL CAPS.
    """
    if script not in _BICAMERAL or mode == "none" or not word:
        return word
    if mode == "upper":
        return word.upper()
    if mode == "title":
        return word[:1].upper() + word[1:]
    if mode == "sentence" and word_idx == 0:
        return word[:1].upper() + word[1:]
    return word


def mix_punctuation(word: str, p: float = 0.15, script: str = "latin") -> str:
    """With probability p, mix punctuation/numbers into the word.

    Pure-number patterns (dates, currency, phone numbers) are only used for
    Latin, since they contain no script-specific characters and would confuse
    LID-1 routing if assigned to other scripts. Scripts with native
    punctuation (danda, Arabic comma, CJK fullwidth) use it ~1/3 of the time.

    Returns the original word unchanged (1-p) of the time.
    """
    if random.random() > p:
        return word
    native = _SCRIPT_PUNCT.get(script)
    if native and random.random() < 0.35:
        return word + random.choice(native)
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
    """True only for a COMPLETE (finalized) MDS chunk.

    MDSWriter writes shard.*.mds files and only writes index.json on a clean
    close. A chunk that crashed mid-write (or was killed by a Modal retry /
    wall-clock cutoff) has shard files but no index.json — it must be
    rewritten, not skipped, or StreamingDataset later can't open it. Keying
    the skip on index.json makes generation safely resumable.
    """
    return (Path(path) / "index.json").exists()


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


# Fraction of lines rendered in the legacy per-word mode (each word its
# own font + vertically normalized to full height). Real documents never
# look like that, but store signs / collage layouts occasionally do, and
# the diversity keeps the model from over-fitting line-consistent style.
RANSOM_PROB = 0.08


def render_content_plan(plan, fonts_by_script, h, mw, aug=None,
                        word_pools=None):
    """Stage 2: Render a content plan into image + metadata.

    Default path renders a typographically consistent line: one font
    size for the whole line, one font/weight per script, words at
    natural metrics composed on a shared baseline (constant x-height —
    'on' next to 'Apply' keeps real proportions instead of each word
    stretching to fill the crop). A small RANSOM_PROB fraction uses the
    legacy per-word pool mode for diversity.

    Words are always rendered clean (white bg, black ink). Augmentation
    applies color/style to the composed line so all words share one style.

    Args:
        plan: list of {"text": str, "script": str} — "whitespace" script means gap
        fonts_by_script: {script: [font_paths]}
        h: image height
        mw: max width
        aug: augmentation (applied to final composed image)
        word_pools: {script: [(pil_img, text, width), ...]} — pre-rendered
            word images for the ransom path.

    Returns:
        (img_tensor, label, group_labels, segments) or None if failed
        segments is a Python list (not JSON) for efficiency.
    """
    if word_pools and random.random() < RANSOM_PROB:
        result = _render_plan_ransom(plan, fonts_by_script, h, mw, word_pools)
    else:
        result = _render_plan_line(plan, fonts_by_script, h, mw)
    if result is None:
        return None
    combined, full_label, group_labels, segments = result

    # Apply augmentation
    if aug is not None:
        combined = aug(combined)

    img_tensor = rgb_to_input(combined)
    return img_tensor, full_label, group_labels, segments


def _render_plan_line(plan, fonts_by_script, h, mw):
    """Line-consistent rendering: shared baseline, one style per line."""
    from src.taxonomy import NUM_GROUPS as BLANK_ID

    # Line-level style. Size distribution mirrors the old crisp/soft mix:
    # mostly render above target height and downsample.
    if random.random() < 0.7:
        font_size = random.randint(int(h * 1.1), h * 2)
    else:
        font_size = random.randint(max(12, h // 2), h - 6)

    line_fonts: dict[str, str] = {}    # script → font for this line
    line_weights: dict[str, int] = {}  # script → weight bucket

    blocks = []  # (img, baseline, text, gid, sid, native_width)
    total_w = 0
    # Width budget works in native units via an estimated final scale;
    # the estimate tightens as taller blocks arrive.
    est_line_h = int(font_size * 1.4)

    def under_budget(extra_w):
        return (total_w + extra_w) * (h / max(est_line_h, 1)) <= mw

    for item in plan:
        text = item["text"]
        script = item["script"]

        if script == "whitespace":
            # Gap scales with font size, like a real space glyph would
            ws_w = max(2, int(font_size * random.uniform(0.18, 0.55)))
            if not under_budget(ws_w):
                break
            blocks.append((None, 0, "", BLANK_ID, 0, ws_w))
            total_w += ws_w
            continue

        # Line font for this script — picked once; per-word fallback only
        # when that font lacks coverage for this specific text.
        font = line_fonts.get(script)
        weight = line_weights.get(script, 0)
        if font is None or not font_covers_text(font, text):
            font = None
            fonts = fonts_by_script.get(script, [])
            for cand in random.sample(fonts, min(4, len(fonts))):
                if font_covers_text(cand, text):
                    font = cand
                    weight = pick_weight(cand, font_size)
                    break
            if font is None:
                return None  # no coverage — skip sample
            line_fonts.setdefault(script, font)
            line_weights.setdefault(script, weight)

        rendered = render_word_baseline(text, font, font_size, weight)
        if rendered is None or not image_has_ink(rendered[0]):
            return None
        img, baseline = rendered

        if not under_budget(img.width):
            # Don't crop a word mid-glyph and keep its full label
            if blocks:
                break
            return None  # single word wider than max width
        gid = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]
        sid = SCRIPT_TO_ID[script]
        blocks.append((img, baseline, text, gid, sid, img.width))
        total_w += img.width
        est_line_h = max(est_line_h, img.height)

    if not blocks or not any(b[2] for b in blocks):
        return None

    full_label = "".join(block[2] for block in blocks if block[2])
    blocks = visual_order_blocks(blocks, text_index=2)
    composed = compose_line_baseline(blocks, h)
    if composed is None:
        return None
    line_img, placed, scale = composed
    final_w = line_img.width
    if final_w > mw:
        return None  # budget estimate overshot — rare, skip sample

    group_labels = np.full(final_w, BLANK_ID, dtype=np.int32)
    segments = []
    for text, gid, sid, x, wnat, _y in placed:
        if not text:
            continue
        o = round(x * scale)
        e = min(final_w, round((x + wnat) * scale))
        if e - o < 1:
            continue
        group_labels[o:e] = gid
        segments.append({"group_id": gid, "script_id": sid,
                         "text": text, "width": e - o, "offset": o})
    if not segments:
        return None
    return line_img, full_label, group_labels, segments


def _render_plan_ransom(plan, fonts_by_script, h, mw, word_pools):
    """Legacy per-word mode: pool images / independent fonts, each word
    vertically normalized to full height. Kept as a small-probability
    diversity mode (signs, collages) — see RANSOM_PROB."""
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

        # Mix pool picks (fast) with live renders of the planned text.
        # Pool-only rendering collapsed diversity to WORD_POOL_SIZE unique
        # (word, font) pairs per script and silently discarded planned
        # segments (numbers, dates, punctuation). Items marked "live"
        # (e.g. _maybe_add_latin_segment output) always render their text.
        pool = word_pools.get(script) if word_pools else None
        use_pool = (pool and not item.get("live")
                    and random.random() < POOL_USE_PROB)
        img = None
        if use_pool:
            img, text, _pw = random.choice(pool)
        else:
            fonts = fonts_by_script.get(script, [])
            for _ in range(min(3, len(fonts))):
                font = random.choice(fonts)
                if not font_covers_text(font, text):
                    continue
                img = render_word(text, font, h, clean=True)
                if img is not None and image_has_ink(img):
                    break
                img = None
            if img is None and pool:
                img, text, _pw = random.choice(pool)
        if img is None:
            return None

        if total_w + img.width > mw:
            # Don't crop a word mid-glyph and keep its full label — that
            # trains the model to hallucinate the cut-off characters.
            # (partial_crop is the intentional, bounded version of this.)
            if blocks:
                break
            return None  # single word wider than max width — skip sample

        gid = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]
        sid = SCRIPT_TO_ID[script]
        blocks.append((img, text, gid, sid, img.width))
        total_w += img.width

    if not blocks or total_w == 0 or not any(b[1] for b in blocks):
        return None

    full_label = "".join(block[1] for block in blocks if block[1])
    blocks = visual_order_blocks(blocks, text_index=1)

    # Compose: concatenate all blocks
    combined = Image.new("RGB", (total_w, h), (255, 255, 255))
    group_labels = np.full(total_w, BLANK_ID, dtype=np.int32)
    segments = []
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

    return combined, full_label, group_labels, segments


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
    # Reaching here means this chunk was NOT skipped as complete, so any dir
    # present is a partial write from a crashed/retried attempt. MDSWriter
    # refuses a non-empty dir (FileExistsError), so wipe first — makes the
    # write idempotent under Modal retries and resumes.
    import shutil
    for d in (t_dir, v_dir):
        if Path(d).exists():
            shutil.rmtree(d)
        Path(d).mkdir(parents=True, exist_ok=True)

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

    # Per-chunk sidecars (widths for batch sizing, script ids for
    # sampling weights). The top-level files are assembled from these in
    # sorted chunk order after generation, which matches
    # LipiStreamingDataset's stream order (pool completion order does
    # not) and survives resumed runs (existing chunks keep their files).
    np.save(str(Path(t_dir) / "widths.npy"),
            np.array(train_widths, dtype=np.int32))
    np.save(str(Path(v_dir) / "widths.npy"),
            np.array(val_widths, dtype=np.int32))
    np.save(str(Path(t_dir) / "script_ids.npy"),
            np.full(len(train_widths), script_id, dtype=np.int32))
    np.save(str(Path(v_dir) / "script_ids.npy"),
            np.full(len(val_widths), script_id, dtype=np.int32))

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
    parser.add_argument("--augment", dest="augment", action="store_true",
                        default=False,
                        help="Bake augmentation into shards (default: off — "
                             "augmentation runs at train time in the "
                             "dataloader; baking it in freezes one "
                             "degradation per sample for all epochs)")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--height", type=int, default=32)
    # 2048 covers full A4 body lines and most table rows at h=32
    # normalization; the line path truncates at the budget (never
    # squashes aspect ratio).
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--punct-prob", type=float, default=0.15,
                        help="Probability of mixing punctuation/numbers into a word (default: 0.15)")
    parser.add_argument("--style-proportions", type=str, default=None,
                        help="Override style mix, e.g. "
                             "'printed=0.5,handwritten=0.15,signage=0.1,"
                             "clean=0.1,degraded=0.15'. Unlisted styles "
                             "keep their default; values are renormalized "
                             "to sum to 1. Tune toward the deployment "
                             "domain (document-heavy → raise printed).")
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
_worker_word_pools = None     # {(font_filter, script): [(pil_img, text, width), ...]}
_worker_group_index = None    # {style: {group_id: [script_info, ...]}}
_worker_mixed_ratio = 0.6     # set from CLI via initializer
_worker_punct_prob = 0.15     # set from CLI via initializer

WORD_POOL_SIZE = 400  # pre-rendered words per (font_filter, script)
POOL_USE_PROB = 0.7   # pool pick vs live render of the planned text

_worker_word_samplers: dict = {}  # script → WordSampler (lazy, per worker)


def _sample_word(script: str, words: list) -> str:
    """Sample a word — 30% frequency-weighted where a corpus table
    exists, uniform otherwise (see WordSampler)."""
    sampler = _worker_word_samplers.get(script)
    if sampler is None or sampler.words is not words:
        sampler = WordSampler(script, words)
        _worker_word_samplers[script] = sampler
    return sampler.sample()


def _init_line_worker(style_configs, mixed_ratio, pool_height=32,
                      punct_prob=0.15):
    """Pool initializer: load shared data and pre-build word pools."""
    global _worker_style_configs, _worker_word_pools
    global _worker_group_index, _worker_mixed_ratio, _worker_punct_prob
    _worker_style_configs = style_configs
    _worker_word_pools = {}
    _worker_mixed_ratio = mixed_ratio
    _worker_punct_prob = punct_prob
    import os
    pid = os.getpid()
    # Pre-build group index for fast mixed-line script selection
    _worker_group_index = {}
    for style, (script_info, _fonts) in style_configs.items():
        by_group = {}
        for info in script_info:
            by_group.setdefault(info[3], []).append(info)
        _worker_group_index[style] = by_group
    # Pre-build word pools for the most common font filter ("regular",
    # used by clean + printed). Other filters build lazily on first use —
    # pools are keyed by (font_filter, script) so handwritten/signage
    # styles actually render with their own fonts.
    t0 = time.time()
    for style, (script_info, _fonts) in style_configs.items():
        if STYLES[style]["font_filter"] != "regular":
            continue
        for script, fonts, words, _gid in script_info:
            _get_word_pool("regular", script, fonts, words,
                           h=pool_height, punct_prob=punct_prob)
        break
    elapsed = time.time() - t0
    total_words = sum(len(p) for p in _worker_word_pools.values())
    print(f"    [worker {pid}] pools ready: {len(_worker_word_pools)} scripts, "
          f"{total_words} words, {elapsed:.1f}s", flush=True)


def _get_word_pool(font_filter, script, fonts, words, h=32, punct_prob=0.15):
    """Get or build a pre-rendered word pool for a (font_filter, script).

    Each pool entry is (pil_image, text, width). Keyed by font filter so
    each style renders with its own font category — a single script-keyed
    pool built from "regular" fonts silently made the handwritten and
    signage font filters dead config.
    """
    key = (font_filter, script)
    if key in _worker_word_pools:
        return _worker_word_pools[key]

    pool = []
    attempts = 0
    target = min(WORD_POOL_SIZE, len(words) * 2)
    while len(pool) < target and attempts < target * 3:
        attempts += 1
        word = _sample_word(script, words)
        # Pool entries carry one script label and render as one segment, so
        # extract a matching maximal run from mixed CJK words instead of
        # assigning the whole word to one Han/Kana head. RTL words take the
        # same path: digit runs must not share a segment with the letters,
        # or the reversed CTC traversal reads the digits backwards.
        if script in HAN_SCRIPTS or script == "kana" or is_rtl_script(script):
            segs = split_by_script(word, script)
            candidates = [text for text, seg_script in segs
                          if seg_script == script and text.strip()]
            if not candidates:
                continue
            word = random.choice(candidates)
        # For non-latin pools: skip words with ASCII chars.
        # split_by_script in plan builders will route those chars to
        # latin, but the pool renders whole words with a single label.
        if script != "latin" and any(is_ascii_cp(ord(c)) for c in word):
            continue
        # Per-word casing (pool entries serve the ransom path, where
        # line-level consistency doesn't apply)
        word = apply_casing(word, script,
                            random.choices(["none", "title", "upper"],
                                           weights=[0.75, 0.15, 0.10], k=1)[0])
        # Mix punctuation for every script (universal patterns keep the
        # native word intact; scripts with native punctuation get danda,
        # Arabic comma, CJK fullwidth, etc.)
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

    Returns total_done. Widths are NOT returned: results arrive in
    completion order (imap_unordered), which does not match the sorted
    chunk order LipiStreamingDataset reads in. Per-chunk sidecar files
    written by save_rendered_samples are assembled after generation
    instead (see assemble_sidecars).
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
    with Pool(processes=n_procs,
              initializer=initializer, initargs=initargs) as pool:
        if initializer:
            # Workers are initializing — first result confirms they're ready
            print(f"  Workers initialized, generating...", flush=True)
        for result in pool.imap_unordered(worker_fn, chunks):
            _, script, n, _tw, _vw = result
            done += n
            elapsed = time.time() - start
            print(f"    total: {done}/{total_est} ({done/elapsed:.0f} img/s)", flush=True)
    elapsed = time.time() - start
    print(f"  Done: {done} {label} images in {elapsed:.0f}s "
          f"({done/max(elapsed,0.1):.0f} img/s)", flush=True)
    return done


def assemble_sidecars(split_dir: Path, name: str = "widths.npy") -> np.ndarray:
    """Concatenate per-chunk sidecar files in sorted chunk order.

    Sorted chunk order is exactly the order LipiStreamingDataset streams
    samples in, so index i here corresponds to dataset[i]. (Widths were
    previously accumulated in pool completion order — a shuffled
    correspondence — and resumed runs dropped skipped chunks' widths
    from the top-level file entirely.)
    """
    parts = []
    for chunk in sorted(split_dir.glob("chunk_*")):
        f = chunk / name
        if not f.exists():
            raise FileNotFoundError(
                f"{f} missing — chunk predates per-chunk sidecars. "
                f"Regenerate this shard directory from scratch.")
        parts.append(np.load(str(f)))
    if not parts:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(parts).astype(np.int32)


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
        aug = RandAugmentOCR(n_ops=2, p=0.5, ops=style_cfg["ops"],
                             chains=style_chains(style))
    t0 = time.time()

    # Collect word pools for this style's font filter (prebuilt at worker
    # init for "regular"; built lazily on first chunk for other filters)
    font_filter = style_cfg["font_filter"]
    word_pools = {}
    for script, fonts, words, _gid in all_script_info:
        pool = _get_word_pool(font_filter, script, fonts, words, h=h,
                              punct_prob=punct_prob)
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
            plan = _build_mixed_line_plan(group_index, primary_info)
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

    render_content_plan renders the planned text live (1 - POOL_USE_PROB)
    of the time and substitutes a pool image otherwise.
    """
    if script_info is None:
        return None

    script, _fonts, words, _gid = script_info
    n_words = random.choices([2, 3, 4, 5, 6, 7, 8],
                             weights=[0.10, 0.20, 0.25, 0.20, 0.15, 0.05, 0.05],
                             k=1)[0]
    casing = sample_line_casing()
    plan = []
    for i in range(n_words):
        if i > 0:
            plan.append({"text": " ", "script": "whitespace"})
        if script != "latin":
            _maybe_add_latin_segment(plan)
        word = apply_casing(_sample_word(script, words), script, casing, i)
        word = mix_punctuation(word, p=_worker_punct_prob, script=script)
        for seg_text, seg_script in split_by_script(word, script):
            plan.append({"text": seg_text, "script": seg_script})
        if script != "latin":
            _maybe_add_latin_segment(plan)
    return plan


def _build_mixed_line_plan(group_index, primary_info=None):
    """Build a content plan for a mixed-script line (2-8 words).

    70% of mixed lines pair the primary script with Latin — the dominant
    real-world mix (local script + English/digits). The rest use 2-4
    random groups, which real text rarely does but the per-pixel router
    still needs to see.

    Args:
        group_index: {group_id: [script_info, ...]} — pre-built index.
        primary_info: script_info of the chunk's primary script.
    """
    n_words = random.choices([2, 3, 4, 5, 6, 7, 8],
                             weights=[0.10, 0.20, 0.25, 0.20, 0.15, 0.05, 0.05],
                             k=1)[0]

    latin_info = None
    for infos in group_index.values():
        for info in infos:
            if info[0] == "latin":
                latin_info = info
                break

    realistic = (primary_info is not None and latin_info is not None
                 and primary_info[0] != "latin" and random.random() < 0.7)
    if realistic:
        # Primary script with Latin words sprinkled in
        chosen = [primary_info if random.random() < 0.7 else latin_info
                  for _ in range(n_words)]
        chosen[0] = primary_info
        if latin_info not in chosen:
            chosen[random.randrange(1, max(2, n_words))] = latin_info
    else:
        n_groups = random.choices([1, 2, 3, 4],
                                  weights=[0.10, 0.45, 0.30, 0.15],
                                  k=1)[0]
        n_groups = min(n_groups, len(group_index))
        n_words = max(n_words, n_groups)

        group_ids = random.sample(list(group_index.keys()),
                                  min(n_groups, len(group_index)))
        group_scripts = [random.choice(group_index[gid]) for gid in group_ids]

        chosen = list(group_scripts)
        for _ in range(n_words - len(chosen)):
            chosen.append(random.choice(group_scripts))
        random.shuffle(chosen)

    casing = sample_line_casing()
    plan = []
    for i, (script, _fonts, words, _gid) in enumerate(chosen):
        if i > 0:
            plan.append({"text": " ", "script": "whitespace"})
        if script != "latin":
            _maybe_add_latin_segment(plan)
            _maybe_add_latin_segment(plan)
        word = apply_casing(_sample_word(script, words), script, casing, i)
        word = mix_punctuation(word, p=_worker_punct_prob, script=script)
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

    # Hard gate: without libraqm, complex scripts (Arabic, Indic, Thai...)
    # render unjoined/reordered and the whole dataset is silently corrupt
    # for those scripts. text_renderer only warns; generation must refuse.
    # Escape hatch for a deliberate Latin/CJK-only render: LIPI_ALLOW_NO_RAQM=1.
    from PIL import features as _pil_features
    if not _pil_features.check("raqm") and \
            os.environ.get("LIPI_ALLOW_NO_RAQM") != "1":
        raise SystemExit(
            "Pillow has no libraqm — complex scripts would render wrong. "
            "Install a Pillow build with libraqm (source-build against "
            "libraqm-dev), or set LIPI_ALLOW_NO_RAQM=1 to generate anyway "
            "(only safe for Latin/CJK-only runs).")

    active_scripts = (list(SCRIPTS) if args.scripts == "all"
                      else [s.strip() for s in args.scripts.split(",")])

    print("Loading word lists...")
    word_lists = load_all_word_lists(active_scripts)

    script_fonts, valid_scripts = discover_fonts(active_scripts, word_lists)

    line_scripts = list(valid_scripts)

    if args.vocab_proportional:
        import math
        script_vocabs = {}
        for script in line_scripts:
            vocab = build_script_vocab(script)
            script_vocabs[script] = len(vocab)
        min_vocab = min(script_vocabs.values())
        print(f"\nVocab-proportional scaling (base={args.samples_per_script}):")
        tasks = []
        for script in line_scripts:
            scale = math.sqrt(script_vocabs[script] / min_vocab)
            target = int(args.samples_per_script * scale)
            tasks.append((script, target))
            print(f"  {script:<15s} vocab={script_vocabs[script]:>5d}  "
                  f"scale={scale:.2f}x  samples={target}")
    else:
        tasks = []
        for script in line_scripts:
            group = SCRIPT_TO_GROUP[script]
            if args.balance_groups:
                scripts_in_group = [s for s in line_scripts if SCRIPT_TO_GROUP[s] == group]
                target = args.samples_per_script // len(scripts_in_group)
            else:
                target = args.samples_per_script
            tasks.append((script, target))

    shard_dir = Path(args.out)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Resolve styles
    if args.style_proportions:
        apply_style_proportions(args.style_proportions)
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
        run_generation_pool(
            chunks, _generate_line_batch, args.workers, "line",
            initializer=_init_line_worker,
            initargs=(style_configs, mixed_ratio, args.height,
                      args.punct_prob))
    else:
        print("All chunks exist. Done.")

    # Character images
    if args.include_chars:
        char_chunks = build_char_chunks(valid_scripts, script_fonts, args,
                                        shard_dir, next_chunk_idx)
        if char_chunks:
            print(f"\n  {len(char_chunks)} char chunks across "
                  f"{min(args.workers, len(char_chunks))} workers\n")
            run_generation_pool(char_chunks, _generate_char_batch,
                                args.workers, "char")
        else:
            print("  All char chunks exist.")

    # Assemble top-level sidecars from per-chunk files, in sorted chunk
    # order (= dataset stream order). Covers resumed chunks too.
    print("\nSaving widths and metadata...", flush=True)
    all_train_widths = assemble_sidecars(shard_dir / "train")
    all_val_widths = assemble_sidecars(shard_dir / "val")
    np.save(str(shard_dir / "train" / "widths.npy"), all_train_widths)
    np.save(str(shard_dir / "val" / "widths.npy"), all_val_widths)
    np.save(str(shard_dir / "train" / "script_ids.npy"),
            assemble_sidecars(shard_dir / "train", "script_ids.npy"))
    np.save(str(shard_dir / "val" / "script_ids.npy"),
            assemble_sidecars(shard_dir / "val", "script_ids.npy"))

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
        "han_split_version": HAN_SPLIT_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
        "height": args.height,
        "max_width": args.max_width,
    }, shard_dir / "metadata.pt")

    n_train = len(all_train_widths)
    n_val = len(all_val_widths)
    print(f"\nDone: {n_train + n_val} samples → {n_train} train + {n_val} val")
    print(f"  {shard_dir}/train  {shard_dir}/val")


if __name__ == "__main__":
    main()
