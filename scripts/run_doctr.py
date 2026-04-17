#!/usr/bin/env python3
"""
End-to-end OCR: docTR FAST detection + Lipi MoE recognition.

Usage:
    python scripts/run_doctr.py path/to/document.pdf [--checkpoint moe_epoch4.pt]
"""

import argparse
import sys
import time
from pathlib import Path

import fitz  # pymupdf
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.color import rgb_to_input
from src.encoding.decompose import decode_ids, script_vocab_size
from src.model.encoder import LipiMoEEncoder
from src.model.lid import (
    GROUPS, GROUP_SCRIPTS, SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID,
    SCRIPT_TO_ID,
)

# Height the model expects
IMG_HEIGHT = 32


def build_vocab_tables():
    """Build group_script_vocab_sizes and group_script_names for all 13 groups."""
    group_script_vocab_sizes = []
    group_script_names = []
    for group_name in GROUPS:
        scripts = GROUP_SCRIPTS[group_name]
        sizes = [script_vocab_size(s) for s in scripts]
        group_script_vocab_sizes.append(sizes)
        group_script_names.append(list(scripts))
    return group_script_vocab_sizes, group_script_names


def load_model(checkpoint_path: str, device: torch.device):
    """Load LipiMoEEncoder from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_config" in ckpt:
        model = LipiMoEEncoder(**ckpt["model_config"])
    else:
        vocab_sizes, script_names = build_vocab_tables()
        model = LipiMoEEncoder(
            dim=512,
            num_groups=len(GROUPS),
            group_script_vocab_sizes=vocab_sizes,
            group_script_names=script_names,
        )

    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model, script_names


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    """Render each PDF page to a PIL image."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def detect_words(page_images: list[Image.Image]):
    """Run docTR FAST detection on page images, return word-level crops."""
    from doctr.models import detection_predictor

    det_model = detection_predictor("fast_base", pretrained=True)

    all_crops = []
    for page_idx, page_img in enumerate(page_images):
        page_np = np.array(page_img)
        # doctr expects (H, W, 3) uint8
        result = det_model([page_np])
        page_preds = result[0]  # dict with 'words' key
        h, w = page_np.shape[:2]

        boxes = page_preds["words"]  # (N, 5) relative coords [xmin,ymin,xmax,ymax,conf]
        if len(boxes) == 0:
            continue

        # Sort by y then x for reading order
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        # Group into lines by y-proximity
        line_height = np.median(boxes[:, 3] - boxes[:, 1]) * h
        order = np.lexsort((centers_x, (centers_y * h / max(line_height, 1)).astype(int)))

        for idx in order:
            xmin, ymin, xmax, ymax = boxes[idx, :4]
            conf = boxes[idx, 4] if boxes.shape[1] > 4 else 1.0
            # Convert relative to absolute
            x1 = max(0, int(xmin * w))
            y1 = max(0, int(ymin * h))
            x2 = min(w, int(xmax * w))
            y2 = min(h, int(ymax * h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = page_img.crop((x1, y1, x2, y2))
            all_crops.append({
                "page": page_idx,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
                "image": crop,
            })
    return all_crops


def prepare_crop(crop_img: Image.Image) -> torch.Tensor:
    """Resize crop to height=32 preserving aspect ratio, convert to RGB tensor."""
    w, h = crop_img.size
    new_w = max(4, int(w * IMG_HEIGHT / h))
    # Width must be multiple of 4 (two 2x downsamples)
    new_w = (new_w + 3) // 4 * 4
    resized = crop_img.resize((new_w, IMG_HEIGHT), Image.BILINEAR)
    return rgb_to_input(resized)  # (3, 32, new_w)


def ctc_decode(logits: torch.Tensor, vocab_size: int) -> list[int]:
    """Greedy CTC decode: argmax, collapse repeats, remove blanks."""
    seq = logits[:, :vocab_size].argmax(dim=-1).tolist()
    ids = []
    prev = -1
    for t in seq:
        if t != 0 and t != prev:
            ids.append(t)
        prev = t
    return ids


def ctc_confidence(logits: torch.Tensor, vocab_size: int) -> float:
    """Compute word-level confidence from CTC logits.

    Takes the softmax probability of the argmax token at each non-blank,
    non-repeat position, then returns the geometric mean (exp of mean log-prob).
    """
    probs = torch.softmax(logits[:, :vocab_size].float(), dim=-1)
    max_probs, max_ids = probs.max(dim=-1)  # (T,)

    # Only score non-blank, non-repeat positions (the actual decoded chars)
    char_probs = []
    prev = -1
    for t in range(max_ids.shape[0]):
        tok = max_ids[t].item()
        if tok != 0 and tok != prev:
            char_probs.append(max_probs[t].item())
        prev = tok

    if not char_probs:
        return 0.0
    # Geometric mean
    log_mean = sum(np.log(p + 1e-10) for p in char_probs) / len(char_probs)
    return float(np.exp(log_mean))


# ---------------------------------------------------------------------------
# Spell checker
# ---------------------------------------------------------------------------

DICT_DIR = Path(__file__).parent.parent / "training_data" / "dictionaries"

# Map scripts to hunspell dictionary files
_SCRIPT_DICT_FILES: dict[str, list[str]] = {
    "latin": ["en_US.dic"],
    "devanagari": ["hi_IN.dic"],
}


def _edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(dp[j], dp[j-1], prev)
            prev = temp
    return dp[n]


class SpellChecker:
    """Simple spell checker using word lists and edit distance.

    Only corrects when:
    1. Model confidence is below threshold
    2. A close match exists (edit distance <= max_edit)
    3. The correction is meaningfully better than the original
    """

    def __init__(self, conf_threshold: float = 0.5, max_edit: int = 2):
        self.conf_threshold = conf_threshold
        self.max_edit = max_edit
        self._vocab: dict[str, set[str]] = {}
        self._index: dict[str, dict[int, list[str]]] = {}  # script -> {length -> [words]}

    def _load_vocab(self, script: str) -> set[str]:
        if script in self._vocab:
            return self._vocab[script]

        words = set()
        files = _SCRIPT_DICT_FILES.get(script, [])
        for fname in files:
            path = DICT_DIR / fname
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                # Hunspell .dic format: word/flags — strip flags
                word = line.split("/")[0]
                if 2 <= len(word) <= 25:
                    words.add(word)

        self._vocab[script] = words
        if words:
            print(f"  Spell checker: {len(words)} words for {script}")
        return words

    def _build_index(self, script: str):
        """Build length-bucketed index for fast lookup."""
        vocab = self._load_vocab(script)
        if script not in self._index:
            buckets: dict[int, list[str]] = {}
            for w in vocab:
                buckets.setdefault(len(w), []).append(w)
            self._index[script] = buckets

    def maybe_correct(self, text: str, confidence: float, script: str) -> tuple[str, bool]:
        """Possibly correct text if confidence is low.

        Returns (corrected_text, was_corrected).
        """
        if confidence >= self.conf_threshold:
            return text, False

        # Don't try to correct very short or number-heavy strings
        if len(text) < 3 or any(c.isdigit() for c in text):
            return text, False

        vocab = self._load_vocab(script)
        if not vocab:
            return text, False
        self._build_index(script)
        buckets = self._index[script]

        # Strip trailing punctuation for lookup, reattach after
        stripped = text.rstrip(".,;:!?\u0964\u0965")  # include danda/double-danda
        trail = text[len(stripped):]
        lookup = stripped.lower() if script == "latin" else stripped

        if len(lookup) < 3:
            return text, False

        # Exact match — no correction needed
        if lookup in vocab:
            return text, False

        # Max edit scales with word length: short words get max 1 edit
        max_edit = 1 if len(lookup) <= 4 else self.max_edit

        # Search only words within ±max_edit length (from bucketed index)
        best_word, best_dist = None, max_edit + 1
        for length in range(max(2, len(lookup) - max_edit),
                            len(lookup) + max_edit + 1):
            for w in buckets.get(length, []):
                d = _edit_distance(lookup, w)
                if d < best_dist:
                    best_dist = d
                    best_word = w
                    if d == 1:
                        break
            if best_dist == 1:
                break

        if best_word is not None and best_dist <= max_edit:
            return best_word + trail, True

        return text, False


def build_script_filter(allowed_scripts: list[str] | None):
    """Build lookup tables for restricting LID-1/LID-2 to specific scripts.

    Returns:
        allowed_group_ids: set of group indices that contain at least one allowed script
        allowed_local_ids: dict[group_id] -> set of local script indices allowed in that group
        None if no filtering requested
    """
    if not allowed_scripts:
        return None

    # Validate script names
    for s in allowed_scripts:
        if s not in SCRIPT_TO_GROUP:
            valid = ", ".join(sorted(SCRIPT_TO_GROUP.keys()))
            raise ValueError(f"Unknown script '{s}'. Valid scripts: {valid}")

    allowed_group_ids = set()
    allowed_local_ids: dict[int, set[int]] = {}

    for script in allowed_scripts:
        group_name = SCRIPT_TO_GROUP[script]
        group_id = GROUP_TO_ID[group_name]
        allowed_group_ids.add(group_id)
        scripts_in_group = GROUP_SCRIPTS[group_name]
        local_id = scripts_in_group.index(script)
        allowed_local_ids.setdefault(group_id, set()).add(local_id)

    return allowed_group_ids, allowed_local_ids


def _decode_one(logits_i, group_id, script_logits_list, batch_idx, mask_offset,
                script_names, script_filter):
    """Decode a single sample from batched output."""
    # LID-2: pick script within group
    local_script_id = 0
    for g, sl, mask in script_logits_list:
        if g == group_id and mask[batch_idx]:
            # Find this sample's index within the group mask
            group_mask_indices = mask[:batch_idx + 1].sum().item() - 1
            if script_filter is not None:
                allowed_local = script_filter[1].get(group_id)
                if allowed_local:
                    masked_sl = sl[int(group_mask_indices)].clone()
                    for s in range(masked_sl.shape[0]):
                        if s not in allowed_local:
                            masked_sl[s] = -float("inf")
                    local_script_id = masked_sl.argmax().item()
                else:
                    local_script_id = sl[int(group_mask_indices)].argmax().item()
            else:
                local_script_id = sl[int(group_mask_indices)].argmax().item()
            break

    group_scripts = script_names[group_id]
    local_script_id = min(local_script_id, len(group_scripts) - 1)
    script_name = group_scripts[local_script_id]
    vs = script_vocab_size(script_name)
    ids = ctc_decode(logits_i, vs)
    text = decode_ids(ids, script_name)
    conf = ctc_confidence(logits_i, vs)
    return script_name, text, conf


@torch.no_grad()
def recognize_crops(model, crops: list[dict], script_names: list[list[str]],
                    device: torch.device, script_filter=None,
                    batch_size: int = 32, spell_checker: SpellChecker | None = None):
    """Run Lipi recognition on crops in batches."""
    results = [None] * len(crops)
    total_model_time = 0.0

    # Prepare all tensors upfront
    prepared = [(i, prepare_crop(c["image"])) for i, c in enumerate(crops)]

    # Sort by width for efficient batching (less padding waste)
    prepared.sort(key=lambda x: x[1].shape[2])

    # Use float16 on MPS/CUDA
    use_amp = device.type in ("mps", "cuda")
    amp_dtype = torch.float16

    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start:batch_start + batch_size]
        indices = [b[0] for b in batch]
        tensors = [b[1] for b in batch]

        # Pad to max width in batch
        max_w = max(t.shape[2] for t in tensors)
        # Round up to multiple of 4
        max_w = (max_w + 3) // 4 * 4
        padded = []
        for t in tensors:
            pad_w = max_w - t.shape[2]
            if pad_w > 0:
                padded.append(torch.nn.functional.pad(t, (0, pad_w)))
            else:
                padded.append(t)

        batch_tensor = torch.stack(padded).to(device)  # (B, 3, 32, max_w)

        # Two-pass strategy for forced routing:
        # Pass 1 (cheap): run just through shared encoder to get group_logits,
        #   then mask to allowed groups.
        # But model doesn't expose partial forward, so instead:
        # Single pass with forced group_ids when filter is active.
        # We run once WITHOUT group_ids to get group_logits, then mask,
        # then only re-run if any were misrouted. With forced routing on
        # every call, we just always pass the forced ids.

        t0 = time.perf_counter()
        if script_filter is not None:
            # First: quick pass to get group logits (full forward unavoidable)
            if use_amp:
                with torch.autocast(device.type, dtype=amp_dtype):
                    output = model(batch_tensor)
            else:
                output = model(batch_tensor)

            # Mask to allowed groups
            allowed_group_ids = script_filter[0]
            masked = output["group_logits"].clone()
            for g in range(len(GROUPS)):
                if g not in allowed_group_ids:
                    masked[:, g] = -float("inf")
            forced = masked.argmax(dim=-1)

            # Only re-run if any were misrouted
            if (forced != output["group_ids"]).any():
                if use_amp:
                    with torch.autocast(device.type, dtype=amp_dtype):
                        output = model(batch_tensor, group_ids=forced)
                else:
                    output = model(batch_tensor, group_ids=forced)
        else:
            if use_amp:
                with torch.autocast(device.type, dtype=amp_dtype):
                    output = model(batch_tensor)
            else:
                output = model(batch_tensor)

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        total_model_time += time.perf_counter() - t0

        # Decode each sample in batch
        all_logits = output["logits"]  # (B, T, max_vocab)
        all_group_ids = output["group_ids"]  # (B,)

        for b, idx in enumerate(indices):
            group_id = all_group_ids[b].item()
            script_name, text, conf = _decode_one(
                all_logits[b], group_id, output["script_logits_per_group"],
                b, 0, script_names, script_filter)

            corrected = False
            if spell_checker is not None:
                text, corrected = spell_checker.maybe_correct(
                    text, conf, script_name)

            results[idx] = {
                "page": crops[idx]["page"],
                "bbox": crops[idx]["bbox"],
                "det_conf": crops[idx]["confidence"],
                "group": GROUPS[group_id],
                "script": script_name,
                "text": text,
                "confidence": conf,
                "corrected": corrected,
            }

    n = len(crops)
    avg_ms = (total_model_time / n * 1000) if n else 0
    print(f"\n  Lipi model: {total_model_time:.2f}s total, "
          f"{avg_ms:.1f}ms/crop, {n} crops ({n // max(1, int(total_model_time))} crops/s)")
    return results


SCRIPT_COLORS = {
    "latin": "#2196F3",
    "devanagari": "#4CAF50",
    "arabic": "#FF9800",
    "han": "#E91E63",
    "kana": "#F06292",
    "korean": "#9C27B0",
    "cyrillic": "#00BCD4",
    "greek": "#00BCD4",
    "hebrew": "#FF5722",
    "bengali": "#8BC34A",
    "gujarati": "#CDDC39",
    "gurmukhi": "#FFC107",
    "kannada": "#009688",
    "telugu": "#3F51B5",
    "malayalam": "#673AB7",
    "tamil": "#795548",
    "thai": "#607D8B",
    "emoji": "#FFEB3B",
}


def _get_font(size: int):
    """Try to load a Unicode-capable font, fall back to default."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Kohinoor.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_results(results: list[dict], page_images: list[Image.Image],
                   output_dir: Path, stem: str):
    """Draw bounding boxes with recognized text on a white canvas."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group results by page
    by_page: dict[int, list[dict]] = {}
    for r in results:
        by_page.setdefault(r["page"], []).append(r)

    for page_idx, page_img in enumerate(page_images):
        w, h = page_img.size
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas, "RGBA")

        page_results = by_page.get(page_idx, [])

        for r in page_results:
            x1, y1, x2, y2 = r["bbox"]
            text = r["text"].strip()
            if not text:
                continue

            color = SCRIPT_COLORS.get(r["script"], "#666666")

            r_c = int(color[1:3], 16)
            g_c = int(color[3:5], 16)
            b_c = int(color[5:7], 16)

            # White fill + colored border
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2,
                           fill=(255, 255, 255, 255))

            # Fit text inside the box
            box_h = y2 - y1
            box_w = x2 - x1
            font_size = max(8, int(box_h * 0.65))
            font = _get_font(font_size)

            # Shrink font if text is wider than the box
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            if text_w > box_w and box_w > 0:
                font_size = max(8, int(font_size * box_w / text_w))
                font = _get_font(font_size)

            # Center text vertically and horizontally inside the box
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = x1 + (box_w - text_w) // 2
            text_y = y1 + (box_h - text_h) // 2

            draw.text((text_x, text_y), text, fill=(r_c, g_c, b_c), font=font)

        out_path = output_dir / f"{stem}_ocr_p{page_idx + 1}.png"
        canvas.convert("RGB").save(out_path)
        print(f"\nVisualization saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="docTR detection + Lipi recognition")
    parser.add_argument("input", type=str, help="Path to PDF or image")
    parser.add_argument("--checkpoint", type=str, default="moe_epoch4.pt")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for visualization (default: same as input)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for recognition (default: 32)")
    parser.add_argument("--spellcheck", action="store_true",
                        help="Enable spell-check correction for low-confidence words")
    parser.add_argument("--spell-threshold", type=float, default=0.5,
                        help="Confidence threshold below which spell-check kicks in (default: 0.5)")
    parser.add_argument("--spell-max-edit", type=int, default=2,
                        help="Maximum edit distance for spell-check corrections (default: 2)")
    parser.add_argument("--scripts", type=str, default=None,
                        help="Comma-separated list of allowed scripts for LID routing "
                             "(e.g. 'devanagari,latin'). Overrides LID-1/LID-2 predictions.")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model, script_names = load_model(args.checkpoint, device)

    # Load document
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".pdf":
        print(f"Rendering PDF at {args.dpi} DPI...")
        page_images = pdf_to_images(str(input_path), dpi=args.dpi)
        print(f"  {len(page_images)} page(s)")
    else:
        page_images = [Image.open(str(input_path)).convert("RGB")]
        print(f"Loaded image: {input_path}")

    # Detect text regions
    print("Running FAST detection...")
    crops = detect_words(page_images)
    print(f"  {len(crops)} word regions detected")

    # Build script filter
    script_filter = None
    if args.scripts:
        allowed = [s.strip() for s in args.scripts.split(",")]
        script_filter = build_script_filter(allowed)
        print(f"Script filter: {allowed}")

    # Spell checker
    checker = None
    if args.spellcheck:
        checker = SpellChecker(conf_threshold=args.spell_threshold,
                               max_edit=args.spell_max_edit)
        print(f"Spell check enabled (threshold={args.spell_threshold}, "
              f"max_edit={args.spell_max_edit})")

    # Recognize
    print("Running Lipi recognition...")
    results = recognize_crops(model, crops, script_names, device,
                              script_filter=script_filter,
                              batch_size=args.batch_size,
                              spell_checker=checker)

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    current_page = -1
    for r in results:
        if r["page"] != current_page:
            current_page = r["page"]
            print(f"\n--- Page {current_page + 1} ---")
        x1, y1, x2, y2 = r["bbox"]
        conf = r.get("confidence", 0)
        mark = " *" if r.get("corrected") else ""
        print(f"  [{r['script']:>12s}] ({x1:4d},{y1:4d})-({x2:4d},{y2:4d})  "
              f"det={r['det_conf']:.2f} conf={conf:.2f}  {r['text']}{mark}")

    # Spell check summary
    if args.spellcheck:
        n_corrected = sum(1 for r in results if r.get("corrected"))
        n_low_conf = sum(1 for r in results if r.get("confidence", 1) < args.spell_threshold)
        print(f"\n  Spell check: {n_corrected} corrected / "
              f"{n_low_conf} low-confidence / {len(results)} total")

    # Render visualization
    output_dir = Path(args.output) if args.output else input_path.parent
    render_results(results, page_images, output_dir, input_path.stem)


if __name__ == "__main__":
    main()
