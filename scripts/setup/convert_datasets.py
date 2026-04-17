#!/usr/bin/env python3
"""
Convert real-world OCR datasets into Lipi shard format.

Reads downloaded datasets from training_data/real_datasets/,
converts to the same (images, labels, script_ids, group_ids) shard
format used by train_moe.py.

Usage:
    python scripts/convert_datasets.py --dataset iam --out data/real_shards
    python scripts/convert_datasets.py --dataset iiit-indic --out data/real_shards
    python scripts/convert_datasets.py --all --out data/real_shards
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, GROUP_TO_ID, SCRIPT_TO_ID
from src.data.color import rgb_to_input
from src.data.rendering import resize_or_pad, image_has_ink

DATA_DIR = Path(__file__).parent.parent / "training_data" / "real_datasets"
SHARD_SIZE = 5000  # images per shard


def process_image(img_path, height=32, max_width=768):
    """Load, validate, and convert an image to model input format.

    Resizes height to target while preserving aspect ratio.
    Width is kept natural (up to max_width). Padding to uniform width
    is done at shard save time, not here.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None

    if img.width < 4 or img.height < 4:
        return None

    # Resize height preserving aspect ratio
    if img.height != height:
        new_width = max(1, int(img.width * height / img.height))
        img = img.resize((new_width, height), Image.BILINEAR)

    # Cap at max_width (very long lines)
    if img.width > max_width:
        img = img.resize((max_width, height), Image.BILINEAR)

    if not image_has_ink(img):
        return None

    return rgb_to_input(img)


def encode_labels_for_shard(labels, script):
    """Pre-encode labels into token IDs."""
    from src.encoding.decompose import encode_text

    encoded = []
    max_len = 0
    for label in labels:
        ids = encode_text(label, script)
        encoded.append(ids)
        max_len = max(max_len, len(ids))

    if max_len == 0:
        max_len = 1
    target_ids = torch.zeros(len(encoded), max_len, dtype=torch.long)
    target_lens = torch.zeros(len(encoded), dtype=torch.long)
    for i, ids in enumerate(encoded):
        target_lens[i] = len(ids)
        if ids:
            target_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return target_ids, target_lens


def save_shards(images, labels, script, out_dir, prefix="real"):
    """Save images and labels as shards with pre-encoded targets."""
    if not images:
        return 0

    script_id = SCRIPT_TO_ID.get(script)
    group_id = GROUP_TO_ID.get(SCRIPT_TO_GROUP.get(script, ""), None)
    if script_id is None or group_id is None:
        print(f"  WARNING: Unknown script '{script}', skipping")
        return 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for i in range(0, len(images), SHARD_SIZE):
        batch_imgs = images[i:i + SHARD_SIZE]
        batch_labels = labels[i:i + SHARD_SIZE]
        n = len(batch_imgs)

        # Pad all images to the max width in this shard
        max_w = max(img.shape[2] for img in batch_imgs)
        padded = []
        for img in batch_imgs:
            if img.shape[2] < max_w:
                pad = torch.full((img.shape[0], img.shape[1], max_w - img.shape[2]),
                                 img.max(), dtype=img.dtype)
                img = torch.cat([img, pad], dim=2)
            padded.append(img)

        target_ids, target_lens = encode_labels_for_shard(batch_labels, script)

        shard_path = out_dir / f"{prefix}_{script}_{i // SHARD_SIZE:04d}.pt"
        torch.save({
            "images": torch.stack(padded),
            "labels": batch_labels,
            "script_ids": torch.full((n,), script_id, dtype=torch.long),
            "group_ids": torch.full((n,), group_id, dtype=torch.long),
            "target_ids": target_ids,
            "target_lens": target_lens,
        }, shard_path)
        total += n

    return total


# ---------------------------------------------------------------------------
# Dataset converters
# ---------------------------------------------------------------------------

CONVERTERS = {}


def register(name, description):
    def decorator(func):
        CONVERTERS[name] = {"fn": func, "description": description}
        return func
    return decorator


@register("iam", "IAM Handwriting Database (English/Latin)")
def convert_iam(out_dir):
    iam_dir = DATA_DIR / "iam"
    images, labels = [], []

    # Try word-level Kaggle format
    words_dir = iam_dir / "words"
    if not words_dir.exists():
        # Try HuggingFace line-level format
        words_dir = iam_dir / "lines"

    if not words_dir.exists():
        # Search for any image directory
        for d in iam_dir.rglob("*"):
            if d.is_dir() and any(d.glob("*.png")):
                words_dir = d
                break

    if not words_dir.exists():
        print(f"  No image directory found in {iam_dir}")
        return

    # Try to find labels file
    labels_file = None
    for candidate in ["words.txt", "labels.txt", "annotations.txt"]:
        f = iam_dir / candidate
        if f.exists():
            labels_file = f
            break

    if labels_file and labels_file.name == "words.txt":
        # IAM words.txt format: id ok/err graylevel x y w h grammar word
        word_labels = {}
        for line in labels_file.read_text(errors="ignore").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 9 and parts[1] == "ok":
                word_id = parts[0]
                word = parts[-1]
                if 2 <= len(word) <= 15:
                    word_labels[word_id] = word

        # Match images to labels
        for img_path in sorted(words_dir.rglob("*.png")):
            word_id = img_path.stem
            if word_id in word_labels:
                tensor = process_image(img_path)
                if tensor is not None:
                    images.append(tensor)
                    labels.append(word_labels[word_id])
    else:
        # Try reading all images with their parent folder name as label hint
        for img_path in sorted(words_dir.rglob("*.png"))[:50000]:
            tensor = process_image(img_path)
            if tensor is not None:
                # Use filename as label if no labels file
                label = img_path.stem.replace("_", " ")
                if 2 <= len(label) <= 15:
                    images.append(tensor)
                    labels.append(label)

    total = save_shards(images, labels, "latin", out_dir, prefix="iam")
    print(f"  IAM: {total} word images converted")


@register("casia-hwdb", "CASIA-HWDB2 Chinese Handwritten Lines (HuggingFace)")
def convert_casia(out_dir):
    casia_dir = DATA_DIR / "casia-hwdb"
    images, labels = [], []

    # HuggingFace Teklia/CASIA-HWDB2-line format
    # Try parquet first (HF datasets default)
    try:
        from datasets import load_dataset
        ds = load_dataset(str(casia_dir))
        for split in ds:
            for item in ds[split]:
                img = item.get("image")
                label = item.get("text", item.get("ground_truth", ""))
                if img and label and 1 <= len(label) <= 50:
                    if isinstance(img, Image.Image):
                        img = img.convert("RGB")
                    else:
                        continue
                    if img.height != 32:
                        new_width = max(1, int(img.width * 32 / img.height))
                        img = img.resize((new_width, 32), Image.BILINEAR)
                    if img.width > 768:
                        img = img.resize((768, 32), Image.BILINEAR)
                    if img.width < 768:
                        padded = Image.new("RGB", (768, 32), (240, 240, 240))
                        padded.paste(img, (0, 0))
                        img = padded
                    if image_has_ink(img):
                        images.append(rgb_to_input(img))
                        labels.append(label)
        total = save_shards(images, labels, "han", out_dir, prefix="casia")
        print(f"  CASIA-HWDB: {total} line images converted")
        return
    except ImportError:
        print("  'datasets' package not installed, trying image directory...")
    except Exception as e:
        print(f"  HuggingFace datasets load failed: {e}, trying image directory...")

    # Fallback: look for image + label file pairs
    for img_dir in [casia_dir, casia_dir / "data", casia_dir / "train"]:
        if not img_dir.exists():
            continue
        for ann_file in sorted(img_dir.rglob("*.txt"))[:50000]:
            if "README" in ann_file.name:
                continue
            base = ann_file.stem
            for ext in [".png", ".jpg", ".jpeg"]:
                img_path = ann_file.parent / (base + ext)
                if img_path.exists():
                    label = ann_file.read_text(errors="ignore").strip()
                    if 1 <= len(label) <= 50:
                        tensor = process_image(img_path)
                        if tensor is not None:
                            images.append(tensor)
                            labels.append(label)
                    break

    total = save_shards(images, labels, "han", out_dir, prefix="casia")
    print(f"  CASIA-HWDB: {total} line images converted")


@register("iiit-indic", "IIIT-INDIC-HW-WORDS (10 Indic scripts)")
def convert_iiit_indic(out_dir):
    base_dir = DATA_DIR / "iiit-indic-hw"

    # Script name mapping (IIIT names → Lipi names)
    name_map = {
        "hindi": "devanagari",
        "bangla": "bengali",
        "gujarati": "gujarati",
        "gurumukhi": "gurmukhi",
        "punjabi": "gurmukhi",
        "kannada": "kannada",
        "malayalam": "malayalam",
        "odia": "odia",
        "oriya": "odia",
        "tamil": "tamil",
        "telugu": "telugu",
        "devanagari": "devanagari",
        "bengali": "bengali",
    }

    for script_dir in sorted(base_dir.rglob("*")):
        if not script_dir.is_dir():
            continue

        dir_name = script_dir.name.lower()
        script = name_map.get(dir_name)
        if not script:
            continue

        images, labels = [], []

        # Look for image files and labels
        label_file = script_dir / "labels.txt"
        if label_file.exists():
            for line in label_file.read_text(errors="ignore").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                img_name, label = parts
                img_path = script_dir / img_name
                if not img_path.exists():
                    img_path = script_dir / "images" / img_name
                if img_path.exists() and 1 <= len(label) <= 20:
                    tensor = process_image(img_path)
                    if tensor is not None:
                        images.append(tensor)
                        labels.append(label)
        else:
            # Try directory-per-word structure or CSV
            for csv_path in script_dir.rglob("*.csv"):
                import csv
                with open(csv_path, encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            img_name, label = row[0], row[1]
                            img_path = csv_path.parent / img_name
                            if img_path.exists() and 1 <= len(label) <= 20:
                                tensor = process_image(img_path)
                                if tensor is not None:
                                    images.append(tensor)
                                    labels.append(label)

            # Try image files with label in filename
            if not images:
                for img_path in sorted(script_dir.rglob("*.png"))[:50000]:
                    tensor = process_image(img_path)
                    if tensor is not None:
                        images.append(tensor)
                        labels.append(img_path.stem)

        if images:
            total = save_shards(images, labels, script, out_dir,
                                prefix=f"iiit_{script}")
            print(f"  {script}: {total} word images converted")


@register("hkr", "HKR Russian/Kazakh Handwriting")
def convert_hkr(out_dir):
    hkr_dir = DATA_DIR / "hkr" / "repo"
    if not hkr_dir.exists():
        hkr_dir = DATA_DIR / "hkr"

    images, labels = [], []

    # HKR has various formats — try common ones
    for img_dir in [hkr_dir / "img", hkr_dir / "data", hkr_dir]:
        if not img_dir.exists():
            continue
        for ann_file in img_dir.rglob("*.json"):
            try:
                with open(ann_file, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if "image" in item and "text" in item:
                            img_path = img_dir / item["image"]
                            if img_path.exists():
                                tensor = process_image(img_path)
                                if tensor is not None:
                                    images.append(tensor)
                                    labels.append(item["text"])
            except Exception:
                continue

    # Try txt annotation format
    if not images:
        for txt_file in hkr_dir.rglob("*.txt"):
            if txt_file.name in ("README.txt", "requirements.txt"):
                continue
            for line in txt_file.read_text(errors="ignore").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_name, label = parts
                    for ext in [".png", ".jpg", ".jpeg"]:
                        img_path = txt_file.parent / (img_name + ext)
                        if img_path.exists():
                            tensor = process_image(img_path)
                            if tensor is not None:
                                images.append(tensor)
                                labels.append(label)
                            break

    total = save_shards(images, labels, "cyrillic", out_dir, prefix="hkr")
    print(f"  HKR: {total} word images converted")


@register("hebrew-htr", "HebHTR Hebrew Handwriting")
def convert_hebrew(out_dir):
    heb_dir = DATA_DIR / "hebrew"
    images, labels = [], []

    for sub in [heb_dir / "hebhtr", heb_dir / "hhd", heb_dir]:
        if not sub.exists():
            continue
        for ann_file in sub.rglob("*.txt"):
            if ann_file.name.startswith("README"):
                continue
            for line in ann_file.read_text(errors="ignore").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_name, label = parts
                    img_path = ann_file.parent / img_name
                    if not img_path.exists():
                        for ext in [".png", ".jpg", ".jpeg", ".tif"]:
                            p = ann_file.parent / (img_name + ext)
                            if p.exists():
                                img_path = p
                                break
                    if img_path.exists() and 1 <= len(label) <= 20:
                        tensor = process_image(img_path)
                        if tensor is not None:
                            images.append(tensor)
                            labels.append(label)

    total = save_shards(images, labels, "hebrew", out_dir, prefix="hebrew")
    print(f"  Hebrew: {total} word images converted")


@register("arabic", "Arabic OCR datasets (Muharaf + OpenITI)")
def convert_arabic(out_dir):
    arabic_dir = DATA_DIR / "arabic"
    images, labels = [], []

    # Muharaf — line-level images with transcriptions
    muharaf = arabic_dir / "muharaf"
    if muharaf.exists():
        for ann_file in muharaf.rglob("*.txt"):
            base = ann_file.stem
            for ext in [".png", ".jpg", ".jpeg", ".tif"]:
                img_path = ann_file.parent / (base + ext)
                if img_path.exists():
                    label = ann_file.read_text(errors="ignore").strip()
                    # Split lines into words
                    for word in label.split():
                        if 1 <= len(word) <= 20:
                            tensor = process_image(img_path)
                            if tensor is not None:
                                images.append(tensor)
                                labels.append(word)
                    break

    total = save_shards(images, labels, "arabic", out_dir, prefix="arabic")
    print(f"  Arabic: {total} word images converted")


@register("ethiopic", "HHD-Ethiopic Historical Manuscripts")
def convert_ethiopic(out_dir):
    eth_dir = DATA_DIR / "ethiopic"
    images, labels = [], []

    # HuggingFace dataset format — look for parquet or image dirs
    for img_dir in [eth_dir, eth_dir / "data", eth_dir / "train"]:
        if not img_dir.exists():
            continue
        for ann_file in img_dir.rglob("*.txt"):
            base = ann_file.stem
            for ext in [".png", ".jpg", ".jpeg", ".tif"]:
                img_path = ann_file.parent / (base + ext)
                if img_path.exists():
                    label = ann_file.read_text(errors="ignore").strip()
                    for word in label.split():
                        if 1 <= len(word) <= 20:
                            tensor = process_image(img_path)
                            if tensor is not None:
                                images.append(tensor)
                                labels.append(word)
                    break

    total = save_shards(images, labels, "ethiopic", out_dir, prefix="ethiopic")
    print(f"  Ethiopic: {total} word images converted")


@register("thai", "iApp Thai Handwriting")
def convert_thai(out_dir):
    thai_dir = DATA_DIR / "thai"
    images, labels = [], []

    # Look for image + label pairs
    for sub in [thai_dir, thai_dir / "data", thai_dir / "train"]:
        if not sub.exists():
            continue
        for ann_file in sorted(sub.rglob("*.txt"))[:10000]:
            base = ann_file.stem
            for ext in [".png", ".jpg", ".jpeg"]:
                img_path = ann_file.parent / (base + ext)
                if img_path.exists():
                    label = ann_file.read_text(errors="ignore").strip()
                    for word in label.split():
                        if 1 <= len(word) <= 20:
                            tensor = process_image(img_path)
                            if tensor is not None:
                                images.append(tensor)
                                labels.append(word)
                    break

    total = save_shards(images, labels, "thai", out_dir, prefix="thai")
    print(f"  Thai: {total} word images converted")


@register("khmer", "KhmerST Scene Text")
def convert_khmer(out_dir):
    khmer_dir = DATA_DIR / "khmer"
    images, labels = [], []

    for sub in [khmer_dir / "khmerst", khmer_dir / "benchmark", khmer_dir]:
        if not sub.exists():
            continue
        for ann_file in sub.rglob("*.txt"):
            if "README" in ann_file.name:
                continue
            for line in ann_file.read_text(errors="ignore").splitlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_name, label = parts
                    img_path = ann_file.parent / img_name
                    if img_path.exists() and 1 <= len(label) <= 20:
                        tensor = process_image(img_path)
                        if tensor is not None:
                            images.append(tensor)
                            labels.append(label)

    total = save_shards(images, labels, "khmer", out_dir, prefix="khmer")
    print(f"  Khmer: {total} word images converted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert real datasets to shards")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=str, default="data/real_shards")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list or (not args.dataset and not args.all):
        print("\nAvailable converters:\n")
        for name, info in sorted(CONVERTERS.items()):
            print(f"  {name:<20s} {info['description']}")
        print(f"\nUsage: python {sys.argv[0]} --dataset <name> --out <dir>")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = list(CONVERTERS.keys()) if args.all else [
        d.strip() for d in args.dataset.split(",")]

    for name in targets:
        if name not in CONVERTERS:
            print(f"Unknown dataset: {name}")
            continue
        print(f"\n{'='*60}")
        print(f"Converting: {name} — {CONVERTERS[name]['description']}")
        print(f"{'='*60}")
        try:
            CONVERTERS[name]["fn"](out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save metadata
    all_scripts = list(set(s for s in SCRIPTS if s != "emoji"
                           and any((out_dir).glob(f"*_{s}_*.pt"))))
    if all_scripts:
        active_groups = []
        seen = set()
        for s in all_scripts:
            g = SCRIPT_TO_GROUP.get(s)
            if g and g not in seen:
                active_groups.append(g)
                seen.add(g)
        torch.save({
            "active_scripts": all_scripts,
            "active_groups": active_groups,
            "script_to_idx": {s: i for i, s in enumerate(all_scripts)},
            "group_to_idx": {g: i for i, g in enumerate(active_groups)},
            "height": 32,
            "max_width": 192,
            "augmented": False,
            "has_labels": True,
            "source": "real_world",
        }, out_dir / "metadata.pt")

    total_shards = len(list(out_dir.glob("*.pt"))) - 1  # minus metadata
    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.pt")) / 1e6
    print(f"\nDone. {total_shards} shards ({total_mb:.0f} MB) in {out_dir}/")


if __name__ == "__main__":
    main()
