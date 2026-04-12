#!/usr/bin/env python3
"""
Run standard OCR benchmarks on a Lipi checkpoint.

Downloads IIIT5K, SVT, IC13, IC15, SVTP, CUTE80 from HuggingFace
and evaluates word accuracy.

Usage:
    python scripts/benchmark.py --resume checkpoints/moe/moe_epoch48.pt
    python scripts/benchmark.py --resume checkpoints/moe/moe_epoch48.pt --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiMoEEncoder
from src.model.lid import GROUPS, NUM_GROUPS, SCRIPT_TO_GROUP
from src.data.color import rgb_to_input
from src.encoding.decompose import decode_ids, script_vocab_size
from src.training.eval import _edit_distance


BENCHMARKS = {
    "IIIT5K": {"split": "test", "n": 3000},
    "SVT":    {"split": "test", "n": 647},
    "IC13":   {"split": "test", "n": 857},
    "IC15":   {"split": "test", "n": 1811},
    "SVTP":   {"split": "test", "n": 645},
    "CUTE80": {"split": "test", "n": 288},
}


def build_vocab_tables():
    from src.encoding.decompose import script_vocab_size
    from src.model.lid import GROUP_SCRIPTS
    group_script_vocab_sizes = []
    group_script_names = []
    for group_name in GROUPS:
        scripts = GROUP_SCRIPTS[group_name]
        sizes = [script_vocab_size(s) for s in scripts]
        group_script_vocab_sizes.append(sizes)
        group_script_names.append(list(scripts))
    return group_script_vocab_sizes, group_script_names


def prepare_image(img: Image.Image, height: int = 32) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    new_w = max(height, int(w * height / h))
    new_w = (new_w + 7) // 8 * 8
    img = img.resize((new_w, height), Image.BILINEAR)
    return rgb_to_input(img)


def ctc_decode(logits: torch.Tensor, vocab_size: int, script: str) -> str:
    seq = logits[:, :vocab_size].argmax(dim=-1).tolist()
    ids = []
    prev = -1
    for t in seq:
        if t != prev and t != 0:
            ids.append(t)
        prev = t
    return decode_ids(ids, script)


def load_benchmark(name: str, cache_dir: str) -> list[tuple[Image.Image, str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from datasets import load_dataset

    print(f"  Loading {name}...")

    if name == "IIIT5K":
        ds = load_dataset("MiXaiLL76/IIIT5K_OCR", split="test", cache_dir=cache_dir)
        return [(sample["image"], sample["text"]) for sample in ds]

    # For other benchmarks, try the OpenOCR collection
    name_map = {
        "SVT": "SVT",
        "IC13": "IC13_857",
        "IC15": "IC15_1811",
        "SVTP": "SVTP",
        "CUTE80": "CUTE80",
    }

    try:
        from huggingface_hub import snapshot_download
        eval_dir = Path(cache_dir) / "openocr_eval"
        if not (eval_dir / "evaluation").exists():
            print(f"  Downloading OpenOCR evaluation data...")
            snapshot_download(
                repo_id="topdu/OpenOCR-Data",
                repo_type="dataset",
                allow_patterns=["evaluation/*"],
                local_dir=str(eval_dir),
            )

        import lmdb
        lmdb_name = name_map.get(name, name)
        lmdb_path = str(eval_dir / "evaluation" / lmdb_name)
        if not Path(lmdb_path).exists():
            print(f"  LMDB not found at {lmdb_path}, skipping {name}")
            return []

        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        samples = []
        with env.begin() as txn:
            n = int(txn.get(b"num-samples").decode())
            for i in range(1, n + 1):
                img_key = f"image-{i:09d}".encode()
                label_key = f"label-{i:09d}".encode()
                img_data = txn.get(img_key)
                label = txn.get(label_key).decode()
                if img_data is None:
                    continue
                from io import BytesIO
                img = Image.open(BytesIO(img_data)).convert("RGB")
                samples.append((img, label))
        env.close()
        return samples
    except Exception as e:
        print(f"  Failed to load {name}: {e}")
        return []


def evaluate_benchmark(model, samples, device, device_type, group_script_vocab_sizes,
                       group_script_names):
    model.eval()
    correct = 0
    total = 0
    total_chars = 0
    correct_chars = 0

    # Latin is group 0, script 0
    latin_vs = group_script_vocab_sizes[0][0]

    batch_size = 64
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        imgs = [prepare_image(img) for img, _ in batch_samples]
        labels = [label for _, label in batch_samples]

        # Pad to same width
        max_w = max(img.shape[2] for img in imgs)
        padded = torch.zeros(len(imgs), 3, 32, max_w, dtype=torch.uint8)
        for i, img in enumerate(imgs):
            padded[i, :, :, :img.shape[2]] = img

        padded = padded.to(device)
        with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
            out = model(padded, group_ids=None)

        logits = out["logits"].float().cpu()
        for i, label in enumerate(labels):
            ref = label.strip().lower()
            dec = ctc_decode(logits[i], latin_vs, "latin").strip().lower()
            if dec == ref:
                correct += 1
            edits = _edit_distance(dec, ref)
            correct_chars += max(0, len(ref) - edits)
            total_chars += len(ref)
            total += 1

    word_acc = 100 * correct / max(total, 1)
    char_acc = 100 * correct_chars / max(total_chars, 1)
    return word_acc, char_acc, total


def main():
    parser = argparse.ArgumentParser(description="Run OCR benchmarks")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache-dir", type=str, default="data/benchmarks")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu")
    else:
        device = torch.device(args.device)
    device_type = device.type
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

    # Build model
    group_script_vocab_sizes, group_script_names = build_vocab_tables()
    if "model_config" in ckpt:
        cfg = ckpt["model_config"]
        print(f"Model config from checkpoint: dim={cfg['dim']}")
        model = LipiMoEEncoder(**cfg).to(device)
    else:
        model = LipiMoEEncoder(
            dim=512,
            num_groups=NUM_GROUPS,
            group_script_vocab_sizes=group_script_vocab_sizes,
            group_script_names=group_script_names,
        ).to(device)

    model_state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(model_state)
    print(f"Loaded checkpoint: {args.resume}")

    # Run benchmarks
    print(f"\n{'='*60}")
    print(f"  Standard OCR Benchmarks (Latin/English)")
    print(f"{'='*60}\n")

    results = {}
    total_correct = 0
    total_samples = 0

    for name in BENCHMARKS:
        samples = load_benchmark(name, args.cache_dir)
        if not samples:
            print(f"  {name}: skipped (no data)")
            continue

        word_acc, char_acc, n = evaluate_benchmark(
            model, samples, device, device_type,
            group_script_vocab_sizes, group_script_names)
        results[name] = (word_acc, char_acc, n)
        total_correct += int(word_acc * n / 100)
        total_samples += n
        print(f"  {name:<10s}  Word: {word_acc:5.1f}%  Char: {char_acc:5.1f}%  ({n} samples)")

    if total_samples > 0:
        avg = 100 * total_correct / total_samples
        print(f"\n  {'Average':<10s}  Word: {avg:5.1f}%  ({total_samples} total samples)")

    # Compare with published results
    print(f"\n{'='*60}")
    print(f"  Reference (published results)")
    print(f"{'='*60}")
    print(f"  {'Model':<20s} {'IIIT5K':>7s} {'SVT':>7s} {'IC13':>7s} {'IC15':>7s} {'SVTP':>7s} {'CUTE':>7s} {'Avg':>7s}")
    print(f"  {'-'*76}")
    print(f"  {'PP-OCRv4':<20s} {'95.0':>7s} {'91.5':>7s} {'95.2':>7s} {'83.5':>7s} {'85.4':>7s} {'87.5':>7s} {'89.7':>7s}")
    print(f"  {'SVTR-Base':<20s} {'96.0':>7s} {'91.5':>7s} {'97.1':>7s} {'85.2':>7s} {'89.9':>7s} {'91.7':>7s} {'91.9':>7s}")
    print(f"  {'PARSeq':<20s} {'97.0':>7s} {'93.6':>7s} {'96.2':>7s} {'86.5':>7s} {'88.9':>7s} {'92.2':>7s} {'92.4':>7s}")
    lipi_avg = f"{avg:.1f}" if total_samples > 0 else "N/A"
    lipi_scores = {name: f"{results[name][0]:.1f}" if name in results else "N/A"
                   for name in BENCHMARKS}
    print(f"  {'Lipi (yours)':<20s} {lipi_scores['IIIT5K']:>7s} {lipi_scores['SVT']:>7s} "
          f"{lipi_scores['IC13']:>7s} {lipi_scores['IC15']:>7s} {lipi_scores['SVTP']:>7s} "
          f"{lipi_scores['CUTE80']:>7s} {lipi_avg:>7s}")


if __name__ == "__main__":
    main()
