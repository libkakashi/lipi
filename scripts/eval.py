#!/usr/bin/env python3
"""
Evaluate a Lipi MoE checkpoint on validation data.

Usage:
    python scripts/eval.py --data data/all_shards --resume checkpoints/moe/moe_epoch2.pt
    python scripts/eval.py --data data/all_shards --resume checkpoints/moe/moe_epoch2.pt --max-batches 100
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiMoEEncoder
from src.model.lid import SCRIPT_TO_GROUP, NUM_GROUPS, GROUPS
from src.training.dataloader import (
    build_script_tokenizers, collate_moe, LipiStreamingDataset,
)
from src.training.eval import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate Lipi MoE checkpoint")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu")
    else:
        device = torch.device(args.device)
    device_type = device.type
    print(f"Device: {device}")

    # Build tokenizers and vocab tables
    active_groups = list(GROUPS)
    all_scripts = list(SCRIPT_TO_GROUP.keys())
    n_groups = NUM_GROUPS

    group_tokenizers, group_script_vocab_sizes, group_script_names = build_script_tokenizers(
        all_scripts, active_groups)

    # Load validation data
    data_path = Path(args.data)
    val_dir = str(data_path / "val")
    val_dataset = LipiStreamingDataset(
        local=val_dir, active_scripts=all_scripts,
        active_groups=active_groups)
    # Subsample to 500 per script for balanced eval
    from collections import Counter
    max_per_script = 500
    script_counts = Counter()
    val_indices = []
    for i in range(len(val_dataset)):
        sid = int(val_dataset._ds[i]["script_id"])
        if script_counts[sid] < max_per_script:
            val_indices.append(i)
            script_counts[sid] += 1
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_moe,
                            pin_memory=(device_type == "cuda"))
    print(f"Val samples: {len(val_subset)} ({max_per_script}/script from {len(val_dataset)} total)")

    # Build model
    model = LipiMoEEncoder(
        dim=args.dim,
        num_groups=n_groups,
        group_script_vocab_sizes=group_script_vocab_sizes,
        group_script_names=group_script_names,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model_state = ckpt["model"] if "model" in ckpt else ckpt

    # Remap legacy keys
    n_4x4 = len(set(k.split(".")[1] for k in model_state if k.startswith("shared_swa_4x4.")))
    for k in list(model_state.keys()):
        if k.startswith("shared_swa_4x4."):
            new_k = k.replace("shared_swa_4x4.", "shared_swa.", 1)
            model_state[new_k] = model_state.pop(k)
        elif k.startswith("shared_swa_4x16."):
            idx = int(k.split(".")[1])
            rest = ".".join(k.split(".")[2:])
            model_state[f"shared_swa.{idx + n_4x4}.{rest}"] = model_state.pop(k)
        elif k.startswith(("proj_shared.", "proj_stem.", "color_proj.")):
            model_state.pop(k)

    # Load with shape matching
    current = model.state_dict()
    loaded = 0
    skipped = []
    for k in current:
        if k in model_state and model_state[k].shape == current[k].shape:
            current[k] = model_state[k]
            loaded += 1
        elif k in model_state:
            skipped.append(f"{k} (shape {model_state[k].shape} vs {current[k].shape})")
    model.load_state_dict(current)
    print(f"Loaded {loaded}/{len(current)} tensors from {args.resume}")
    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped[:5]}")

    # AMP
    use_amp = device_type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device_type == "cuda" else torch.float16

    # Evaluate
    print(f"\n  Eval:")
    evaluate(model, val_loader, group_tokenizers, group_script_names,
             active_groups, device, device_type, use_amp, amp_dtype,
             group_script_vocab_sizes=group_script_vocab_sizes)


if __name__ == "__main__":
    main()
