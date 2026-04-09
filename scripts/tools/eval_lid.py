#!/usr/bin/env python3
"""
Evaluate LID-1 accuracy with per-group breakdown and confusion matrix.

Usage:
    python scripts/eval_lid.py --data data/shards_v3 --resume checkpoints/moe/moe_epoch2.pt
    python scripts/eval_lid.py --data data/shards_v3 --resume ckpt.pt --shared-dim 256
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPT_TO_GROUP, LIDCoarse
from src.model.stem import ResNetStem
from src.model.attention import SWABlock
from src.training.dataloader import (
    load_shards, build_script_tokenizers, encode_labels,
    remap_ids, MoEDataset, collate_moe,
)


class SharedEncoder(torch.nn.Module):
    def __init__(self, dim, num_groups):
        super().__init__()
        shared_dim = dim // 2
        self.stem = ResNetStem(out_channels=shared_dim)
        self.shared_swa = torch.nn.ModuleList()
        for i in range(4):
            self.shared_swa.append(
                SWABlock(dim=shared_dim, num_heads=shared_dim // 32,
                         window_h=8, window_w=8,
                         shift=(i % 2 == 1), mlp_ratio=4))
        for i in range(2):
            self.shared_swa.append(
                SWABlock(dim=shared_dim, num_heads=shared_dim // 32,
                         window_h=8, window_w=32,
                         shift=(i % 2 == 1), mlp_ratio=4))
        self.lid_coarse = LIDCoarse(in_channels=shared_dim, num_groups=num_groups)

    def forward(self, images):
        x = images.float() / 255.0 if images.dtype == torch.uint8 else images
        x = self.stem(x)
        _, C, h, w = x.shape
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        for block in self.shared_swa:
            x = block(x, h=h, w=w)
        return self.lid_coarse.forward_seq(x)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LID-1")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit eval batches (0 = all)")
    parser.add_argument("--dim", type=int, default=256)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    device_type = device.type

    # Data
    data_path = Path(args.data)
    images, labels, script_ids_global, group_ids_global, meta, _, _ = load_shards(data_path)
    active_scripts = meta["active_scripts"]

    active_groups = []
    seen = set()
    for s in active_scripts:
        g = SCRIPT_TO_GROUP.get(s)
        if g and g not in seen:
            active_groups.append(g)
            seen.add(g)
    n_groups = len(active_groups)

    group_ids, local_script_ids, _ = remap_ids(
        active_scripts, active_groups, script_ids_global, group_ids_global)
    group_tokenizers, _, _ = build_script_tokenizers(active_scripts, active_groups)
    target_tensor, target_len_tensor = encode_labels(
        labels, group_ids, local_script_ids, active_groups, group_tokenizers)

    valid = target_len_tensor > 0
    if (~valid).sum().item() > 0:
        keep = valid.nonzero(as_tuple=True)[0]
        images = images[keep]
        group_ids = group_ids[keep]
        local_script_ids = local_script_ids[keep]
        labels = [labels[i] for i in keep.tolist()]
        target_tensor = target_tensor[keep]
        target_len_tensor = target_len_tensor[keep]

    dataset = MoEDataset(images, target_tensor, target_len_tensor,
                         group_ids, local_script_ids, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_moe, pin_memory=(device_type == "cuda"))

    # Model
    model = SharedEncoder(
        dim=args.dim,
        num_groups=n_groups,
    ).to(device)

    # Load weights
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model_state = model.state_dict()
    ckpt_state = ckpt["model"]
    # Remap legacy keys from old architecture
    n_4x4 = len(set(k.split(".")[1] for k in ckpt_state if k.startswith("shared_swa_4x4.")))
    for k in list(ckpt_state.keys()):
        if k.startswith("shared_swa_4x4."):
            new_k = k.replace("shared_swa_4x4.", "shared_swa.", 1)
            ckpt_state[new_k] = ckpt_state.pop(k)
        elif k.startswith("shared_swa_4x16."):
            idx = int(k.split(".")[1])
            rest = ".".join(k.split(".")[2:])
            ckpt_state[f"shared_swa.{idx + n_4x4}.{rest}"] = ckpt_state.pop(k)
        elif k.startswith(("proj_shared.", "proj_stem.")):
            # proj_stem removed — stem outputs shared_dim directly
            ckpt_state.pop(k)
    loaded = 0
    for k in model_state:
        if k in ckpt_state and ckpt_state[k].shape == model_state[k].shape:
            model_state[k] = ckpt_state[k]
            loaded += 1
    model.load_state_dict(model_state)
    del ckpt, ckpt_state
    print(f"Loaded {loaded}/{len(model_state)} tensors from {args.resume}")

    # Eval
    model.eval()
    use_amp = device_type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # Per-group stats
    group_correct = Counter()
    group_total = Counter()
    confusion = {}  # (true, pred) → count

    with torch.no_grad():
        for batch_idx, (imgs, targets, tgt_lens, gids, sids, _labels) in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            gids = gids.to(device, non_blocking=True)

            with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                logits = model(imgs)
            preds = logits.argmax(-1)

            for true, pred in zip(gids.cpu().tolist(), preds.cpu().tolist()):
                group_total[true] += 1
                if true == pred:
                    group_correct[true] += 1
                confusion[(true, pred)] = confusion.get((true, pred), 0) + 1

    # Results
    total_correct = sum(group_correct.values())
    total_samples = sum(group_total.values())
    overall_acc = 100 * total_correct / max(total_samples, 1)

    print(f"\n{'='*60}")
    print(f"LID-1 Evaluation: {overall_acc:.2f}% ({total_correct}/{total_samples})")
    print(f"{'='*60}")

    print(f"\n{'Group':<20s} {'Accuracy':>10s} {'Correct':>10s} {'Total':>10s}")
    print("-" * 52)
    for g in range(n_groups):
        acc = 100 * group_correct[g] / max(group_total[g], 1)
        name = active_groups[g] if g < len(active_groups) else f"group{g}"
        print(f"{name:<20s} {acc:>9.2f}% {group_correct[g]:>10d} {group_total[g]:>10d}")

    # Confusion matrix
    print(f"\n{'='*60}")
    print("Confusion Matrix (rows=true, cols=predicted)")
    print(f"{'='*60}\n")

    # Header
    short_names = [g[:6] for g in active_groups]
    header = f"{'':>12s} " + " ".join(f"{s:>6s}" for s in short_names)
    print(header)
    print("-" * len(header))

    for g_true in range(n_groups):
        row = f"{active_groups[g_true][:12]:>12s} "
        for g_pred in range(n_groups):
            count = confusion.get((g_true, g_pred), 0)
            if count == 0:
                row += f"{'·':>6s} "
            elif g_true == g_pred:
                row += f"{count:>6d} "
            else:
                row += f"{count:>6d} "
        # Show accuracy at end
        acc = 100 * group_correct[g_true] / max(group_total[g_true], 1)
        row += f"  {acc:.1f}%"
        print(row)

    # Show worst confusions
    print(f"\nTop misclassifications:")
    misclass = [(k, v) for k, v in confusion.items() if k[0] != k[1]]
    misclass.sort(key=lambda x: -x[1])
    for (true, pred), count in misclass[:10]:
        true_name = active_groups[true] if true < len(active_groups) else f"g{true}"
        pred_name = active_groups[pred] if pred < len(active_groups) else f"g{pred}"
        pct = 100 * count / group_total[true]
        print(f"  {true_name} → {pred_name}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
