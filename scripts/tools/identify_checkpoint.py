#!/usr/bin/env python3
"""
Identify the architecture format of .pt checkpoint files in a directory.

Classifies each checkpoint by its state_dict key signatures:
  - pre-LID-0: has shared_b.*, no super_b.*, no lid0_head.*, no lid1_attn.*
  - LID-0 era: has super_b.{0..4}.* and/or lid0_head.*
  - post-rollback: has shared_b.* and lid1_attn.* (no super_b / lid0_head)
  - unknown: neither shared_b nor super_b (very old or non-MoE)

Also reports the num_groups and training epoch stored in the checkpoint.

Usage:
    python scripts/tools/identify_checkpoint.py checkpoints/moe
    python scripts/tools/identify_checkpoint.py checkpoints/ --recursive
"""

import argparse
import re
from pathlib import Path

import torch


def classify(path: Path) -> tuple[str, int | None, int | str]:
    """Return (label, epoch, num_groups)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return f"ERROR: {e.__class__.__name__}", None, "?"

    state = ckpt.get("model", ckpt)
    keys = list(state.keys()) if hasattr(state, "keys") else []

    has_shared_b = any(k.startswith("shared_b.") for k in keys)
    has_super_b = any(k.startswith("super_b.") for k in keys)
    has_lid0_head = any(k.startswith("lid0_head.") for k in keys)
    has_lid1_attn = any(k.startswith("lid1_attn.") for k in keys)
    has_group_experts = any(k.startswith("group_local_blocks.") for k in keys)

    # Try to extract num_groups. Prefer model_config; fall back to counting
    # group experts.
    cfg = ckpt.get("model_config", {}) or {}
    num_groups = cfg.get("num_groups")
    if num_groups is None and has_group_experts:
        gids = set()
        for k in keys:
            m = re.match(r"group_local_blocks\.\d+\.expert_attns\.(\d+)\.", k)
            if m:
                gids.add(int(m.group(1)))
        num_groups = max(gids) + 1 if gids else None

    epoch = ckpt.get("epoch")
    if epoch is None:
        # Try to derive from filename (e.g. moe_epoch20.pt)
        m = re.search(r"epoch(\d+)", path.name)
        if m:
            epoch = int(m.group(1))

    if has_super_b or has_lid0_head:
        label = "LID-0 era"
    elif has_lid1_attn:
        label = "post-rollback"
    elif has_shared_b:
        label = "pre-LID-0"
    elif has_group_experts:
        label = "very old (no shared_b)"
    else:
        label = "unknown"

    return label, epoch, (num_groups if num_groups is not None else "?")


def main():
    parser = argparse.ArgumentParser(
        description="Classify .pt checkpoints by architecture format")
    parser.add_argument("dir", help="Directory to scan")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recurse into subdirectories")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    assert root.is_dir(), f"Not a directory: {root}"
    pattern = "**/*.pt" if args.recursive else "*.pt"
    files = sorted(root.glob(pattern))
    if not files:
        print(f"No .pt files found in {root}")
        return

    print(f"Scanning {len(files)} .pt file(s) in {root}\n")

    rows = []
    for f in files:
        rel = f.relative_to(root)
        size_mb = f.stat().st_size / 1e6
        label, epoch, ng = classify(f)
        rows.append((rel, size_mb, label, epoch, ng))

    name_w = max(len(str(r[0])) for r in rows)
    label_w = max(len(r[2]) for r in rows)
    print(f"  {'name':<{name_w}s}  {'size':>8s}  "
          f"{'format':<{label_w}s}  epoch  groups")
    print(f"  {'─' * (name_w + 8 + label_w + 20)}")
    for rel, size_mb, label, epoch, ng in rows:
        epoch_str = str(epoch) if epoch is not None else "?"
        print(f"  {str(rel):<{name_w}s}  {size_mb:>6.1f}MB  "
              f"{label:<{label_w}s}  {epoch_str:>5s}  {ng}")

    pre = [r for r in rows if r[2] == "pre-LID-0"]
    if not pre:
        print("\nNo pre-LID-0 checkpoints found.")
        return

    # Pick the one with the highest epoch (most trained before the LID-0 switch)
    pre_sorted = sorted(
        pre, key=lambda r: (r[3] if isinstance(r[3], int) else -1))
    latest = pre_sorted[-1]
    print()
    print(f"Most recent pre-LID-0 checkpoint: {latest[0]}")
    print(f"  epoch={latest[3]}, num_groups={latest[4]}, size={latest[1]:.1f}MB")
    print()
    print("To use it as the starting point for the rolled-back architecture:")
    print(f"  python scripts/migrate_checkpoint.py \\")
    print(f"    --input {root / latest[0]} \\")
    print(f"    --output {root / (latest[0].stem + '_migrated.pt')}")


if __name__ == "__main__":
    main()
