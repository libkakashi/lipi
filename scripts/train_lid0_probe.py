#!/usr/bin/env python3
"""LID-0 super-group probe.

Builds a mini model = ConvStem + shared_a + shared_b + (pool h=4->1) + LID-0 head,
trains it on an existing MDS dataset (generated via scripts/generate.py),
and reports per-super-group accuracy.

The probe answers: can a classifier on top of shared_b features distinguish
between visual super-groups (alphabetic vs CJK vs brahmic etc.)? If yes,
the precondition for super-group MoE routing at shared_c is satisfied.

Run on MPS (Apple Silicon) by default.

Usage:
    python scripts/generate.py --samples-per-script 400 --out data/probe_shards
    python scripts/train_lid0_probe.py --data data/probe_shards --steps 1500
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import ConvStem, SWABlock, _patch_merge_h
from src.taxonomy import GROUPS, NUM_GROUPS, SCRIPT_TO_GROUP
from src.training.dataloader import LipiStreamingDataset


# ---------------------------------------------------------------------------
# Super-group mapping (6 super-groups by visual family)
# ---------------------------------------------------------------------------

SUPER_GROUPS = [
    "alphabetic",   # latin, cyrillic_greek, caucasus
    "abjad",        # arabic, hebrew
    "cjk",          # han, kana, korean
    "brahmic",      # ne_indic, dravidian_north, dravidian_south, se_asian
    "other",        # ethiopic, tibetan
    "emoji",        # emoji
]
NUM_SUPER_GROUPS = len(SUPER_GROUPS)
BLANK_SUPER = NUM_SUPER_GROUPS  # whitespace / padding class id (not a real class)

GROUP_NAME_TO_SUPER = {
    "latin":           0,
    "cyrillic_greek":  0,
    "caucasus":        0,
    "arabic":          1,
    "hebrew":          1,
    "han":             2,
    "kana":            2,
    "korean":          2,
    "ne_indic":        3,
    "dravidian_north": 3,
    "dravidian_south": 3,
    "se_asian":        3,
    "ethiopic":        4,
    "tibetan":         4,
    "emoji":           5,
}
assert set(GROUP_NAME_TO_SUPER) == set(GROUPS), \
    f"mismatch: {set(GROUP_NAME_TO_SUPER) ^ set(GROUPS)}"

# Build local_group_id -> super_group_id (GROUPS list defines the order).
GROUP_ID_TO_SUPER = torch.tensor(
    [GROUP_NAME_TO_SUPER[g] for g in GROUPS] + [BLANK_SUPER],
    dtype=torch.long,
)  # shape (NUM_GROUPS + 1,)


def frame_groups_to_supers(
    group_labels: torch.Tensor, pad_ignore: int = -100
) -> torch.Tensor:
    """Map per-frame group IDs -> per-frame super-group IDs.

    Frames with PAD_IGNORE (-100) stay as -100. Blank frames (NUM_GROUPS)
    map to BLANK_SUPER so CrossEntropy with ignore_index=BLANK_SUPER skips
    them. Real group IDs map through GROUP_ID_TO_SUPER.
    """
    out = torch.full_like(group_labels, pad_ignore)
    valid = group_labels != pad_ignore
    out[valid] = GROUP_ID_TO_SUPER.to(group_labels.device)[group_labels[valid]]
    return out


# ---------------------------------------------------------------------------
# Probe model: stem + shared_a + shared_b + pool + lid0_attn + classifier
# ---------------------------------------------------------------------------

class LID0Probe(nn.Module):
    def __init__(self, num_super_groups: int = NUM_SUPER_GROUPS,
                 dim: int = 128, mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.stem = ConvStem(in_ch=3, out_ch=dim)
        self.shared_a = nn.ModuleList([
            SWABlock(dim=dim, num_heads=max(dim // 64, 1),
                     window_h=8, window_w=16, shift=(i % 2 == 1),
                     mlp_ratio=mlp_ratio, drop_path=0.0,
                     layer_scale_init=1.0)
            for i in range(2)
        ])
        self.merge_a = nn.Linear(dim * 2, dim)
        self.shared_b = nn.ModuleList([
            SWABlock(dim=dim, num_heads=max(dim // 64, 1),
                     window_h=4, window_w=32, shift=(i % 2 == 1),
                     mlp_ratio=mlp_ratio, drop_path=0.0,
                     layer_scale_init=1.0)
            for i in range(2)
        ])
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.lid0_attn = SWABlock(
            dim=dim, num_heads=max(dim // 64, 1),
            window_h=1, window_w=32, shift=False,
            mlp_ratio=mlp_ratio, drop_path=0.0, layer_scale_init=1.0,
        )
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_super_groups + 1),  # +1 for blank class
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        x = images.float() / 255.0 if images.dtype == torch.uint8 else images
        x = self.stem(x)
        _, C, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        for blk in self.shared_a:
            x = blk(x, h, w)
        x, h = _patch_merge_h(x, h, w, self.merge_a)
        for blk in self.shared_b:
            x = blk(x, h, w)
        d = x.shape[-1]
        x = x.reshape(B, h, w, d).permute(0, 3, 1, 2)
        x = self.h_pool(x).squeeze(2).permute(0, 2, 1)  # (B, w, d)
        x = self.lid0_attn(x, 1, w)
        return self.head(x)  # (B, w, num_super_groups + 1)


# ---------------------------------------------------------------------------
# Collate — simplified from the full MoE collate; we only need imgs + glabels
# ---------------------------------------------------------------------------

PAD_IGNORE = -100


FIXED_WIDTH = 1024  # All batches padded/truncated here so MPS compiles once.


def collate_probe(batch):
    """Pad/truncate images + group_labels to FIXED_WIDTH. Drop everything else."""
    imgs, _tids, _tlens, _gids, _sids, _labels, glabels, _segs = zip(*batch)
    max_w = FIXED_WIDTH
    padded_imgs = []
    for img in imgs:
        if img.shape[2] < max_w:
            pad = torch.zeros(img.shape[0], img.shape[1], max_w - img.shape[2],
                              dtype=img.dtype)
            img = torch.cat([img, pad], dim=2)
        elif img.shape[2] > max_w:
            img = img[:, :, :max_w]
        padded_imgs.append(img)
    padded_gl = []
    # Per-frame labels are at W resolution; stem downsamples to W/2 later,
    # but the dataset stores them at W. We'll handle downsampling in the
    # training loop (simple slicing).
    for gl in glabels:
        if gl.shape[0] < max_w:
            pad = torch.full((max_w - gl.shape[0],), PAD_IGNORE,
                             dtype=gl.dtype)
            gl = torch.cat([gl, pad])
        else:
            gl = gl[:max_w]
        padded_gl.append(gl)
    return torch.stack(padded_imgs), torch.stack(padded_gl)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to MDS shard dir (with train/ and val/)")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--max-batch-size", type=int, default=64,
                        help="Batch size. Every batch is padded to FIXED_WIDTH=1024.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="MosaicML streaming + MPS hangs with workers > 0.")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    data_path = Path(args.data)
    all_scripts = list(SCRIPT_TO_GROUP.keys())
    active_groups = list(GROUPS)

    print(f"\nLoading MDS datasets from {data_path}...")
    train_ds = LipiStreamingDataset(
        local=str(data_path / "train"),
        active_scripts=all_scripts,
        active_groups=active_groups,
    )
    val_ds = LipiStreamingDataset(
        local=str(data_path / "val"),
        active_scripts=all_scripts,
        active_groups=active_groups,
    )
    print(f"  Train: {len(train_ds)},  Val: {len(val_ds)}")

    # Fixed width padding means every batch has identical shape — no need
    # for width-sorted budgeting. Simple shuffled DataLoader is fine.
    train_loader = DataLoader(
        train_ds, batch_size=args.max_batch_size, shuffle=True,
        collate_fn=collate_probe, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        val_ds, batch_size=args.max_batch_size, shuffle=False,
        collate_fn=collate_probe, num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0))

    print("\nBuilding model...")
    model = LID0Probe().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Probe params: {n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\nTraining for {args.steps} steps (max_batch_size={args.max_batch_size})...")
    t0 = time.time()
    train_iter = iter(train_loader)
    running_loss = 0.0
    running_acc = 0.0
    n_acc = 0
    for step in range(1, args.steps + 1):
        try:
            x, glabels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, glabels = next(train_iter)

        # group_labels (per-frame @ W) -> super-group labels @ W.
        # Stem downsamples to W/2 and we predict at W/2, so take every
        # 2nd label (rounded to match stem stride).
        supers_w = frame_groups_to_supers(glabels)  # (B, W)
        supers_t = supers_w[:, ::2]                 # (B, W/2)

        x = x.to(device)
        supers_t = supers_t.to(device)

        model.train()
        logits = model(x)  # (B, T, num_super_groups+1)
        T = min(logits.shape[1], supers_t.shape[1])
        logits = logits[:, :T]
        target = supers_t[:, :T]
        loss = F.cross_entropy(
            logits.reshape(-1, NUM_SUPER_GROUPS + 1),
            target.reshape(-1),
            ignore_index=PAD_IGNORE,
        )

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        with torch.no_grad():
            # Frame-level accuracy over non-blank, non-pad frames
            pred = logits.argmax(dim=-1)
            valid = (target != PAD_IGNORE) & (target != BLANK_SUPER)
            if valid.any():
                acc = (pred[valid] == target[valid]).float().mean().item()
                running_acc += acc
                n_acc += 1
        running_loss += loss.item()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            avg_loss = running_loss / args.log_interval
            avg_acc = running_acc / max(n_acc, 1)
            running_loss = 0.0
            running_acc = 0.0
            n_acc = 0
            print(f"  step {step:5d}/{args.steps}  loss={avg_loss:.4f}  "
                  f"acc={avg_acc:.3f}  ({steps_per_sec:.2f} steps/s)")

        if step % args.eval_interval == 0 or step == args.steps:
            evaluate(model, val_loader, device,
                     full=(step == args.steps))


@torch.no_grad()
def evaluate(model, val_loader, device, full: bool = False):
    model.eval()
    cm = torch.zeros(NUM_SUPER_GROUPS, NUM_SUPER_GROUPS, dtype=torch.long)
    sample_correct = 0
    sample_total = 0
    frame_correct = 0
    frame_total = 0
    for x, glabels in val_loader:
        supers_w = frame_groups_to_supers(glabels)
        supers_t = supers_w[:, ::2].to(device)
        x = x.to(device)
        logits = model(x)
        T = min(logits.shape[1], supers_t.shape[1])
        logits = logits[:, :T]
        target = supers_t[:, :T]

        valid = (target != PAD_IGNORE) & (target != BLANK_SUPER)
        pred = logits.argmax(dim=-1)

        # Frame-level
        if valid.any():
            frame_correct += (pred[valid] == target[valid]).sum().item()
            frame_total += int(valid.sum().item())

        # Sample-level: majority prediction among valid frames per sample
        B = pred.shape[0]
        for b in range(B):
            mask = valid[b]
            if not mask.any():
                continue
            # Target is the mode (should be uniform in monolingual samples
            # but mixed samples may not be; take mode either way).
            t_vals = target[b][mask]
            p_vals = pred[b][mask]
            t_mode = int(torch.mode(t_vals).values.item())
            p_mode = int(torch.mode(p_vals).values.item())
            if 0 <= t_mode < NUM_SUPER_GROUPS and 0 <= p_mode < NUM_SUPER_GROUPS:
                cm[t_mode, p_mode] += 1
                sample_total += 1
                if t_mode == p_mode:
                    sample_correct += 1

    frame_acc = frame_correct / max(frame_total, 1)
    sample_acc = sample_correct / max(sample_total, 1)
    print(f"  [val] frame_acc={frame_acc:.4f}  sample_acc={sample_acc:.4f}  "
          f"(n={sample_total})")

    if full:
        print("\n  Per-super-group sample accuracy:")
        for sg in range(NUM_SUPER_GROUPS):
            n = cm[sg].sum().item()
            if n == 0:
                continue
            acc = cm[sg, sg].item() / n
            print(f"    {sg} {SUPER_GROUPS[sg]:<12s} n={n:4d}  acc={acc:.3f}")

        print("\n  Confusion matrix (rows=true, cols=pred):")
        header = "              " + "  ".join(
            f"{n[:6]:>6s}" for n in SUPER_GROUPS)
        print(header)
        for sg in range(NUM_SUPER_GROUPS):
            row = "  ".join(f"{cm[sg, p].item():>6d}"
                            for p in range(NUM_SUPER_GROUPS))
            print(f"    {SUPER_GROUPS[sg][:12]:<12s} {row}")


if __name__ == "__main__":
    main()
