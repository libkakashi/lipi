#!/usr/bin/env python3
"""
Experiment: HGNetV2 backbone + LID-1 only.

Tests whether a pretrained backbone gives good script identification
without any attention blocks.

Usage:
    python scripts/experiments/hgnet_lid1.py --data data/all_shards
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.lid import LIDCoarse, NUM_GROUPS, GROUPS, SCRIPT_TO_GROUP
from src.training.dataloader import (
    build_script_tokenizers, collate_moe, LipiStreamingDataset,
)

import timm


class HGNetLID1(nn.Module):
    """HGNetV2 backbone + LID-1 classifier. No attention, no CTC."""

    def __init__(self, dim: int = 256, num_groups: int = NUM_GROUPS):
        super().__init__()
        # Pretrained HGNetV2-B0 backbone
        self.backbone = timm.create_model(
            'hgnetv2_b0.ssld_stage1_in22k_in1k',
            pretrained=True,
            features_only=True,
        )
        # Stage 2 output: 512ch, h=2, w=W/8 for 32px input
        # We'll use stage 2 (h=2) for richer spatial info
        backbone_ch = 512  # stage 2 channels

        # Adapt input from 2ch (L+a) to 3ch (backbone expects RGB)
        self.input_proj = nn.Conv2d(2, 3, kernel_size=1)

        # Project backbone features to dim
        self.proj = nn.Linear(backbone_ch, dim)

        # LID-1 classifier
        self.lid = LIDCoarse(in_channels=dim, num_groups=num_groups)

    def forward(self, images):
        # Dequantize if uint8
        x = images.float() / 255.0 if images.dtype == torch.uint8 else images

        # 2ch → 3ch for pretrained backbone
        x = self.input_proj(x)

        # Run backbone, take stage 2 (h=2)
        feats = self.backbone(x)
        x = feats[2]  # (B, 512, h=2, w=W/8)

        # Pool + project
        B, C, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)  # (B, T, 512)
        x = self.proj(x)  # (B, T, dim)

        # LID-1
        return self.lid.forward_seq(x)  # (B, num_groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone, only train input_proj + proj + LID")
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

    # Data
    data_path = Path(args.data)
    active_groups = list(GROUPS)
    all_scripts = list(SCRIPT_TO_GROUP.keys())

    train_dataset = LipiStreamingDataset(
        local=str(data_path / "train"), active_scripts=all_scripts,
        active_groups=active_groups)
    val_dataset = LipiStreamingDataset(
        local=str(data_path / "val"), active_scripts=all_scripts,
        active_groups=active_groups)

    # Subsample val to 200 per script
    from collections import Counter
    script_counts = Counter()
    val_indices = []
    for i in range(len(val_dataset)):
        sid = int(val_dataset._ds[i]["script_id"])
        if script_counts[sid] < 200:
            val_indices.append(i)
            script_counts[sid] += 1
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_moe,
                              pin_memory=(device_type == "cuda"))
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_moe,
                            pin_memory=(device_type == "cuda"))

    print(f"Train: {len(train_dataset)}, Val: {len(val_subset)}")

    # Model
    model = HGNetLID1(dim=256).to(device)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Backbone frozen. Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"All trainable: {total/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01)
    ce_loss = nn.CrossEntropyLoss()

    use_amp = device_type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device_type == "cuda" else torch.float16

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_idx, (imgs, targets, tgt_lens, gids, sids, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            gids = gids.to(device, non_blocking=True)

            with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                logits = model(imgs)
                loss = ce_loss(logits, gids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(-1)
            correct += (pred == gids).sum().item()
            total += gids.shape[0]

            if (batch_idx + 1) % 50 == 0:
                acc = 100 * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                elapsed = time.time() - t0
                print(f"  [{epoch}/{args.epochs}] {batch_idx+1} "
                      f"loss={avg_loss:.4f} acc={acc:.1f}% "
                      f"({total/elapsed:.0f} img/s)")

            if batch_idx >= 500:
                break

        # Val
        model.eval()
        val_correct = 0
        val_total = 0
        group_correct = [0] * NUM_GROUPS
        group_total = [0] * NUM_GROUPS

        with torch.no_grad():
            for imgs, targets, tgt_lens, gids, sids, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                gids = gids.to(device, non_blocking=True)

                with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                    logits = model(imgs)

                pred = logits.argmax(-1)
                val_correct += (pred == gids).sum().item()
                val_total += gids.shape[0]

                for g in range(NUM_GROUPS):
                    mask = (gids == g)
                    if mask.any():
                        group_total[g] += mask.sum().item()
                        group_correct[g] += (pred[mask] == g).sum().item()

        val_acc = 100 * val_correct / max(val_total, 1)
        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch}: val LID-1 = {val_acc:.1f}% ({elapsed:.0f}s)")
        for g, name in enumerate(active_groups):
            if group_total[g] > 0:
                g_acc = 100 * group_correct[g] / group_total[g]
                print(f"    {name:<20s} {g_acc:5.1f}% ({group_total[g]})")
        print()


if __name__ == "__main__":
    main()
