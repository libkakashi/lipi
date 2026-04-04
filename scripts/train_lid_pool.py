#!/usr/bin/env python3
"""
Train shared encoder + LID-1 in isolation. No experts loaded.

Loads only the shared weights from a checkpoint, trains with LID-1
loss only. Saves updated shared weights that can be merged back
into full training via --resume.

Usage:
    python scripts/train_lid_pool.py --data data/shards_v3 --resume ckpt.pt --epochs 3 --train-swa
    python scripts/train_lid_pool.py --data data/shards_v3 --resume ckpt.pt --epochs 3  # LID-1 only
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_util
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import SCRIPT_TO_GROUP, LIDCoarse
from src.model.stem import ResNetStem
from src.model.attention import SWABlock
from src.data.color import ColorProjection
from src.training.moe_data import (
    load_shards, build_script_tokenizers, encode_labels,
    remap_ids, MoEDataset, collate_moe,
)


class SharedEncoder(nn.Module):
    """Lightweight shared encoder — only the parts needed for LID-1."""

    def __init__(self, stem_depth, shared_dim, shared_blocks_4x4,
                 shared_blocks_4x16, num_groups, stem_channels=64):
        super().__init__()
        self.color_proj = ColorProjection()
        self.stem = ResNetStem(out_channels=stem_channels, depth=stem_depth)
        self.proj_shared = nn.Linear(stem_channels, shared_dim)

        self.shared_swa = nn.ModuleList()
        for i in range(shared_blocks_4x4):
            self.shared_swa.append(
                SWABlock(dim=shared_dim, num_heads=shared_dim // 32,
                         window_h=4, window_w=4,
                         shift=(i % 2 == 1), mlp_ratio=4))
        for i in range(shared_blocks_4x16):
            self.shared_swa.append(
                SWABlock(dim=shared_dim, num_heads=shared_dim // 32,
                         window_h=4, window_w=16,
                         shift=(i % 2 == 1), mlp_ratio=4))

        self.lid_coarse = LIDCoarse(in_channels=shared_dim, num_groups=num_groups)

    def forward(self, images):
        x = self.color_proj(images)
        x = self.stem(x)
        _, C, h, w = x.shape
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
        x = self.proj_shared(x)
        for block in self.shared_swa:
            if self.training and torch.is_grad_enabled():
                x = ckpt_util.checkpoint(block, x, h, w, use_reentrant=False)
            else:
                x = block(x, h=h, w=w)
        return self.lid_coarse.forward_seq(x)


def main():
    parser = argparse.ArgumentParser(description="Train shared encoder + LID-1")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--train-swa", action="store_true",
                        help="Train shared SWA + stem, not just LID-1")
    parser.add_argument("--save-path", type=str, default=None)
    # Model dims
    parser.add_argument("--shared-dim", type=int, default=384)
    parser.add_argument("--shared-blocks-4x4", type=int, default=4)
    parser.add_argument("--shared-blocks-4x16", type=int, default=4)
    parser.add_argument("--stem-depth", type=int, default=3)
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    device_type = device.type
    print(f"Device: {device}")

    # --- Data ---
    data_path = Path(args.data)
    print(f"\nLoading data from {data_path}/...")
    images, labels, script_ids_global, group_ids_global, meta = load_shards(data_path)
    active_scripts = meta["active_scripts"]

    active_groups = []
    seen = set()
    for s in active_scripts:
        g = SCRIPT_TO_GROUP.get(s)
        if g and g not in seen:
            active_groups.append(g)
            seen.add(g)
    n_groups = len(active_groups)
    print(f"Groups: {n_groups} -> {active_groups}")

    group_ids, local_script_ids, _ = remap_ids(
        active_scripts, active_groups, script_ids_global, group_ids_global)

    group_tokenizers, _, _ = build_script_tokenizers(active_scripts, active_groups)

    target_tensor, target_len_tensor = encode_labels(
        labels, group_ids, local_script_ids, active_groups, group_tokenizers)

    valid = target_len_tensor > 0
    n_filtered = (~valid).sum().item()
    if n_filtered > 0:
        keep = valid.nonzero(as_tuple=True)[0]
        images = images[keep]
        group_ids = group_ids[keep]
        local_script_ids = local_script_ids[keep]
        labels = [labels[i] for i in keep.tolist()]
        target_tensor = target_tensor[keep]
        target_len_tensor = target_len_tensor[keep]

    dataset = MoEDataset(images, target_tensor, target_len_tensor,
                         group_ids, local_script_ids, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_moe, pin_memory=(device_type == "cuda"))
    print(f"Samples: {len(dataset)}, Batches: {len(loader)}")

    # --- Shared encoder only ---
    model = SharedEncoder(
        stem_depth=args.stem_depth,
        shared_dim=args.shared_dim,
        shared_blocks_4x4=args.shared_blocks_4x4,
        shared_blocks_4x16=args.shared_blocks_4x16,
        num_groups=n_groups,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Shared encoder: {params/1e6:.1f}M params")

    # Load shared weights from checkpoint
    print(f"\nLoading shared weights from {args.resume}...")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    ckpt_state = ckpt["model"]
    del ckpt

    # Match keys: checkpoint has full model keys, we only have shared
    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    new_keys = []
    for k in model_state:
        if k in ckpt_state and ckpt_state[k].shape == model_state[k].shape:
            model_state[k] = ckpt_state[k]
            loaded += 1
        else:
            new_keys.append(k)
            skipped += 1
    model.load_state_dict(model_state)
    del ckpt_state

    print(f"  Loaded: {loaded} tensors, New (random init): {skipped}")
    if new_keys:
        for k in new_keys[:10]:
            print(f"    {k}")

    # Freeze/unfreeze
    if args.train_swa:
        mode = "shared encoder + LID-1"
    else:
        for name, param in model.named_parameters():
            if "lid_coarse" not in name:
                param.requires_grad = False
        mode = "LID-1 only"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable/1e6:.1f}M ({mode}), Frozen: {frozen/1e6:.1f}M")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    ce_loss_fn = nn.CrossEntropyLoss()

    use_amp = device_type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"TRAINING {mode.upper()}: {args.epochs} epochs, lr={args.lr}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        n_batches = 0

        for batch_idx, (imgs, targets, tgt_lens, gids, sids, _labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            gids = gids.to(device, non_blocking=True)

            with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                group_logits = model(imgs)

            loss = ce_loss_fn(group_logits, gids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pred = group_logits.argmax(-1)
            total_correct += (pred == gids).sum().item()
            total_samples += imgs.shape[0]
            total_loss += loss.item()
            n_batches += 1

            if n_batches % args.log_interval == 0:
                acc = 100 * total_correct / total_samples
                avg_loss = total_loss / n_batches
                print(f"  [{epoch}/{args.epochs}] batch {batch_idx+1}/{len(loader)}  "
                      f"lid1_loss={avg_loss:.4f}  lid1_acc={acc:.2f}%")

        elapsed = time.time() - t0
        acc = 100 * total_correct / total_samples
        avg_loss = total_loss / n_batches
        print(f"\nEpoch {epoch}: lid1_loss={avg_loss:.4f}  lid1_acc={acc:.2f}%  time={elapsed:.0f}s")

        # Save — merge trained weights back into full checkpoint format
        save_path = args.save_path or args.resume
        ckpt_path = str(Path(save_path).parent /
                        f"{Path(save_path).stem}_lid_ep{epoch}.pt")

        # Reload full checkpoint, update shared weights, save
        full_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        full_state = full_ckpt["model"]
        for k, v in model.state_dict().items():
            if k in full_state:
                full_state[k] = v.cpu()
        torch.save({
            "model": full_state,
            "epoch": 0,
            "args": full_ckpt.get("args", vars(args)),
        }, ckpt_path)
        del full_ckpt, full_state
        print(f"  Saved: {ckpt_path}")

    print(f"\nDone. Resume full training with --resume {ckpt_path}")


if __name__ == "__main__":
    main()
