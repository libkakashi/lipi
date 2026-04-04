#!/usr/bin/env python3
"""
Train only the LID-1 spatial projection layer.

Loads a checkpoint, freezes everything except lid_coarse (spatial_pool +
classifier), and trains with LID-1 loss only. Fast — no CTC, no expert
gradient, just classification.

Usage:
    python scripts/train_lid_pool.py --data data/shards_v3 --resume checkpoints/moe/moe_epoch4.pt --epochs 3
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.moe_encoder import LipiMoEEncoder
from src.model.lid import SCRIPT_TO_GROUP
from src.training.moe_data import (
    load_shards, build_script_tokenizers, encode_labels,
    remap_ids, MoEDataset, collate_moe,
)


def main():
    parser = argparse.ArgumentParser(description="Train LID-1 spatial projection")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--train-swa", action="store_true",
                        help="Also train shared SWA blocks, not just LID-1")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Save updated checkpoint (default: overwrite resume path)")
    # Model dims (must match checkpoint)
    parser.add_argument("--shared-dim", type=int, default=288)
    parser.add_argument("--shared-blocks-4x4", type=int, default=8)
    parser.add_argument("--shared-blocks-4x16", type=int, default=4)
    parser.add_argument("--stage1-dim", type=int, default=288)
    parser.add_argument("--stage1-blocks", type=int, default=12)
    parser.add_argument("--stage2-dim", type=int, default=576)
    parser.add_argument("--stage2-blocks", type=int, default=8)
    parser.add_argument("--head-hidden", type=int, default=384)
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
    print(f"Scripts: {active_scripts}")

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

    print("Building tokenizers...")
    group_tokenizers, group_script_vocab_sizes, group_script_names = build_script_tokenizers(
        active_scripts, active_groups)

    print("Encoding labels...")
    target_tensor, target_len_tensor = encode_labels(
        labels, group_ids, local_script_ids, active_groups, group_tokenizers)

    # Filter empty labels
    valid = target_len_tensor > 0
    n_filtered = (~valid).sum().item()
    if n_filtered > 0:
        keep = valid.nonzero(as_tuple=True)[0]
        images = images[keep]
        group_ids = group_ids[keep]
        local_script_ids = local_script_ids[keep]
        labels = [labels[i] for i in keep.tolist()]

    dataset = MoEDataset(images, target_tensor[valid] if n_filtered else target_tensor,
                         target_len_tensor[valid] if n_filtered else target_len_tensor,
                         group_ids, local_script_ids, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_moe, pin_memory=(device_type == "cuda"))
    print(f"Samples: {len(dataset)}, Batches: {len(loader)}")

    # --- Model ---
    model = LipiMoEEncoder(
        stem_depth=args.stem_depth,
        shared_dim=args.shared_dim,
        shared_blocks_4x4=args.shared_blocks_4x4,
        shared_blocks_4x16=args.shared_blocks_4x16,
        stage1_dim=args.stage1_dim,
        stage1_blocks=args.stage1_blocks,
        stage2_dim=args.stage2_dim,
        stage2_blocks=args.stage2_blocks,
        num_groups=n_groups,
        group_script_vocab_sizes=group_script_vocab_sizes,
        group_script_names=group_script_names,
        head_hidden=args.head_hidden,
    ).to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model_state = ckpt["model"]
    missing, unexpected = [], []
    current = model.state_dict()
    for k in list(model_state.keys()):
        if k in current and model_state[k].shape != current[k].shape:
            del model_state[k]
            missing.append(k)
    result = model.load_state_dict(model_state, strict=False)
    new_keys = [k for k in current if k not in model_state]
    if new_keys:
        print(f"  New layers (randomly initialized):")
        for k in new_keys:
            print(f"    {k}: {current[k].shape}")
    if missing:
        print(f"  Skipped {len(missing)} shape-mismatched layers")
    del ckpt

    # --- Freeze experts, train shared encoder + LID-1 ---
    expert_keys = ("stage1.", "stage2.", "ctc_modules.")
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if args.train_swa:
            # Train everything except expert blocks
            if any(k in name for k in expert_keys):
                param.requires_grad = False
                frozen += param.numel()
            else:
                param.requires_grad = True
                trainable += param.numel()
        else:
            # Train only LID-1
            if "lid_coarse" in name:
                param.requires_grad = True
                trainable += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()

    mode = "shared encoder + LID-1" if args.train_swa else "LID-1 only"
    print(f"\nFrozen: {frozen/1e6:.1f}M params")
    print(f"Trainable: {trainable/1e6:.1f}M params ({mode})")

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

            # Only forward through shared encoder + LID-1 (experts are frozen)
            with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                # Run shared encoder
                x = model.color_proj(imgs)
                x = model.stem(x)
                _, C, h, w = x.shape
                B = x.shape[0]
                x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
                x = model.proj_shared(x)
                for block in model.shared_swa_4x4:
                    x = block(x, h=h, w=w)
                for block in model.shared_swa_4x16:
                    x = block(x, h=h, w=w)
                # LID-1 classification
                group_logits = model.lid_coarse.forward_seq(x)

            loss = ce_loss_fn(group_logits, gids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            pred = group_logits.argmax(-1)
            total_correct += (pred == gids).sum().item()
            total_samples += B
            total_loss += loss.item()
            n_batches += 1

            if n_batches % args.log_interval == 0:
                acc = 100 * total_correct / total_samples
                avg_loss = total_loss / n_batches
                print(f"  [{epoch}/{args.epochs}] batch {batch_idx+1}/{len(loader)}  "
                      f"lid1_loss={avg_loss:.4f}  lid1_acc={acc:.1f}%")

        elapsed = time.time() - t0
        acc = 100 * total_correct / total_samples
        avg_loss = total_loss / n_batches
        print(f"\nEpoch {epoch}: lid1_loss={avg_loss:.4f}  lid1_acc={acc:.1f}%  time={elapsed:.0f}s")

        # Save after every epoch
        save_path = args.save_path or args.resume
        ckpt_path = save_path.replace(".pt", f"_lid_ep{epoch}.pt") if args.save_path else str(
            Path(save_path).parent / f"{Path(save_path).stem}_lid_ep{epoch}.pt")
        full_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            "model": full_state,
            "epoch": 0,
            "args": vars(args),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
