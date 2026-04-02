#!/usr/bin/env python3
"""
Train Lipi MoE Encoder end-to-end.

Usage:
    python scripts/train_moe.py --data data/shards --epochs 15 --batch-size 192
    python scripts/train_moe.py --data data/shards --epochs 20 --resume checkpoints/moe/moe_epoch15.pt
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
from src.training.moe_losses import compute_lid1_loss, compute_lid2_loss, compute_ctc_loss
from src.training.routing import get_predicted_script_ids, build_routing_masks
from src.training.cpu_offload import CPUOffloadOptimizer
from src.training.moe_eval import evaluate


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, base_optimizer, scheduler, scaler,
                    ce_loss_fn, device, device_type, use_amp, amp_dtype,
                    epoch, total_epochs, grad_accum, log_interval,
                    lid1_weight, group_script_vocabs):
    model.train()
    n_batches = 0

    # Accumulate losses on GPU — avoid .item() sync every batch
    ctc_loss_accum = torch.zeros(1, device=device)
    lid1_loss_accum = torch.zeros(1, device=device)
    total_loss_accum = torch.zeros(1, device=device)
    log_ctc = torch.zeros(1, device=device)
    log_lid1 = torch.zeros(1, device=device)
    log_lid2 = torch.zeros(1, device=device)
    log_total = torch.zeros(1, device=device)
    log_count = 0

    for batch_idx, (imgs, targets, tgt_lens, gids, sids, _labels) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        tgt_lens = tgt_lens.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)
        sids = sids.to(device, non_blocking=True)

        # Forward — predicted routing (LID-1 and LID-2 decide)
        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(imgs, group_ids=None, script_ids=None)

        # LID-1 loss (all samples)
        lid1_loss = compute_lid1_loss(out["group_logits"], gids, ce_loss_fn)

        # Build routing masks
        pred_sids = get_predicted_script_ids(out["script_logits_per_group"], sids)
        lid1_ok, ctc_ok = build_routing_masks(
            out["group_ids"], gids, pred_sids, sids, tgt_lens, out["lengths"])

        # CTC loss (both LID-1 and LID-2 correct, per-script vocab slicing)
        ctc_loss = compute_ctc_loss(
            out["logits"], targets, out["lengths"], tgt_lens,
            ctc_ok, gids, sids, group_script_vocabs)

        # LID-2 loss (LID-1 correct, learns from LID-2 mistakes)
        lid2_loss = compute_lid2_loss(
            out["script_logits_per_group"], sids, lid1_ok, ce_loss_fn)

        loss = ctc_loss + lid1_weight * lid1_loss.float() + lid2_loss.float()

        if grad_accum > 1:
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(base_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25.0)
            old_scale = scaler.get_scale()
            optimizer.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:
                scheduler.step()

        # Accumulate on GPU — no sync
        mult = float(grad_accum) if grad_accum > 1 else 1.0
        ctc_loss_accum += ctc_loss.detach()
        lid1_loss_accum += lid1_loss.detach()
        total_loss_accum += loss.detach() * mult
        log_ctc += ctc_loss.detach()
        log_lid1 += lid1_loss.detach()
        log_lid2 += lid2_loss.detach()
        log_total += loss.detach() * mult
        n_batches += 1
        log_count += 1

        if n_batches % log_interval == 0:
            avg_ctc = log_ctc.item() / log_count
            avg_lid1 = log_lid1.item() / log_count
            avg_lid2 = log_lid2.item() / log_count
            avg_total = log_total.item() / log_count
            lr = scheduler.get_last_lr()[0]
            steps = len(train_loader)
            print(f"  [{epoch}/{total_epochs}] batch {batch_idx+1}/{steps}  "
                  f"loss={avg_total:.4f} "
                  f"(ctc={avg_ctc:.4f} lid1={avg_lid1:.4f} lid2={avg_lid2:.4f})  "
                  f"lr={lr:.2e}")
            log_ctc.zero_()
            log_lid1.zero_()
            log_lid2.zero_()
            log_total.zero_()
            log_count = 0

    if n_batches == 0:
        return {}

    return {
        "ctc": ctc_loss_accum.item() / n_batches,
        "lid1": lid1_loss_accum.item() / n_batches,
        "total": total_loss_accum.item() / n_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Lipi MoE Encoder")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--scripts", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="checkpoints/moe")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    # Model
    parser.add_argument("--stem-depth", type=int, default=3)
    parser.add_argument("--shared-dim", type=int, default=288)
    parser.add_argument("--shared-blocks-4x4", type=int, default=8)
    parser.add_argument("--shared-blocks-4x16", type=int, default=4)
    parser.add_argument("--stage1-dim", type=int, default=288)
    parser.add_argument("--stage1-blocks", type=int, default=12)
    parser.add_argument("--stage2-dim", type=int, default=576)
    parser.add_argument("--stage2-blocks", type=int, default=8)
    parser.add_argument("--head-hidden", type=int, default=384)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--lid1-weight", type=float, default=1.0)
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload optimizer states to CPU (frees ~5-7GB VRAM)")
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
    if args.scripts != "all":
        selected = set(s.strip() for s in args.scripts.split(","))
        active_scripts = [s for s in active_scripts if s in selected]
    print(f"Scripts: {active_scripts}")

    # Active groups
    active_groups = []
    seen = set()
    for s in active_scripts:
        g = SCRIPT_TO_GROUP.get(s)
        if g and g not in seen:
            active_groups.append(g)
            seen.add(g)
    n_groups = len(active_groups)
    print(f"Groups: {n_groups} -> {active_groups}")

    # Remap IDs
    group_ids, local_script_ids, global_to_local_group = remap_ids(
        active_scripts, active_groups, script_ids_global, group_ids_global)

    # Tokenizers (fixed vocabs from Unicode ranges, not data-dependent)
    print("\nBuilding per-script tokenizers...")
    group_tokenizers, group_script_vocab_sizes, group_script_names = build_script_tokenizers(
        active_scripts, active_groups)
    print(f"  Per-script vocab sizes: {group_script_vocab_sizes}")

    # Encode labels
    print("Pre-encoding labels...")
    target_tensor, target_len_tensor = encode_labels(
        labels, group_ids, local_script_ids, active_groups, group_tokenizers)
    print(f"  Max label length: {target_len_tensor.max().item()}")

    # Pre-filter empty/too-long labels
    max_enc_len = images.shape[3] // 4
    valid = (target_len_tensor > 0) & (target_len_tensor <= max_enc_len)
    n_filtered = (~valid).sum().item()
    if n_filtered > 0:
        keep = valid.nonzero(as_tuple=True)[0]
        images = images[keep]
        target_tensor = target_tensor[keep]
        target_len_tensor = target_len_tensor[keep]
        group_ids = group_ids[keep]
        local_script_ids = local_script_ids[keep]
        labels = [labels[i] for i in keep.tolist()]
        print(f"  Filtered {n_filtered} samples (empty or too long for CTC)")

    # Dataset + split
    dataset = MoEDataset(images, target_tensor, target_len_tensor,
                         group_ids, local_script_ids, labels)
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_moe, pin_memory=(device_type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_moe, pin_memory=(device_type == "cuda"))

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

    def vram(label=""):
        if device_type == "cuda":
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            print(f"  VRAM [{label}]: {a:.2f} GB allocated, {r:.2f} GB reserved")

    vram("after model to device (fp32)")

    # Cast to bf16 — halves param + gradient memory.
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
        torch.cuda.empty_cache()
        vram("after bf16 cast")

    total_params = sum(p.numel() for p in model.parameters())
    p0 = next(model.parameters())
    print(f"Model: {total_params / 1e6:.1f}M params ({n_groups} groups), dtype={p0.dtype}")

    # NOTE: torch.compile moved AFTER optimizer creation + resume.
    # Compiling before optimizer changes parameter structure, breaking
    # optimizer state dict loading from non-compiled checkpoints.

    # --- Optimizer + Scheduler ---
    vram("before optimizer")
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    vram("after optimizer init")
    if args.cpu_offload and device_type == "cuda":
        optimizer = CPUOffloadOptimizer(base_optimizer)
        print("Optimizer states offloaded to CPU (~5-7GB VRAM freed)")
    else:
        optimizer = base_optimizer

    use_amp = device_type in ("cuda", "mps")
    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        print(f"AMP: {amp_dtype}")
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(steps_per_epoch, total_steps // 10)
    warmup = torch.optim.lr_scheduler.LinearLR(
        base_optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        base_optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # Resume
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Partial load: skip mismatched layers (e.g., CTC proj after vocab change)
        model_state = ckpt["model"]
        current_state = model.state_dict()
        skipped = []
        for k in list(model_state.keys()):
            if k in current_state and model_state[k].shape != current_state[k].shape:
                skipped.append(k)
                del model_state[k]
        if skipped:
            print(f"  Skipped {len(skipped)} mismatched layers (vocab changed):")
            for k in skipped[:5]:
                print(f"    {k}")
            if len(skipped) > 5:
                print(f"    ... and {len(skipped) - 5} more")
        model.load_state_dict(model_state, strict=False)
        if not skipped:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print("  Skipping optimizer state (vocab changed, momentum shapes stale)")
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt.get("epoch", 0) + 1

        # Always rebuild scheduler on resume — checkpoint might have a different
        # scheduler type (CosineAnnealingLR vs SequentialLR) or different total epochs.
        print(f"  Rebuilding scheduler from epoch {start_epoch}")
        for pg in base_optimizer.param_groups:
            pg["lr"] = args.lr
        remaining_steps = steps_per_epoch * (args.epochs - start_epoch + 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_optimizer, T_max=max(remaining_steps, 1), eta_min=1e-6)

        del ckpt
        print(f"  Resumed at epoch {start_epoch}, lr={base_optimizer.param_groups[0]['lr']:.2e}")
        torch.cuda.empty_cache()
        vram("after resume")

    # torch.compile after optimizer + resume (avoids param group mismatch)
    if device_type == "cuda" and not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        vram("after compile")

    # --- Train ---
    ce_loss_fn = nn.CrossEntropyLoss()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    eff_batch = args.batch_size * args.grad_accum
    print(f"\n{'=' * 60}")
    print(f"TRAINING: epochs {start_epoch}-{args.epochs}, lr={args.lr}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} = {eff_batch} effective")
    print(f"  Losses: CTC x1.0 + LID1 x{args.lid1_weight}")
    print(f"  Routing: predicted (skip CTC on LID-1/LID-2 misroutes)")
    print(f"  Per-script vocabs: {group_script_vocab_sizes}")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        metrics = train_one_epoch(
            model, train_loader, optimizer, base_optimizer, scheduler, scaler,
            ce_loss_fn, device, device_type, use_amp, amp_dtype,
            epoch, args.epochs, args.grad_accum, args.log_interval,
            lid1_weight=args.lid1_weight,
            group_script_vocabs=group_script_vocab_sizes)

        elapsed = time.time() - t0
        if metrics:
            print(f"\nEpoch {epoch}/{args.epochs}: "
                  f"ctc={metrics['ctc']:.4f} lid1={metrics['lid1']:.4f}  "
                  f"time={elapsed:.0f}s")

        # Save (model to CPU to avoid OOM)
        ckpt_path = save_dir / f"moe_epoch{epoch}.pt"
        torch.save({
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # Eval
        print(f"\n  Eval epoch {epoch}:")
        evaluate(model, val_loader, group_tokenizers, group_script_names,
                 active_groups, device, device_type, use_amp, amp_dtype)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
