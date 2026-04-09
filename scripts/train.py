#!/usr/bin/env python3
"""
Train Lipi MoE Encoder end-to-end.

Usage:
    python scripts/train_moe.py --data data/shards --epochs 15 --batch-size 192
    python scripts/train_moe.py --data data/shards --epochs 20 --resume checkpoints/moe/moe_epoch15.pt
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiMoEEncoder
from src.model.memory import estimate_pixel_budget
from src.model.lid import SCRIPT_TO_GROUP, NUM_GROUPS, GROUPS
from src.training.dataloader import (
    build_script_tokenizers, collate_moe, WidthSortedBatchSampler,
    LipiStreamingDataset,
)
from src.training.losses import (
    compute_lid1_loss, compute_lid2_loss, compute_ctc_loss,
    compute_regional_token_loss,
)
from src.training.eval import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vram(label="", device_type="cuda"):
    if device_type == "cuda":
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        print(f"  VRAM [{label}]: {a:.2f} GB allocated, {r:.2f} GB reserved")



# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Lipi MoE Encoder")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--scripts", type=str, default="all")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=192,
                        help="Max batch size (actual size varies by width)")
    parser.add_argument("--vram", type=float, default=32,
                        help="GPU VRAM in GB (used to auto-size batches)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="checkpoints/moe")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    # Model
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--lid1-weight", type=float, default=1.0)
    parser.add_argument("--align-ce-weight", type=float, default=0.0,
                        help="Weight for forced-alignment CE loss. Gives partial credit "
                             "for multi-token CJK chars. Recommended: 0.1-0.3")
    parser.add_argument("--routing-penalty", type=float, default=0.0,
                        help="Extra LID-1 weight proportional to misroute rate. "
                             "Effective weight = lid1_weight + penalty * (1 - ctc_ok_frac)")
    parser.add_argument("--expert-lr", type=float, default=None,
                        help="Separate learning rate for expert params. Default: same as --lr")
    parser.add_argument("--detach-epochs", type=int, default=0,
                        help="Number of epochs to detach shared→expert gradient. "
                             "LID-1 gets undivided shared encoder, CTC trains experts only.")
    parser.add_argument("--freeze-except", type=str, default=None,
                        choices=["experts", "experts+ctc", "ctc", "shared"],
                        help="Freeze everything except: 'experts' (stage1+stage2 only), "
                             "'experts+ctc' (stage1+stage2+ctc heads+lid2), "
                             "'ctc' (ctc heads only), "
                             "or 'shared' (shared SWA + LID-1 only)")
    args = parser.parse_args()

    # Validation
    assert args.epochs > 0, f"--epochs must be > 0, got {args.epochs}"
    assert args.batch_size > 0, f"--batch-size must be > 0, got {args.batch_size}"
    assert args.lr > 0, f"--lr must be > 0, got {args.lr}"
    assert 0 < args.val_split < 1, f"--val-split must be in (0, 1), got {args.val_split}"
    assert args.grad_accum >= 1, f"--grad-accum must be >= 1, got {args.grad_accum}"
    assert Path(args.data).exists(), f"--data path does not exist: {args.data}"
    if args.resume:
        assert Path(args.resume).exists(), f"--resume path does not exist: {args.resume}"

    return args


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    device_type = device.type
    print(f"Device: {device}")
    return device, device_type


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def load_and_prepare_data(args, device):
    device_type = device.type
    data_path = Path(args.data)
    print(f"\nLoading data from {data_path}/...")

    # Load metadata
    meta_path = data_path / "metadata.pt"
    if not meta_path.exists():
        # Check parent for metadata (MDS dirs are data_path/train, data_path/val)
        meta_path = data_path.parent / "metadata.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, weights_only=False)
        active_scripts = meta["active_scripts"]
    else:
        active_scripts = list(SCRIPT_TO_GROUP.keys())

    if args.scripts != "all":
        selected = set(s.strip() for s in args.scripts.split(","))
        active_scripts = [s for s in active_scripts if s in selected]
    print(f"Scripts: {active_scripts}")

    # Active data groups (from data scripts)
    active_data_groups = set()
    for s in active_scripts:
        g = SCRIPT_TO_GROUP.get(s)
        if g:
            active_data_groups.add(g)
    assert active_data_groups, "No active groups found"
    print(f"Data groups: {sorted(active_data_groups)}")

    # Always build model with ALL 13 groups for checkpoint compatibility.
    active_groups = list(GROUPS)
    all_scripts = list(SCRIPT_TO_GROUP.keys())
    n_groups = NUM_GROUPS
    print(f"Model groups: {n_groups} (full architecture, data for {len(active_data_groups)})")

    # Tokenizers for all groups
    print("\nBuilding per-script tokenizers...")
    group_tokenizers, group_script_vocab_sizes, group_script_names = build_script_tokenizers(
        all_scripts, active_groups)
    print(f"  Per-script vocab sizes: {group_script_vocab_sizes}")

    # MDS streaming datasets
    train_dir = str(data_path / "train")
    val_dir = str(data_path / "val")
    print(f"Loading MDS datasets from {data_path}/...")

    train_dataset = LipiStreamingDataset(
        local=train_dir, active_scripts=all_scripts,
        active_groups=active_groups)
    val_dataset = LipiStreamingDataset(
        local=val_dir, active_scripts=all_scripts,
        active_groups=active_groups)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")


    train_widths = np.load(str(Path(train_dir) / "widths.npy"))

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_moe,
                            pin_memory=(device_type == "cuda"))

    return {
        "train_dataset": train_dataset,
        "train_widths": train_widths,
        "val_loader": val_loader,
        "n_groups": n_groups,
        "active_groups": active_groups,
        "group_tokenizers": group_tokenizers,
        "group_script_vocab_sizes": group_script_vocab_sizes,
        "group_script_names": group_script_names,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(args, n_groups, group_script_vocab_sizes, group_script_names, device):
    device_type = device.type
    model = LipiMoEEncoder(
        dim=args.dim,
        num_groups=n_groups,
        group_script_vocab_sizes=group_script_vocab_sizes,
        group_script_names=group_script_names,
    ).to(device)

    vram("after model to device (fp32)", device_type)

    # NOTE: Do NOT cast model to bf16. Autocast handles bf16 forward/backward
    # while keeping fp32 params for optimizer precision. Casting to bf16
    # makes Adam's m/v states bf16 (7-bit mantissa) — not enough precision
    # for stable convergence.

    total_params = sum(p.numel() for p in model.parameters())
    p0 = next(model.parameters())
    print(f"Model: {total_params / 1e6:.1f}M params ({n_groups} groups), dtype={p0.dtype}")

    return model


# ---------------------------------------------------------------------------
# Optimizer and scheduler
# ---------------------------------------------------------------------------

def build_optimizer_and_scheduler(args, model, device_type, steps_per_epoch):
    vram("before optimizer", device_type)
    expert_lr = args.expert_lr or args.lr
    if expert_lr != args.lr:
        shared_params = [p for n, p in model.named_parameters()
                         if not any(k in n for k in ("stage1.", "stage2.", "ctc_modules."))]
        expert_params = [p for n, p in model.named_parameters()
                         if any(k in n for k in ("stage1.", "stage2.", "ctc_modules."))]
        base_optimizer = torch.optim.AdamW([
            {"params": shared_params, "lr": args.lr},
            {"params": expert_params, "lr": expert_lr},
        ], weight_decay=0.01)
        print(f"Optimizer: shared lr={args.lr}, expert lr={expert_lr}")
    else:
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    vram("after optimizer init", device_type)
    optimizer = base_optimizer

    use_amp = device_type in ("cuda", "mps")
    if device_type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('high')
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        print(f"AMP: {amp_dtype}")
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False)

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(steps_per_epoch, total_steps // 10)
    warmup = torch.optim.lr_scheduler.LinearLR(
        base_optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        base_optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    return {
        "optimizer": optimizer,
        "base_optimizer": base_optimizer,
        "scaler": scaler,
        "scheduler": scheduler,
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
    }


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def resume_from_checkpoint(args, model, optimizer, base_optimizer, scaler, scheduler,
                           steps_per_epoch, device_type):
    print(f"\nResuming from {args.resume}...")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    # Partial load: skip mismatched layers (e.g., CTC proj after vocab change)
    model_state = ckpt["model"]

    # Remap legacy key names from old architecture
    n_4x4 = len(set(k.split(".")[1] for k in model_state if k.startswith("shared_swa_4x4.")))
    remapped = 0
    for k in list(model_state.keys()):
        if k.startswith("shared_swa_4x4."):
            new_k = k.replace("shared_swa_4x4.", "shared_swa.", 1)
            model_state[new_k] = model_state.pop(k)
            remapped += 1
        elif k.startswith("shared_swa_4x16."):
            idx = int(k.split(".")[1])
            rest = ".".join(k.split(".")[2:])
            new_k = f"shared_swa.{idx + n_4x4}.{rest}"
            model_state[new_k] = model_state.pop(k)
            remapped += 1
        elif k.startswith("proj_shared."):
            model_state[k.replace("proj_shared.", "proj_stem.", 1)] = model_state.pop(k)
            remapped += 1
    if remapped:
        print(f"  Remapped {remapped} legacy checkpoint keys")

    current_state = model.state_dict()
    skipped = []
    for k in list(model_state.keys()):
        if k in current_state and model_state[k].shape != current_state[k].shape:
            skipped.append(k)
            del model_state[k]
    if skipped:
        print(f"  Skipped {len(skipped)} shape-mismatched layers:")
        for k in skipped[:5]:
            print(f"    {k}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    missing = [k for k in current_state if k not in model_state]
    if missing:
        print(f"  {len(missing)} layers missing from checkpoint (randomly initialized):")
        for k in missing[:10]:
            print(f"    {k}: {current_state[k].shape}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    model.load_state_dict(model_state, strict=False)
    # Skip optimizer state if layers were skipped OR if checkpoint was
    # from a different dtype (e.g., bf16 model -> fp32 model)
    ckpt_dtype = None
    for v in model_state.values():
        if v.is_floating_point():
            ckpt_dtype = v.dtype
            break
    model_dtype = next(model.parameters()).dtype
    dtype_changed = ckpt_dtype is not None and ckpt_dtype != model_dtype
    # Check param group count matches (e.g., checkpoint had 1 group, now we have 2)
    ckpt_groups = len(ckpt.get("optimizer", {}).get("param_groups", []))
    cur_groups = len(base_optimizer.param_groups)
    groups_changed = ckpt_groups != cur_groups
    freeze_mode = hasattr(args, 'freeze_except') and args.freeze_except is not None
    if "optimizer" not in ckpt:
        print("  No optimizer state in checkpoint (fresh optimizer)")
    elif skipped or dtype_changed or groups_changed or freeze_mode:
        reasons = []
        if skipped:
            reasons.append("vocab changed")
        if dtype_changed:
            reasons.append(f"dtype changed ({ckpt_dtype}→{model_dtype})")
        if groups_changed:
            reasons.append(f"param groups changed ({ckpt_groups}→{cur_groups})")
        if freeze_mode:
            reasons.append(f"freeze mode ({args.freeze_except})")
        print(f"  Skipping optimizer state ({', '.join(reasons)})")
    else:
        optimizer.load_state_dict(ckpt["optimizer"])
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
    if device_type == "cuda":
        torch.cuda.empty_cache()
    vram("after resume", device_type)

    return start_epoch, scheduler


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, save_dir):
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



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, base_optimizer, scheduler, scaler,
                    ce_loss_fn, device, device_type, use_amp, amp_dtype,
                    epoch, total_epochs, grad_accum, log_interval,
                    lid1_weight, group_script_vocabs,
                    detach_for_experts=False, align_ce_weight=0.0,
                    save_dir=None, args=None):
    model.train()
    steps = len(train_loader)
    n_batches = 0
    oom_skipped = 0
    oom_retried = 0

    # Cache param split for grad clipping (avoid iterating named_parameters every step)
    shared_params = []
    expert_params = []
    for name, p in model.named_parameters():
        if any(k in name for k in ("stage1.", "stage2.", "ctc_modules.")):
            expert_params.append(p)
        else:
            shared_params.append(p)

    # Accumulate losses on GPU — avoid .item() sync every batch
    ctc_loss_accum = torch.zeros(1, device=device)
    lid1_loss_accum = torch.zeros(1, device=device)
    total_loss_accum = torch.zeros(1, device=device)
    log_ctc = torch.zeros(1, device=device)
    log_lid1 = torch.zeros(1, device=device)
    log_lid2 = torch.zeros(1, device=device)
    log_ace = torch.zeros(1, device=device)
    log_total = torch.zeros(1, device=device)
    log_count = 0
    log_time = time.time()
    shared_norm = 0.0
    expert_norm = 0.0

    def _forward_backward(imgs_, targets_, tgt_lens_, gids_, sids_, scale):
        """Run forward + backward on a (sub-)batch. scale adjusts loss."""
        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(imgs_, group_ids=gids_, script_ids=sids_,
                        detach_for_experts=detach_for_experts)

        lid1_loss = compute_lid1_loss(out["group_logits"], gids_, ce_loss_fn)

        all_ok = (tgt_lens_ <= out["lengths"]) & (tgt_lens_ > 0)
        ctc_loss = compute_ctc_loss(
            out["logits"], targets_, out["lengths"], tgt_lens_,
            all_ok, gids_, sids_, group_script_vocabs)

        all_true = torch.ones(imgs_.shape[0], dtype=torch.bool, device=device)
        lid2_loss = compute_lid2_loss(
            out["script_logits_per_group"], sids_, all_true, ce_loss_fn)

        if align_ce_weight > 0:
            ace_loss = compute_regional_token_loss(
                out["logits"], targets_, out["lengths"], tgt_lens_,
                all_ok, gids_, sids_, group_script_vocabs)
        else:
            ace_loss = torch.zeros(1, device=device)

        loss = (ctc_loss
                + lid1_weight * lid1_loss.float()
                + lid2_loss.float()
                + align_ce_weight * ace_loss.float())

        loss = loss * scale
        if grad_accum > 1:
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        return ctc_loss, lid1_loss, lid2_loss, ace_loss, loss

    for batch_idx, (imgs, targets, tgt_lens, gids, sids, _labels) in enumerate(train_loader):
        if batch_idx < 5:
            print(f"    [shape] batch {batch_idx}: imgs={list(imgs.shape)} "
                  f"B={imgs.shape[0]} W={imgs.shape[3]}", flush=True)

        # Keep CPU refs for OOM retry before moving to GPU
        cpu_imgs, cpu_targets, cpu_tgt_lens = imgs, targets, tgt_lens
        cpu_gids, cpu_sids = gids, sids

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        tgt_lens = tgt_lens.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)
        sids = sids.to(device, non_blocking=True)

        try:
            ctc_loss, lid1_loss, lid2_loss, ace_loss, loss = \
                _forward_backward(imgs, targets, tgt_lens, gids, sids, scale=1.0)
        except torch.cuda.OutOfMemoryError:
            # Clean up the failed forward/backward
            del imgs, targets, tgt_lens, gids, sids
            try:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            except Exception:
                print("  ** CUDA context corrupted after OOM, exiting",
                      flush=True)
                sys.exit(1)

            # Retry: split CPU batch in half, run each half separately
            B = cpu_imgs.shape[0]
            half = B // 2
            if half == 0:
                oom_skipped += 1
                print(f"  ** OOM at batch {batch_idx+1}/{steps} "
                      f"(B=1) — skipping [{oom_skipped} this epoch]",
                      flush=True)
                del cpu_imgs, cpu_targets, cpu_tgt_lens, cpu_gids, cpu_sids
                continue

            print(f"  ** OOM at batch {batch_idx+1}/{steps} "
                  f"(B={B} W={cpu_imgs.shape[3]}) — "
                  f"retrying as 2×{half}", flush=True)
            oom_retried += 1

            retry_ok = True
            for sub_start in (0, half):
                sub_end = sub_start + half if sub_start == 0 else B
                s_imgs = cpu_imgs[sub_start:sub_end].to(device)
                s_targets = cpu_targets[sub_start:sub_end].to(device)
                s_tgt_lens = cpu_tgt_lens[sub_start:sub_end].to(device)
                s_gids = cpu_gids[sub_start:sub_end].to(device)
                s_sids = cpu_sids[sub_start:sub_end].to(device)
                try:
                    sub_ctc, sub_lid1, sub_lid2, sub_ace, sub_loss = \
                        _forward_backward(s_imgs, s_targets, s_tgt_lens,
                                          s_gids, s_sids,
                                          scale=(sub_end - sub_start) / B)
                except torch.cuda.OutOfMemoryError:
                    # Half still OOMs — give up on this batch
                    del s_imgs, s_targets, s_tgt_lens, s_gids, s_sids
                    try:
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                    except Exception:
                        print("  ** CUDA context corrupted after OOM, exiting",
                              flush=True)
                        sys.exit(1)
                    oom_skipped += 1
                    print(f"  ** OOM retry also failed — "
                          f"skipping [{oom_skipped} this epoch]", flush=True)
                    retry_ok = False
                    break
                del s_imgs, s_targets, s_tgt_lens, s_gids, s_sids

            del cpu_imgs, cpu_targets, cpu_tgt_lens, cpu_gids, cpu_sids
            if not retry_ok:
                continue
            # Use last sub-batch losses for logging (approximate)
            ctc_loss, lid1_loss = sub_ctc, sub_lid1
            lid2_loss, ace_loss, loss = sub_lid2, sub_ace, sub_loss

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(base_optimizer)
            shared_norm = torch.nn.utils.clip_grad_norm_(shared_params, max_norm=25.0)
            expert_norm = torch.nn.utils.clip_grad_norm_(expert_params, max_norm=25.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:
                scheduler.step()

        # Accumulate on GPU — no sync
        mult = float(grad_accum)
        ctc_loss_accum += ctc_loss.detach()
        lid1_loss_accum += lid1_loss.detach()
        total_loss_accum += loss.detach() * mult
        log_ctc += ctc_loss.detach()
        log_lid1 += lid1_loss.detach()
        log_lid2 += lid2_loss.detach()
        log_ace += ace_loss.detach()
        log_total += loss.detach() * mult
        n_batches += 1
        log_count += 1

        if save_dir and n_batches % 500 == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, args, save_dir)

        if n_batches % log_interval == 0:
            avg_ctc = log_ctc.item() / log_count
            avg_lid1 = log_lid1.item() / log_count
            avg_lid2 = log_lid2.item() / log_count
            avg_ace = log_ace.item() / log_count
            avg_total = log_total.item() / log_count
            # LID-1 accuracy (predicted vs ground truth)
            pred_gids = out["group_logits"].argmax(-1)
            lid1_acc = (pred_gids == gids).float().mean().item() * 100
            # LID-2 accuracy (all samples in multi-script groups)
            lid2_correct = 0
            lid2_total = 0
            for _g, script_logits, group_mask in out["script_logits_per_group"]:
                if script_logits is not None:
                    pred = script_logits.argmax(-1)
                    true = sids[group_mask]
                    lid2_correct += (pred == true).sum().item()
                    lid2_total += true.shape[0]
            lid2_acc = 100 * lid2_correct / max(lid2_total, 1)
            elapsed = time.time() - log_time
            ms_per_step = elapsed / log_count * 1000
            samples_per_sec = imgs.shape[0] * log_count / elapsed
            ace_str = f"  ace {avg_ace:.4f}" if align_ce_weight > 0 else ""
            batch_str = f"{batch_idx+1}/{steps}"
            print(
                f"  [{epoch:>2}/{total_epochs}] {batch_str:>9}  "
                f"loss {avg_total:.4f}  "
                f"ctc {avg_ctc:.4f}  lid1 {avg_lid1:.4f}  lid2 {avg_lid2:.4f}{ace_str}  "
                f"| acc {lid1_acc:5.1f}% {lid2_acc:5.1f}%  "
                f"| {samples_per_sec:.0f} img/s  {ms_per_step:.0f}ms/step  "
                f"gnorm {shared_norm:.1f}/{expert_norm:.1f}"
            )
            log_time = time.time()
            log_ctc.zero_()
            log_lid1.zero_()
            log_lid2.zero_()
            log_ace.zero_()
            log_total.zero_()
            log_count = 0

    if oom_retried > 0 or oom_skipped > 0:
        print(f"  ** OOM: {oom_retried} retried (halved), "
              f"{oom_skipped} skipped this epoch")

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
    args = parse_args()
    device, device_type = resolve_device(args)

    data = load_and_prepare_data(args, device)
    model = build_model(args, data["n_groups"], data["group_script_vocab_sizes"],
                        data["group_script_names"], device)

    # Selective freezing
    if args.freeze_except:
        expert_keys = ("stage1.", "stage2.")
        frozen = 0
        trainable = 0
        for name, param in model.named_parameters():
            if args.freeze_except == "experts":
                # Train only expert SWA blocks
                param.requires_grad = any(k in name for k in expert_keys)
            elif args.freeze_except == "experts+ctc":
                # Train expert SWA + CTC heads + LID-2
                param.requires_grad = any(k in name for k in
                    ("stage1.", "stage2.", "ctc_modules."))
            elif args.freeze_except == "ctc":
                # Train only CTC heads + LID-2
                param.requires_grad = "ctc_modules." in name
            elif args.freeze_except == "shared":
                # Train only shared encoder + LID-1
                param.requires_grad = not any(k in name for k in
                    ("stage1.", "stage2.", "ctc_modules."))
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()
        print(f"Freeze mode: training {args.freeze_except} only")
        print(f"  Trainable: {trainable/1e6:.1f}M, Frozen: {frozen/1e6:.1f}M")

    # Build train_loader with VRAM-estimated pixel budget

    train_widths = data["train_widths"]
    max_width = int(train_widths.max())

    pixel_budget = estimate_pixel_budget(model, vram_gb=args.vram)
    max_batch_at_widest = pixel_budget // max_width
    # Cap at --batch-size: pixel budget handles width scaling, but there's
    # per-sample overhead (autograd nodes, CTC loss, routing) that doesn't
    # scale with width. --batch-size caps the max samples in any batch.
    max_batch_size = min(args.batch_size, max(max_batch_at_widest, 1))

    train_batch_sampler = WidthSortedBatchSampler(
        train_widths, max_batch_size, max_width=0,
        pixel_budget=pixel_budget)
    batch_sizes = [len(b) for b in train_batch_sampler._batches]
    print(f"  Batching: {len(train_batch_sampler)} batches, "
          f"size {min(batch_sizes)}-{max(batch_sizes)} "
          f"(max={max_batch_size}, budget={pixel_budget}px)")

    train_loader = DataLoader(data["train_dataset"], batch_sampler=train_batch_sampler,
                              collate_fn=collate_moe,
                              pin_memory=(device_type == "cuda"))

    steps_per_epoch = len(train_loader)
    opt = build_optimizer_and_scheduler(args, model, device_type, steps_per_epoch)

    # NOTE: torch.compile moved AFTER optimizer creation + resume.
    # Compiling before optimizer changes parameter structure, breaking
    # optimizer state dict loading from non-compiled checkpoints.

    start_epoch = 1
    if args.resume:
        start_epoch, opt["scheduler"] = resume_from_checkpoint(
            args, model, opt["optimizer"], opt["base_optimizer"], opt["scaler"],
            opt["scheduler"], steps_per_epoch, device_type)

    # torch.compile after optimizer + resume (avoids param group mismatch)
    if device_type == "cuda" and not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        vram("after compile", device_type)

    ce_loss_fn = nn.CrossEntropyLoss()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    eff_batch = f"{min(batch_sizes)}-{max(batch_sizes)}"
    print(f"\n{'=' * 60}")
    print(f"TRAINING: epochs {start_epoch}-{args.epochs}, lr={args.lr}")
    print(f"  Batch: {eff_batch} x {args.grad_accum} (pixel-budgeted)")
    ace_str = f" + ACE x{args.align_ce_weight}" if args.align_ce_weight > 0 else ""
    print(f"  Losses: CTC x1.0 + LID1 x{args.lid1_weight}{ace_str}")
    print(f"  Routing: ground truth (CTC on all samples)")
    print(f"  Per-script vocabs: {data['group_script_vocab_sizes']}")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        detach = epoch <= args.detach_epochs
        if detach:
            print(f"  [detach mode: CTC gradient stops at expert boundary, epoch {epoch}/{args.detach_epochs}]")
        metrics = train_one_epoch(
            model, train_loader, opt["optimizer"], opt["base_optimizer"],
            opt["scheduler"], opt["scaler"], ce_loss_fn, device, device_type,
            opt["use_amp"], opt["amp_dtype"], epoch, args.epochs, args.grad_accum,
            args.log_interval, lid1_weight=args.lid1_weight,
            group_script_vocabs=data["group_script_vocab_sizes"],
            detach_for_experts=detach,
            align_ce_weight=args.align_ce_weight,
            save_dir=save_dir, args=args)

        elapsed = time.time() - t0
        if metrics:
            print(f"\nEpoch {epoch}/{args.epochs}: "
                  f"ctc={metrics['ctc']:.4f} lid1={metrics['lid1']:.4f}  "
                  f"time={elapsed:.0f}s")

        save_checkpoint(model, opt["optimizer"], opt["scheduler"], opt["scaler"],
                        epoch, args, save_dir)

        print(f"\n  Eval epoch {epoch}:")
        evaluate(model, data["val_loader"], data["group_tokenizers"],
                 data["group_script_names"], data["active_groups"],
                 device, device_type, opt["use_amp"], opt["amp_dtype"])

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
