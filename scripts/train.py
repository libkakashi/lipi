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
import torch.nn.functional as F
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
    compute_lid1_loss, compute_ctc_loss,
    compute_ctc_loss_segments, compute_regional_token_loss,
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
    parser.add_argument("--dim", type=int, default=512)
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
                        choices=["experts", "experts+ctc", "ctc", "backbone"],
                        help="Freeze everything except specified components")
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
    print(f"  Train: {len(train_dataset)}, Val (full): {len(val_dataset)}")

    # Subsample val to 500 per script for fast, balanced eval
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
    print(f"  Val (subsampled): {len(val_subset)} ({max_per_script}/script)")
    for sid, count in sorted(script_counts.items()):
        sname = all_scripts[sid] if sid < len(all_scripts) else f"id={sid}"
        print(f"    {sname:<18s} {count:>6d}")

    train_widths = np.load(str(Path(train_dir) / "widths.npy"))

    eval_batch_size = min(args.batch_size, 128)
    val_loader = DataLoader(val_subset, batch_size=eval_batch_size, shuffle=True,
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
                         if not any(k in n for k in ("expert_blocks.", "ctc_modules."))]
        expert_params = [p for n, p in model.named_parameters()
                         if any(k in n for k in ("expert_blocks.", "ctc_modules."))]
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
    # Check param groups match (count and size)
    ckpt_opt = ckpt.get("optimizer", {})
    ckpt_groups = len(ckpt_opt.get("param_groups", []))
    cur_groups = len(base_optimizer.param_groups)
    ckpt_param_count = len(ckpt_opt.get("state", {}))
    cur_param_count = sum(len(pg["params"]) for pg in base_optimizer.param_groups)
    groups_changed = ckpt_groups != cur_groups or ckpt_param_count != cur_param_count
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
        "model_config": model.config,
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
                    lid1_weight, group_script_vocabs, group_script_names,
                    detach_for_experts=False, align_ce_weight=0.0,
                    save_dir=None, args=None):
    model.train()
    steps = len(train_loader)
    n_batches = 0
    oom_skipped = 0

    # Cache param split for grad clipping (avoid iterating named_parameters every step)
    shared_params = []
    expert_params = []
    for name, p in model.named_parameters():
        if any(k in name for k in ("expert_blocks.", "ctc_modules.")):
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
    log_lid1_correct = 0
    log_lid1_total = 0
    log_lid2_correct = 0
    log_lid2_total = 0
    log_time = time.time()
    shared_norm = 0.0
    expert_norm = 0.0

    def _forward_backward(imgs_, targets_, tgt_lens_, gids_, sids_, scale,
                          group_labels_=None, segments_=None):
        """Run forward + backward on a (sub-)batch. scale adjusts loss."""
        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            # Per-frame group labels: -100 = padding (loss ignores),
            # NUM_GROUPS = whitespace (learnable). For model routing,
            # both padding and whitespace should skip expert blocks.
            T_est = imgs_.shape[3] // 2
            gl_frames = group_labels_[:, ::2][:, :T_est]
            gl_for_model = gl_frames.clone()
            gl_for_model[gl_for_model < 0] = NUM_GROUPS  # padding → blank for routing

            # Build per-frame script labels from segment metadata
            # so CTC heads use ground truth script routing during training
            B_cur = imgs_.shape[0]
            sl_frames = torch.zeros(B_cur, T_est, dtype=torch.long, device=device)
            for b in range(B_cur):
                for seg in segments_[b]:
                    frame_start = seg["offset"] // 2
                    frame_end = min((seg["offset"] + seg["width"] + 1) // 2, T_est)
                    sl_frames[b, frame_start:frame_end] = seg["script_id"]

            out = model(imgs_, group_ids=gl_for_model, script_ids=sl_frames,
                        detach_for_experts=detach_for_experts)

        # LID-1 loss (gl_frames has -100 for padding → ignored by CE)
        T = out["group_logits"].shape[1]
        gl_frames = gl_frames[:, :T]
        lid1_loss = compute_lid1_loss(out["group_logits"], gl_frames, ce_loss_fn)

        # CTC loss: batch single-script, per-segment for mixed
        mixed_indices = [i for i, s in enumerate(segments_) if len(s) > 1]
        single_indices = [i for i, s in enumerate(segments_) if len(s) == 1]

        ctc_loss = torch.zeros(1, device=device)
        ctc_parts = 0

        if single_indices:
            si = torch.tensor(single_indices, device=device)
            all_ok = (tgt_lens_[si] <= out["lengths"][si]) & (tgt_lens_[si] > 0)
            single_loss = compute_ctc_loss(
                out["logits"][si], targets_[si], out["lengths"][si], tgt_lens_[si],
                all_ok, gids_[si], sids_[si], group_script_vocabs)
            ctc_loss = ctc_loss + single_loss
            ctc_parts += 1

        if mixed_indices:
            mixed_segs = [segments_[i] for i in mixed_indices]
            mi = torch.tensor(mixed_indices, device=device)
            mixed_loss = compute_ctc_loss_segments(
                out["logits"][mi], mixed_segs, out["lengths"][mi],
                group_script_names, group_script_vocabs)
            ctc_loss = ctc_loss + mixed_loss
            ctc_parts += 1

        if ctc_parts > 1:
            ctc_loss = ctc_loss / ctc_parts

        # LID-2 loss: per-frame CE within multi-script groups
        # Uses ground truth sl_frames as targets, same ignore_index=-100 as LID-1
        lid2_loss = torch.zeros(1, device=device)
        lid2_count = 0
        T_lid2 = out["group_logits"].shape[1]
        sl_for_loss = sl_frames[:, :T_lid2]
        gl_for_loss = gl_for_model[:, :T_lid2]
        for g_str, lid2_logits in out.get("lid2_logits_per_group", {}).items():
            g = int(g_str)
            # Only compute on frames that belong to this group
            g_mask = (gl_for_loss == g)
            if not g_mask.any():
                continue
            # lid2_logits: (B, T, n_scripts_in_group)
            pred = lid2_logits[g_mask]  # (N_frames, n_scripts)
            target = sl_for_loss[g_mask]  # (N_frames,)
            lid2_loss = lid2_loss + F.cross_entropy(pred, target, label_smoothing=0.1)
            lid2_count += 1
        if lid2_count > 0:
            lid2_loss = lid2_loss / lid2_count

        if align_ce_weight > 0:
            all_ok = (tgt_lens_ <= out["lengths"]) & (tgt_lens_ > 0)
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
        # Detach logits for logging — prevents autograd graph from leaking
        # Detach LID-2 logits for logging
        detached_lid2 = {
            g: lg.detach() for g, lg in out.get("lid2_logits_per_group", {}).items()
        }
        return (ctc_loss, lid1_loss, lid2_loss, ace_loss, loss,
                out["group_logits"].detach(), detached_lid2,
                gl_for_model.detach(), sl_frames.detach())

    for batch_idx, batch in enumerate(train_loader):
        imgs, targets, tgt_lens, gids, sids, _labels, group_labels, segments = batch
        if batch_idx < 5:
            print(f"    [shape] batch {batch_idx}: B={imgs.shape[0]} "
                  f"C={imgs.shape[1]} H={imgs.shape[2]} W={imgs.shape[3]}", flush=True)
        if batch_idx == 0 and imgs.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB images, got {imgs.shape[1]} channels. "
                             f"Regenerate data with current code.")

        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        tgt_lens = tgt_lens.to(device, non_blocking=True)
        gids = gids.to(device, non_blocking=True)
        sids = sids.to(device, non_blocking=True)
        if group_labels is not None:
            group_labels = group_labels.to(device, non_blocking=True)

        try:
            (ctc_loss, lid1_loss, lid2_loss, ace_loss, loss,
             group_logits, lid2_logits, gt_groups, gt_scripts) = \
                _forward_backward(imgs, targets, tgt_lens, gids, sids, scale=1.0,
                                  group_labels_=group_labels, segments_=segments)
        except torch.cuda.OutOfMemoryError:
            del imgs, targets, tgt_lens, gids, sids, group_labels, segments
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            oom_skipped += 1
            print(f"  ** OOM at batch {batch_idx+1}/{steps} — "
                  f"skipping [{oom_skipped} this epoch]", flush=True)
            continue

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

        # Accumulate LID-1 and LID-2 accuracy across logging interval
        with torch.no_grad():
            fp = group_logits.argmax(dim=-1)
            T_acc = fp.shape[1]
            fl = group_labels[:, ::2][:, :T_acc].to(fp.device)
            non_pad = (fl >= 0)
            if non_pad.any():
                log_lid1_correct += (fp[non_pad] == fl[non_pad]).sum().item()
                log_lid1_total += non_pad.sum().item()

            # LID-2: per-frame accuracy within multi-script groups
            for g_str, lid2_log in lid2_logits.items():
                g = int(g_str)
                g_mask = (gt_groups[:, :T_acc] == g)
                if g_mask.any():
                    pred_s = lid2_log[:, :T_acc][g_mask].argmax(dim=-1)
                    true_s = gt_scripts[:, :T_acc][g_mask]
                    log_lid2_correct += (pred_s == true_s).sum().item()
                    log_lid2_total += true_s.shape[0]

        if save_dir and n_batches % 500 == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, args, save_dir)

        if n_batches % log_interval == 0:
            avg_ctc = log_ctc.item() / log_count
            avg_lid1 = log_lid1.item() / log_count
            avg_lid2 = log_lid2.item() / log_count
            avg_ace = log_ace.item() / log_count
            avg_total = log_total.item() / log_count
            # LID-1 accuracy (accumulated across logging interval)
            lid1_acc = 100 * log_lid1_correct / max(log_lid1_total, 1)
            # LID-2 accuracy (accumulated across logging interval)
            lid2_acc = 100 * log_lid2_correct / max(log_lid2_total, 1)
            # Sanity check: warn if accumulator empty (possible bug)
            if log_lid1_total == 0:
                print(f"  WARN: log_lid1_total=0 at step {batch_idx+1}", flush=True)
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
            log_lid1_correct = 0
            log_lid1_total = 0
            log_lid2_correct = 0
            log_lid2_total = 0

    if oom_skipped > 0:
        print(f"  ** OOM: {oom_skipped} batches skipped this epoch")

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
        frozen = 0
        trainable = 0
        for name, param in model.named_parameters():
            if args.freeze_except == "experts":
                param.requires_grad = "expert_blocks." in name
            elif args.freeze_except == "experts+ctc":
                param.requires_grad = any(k in name for k in
                    ("expert_blocks.", "ctc_modules."))
            elif args.freeze_except == "ctc":
                param.requires_grad = "ctc_modules." in name
            elif args.freeze_except == "backbone":
                param.requires_grad = "backbone." in name
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

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # ignore_index=-100 skips padding
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
            group_script_names=data["group_script_names"],
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

        # Move optimizer state to CPU to free VRAM for eval
        optimizer = opt["optimizer"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        torch.cuda.empty_cache()

        print(f"\n  Eval epoch {epoch}:")
        evaluate(model, data["val_loader"], data["group_tokenizers"],
                 data["group_script_names"], data["active_groups"],
                 device, device_type, opt["use_amp"], opt["amp_dtype"],
                 group_script_vocab_sizes=data["group_script_vocab_sizes"])

        # Move optimizer state back to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
