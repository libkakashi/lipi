#!/usr/bin/env python3
"""
Train Lipi MoE Encoder end-to-end.

Usage:
    python scripts/train.py --data data/shards --epochs 15 --batch-size 192
    python scripts/train.py --data data/shards --epochs 20 --resume checkpoints/moe/moe_epoch15.pt
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model.encoder import LipiMoEEncoder
from src.model.memory import estimate_pixel_budget
from src.taxonomy import SCRIPT_TO_GROUP, NUM_GROUPS, GROUPS
from src.training.dataloader import (
    build_script_tokenizers, collate_moe, WidthSortedBatchSampler,
    LipiStreamingDataset,
)
from src.training.losses import (
    compute_lid1_loss, compute_lid2_loss, compute_ctc_loss_segments,
    compute_consistency_loss,
)
from src.training.ema import ModelEMA
from src.training.routing import build_frame_labels_from_segments
from src.training.eval import evaluate


# Parameter-name prefixes for the "expert" param split (distinct LR / clip
# group from the shared backbone and LID heads). Must be kept in sync with
# encoder.py module names. group_layers / script_layers each contain a
# stack of MoELayers — attention/shared_mlp inside them are formally
# "shared", but keeping the whole MoE stack together makes the split
# clean and matches how experts scale with num_scripts.
EXPERT_PARAM_PREFIXES = (
    "group_layers.", "script_layers.",
    "ctc_modules.",
)


def _is_expert_param(name: str) -> bool:
    return any(k in name for k in EXPERT_PARAM_PREFIXES)


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
    parser.add_argument("--vram-safety", type=float, default=0.85,
                        help="Fraction of calibrated budget to actually use. "
                             "Lower if you see OOM batches; raise toward 1.0 "
                             "if OOMs never fire but throughput is low.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="checkpoints/moe")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--skip-backbone-load", action="store_true",
                        help="When resuming, skip loading shared/merge/stem/LID "
                             "weights. Loads only experts + CTC heads.")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--train-aug", dest="train_aug", action="store_true",
                        default=True,
                        help="Apply augmentation on the fly in the train "
                             "dataloader (default: on). Expects shards "
                             "generated WITHOUT baked-in augmentation.")
    parser.add_argument("--no-train-aug", dest="train_aug",
                        action="store_false")
    parser.add_argument("--train-aug-p", type=float, default=0.75,
                        help="Probability a train sample is augmented "
                             "(default: 0.75)")
    parser.add_argument("--num-workers", type=int, default=12,
                        help="DataLoader worker processes. Set 0 to disable "
                             "multiprocessing (useful for debugging). Lower "
                             "this if you have <16 CPU cores.")
    parser.add_argument("--script-sample-beta", type=float, default=0.5,
                        help="Resample the train set each epoch so script s "
                             "gets sampling mass ∝ vocab_size(s)^beta. "
                             "beta=0 gives every script equal mass; larger "
                             "beta shifts visits toward large-vocab scripts "
                             "(han, korean) that collapsed under flat "
                             "sampling. Negative disables resampling "
                             "(natural shard distribution). Needs the "
                             "script_ids.npy sidecar (regenerated shards).")
    # Model
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="Weight EMA decay per optimizer step; eval and "
                             "the 'ema' checkpoint entry use the averaged "
                             "weights. 0 disables. Costs one fp32 copy of "
                             "the params in VRAM.")
    parser.add_argument("--ctc-weight", type=float, default=1.0,
                        help="Set to 0 to disable CTC loss. Useful for "
                             "pretraining stem + shared SWA on LID alone "
                             "before joint training — random features into "
                             "CTC cause blank collapse, structured features "
                             "from LID-pretraining let CTC escape cleanly.")
    parser.add_argument("--consistency-weight", type=float, default=0.0,
                        help="Two-view consistency: each sample is "
                             "augmented twice and the symmetric KL between "
                             "the views' CTC + LID-1 posteriors is added "
                             "to the loss. Directly optimizes 'same text "
                             "under any degradation → same output'. "
                             "Doubles forward compute (pixel budget is "
                             "halved automatically); requires train-aug. "
                             "0 disables; 0.5 is a sensible starting value.")
    parser.add_argument("--inter-ctc-weight", type=float, default=0.3,
                        help="Auxiliary CTC loss on the features after the "
                             "group MoE stack (before script experts), "
                             "through the same norm + CTC heads (no new "
                             "params, training-only). Regularizes the trunk "
                             "and forces character info to exist before "
                             "script specialization. 0 disables.")
    parser.add_argument("--lid1-weight", type=float, default=1.0)
    parser.add_argument("--lid2-weight", type=float, default=1.0,
                        help="Set to 0 to disable LID-2 loss "
                             "(useful for CTC warmup with --lid1-weight 0)")
    parser.add_argument("--route-sample-max", type=float, default=0.25,
                        help="Scheduled sampling for expert routing: per-"
                             "frame probability of routing by predicted "
                             "LID instead of ground truth ramps linearly "
                             "from 0 (epoch 1) to this value (final epoch). "
                             "Losses always use GT labels. 0 disables.")
    parser.add_argument("--expert-lr", type=float, default=None,
                        help="Separate learning rate for expert params. Default: same as --lr")
    parser.add_argument("--detach-epochs", type=int, default=0,
                        help="Number of epochs to detach shared→expert gradient. "
                             "LID-1 gets undivided shared encoder, CTC trains experts only.")
    parser.add_argument("--freeze-except", type=str, default=None,
                        help="Freeze everything except specified components. "
                             "Comma-separated. Valid: experts, ctc, backbone, "
                             "lid (both heads), lid1 / lid2 (individual heads).")
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

def build_script_sample_weights(train_dir: str, beta: float) -> np.ndarray | None:
    """Per-sample weights giving script s total sampling mass ∝ vocab^beta.

    Motivation: flat per-script sampling collapsed CJK (han has 3811
    classes vs emoji's 107 but got the same number of visits). Weighting
    mass by vocab_size^beta gives complex scripts proportionally more
    visits; beta=0.5 ≈ 6x more han than armenian rather than 25x (beta=1)
    or 1x (beta=0).

    Returns None (resampling disabled) when the script_ids.npy sidecar is
    missing — shards generated before the sidecar existed.
    """
    from src.encoding.decompose import script_vocab_size
    from src.taxonomy import SCRIPTS

    sids_path = Path(train_dir) / "script_ids.npy"
    if not sids_path.exists():
        print("  [script-resample] script_ids.npy not found — disabled. "
              "Regenerate shards to enable script-balanced sampling.")
        return None

    sids = np.load(str(sids_path))
    uniq, counts = np.unique(sids, return_counts=True)

    # mass_s ∝ vocab^beta; per-sample weight = script mass / script count
    masses = {}
    for sid, n_s in zip(uniq.tolist(), counts.tolist()):
        name = SCRIPTS[sid] if sid < len(SCRIPTS) else None
        vs = script_vocab_size(name) if name else 1
        masses[sid] = max(vs, 1) ** beta

    total_mass = sum(masses.values())
    weights = np.zeros(len(sids), dtype=np.float64)
    print(f"  [script-resample] beta={beta} — per-epoch visit factor "
          f"(share of epoch / natural share):")
    for sid, n_s in zip(uniq.tolist(), counts.tolist()):
        share = masses[sid] / total_mass
        weights[sids == sid] = share / n_s
        factor = share * len(sids) / n_s
        name = SCRIPTS[sid] if sid < len(SCRIPTS) else f"id={sid}"
        print(f"    {name:<12s} {n_s:>8d} samples  x{factor:.2f}")
    return weights


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

    # Always build model with ALL groups for checkpoint compatibility.
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

    use_two_views = args.consistency_weight > 0 and args.train_aug
    if args.consistency_weight > 0 and not args.train_aug:
        print("  WARNING: --consistency-weight needs --train-aug; disabled.")
    train_dataset = LipiStreamingDataset(
        local=train_dir, active_scripts=all_scripts,
        active_groups=active_groups,
        augment=args.train_aug, augment_p=args.train_aug_p,
        two_views=use_two_views)
    val_dataset = LipiStreamingDataset(
        local=val_dir, active_scripts=all_scripts,
        active_groups=active_groups)  # val stays clean
    print(f"  Train: {len(train_dataset)}, Val (full): {len(val_dataset)}"
          f"{'  [train-time aug p=%.2f]' % args.train_aug_p if args.train_aug else ''}")

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

    # Script-balanced sampling weights (None → natural distribution)
    train_sample_weights = None
    if args.script_sample_beta >= 0:
        train_sample_weights = build_script_sample_weights(
            train_dir, args.script_sample_beta)

    eval_batch_size = min(args.batch_size, 128)
    val_loader = DataLoader(val_subset, batch_size=eval_batch_size, shuffle=True,
                            collate_fn=collate_moe,
                            pin_memory=(device_type == "cuda"))

    return {
        "train_dataset": train_dataset,
        "train_widths": train_widths,
        "train_sample_weights": train_sample_weights,
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
                         if not _is_expert_param(n)]
        expert_params = [p for n, p in model.named_parameters()
                         if _is_expert_param(n)]
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
        # With WidthSortedBatchSampler, shapes cluster per-bucket and
        # torch.compile handles dynamic shapes, so cudnn autotune pays off.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        print(f"AMP: {amp_dtype}")
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False)

    total_steps = steps_per_epoch * args.epochs
    # No LR warmup: with identity-init experts and small CTC head init,
    # early gradients are already well-behaved, and low-LR warmup just
    # parks CTC in the blank-collapse basin. AdamW's bias correction
    # handles moment initialization in its first ~100 steps internally.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
    print(f"LR schedule: cosine {total_steps} steps "
          f"({args.lr:.1e} → 1e-6), no warmup")

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
    model_state = ckpt["model"]

    # Optionally drop backbone keys so they keep fresh init
    if getattr(args, 'skip_backbone_load', False):
        backbone_prefixes = (
            "stem.", "convA.", "convB.", "blur_ab.", "blur_bc.",
            "swac_in_proj.", "swa_c.", "swa_d.",
            "merge_cd.", "merge_d1.",
            "group_head.", "group_h_pool.", "lid1_attn.",
            "lid2_heads.", "norm.",
        )
        dropped = [k for k in model_state if any(k.startswith(p) for p in backbone_prefixes)]
        for k in dropped:
            del model_state[k]
        print(f"  --skip-backbone-load: dropped {len(dropped)} backbone keys")

    # Skip mismatched shapes (e.g., CTC proj after vocab change)
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
    ema_state = ckpt.get("ema")
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

    return start_epoch, scheduler, ema_state


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, args, save_dir,
                    ema=None):
    ckpt_path = save_dir / f"moe_epoch{epoch}.pt"
    payload = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_config": model.config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, ckpt_path)
    print(f"  Saved: {ckpt_path}")



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, base_optimizer, scheduler, scaler,
                    ce_loss_fn, device, device_type, use_amp, amp_dtype,
                    epoch, total_epochs, grad_accum, log_interval,
                    ctc_weight, lid1_weight, lid2_weight,
                    group_script_vocabs, group_script_names,
                    detach_for_experts=False,
                    save_dir=None, args=None,
                    ema=None, ema_model=None,
                    inter_ctc_weight=0.0, route_sample_p=0.0,
                    consistency_weight=0.0):
    model.train()
    steps = len(train_loader)
    n_batches = 0
    oom_skipped = 0

    # Cache param split for grad clipping (avoid iterating named_parameters every step)
    shared_params = []
    expert_params = []
    for name, p in model.named_parameters():
        if _is_expert_param(name):
            expert_params.append(p)
        else:
            shared_params.append(p)

    # Accumulate losses and accuracy on GPU — avoid .item() sync every batch.
    # All *_accum / log_* tensors are 0-d GPU longs/floats; we sync to CPU
    # only at log interval / epoch end.
    ctc_loss_accum = torch.zeros(1, device=device)
    lid1_loss_accum = torch.zeros(1, device=device)
    total_loss_accum = torch.zeros(1, device=device)
    log_ctc = torch.zeros(1, device=device)
    log_ictc = torch.zeros(1, device=device)
    log_cons = torch.zeros(1, device=device)
    log_lid1 = torch.zeros(1, device=device)
    log_lid2 = torch.zeros(1, device=device)
    log_total = torch.zeros(1, device=device)
    log_count = 0
    log_lid1_correct = torch.zeros((), dtype=torch.long, device=device)
    log_lid1_total = torch.zeros((), dtype=torch.long, device=device)
    log_lid2_correct = torch.zeros((), dtype=torch.long, device=device)
    log_lid2_total = torch.zeros((), dtype=torch.long, device=device)
    log_time = time.time()
    shared_norm = 0.0
    expert_norm = 0.0

    def _forward_backward(imgs_, targets_, tgt_lens_, gids_, sids_, scale,
                          group_labels_=None, segments_=None,
                          imgs2_=None, aligned_=None):
        """Run forward + backward on a (sub-)batch. scale adjusts loss."""
        with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
            # Per-frame group labels: -100 = padding (loss ignores),
            # NUM_GROUPS = whitespace (learnable). For model routing,
            # both padding and whitespace should skip expert blocks.
            T_est = imgs_.shape[3] // 4
            # LID-1 CE target: keep -100 for padding so ignore_index works
            gl_frames = group_labels_[:, ::4][:, :T_est]
            # Model routing + CTC segment ranges: derived from segments so
            # both see identical frame boundaries.
            gl_for_model, sl_frames = build_frame_labels_from_segments(
                segments_, T_est, NUM_GROUPS, device)

            out = model(imgs_, group_ids=gl_for_model, script_ids=sl_frames,
                        detach_for_experts=detach_for_experts,
                        compute_ctc=(ctc_weight != 0),
                        inter_ctc=(inter_ctc_weight != 0 and ctc_weight != 0),
                        route_sample_p=route_sample_p)

            # Second view for consistency: GT routing, no scheduled
            # sampling — the reference posterior view 1 is pulled toward.
            out2 = None
            if consistency_weight != 0 and imgs2_ is not None:
                out2 = model(imgs2_, group_ids=gl_for_model,
                             script_ids=sl_frames,
                             detach_for_experts=detach_for_experts,
                             compute_ctc=(ctc_weight != 0))

        # LID-1 loss. Skip when weight is 0.
        T = out["group_logits"].shape[1]
        gl_frames = gl_frames[:, :T]
        if lid1_weight != 0:
            lid1_loss = compute_lid1_loss(
                out["group_logits"], gl_frames, ce_loss_fn)
        else:
            lid1_loss = torch.zeros(1, device=device)

        # CTC loss: per-segment for all lines (handles both single and mixed script)
        # Skip CTC loss computation entirely when its weight is 0
        # (LID-only pretraining) — the Python-heavy per-segment loop is
        # wasted work otherwise.
        if ctc_weight != 0:
            ctc_loss = compute_ctc_loss_segments(
                out["logits"], segments_, out["lengths"],
                group_script_names, group_script_vocabs)
        else:
            ctc_loss = torch.zeros(1, device=device)

        # Intermediate CTC on the pre-script-stack features (same segments)
        if "inter_logits" in out:
            inter_ctc_loss = compute_ctc_loss_segments(
                out["inter_logits"], segments_, out["lengths"],
                group_script_names, group_script_vocabs)
        else:
            inter_ctc_loss = torch.zeros(1, device=device)

        # LID-2 loss: per-frame CE within multi-script groups. Skip when
        # weight is 0 or when lid2_logits_per_group is empty.
        if lid2_weight != 0:
            lid2_loss = compute_lid2_loss(
                out.get("lid2_logits_per_group", {}),
                gl_for_model[:, :T], sl_frames[:, :T])
        else:
            lid2_loss = torch.zeros(1, device=device)

        # Two-view consistency: symmetric KL between the views' posteriors
        if out2 is not None:
            cons_loss = compute_consistency_loss(
                out["logits"] if ctc_weight != 0 else None,
                out2["logits"] if ctc_weight != 0 else None,
                out["group_logits"], out2["group_logits"],
                out2["flat_scripts"], gl_frames, aligned_,
                group_script_vocabs)
        else:
            cons_loss = torch.zeros(1, device=device)

        loss = (ctc_weight * ctc_loss
                + inter_ctc_weight * inter_ctc_loss
                + lid1_weight * lid1_loss.float()
                + lid2_weight * lid2_loss.float()
                + consistency_weight * cons_loss.float())

        loss = loss * scale
        if grad_accum > 1:
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        # Detach logits for logging — prevents autograd graph from leaking
        # Detach LID-2 logits for logging
        detached_lid2 = {
            g: lg.detach() for g, lg in out.get("lid2_logits_per_group", {}).items()
        }
        return (ctc_loss, inter_ctc_loss, lid1_loss, lid2_loss, cons_loss,
                loss, out["group_logits"].detach(), detached_lid2,
                gl_for_model.detach(), sl_frames.detach())

    for batch_idx, batch in enumerate(train_loader):
        if len(batch) == 10:  # two-view consistency batches
            (imgs, targets, tgt_lens, gids, sids, _labels, group_labels,
             segments, imgs2, aligned) = batch
        else:
            (imgs, targets, tgt_lens, gids, sids, _labels, group_labels,
             segments) = batch
            imgs2 = aligned = None
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
        if imgs2 is not None:
            imgs2 = imgs2.to(device, non_blocking=True)
            aligned = aligned.to(device, non_blocking=True)

        try:
            (ctc_loss, inter_ctc_loss, lid1_loss, lid2_loss, cons_loss,
             loss, group_logits, lid2_logits,
             gt_groups, gt_scripts) = \
                _forward_backward(imgs, targets, tgt_lens, gids, sids, scale=1.0,
                                  group_labels_=group_labels, segments_=segments,
                                  imgs2_=imgs2, aligned_=aligned)
        except torch.cuda.OutOfMemoryError:
            del imgs, targets, tgt_lens, gids, sids, group_labels, segments
            del imgs2, aligned
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
                # EMA tracks the uncompiled module (same tensors as the
                # compiled wrapper, but stable parameter names).
                if ema is not None:
                    ema.update(ema_model)

        # Accumulate on GPU — no sync
        mult = float(grad_accum)
        ctc_loss_accum += ctc_loss.detach()
        lid1_loss_accum += lid1_loss.detach()
        total_loss_accum += loss.detach() * mult
        log_ctc += ctc_loss.detach()
        log_ictc += inter_ctc_loss.detach()
        log_cons += cons_loss.detach()
        log_lid1 += lid1_loss.detach()
        log_lid2 += lid2_loss.detach()
        log_total += loss.detach() * mult
        n_batches += 1
        log_count += 1

        # Accumulate LID-1 and LID-2 accuracy on GPU — no per-batch sync.
        # Skip trackers whose loss weight is 0.
        with torch.no_grad():
            T_acc = group_logits.shape[1]

            if lid1_weight != 0:
                fp = group_logits.argmax(dim=-1)
                fl = group_labels[:, ::4][:, :T_acc].to(fp.device)
                non_pad = (fl >= 0)
                log_lid1_correct += ((fp == fl) & non_pad).sum()
                log_lid1_total += non_pad.sum()

            if lid2_weight != 0:
                gt_groups_T = gt_groups[:, :T_acc]
                gt_scripts_T = gt_scripts[:, :T_acc]
                for g_str, lid2_log in lid2_logits.items():
                    g = int(g_str)
                    g_mask = (gt_groups_T == g)
                    pred_s = lid2_log[:, :T_acc].argmax(dim=-1)
                    log_lid2_correct += ((pred_s == gt_scripts_T) & g_mask).sum()
                    log_lid2_total += g_mask.sum()

        if save_dir and n_batches % 500 == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, args, save_dir, ema=ema)

        if n_batches % log_interval == 0:
            # Batch all the per-interval counters into one GPU→CPU transfer.
            stats = torch.stack([
                log_ctc.squeeze(), log_ictc.squeeze(), log_cons.squeeze(),
                log_lid1.squeeze(), log_lid2.squeeze(), log_total.squeeze(),
                log_lid1_correct.float(), log_lid1_total.float(),
                log_lid2_correct.float(), log_lid2_total.float(),
            ]).tolist()
            (avg_ctc, avg_ictc, avg_cons, avg_lid1, avg_lid2, avg_total,
             lid1_c, lid1_t, lid2_c, lid2_t) = stats
            avg_ctc /= log_count
            avg_ictc /= log_count
            avg_cons /= log_count
            avg_lid1 /= log_count
            avg_lid2 /= log_count
            avg_total /= log_count
            lid1_acc = 100 * lid1_c / max(lid1_t, 1)
            lid2_acc = 100 * lid2_c / max(lid2_t, 1)
            elapsed = time.time() - log_time
            ms_per_step = elapsed / log_count * 1000
            samples_per_sec = imgs.shape[0] * log_count / elapsed
            batch_str = f"{batch_idx+1}/{steps}"
            ictc_str = f"ictc {avg_ictc:.4f}  " if inter_ctc_weight != 0 else ""
            cons_str = f"cons {avg_cons:.4f}  " if consistency_weight != 0 else ""
            print(
                f"  [{epoch:>2}/{total_epochs}] {batch_str:>9}  "
                f"loss {avg_total:.4f}  "
                f"ctc {avg_ctc:.4f}  {ictc_str}{cons_str}"
                f"lid1 {avg_lid1:.4f}  lid2 {avg_lid2:.4f}  "
                f"| acc {lid1_acc:5.1f}% {lid2_acc:5.1f}%  "
                f"| {samples_per_sec:.0f} img/s  {ms_per_step:.0f}ms/step  "
                f"gnorm {shared_norm:.1f}/{expert_norm:.1f}"
            )
            log_time = time.time()
            log_ctc.zero_()
            log_ictc.zero_()
            log_cons.zero_()
            log_lid1.zero_()
            log_lid2.zero_()
            log_total.zero_()
            log_count = 0
            log_lid1_correct.zero_()
            log_lid1_total.zero_()
            log_lid2_correct.zero_()
            log_lid2_total.zero_()

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
        components = set(c.strip() for c in args.freeze_except.split(","))
        valid = {"experts", "ctc", "backbone", "lid", "lid1", "lid2"}
        bad = components - valid
        assert not bad, f"Unknown --freeze-except components: {bad}. Valid: {valid}"

        # Build set of prefixes to unfreeze. `lid` is an alias for both
        # heads; `lid1` and `lid2` target individual heads. lid1_attn
        # rides with lid1 since it's a pre-classifier attention block
        # dedicated to LID-1 script-family extraction.
        unfreeze_prefixes = []
        if "lid" in components or "lid1" in components:
            unfreeze_prefixes.extend(["lid1_attn.", "group_head."])
        if "lid" in components or "lid2" in components:
            unfreeze_prefixes.append("lid2_heads.")
        if "ctc" in components:
            unfreeze_prefixes.append("ctc_modules.")
        if "experts" in components:
            unfreeze_prefixes.extend([
                k for k in EXPERT_PARAM_PREFIXES if k != "ctc_modules."])
        if "backbone" in components:
            unfreeze_prefixes.extend([
                "stem.", "convA.", "convB.", "blur_ab.", "blur_bc.",
                "swac_in_proj.", "swa_c.", "swa_d.",
                "merge_cd.", "merge_d1.", "norm."])

        frozen = 0
        trainable = 0
        for name, param in model.named_parameters():
            param.requires_grad = any(p in name for p in unfreeze_prefixes)
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()
        print(f"Freeze mode: training {args.freeze_except} only")
        print(f"  Trainable: {trainable/1e6:.1f}M, Frozen: {frozen/1e6:.1f}M")

    # Build train_loader with VRAM-estimated pixel budget

    train_widths = data["train_widths"]
    max_width = int(train_widths.max())

    # Match the VRAM calibration to the actual training forward: when
    # ctc_weight=0 the CTC path is skipped, so the activation footprint
    # is smaller.
    pixel_budget = estimate_pixel_budget(
        model, vram_gb=args.vram, compute_ctc=(args.ctc_weight != 0),
        safety=args.vram_safety)
    if args.consistency_weight > 0 and args.train_aug:
        # Two-view consistency runs a second forward per batch
        pixel_budget //= 2
    max_batch_at_widest = pixel_budget // max_width
    # Cap at --batch-size: pixel budget handles width scaling, but there's
    # per-sample overhead (autograd nodes, CTC loss, routing) that doesn't
    # scale with width. --batch-size caps the max samples in any batch.
    max_batch_size = min(args.batch_size, max(max_batch_at_widest, 1))

    train_batch_sampler = WidthSortedBatchSampler(
        train_widths, max_batch_size, max_width=0,
        pixel_budget=pixel_budget,
        sample_weights=data["train_sample_weights"])
    batch_sizes = [len(b) for b in train_batch_sampler._batches]
    print(f"  Batching: {len(train_batch_sampler)} batches, "
          f"size {min(batch_sizes)}-{max(batch_sizes)} "
          f"(max={max_batch_size}, budget={pixel_budget}px)")

    train_loader = DataLoader(data["train_dataset"], batch_sampler=train_batch_sampler,
                              collate_fn=collate_moe,
                              num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None,
                              pin_memory=(device_type == "cuda"))

    steps_per_epoch = len(train_loader)
    opt = build_optimizer_and_scheduler(args, model, device_type, steps_per_epoch)

    # NOTE: torch.compile moved AFTER optimizer creation + resume.
    # Compiling before optimizer changes parameter structure, breaking
    # optimizer state dict loading from non-compiled checkpoints.

    start_epoch = 1
    ema_state = None
    if args.resume:
        start_epoch, opt["scheduler"], ema_state = resume_from_checkpoint(
            args, model, opt["optimizer"], opt["base_optimizer"], opt["scaler"],
            opt["scheduler"], steps_per_epoch, device_type)

    # Weight EMA: snapshot after resume so a fresh EMA starts from the
    # loaded weights, and before compile so parameter names are stable.
    ema = None
    base_model = model
    if args.ema_decay > 0:
        ema = ModelEMA(base_model, decay=args.ema_decay)
        if ema_state is not None:
            ema.load_state_dict(ema_state)
            print(f"  Resumed EMA state ({ema.updates} updates)")
        print(f"EMA: decay={args.ema_decay} (eval uses averaged weights)")

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
    print(f"  Losses: CTC x{args.ctc_weight} + interCTC x{args.inter_ctc_weight} "
          f"+ LID1 x{args.lid1_weight} + LID2 x{args.lid2_weight}"
          + (f" + consistency x{args.consistency_weight}"
             if args.consistency_weight > 0 else ""))
    print(f"  Routing: ground truth + scheduled sampling "
          f"(0 → {args.route_sample_max} over training)")
    print(f"  Per-script vocabs: {data['group_script_vocab_sizes']}")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        detach = epoch <= args.detach_epochs
        if detach:
            print(f"  [detach mode: CTC gradient stops at expert boundary, epoch {epoch}/{args.detach_epochs}]")
        # Scheduled routing sampling: linear ramp 0 → max over training
        route_p = args.route_sample_max * (epoch - 1) / max(args.epochs - 1, 1)
        if route_p > 0:
            print(f"  [scheduled routing sampling: p={route_p:.3f}]")
        metrics = train_one_epoch(
            model, train_loader, opt["optimizer"], opt["base_optimizer"],
            opt["scheduler"], opt["scaler"], ce_loss_fn, device, device_type,
            opt["use_amp"], opt["amp_dtype"], epoch, args.epochs, args.grad_accum,
            args.log_interval,
            ctc_weight=args.ctc_weight,
            lid1_weight=args.lid1_weight, lid2_weight=args.lid2_weight,
            group_script_vocabs=data["group_script_vocab_sizes"],
            group_script_names=data["group_script_names"],
            detach_for_experts=detach,
            save_dir=save_dir, args=args,
            ema=ema, ema_model=base_model,
            inter_ctc_weight=args.inter_ctc_weight,
            route_sample_p=route_p,
            consistency_weight=args.consistency_weight)

        elapsed = time.time() - t0
        if metrics:
            print(f"\nEpoch {epoch}/{args.epochs}: "
                  f"ctc={metrics['ctc']:.4f} lid1={metrics['lid1']:.4f}  "
                  f"time={elapsed:.0f}s")

        save_checkpoint(model, opt["optimizer"], opt["scheduler"], opt["scaler"],
                        epoch, args, save_dir, ema=ema)

        # Move optimizer state to CPU to free VRAM for eval
        optimizer = opt["optimizer"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        torch.cuda.empty_cache()

        print(f"\n  Eval epoch {epoch}:"
              + (" (EMA weights)" if ema is not None else ""))
        if ema is not None:
            # Swap averaged weights into the (shared) parameter tensors;
            # the compiled wrapper sees them too.
            with ema.average_parameters(base_model):
                evaluate(model, data["val_loader"], data["group_tokenizers"],
                         data["group_script_names"], data["active_groups"],
                         device, device_type, opt["use_amp"], opt["amp_dtype"],
                         group_script_vocab_sizes=data["group_script_vocab_sizes"])
        else:
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
