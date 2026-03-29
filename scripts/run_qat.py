#!/usr/bin/env python3
"""
Phase 3: Quantization-Aware Training.

Quantize backbone to NVFP4 (Blackwell) or FP8 (Hopper/Ampere),
keep GRU at FP8, fine-tune to recover accuracy.

Usage:
    python scripts/run_qat.py \
        --backbone checkpoints/phase1/phase1_best.pt \
        --adapter checkpoints/phase2/en/adapter_best.pt \
        --train_dir training_data/datasets/en_synth \
        --test_dir training_data/external/test \
        --epochs 5
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.lora import inject_lora
from src.model.decode import greedy_decode
from src.training.loss import rnnt_loss
from src.training.foundation_trainer import FoundationTrainer
from src.quantization.qat import apply_qat_config
from src.quantization.polar import apply_polar_rotation
from src.data.bigrams import LipiTokenizer
from src.data.parseq_lmdb import PARSeqLMDB, discover_parseq_structure
from src.data.dataset import collate_ocr


def evaluate(encoder, pred_net, joint_net, tokenizer, test_dir, device):
    """Quick eval on standard benchmarks."""
    encoder.eval()
    pred_net.eval()
    joint_net.eval()

    benchmarks = discover_parseq_structure(test_dir)
    standard = ["IIIT5k", "SVT", "IC13_1015", "IC15_2077", "SVTP", "CUTE80"]
    results = {}

    for name in standard:
        path = benchmarks.get(name)
        if not path:
            continue

        dataset = PARSeqLMDB(path)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_ocr)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_imgs, batch_labels, widths in loader:
                batch_imgs = batch_imgs.to(device)
                features, _ = encoder(batch_imgs)
                decoded_ids = greedy_decode(features, pred_net, joint_net, max_tokens=25)
                for pred_ids, label in zip(decoded_ids, batch_labels):
                    if tokenizer.decode(pred_ids).lower() == label.lower():
                        correct += 1
                    total += 1

        acc = correct / max(total, 1) * 100
        results[name] = acc
        dataset.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3: QAT")
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--adapter", type=str, help="Optional adapter checkpoint")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--vocab", type=str, help="Vocabulary JSON")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--apply_polar", action="store_true", help="Apply PolarQuant before QAT")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phase3")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Detect hardware capability
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU: {gpu_name}, compute capability {capability}")
        if capability[0] >= 10:
            print("  Blackwell detected — NVFP4 quantization available")
            quant_mode = "nvfp4"
        elif capability[0] >= 9:
            print("  Hopper detected — FP8 quantization")
            quant_mode = "fp8"
        else:
            print("  Ampere/older — FP8 simulation")
            quant_mode = "fp8"
    else:
        quant_mode = "simulated"
        print("  CPU — simulated quantization only")

    # Load backbone
    print(f"\nLoading backbone from {args.backbone}...")
    encoder = FoundationTrainer.load_backbone(args.backbone, device="cpu")

    # Load tokenizer
    if args.vocab and Path(args.vocab).exists():
        tokenizer = LipiTokenizer.load(args.vocab)
    else:
        tokenizer = LipiTokenizer.build_character_level("en")

    # Load adapter if provided
    if args.adapter and Path(args.adapter).exists():
        print(f"Loading adapter from {args.adapter}...")
        encoder = inject_lora(encoder)
        state = torch.load(args.adapter, map_location="cpu", weights_only=True)
        lora_state = state.get("lora", {})
        model_state = encoder.state_dict()
        model_state.update(lora_state)
        encoder.load_state_dict(model_state, strict=False)

        pred_net = PredictionNetwork(vocab_size=tokenizer.vocab_size)
        pred_net.load_state_dict(state["pred_net"])
        joint_net = JointNetwork(vocab_size=tokenizer.vocab_size)
        joint_net.load_state_dict(state["joint_net"])
    else:
        pred_net = PredictionNetwork(vocab_size=tokenizer.vocab_size)
        joint_net = JointNetwork(vocab_size=tokenizer.vocab_size)

    # Step 1: PolarQuant rotation (optional)
    if args.apply_polar:
        print("\nApplying PolarQuant rotation...")
        encoder, rotations = apply_polar_rotation(encoder)
        print(f"  Rotated {len(rotations)} modules")

    # Step 2: Apply QAT config
    print(f"\nApplying QAT config (mode: {quant_mode})...")
    encoder = apply_qat_config(encoder)

    # Move to device
    encoder = encoder.to(device)
    pred_net = pred_net.to(device)
    joint_net = joint_net.to(device)

    # Pre-QAT evaluation
    print("\nPre-QAT Evaluation:")
    pre_results = evaluate(encoder, pred_net, joint_net, tokenizer, args.test_dir, device)
    for name, acc in sorted(pre_results.items()):
        print(f"  {name}: {acc:.1f}%")

    # Training data
    train_dataset = PARSeqLMDB(args.train_dir)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_ocr, drop_last=True, num_workers=4, pin_memory=True,
    )

    # QAT fine-tuning
    # Unfreeze all params for QAT (even previously frozen backbone)
    for p in encoder.parameters():
        p.requires_grad = True

    all_params = (
        list(encoder.parameters()) +
        list(pred_net.parameters()) +
        list(joint_net.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nQAT Training: {args.epochs} epochs, lr={args.lr}")

    encoder.train()
    pred_net.train()
    joint_net.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(loader, desc=f"QAT Epoch {epoch}/{args.epochs}")
        for batch_imgs, batch_labels, widths in pbar:
            batch_imgs = batch_imgs.to(device, non_blocking=True)

            target_ids = [tokenizer.encode(l) for l in batch_labels]
            target_lengths = torch.tensor([len(ids) for ids in target_ids], device=device)
            if (target_lengths == 0).any():
                continue

            max_tgt = target_lengths.max().item()
            targets = torch.zeros(len(batch_labels), max_tgt, dtype=torch.long, device=device)
            for i, ids in enumerate(target_ids):
                if len(ids) > 0:
                    targets[i, :len(ids)] = torch.tensor(ids, device=device)

            features, _ = encoder(batch_imgs)
            B, T = features.shape[0], features.shape[1]
            enc_lengths = torch.full((B,), T, dtype=torch.long, device=device)

            blank = torch.zeros(B, 1, dtype=torch.long, device=device)
            pred_input = torch.cat([blank, targets], dim=1)
            pred_out, _ = pred_net(pred_input)

            logits = joint_net(features.unsqueeze(2), pred_out.unsqueeze(1))
            loss = rnnt_loss(logits, targets, enc_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"QAT Epoch {epoch}: avg_loss={avg_loss:.4f}")

        # Evaluate
        post_results = evaluate(encoder, pred_net, joint_net, tokenizer, args.test_dir, device)
        for name, acc in sorted(post_results.items()):
            pre = pre_results.get(name, 0)
            delta = acc - pre
            print(f"  {name}: {acc:.1f}% (delta: {delta:+.1f}%)")

    # Save
    torch.save({
        "encoder": encoder.state_dict(),
        "pred_net": pred_net.state_dict(),
        "joint_net": joint_net.state_dict(),
        "quant_mode": quant_mode,
    }, save_dir / "qat_final.pt")

    print(f"\nQAT complete. Saved to {save_dir / 'qat_final.pt'}")

    # Accuracy drop summary
    print(f"\nAccuracy drop from quantization (after QAT):")
    for name in sorted(pre_results.keys()):
        pre = pre_results[name]
        post = post_results.get(name, 0)
        drop = pre - post
        status = "OK" if drop < 1.5 else "WARN"
        print(f"  [{status}] {name}: {pre:.1f}% -> {post:.1f}% (drop: {drop:.1f}%)")


if __name__ == "__main__":
    main()
