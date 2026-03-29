#!/usr/bin/env python3
"""
Evaluate model on STR benchmarks.

Tests on standard benchmarks: IIIT5K, SVT, IC13, IC15, SVTP, CUTE80.
Reports per-benchmark and average word accuracy.

Usage:
    python scripts/benchmark.py \
        --backbone checkpoints/phase1/phase1_best.pt \
        --adapter checkpoints/phase2/en/adapter_best.pt \
        --vocab training_data/vocabs/en_2k.json \
        --data_dir training_data/datasets/benchmarks
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.encoder import LipiEncoder
from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.lora import inject_lora
from src.model.decode import greedy_decode
from src.data.bigrams import LipiTokenizer
from src.data.dataset import LMDBDataset, collate_ocr
from src.training.foundation_trainer import FoundationTrainer, CTCHead
from src.training.loss import ctc_loss


def evaluate_ctc(encoder, ctc_head, tokenizer, data_loader, device):
    """Evaluate with CTC decoding."""
    encoder.eval()
    ctc_head.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_imgs, batch_labels, widths in data_loader:
            batch_imgs = batch_imgs.to(device)
            features, enc_lengths = encoder(batch_imgs)
            logits = ctc_head(features)
            preds = logits.argmax(dim=-1)

            for i, label in enumerate(batch_labels):
                p = preds[i].tolist()
                collapsed = [p[0]] + [p[j] for j in range(1, len(p)) if p[j] != p[j-1]]
                collapsed = [x for x in collapsed if x != 0]
                decoded = tokenizer.decode(collapsed)
                if decoded.lower() == label.lower():
                    correct += 1
                total += 1

    return correct, total


def evaluate_rnnt(encoder, pred_net, joint_net, tokenizer, data_loader, device):
    """Evaluate with RNN-T greedy decoding."""
    encoder.eval()
    pred_net.eval()
    joint_net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_imgs, batch_labels, widths in data_loader:
            batch_imgs = batch_imgs.to(device)
            features, _ = encoder(batch_imgs)
            decoded_ids = greedy_decode(features, pred_net, joint_net, max_tokens=25)

            for pred_ids, label in zip(decoded_ids, batch_labels):
                decoded = tokenizer.decode(pred_ids)
                if decoded.lower() == label.lower():
                    correct += 1
                total += 1

    return correct, total


def main():
    parser = argparse.ArgumentParser(description="Evaluate on STR benchmarks")
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--adapter", type=str, help="Phase 2 adapter checkpoint")
    parser.add_argument("--vocab", type=str, help="Vocabulary JSON")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mode", choices=["ctc", "rnnt"], default="rnnt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Load tokenizer
    if args.vocab and Path(args.vocab).exists():
        tokenizer = LipiTokenizer.load(args.vocab)
    else:
        tokenizer = LipiTokenizer.build_character_level("en")

    # Load model
    encoder = FoundationTrainer.load_backbone(args.backbone, device="cpu")

    if args.adapter and Path(args.adapter).exists() and args.mode == "rnnt":
        # Phase 2: LoRA + RNN-T
        encoder_lora = inject_lora(encoder)
        adapter_state = torch.load(args.adapter, map_location="cpu", weights_only=True)

        # Load LoRA weights
        lora_state = adapter_state.get("lora", {})
        model_state = encoder_lora.state_dict()
        model_state.update(lora_state)
        encoder_lora.load_state_dict(model_state, strict=False)
        encoder_lora = encoder_lora.to(device)

        pred_net = PredictionNetwork(vocab_size=tokenizer.vocab_size)
        pred_net.load_state_dict(adapter_state["pred_net"])
        pred_net = pred_net.to(device)

        joint_net = JointNetwork(vocab_size=tokenizer.vocab_size)
        joint_net.load_state_dict(adapter_state["joint_net"])
        joint_net = joint_net.to(device)

        eval_fn = lambda loader: evaluate_rnnt(
            encoder_lora, pred_net, joint_net, tokenizer, loader, device
        )
    else:
        # Phase 1: CTC
        encoder = encoder.to(device)
        ctc_head = CTCHead(encoder.output_dim, tokenizer.vocab_size).to(device)

        if args.adapter and Path(args.adapter).exists():
            state = torch.load(args.adapter, map_location="cpu", weights_only=True)
            if "ctc_head" in state:
                ctc_head.load_state_dict(state["ctc_head"])

        eval_fn = lambda loader: evaluate_ctc(
            encoder, ctc_head, tokenizer, loader, device
        )

    # Evaluate on each benchmark
    data_dir = Path(args.data_dir)
    benchmarks = ["IIIT5k_3000", "SVT", "IC13_1015", "IC15_2077", "SVTP", "CUTE80"]
    results = {}

    print(f"\n{'='*60}")
    print(f"{'Benchmark':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'='*60}")

    total_correct = 0
    total_total = 0

    for bench in benchmarks:
        bench_path = data_dir / bench
        if not bench_path.exists():
            print(f"{bench:<20} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue

        dataset = LMDBDataset(bench_path)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ocr,
        )

        correct, total = eval_fn(loader)
        acc = correct / max(total, 1) * 100
        results[bench] = acc

        print(f"{bench:<20} {correct:>8} {total:>8} {acc:>9.1f}%")
        total_correct += correct
        total_total += total

    if total_total > 0:
        avg = total_correct / total_total * 100
        print(f"{'='*60}")
        print(f"{'AVERAGE':<20} {total_correct:>8} {total_total:>8} {avg:>9.1f}%")


if __name__ == "__main__":
    main()
