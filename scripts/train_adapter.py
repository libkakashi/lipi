#!/usr/bin/env python3
"""
Phase 2: Train LoRA adapter + RNN-T head for a specific language.

Usage:
    python scripts/train_adapter.py --backbone checkpoints/phase1/phase1_best.pt \
        --lang hi --data_dir training_data/datasets/hindi --vocab training_data/vocabs/hi_2k.json
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
from src.data.bigrams import LipiTokenizer
from src.data.dataset import LMDBDataset, collate_ocr
from src.training.adapter_trainer import AdapterTrainer
from src.training.foundation_trainer import FoundationTrainer


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Adapter training")
    parser.add_argument("--backbone", type=str, required=True, help="Phase 1 backbone checkpoint")
    parser.add_argument("--lang", type=str, required=True, help="Language/script ID")
    parser.add_argument("--data_dir", type=str, required=True, help="LMDB dataset directory")
    parser.add_argument("--vocab", type=str, help="Vocabulary JSON path")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lora_lr", type=float, default=1e-3)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"checkpoints/phase2/{args.lang}"

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Language: {args.lang}")
    print(f"Device: {device}")
    print(f"LoRA rank: {args.lora_rank}")

    # Load backbone
    print(f"Loading backbone from: {args.backbone}")
    backbone = FoundationTrainer.load_backbone(args.backbone, device="cpu")

    # Tokenizer
    if args.vocab and Path(args.vocab).exists():
        tokenizer = LipiTokenizer.load(args.vocab)
    else:
        print(f"No vocab found, building character-level for {args.lang}")
        tokenizer = LipiTokenizer.build_character_level(args.lang)

    print(f"Vocab size: {tokenizer.vocab_size}")

    # Build trainer
    trainer = AdapterTrainer(
        backbone=backbone,
        tokenizer=tokenizer,
        lora_rank=args.lora_rank,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        device=device,
    )

    lora_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in trainer.pred_net.parameters()) + \
                  sum(p.numel() for p in trainer.joint_net.parameters())
    print(f"LoRA params: {lora_params/1e6:.3f}M")
    print(f"Head params: {head_params/1e6:.3f}M")

    # Data
    dataset = LMDBDataset(args.data_dir, augment=True)
    print(f"Dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ocr,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Train
    history = trainer.train(
        train_loader=loader,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )

    print(f"\nTraining complete for {args.lang}")
    print(f"Adapter saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
