#!/usr/bin/env python3
"""
Phase 1: Train backbone with CTC head.

Usage:
    python scripts/train_foundation.py --config configs/training/phase1_foundation.yaml
    python scripts/train_foundation.py --data_dir training_data/datasets/synth --epochs 5 --batch_size 64
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
from src.training.foundation_trainer import FoundationTrainer


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Foundation training")
    parser.add_argument("--config", type=str, help="YAML config")
    parser.add_argument("--data_dir", type=str, help="LMDB dataset directory")
    parser.add_argument("--vocab", type=str, help="Vocabulary JSON path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")

    # Build or load tokenizer
    if args.vocab and Path(args.vocab).exists():
        tokenizer = LipiTokenizer.load(args.vocab)
    else:
        print("No vocab specified, using character-level English")
        tokenizer = LipiTokenizer.build_character_level("en")

    print(f"Vocab size: {tokenizer.vocab_size}")

    # Build model
    encoder = LipiEncoder()
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {total_params/1e6:.2f}M")

    # Build trainer
    trainer = FoundationTrainer(
        encoder=encoder,
        tokenizer=tokenizer,
        lr=args.lr,
        device=device,
    )

    # Data
    if args.data_dir and Path(args.data_dir).exists():
        dataset = LMDBDataset(args.data_dir, augment=True)
        print(f"Dataset: {len(dataset)} samples from {args.data_dir}")
    else:
        print("ERROR: No valid data directory specified.")
        print("Generate data first with: python scripts/generate_synth.py")
        sys.exit(1)

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

    print(f"\nTraining complete. Best loss: {min(history['epoch_loss']):.4f}")
    print(f"Backbone saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
