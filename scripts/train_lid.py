#!/usr/bin/env python3
"""
Train the Micro-LID script classifier.

Uses script-labeled word crops (synthetic or real).
Simple cross-entropy classification — 11 classes with distinct visual features.

Usage:
    python scripts/train_lid.py --data_dir training_data/datasets/lid_crops --epochs 20
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lid import MicroLID, SCRIPT_NAMES, NUM_SCRIPTS
from src.data.dataset import preprocess_crop


class LIDDataset(Dataset):
    """Dataset for LID training: (image, script_id) pairs."""

    def __init__(self, data_dir: str, target_height: int = 32):
        self.data_dir = Path(data_dir)
        self.target_height = target_height
        self.samples = []  # [(image_path, script_idx)]

        for script_idx, script_name in enumerate(SCRIPT_NAMES):
            script_dir = self.data_dir / script_name
            if script_dir.exists():
                for img_path in script_dir.glob("*.png"):
                    self.samples.append((img_path, script_idx))
                for img_path in script_dir.glob("*.jpg"):
                    self.samples.append((img_path, script_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, script_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        crop = preprocess_crop(img, self.target_height)
        return crop, script_idx


def lid_collate(batch):
    """Collate LID samples with padding."""
    images, labels = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    B = len(images)

    padded = torch.zeros(B, 3, images[0].shape[1], max_w, dtype=torch.float32)
    for i, img in enumerate(images):
        import numpy as np
        t = torch.from_numpy(img) if isinstance(img, np.ndarray) else img
        padded[i, :, :, :t.shape[2]] = t

    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels


def main():
    parser = argparse.ArgumentParser(description="Train Micro-LID classifier")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_path", type=str, default="checkpoints/lid.pt")
    args = parser.parse_args()

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

    model = MicroLID().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"LID params: {params}")

    dataset = LIDDataset(args.data_dir)
    if len(dataset) == 0:
        print(f"ERROR: No data found in {args.data_dir}")
        print(f"Expected subdirectories: {SCRIPT_NAMES}")
        sys.exit(1)

    print(f"Dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lid_collate, num_workers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]

        acc = correct / max(total, 1) * 100
        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.1f}%")

    # Save
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved to: {args.save_path}")


if __name__ == "__main__":
    main()
