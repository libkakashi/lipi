#!/usr/bin/env python3
"""Diagnose GPU memory usage."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import gc

print("=" * 60)
print("GPU MEMORY DIAGNOSTIC")
print("=" * 60)

if not torch.cuda.is_available():
    print("No CUDA available")
    exit()

device = torch.cuda.current_device()
print(f"Device: {torch.cuda.get_device_name(device)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

# Clear any leftover allocations
gc.collect()
torch.cuda.empty_cache()

allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"After cache clear — Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Test 1: Can we allocate model?
print("\n--- Test 1: Model allocation ---")
from src.model.encoder import LipiEncoder
from src.training.foundation_trainer import CTCHead

encoder = LipiEncoder().cuda()
ctc_head = CTCHead(384, 96).cuda()
allocated = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded: {allocated:.2f} GB")

# Test 2: Optimizer
print("\n--- Test 2: Optimizer ---")
params = list(encoder.parameters()) + list(ctc_head.parameters())
optimizer = torch.optim.AdamW(params, lr=3e-4)
# Optimizer doesn't allocate until first step

# Test 3: Forward pass at different batch sizes
print("\n--- Test 3: Forward pass ---")
encoder.train()
ctc_head.train()

for bs in [32, 64, 128, 256, 400, 600, 800]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        x = torch.randn(bs, 3, 32, 128, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            features, enc_lengths = encoder(x)
            logits = ctc_head(features)

        # Fake CTC loss backward
        loss = logits.sum()
        loss.backward()

        peak = torch.cuda.max_memory_allocated() / 1e9
        current = torch.cuda.memory_allocated() / 1e9
        print(f"  B={bs:4d}: peak={peak:.2f} GB, current={current:.2f} GB ✓")

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        del x, features, logits, loss
    except torch.cuda.OutOfMemoryError:
        print(f"  B={bs:4d}: OOM ✗")
        torch.cuda.empty_cache()
        break

# Test 4: CPU loader (PARSeqLMDB with preload_raw)
print("\n--- Test 4: CPU loader with preload_raw ---")
gc.collect()
torch.cuda.empty_cache()

import psutil
ram_before = psutil.Process().memory_info().rss / 1e9
gpu_before = torch.cuda.memory_allocated() / 1e9

from src.data.parseq_lmdb import PARSeqLMDB
from src.data.dataset import collate_ocr
from torch.utils.data import DataLoader

dataset = PARSeqLMDB("training_data/external/train/synth/MJ/train", augment=True)
print(f"  Dataset: {len(dataset)} samples")

dataset.preload_raw(max_samples=100000)
ram_after_preload = psutil.Process().memory_info().rss / 1e9
print(f"  RAM after preload: {ram_before:.1f} -> {ram_after_preload:.1f} GB (+{ram_after_preload-ram_before:.1f} GB)")

loader = DataLoader(
    dataset, batch_size=600, shuffle=True, collate_fn=collate_ocr,
    drop_last=True, num_workers=16, pin_memory=True, persistent_workers=True,
)
ram_after_loader = psutil.Process().memory_info().rss / 1e9
print(f"  RAM after DataLoader init: {ram_after_loader:.1f} GB")

# Get a few batches and check GPU
print("  Loading batches...")
for i, (imgs, labels, widths) in enumerate(loader):
    imgs = imgs.cuda(non_blocking=True)

    gpu_now = torch.cuda.memory_allocated() / 1e9
    ram_now = psutil.Process().memory_info().rss / 1e9
    print(f"    Batch {i}: GPU={gpu_now:.2f} GB, RAM={ram_now:.1f} GB, img_shape={imgs.shape}")

    # Do a forward+backward like real training
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        features, enc_lengths = encoder(imgs)
        logits = ctc_head(features)
    loss = logits.float().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    gpu_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"           GPU peak={gpu_peak:.2f} GB")

    del imgs, features, logits, loss
    torch.cuda.empty_cache()

    if i >= 4:
        break

ram_final = psutil.Process().memory_info().rss / 1e9
gpu_final = torch.cuda.memory_allocated() / 1e9
print(f"\n  Final: GPU={gpu_final:.2f} GB, RAM={ram_final:.1f} GB")

# Check per-worker memory
print(f"\n  Note: 16 persistent workers each fork the process.")
print(f"  If preload_raw holds 3M images * ~5KB = ~15GB in the parent,")
print(f"  each fork copies that on write. Check system RAM usage.")
