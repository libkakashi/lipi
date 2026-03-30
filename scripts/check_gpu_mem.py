#!/usr/bin/env python3
"""Diagnose GPU memory usage."""

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
print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_mem / 1e9:.1f} GB")

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

# Test 4: With DALI loaded
print("\n--- Test 4: DALI pipeline memory ---")
gc.collect()
torch.cuda.empty_cache()

before = torch.cuda.memory_allocated() / 1e9
try:
    from src.data.dali_pipeline import DALIOCRLoader
    loader = DALIOCRLoader(
        "training_data/external/train/synth/MJ/train",
        batch_size=256, max_samples=100000, device_id=0,
    )
    after_init = torch.cuda.memory_allocated() / 1e9
    print(f"  DALI init: {before:.2f} -> {after_init:.2f} GB (+{after_init-before:.2f} GB)")

    # Get one batch
    it = iter(loader)
    batch = next(it)
    after_batch = torch.cuda.memory_allocated() / 1e9
    print(f"  After 1 batch: {after_batch:.2f} GB (+{after_batch-after_init:.2f} GB)")
    del batch, it, loader
except Exception as e:
    print(f"  DALI error: {e}")

gc.collect()
torch.cuda.empty_cache()
final = torch.cuda.memory_allocated() / 1e9
print(f"\n  Final after cleanup: {final:.2f} GB")
