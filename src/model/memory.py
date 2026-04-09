"""VRAM budget estimation for LipiMoEEncoder.

Estimates the maximum pixel budget (B*W) that fits in GPU memory
by running a calibration forward+backward pass on a small batch.
"""

import torch
import torch.nn as nn


def estimate_pixel_budget(model, vram_gb: float = 32) -> int:
    """Estimate max pixel budget (B*W) by measuring actual VRAM usage.

    Runs a small calibration batch to measure real bytes-per-pixel,
    then extrapolates to the available VRAM.
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        # Can't measure — return a conservative default
        print("  VRAM estimate: non-CUDA device, using default budget")
        return 50000

    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    fixed = model_bytes * 4  # params + grads + adam m + adam v

    # Calibration: run a small batch and measure peak VRAM
    cal_B, cal_H, cal_W = 16, 32, 128
    cal_pixels = cal_B * cal_W

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)

    # Build calibration inputs
    num_groups = model.num_groups
    cal_imgs = torch.randn(cal_B, 2, cal_H, cal_W, device=device, dtype=torch.float32)
    cal_gids = torch.randint(0, num_groups, (cal_B,), device=device)
    cal_sids = torch.zeros(cal_B, dtype=torch.long, device=device)

    # Forward pass with autocast (matches training)
    was_training = model.training
    model.train()
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(cal_imgs, group_ids=cal_gids, script_ids=cal_sids)
            # Simulate loss backward to capture full memory footprint
            dummy_loss = out["logits"].sum() + out["group_logits"].sum()
        dummy_loss.backward()
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()

    peak = torch.cuda.max_memory_allocated(device)
    activation_bytes = peak - baseline

    # Clean up
    del cal_imgs, cal_gids, cal_sids, out, dummy_loss
    torch.cuda.empty_cache()

    bytes_per_pixel = activation_bytes / cal_pixels
    available = vram_gb * 1e9 - fixed
    # Reserve 25% for fragmentation and other overhead
    pixel_budget = int(available * 0.75 / bytes_per_pixel)

    print(f"  VRAM estimate: {model_bytes/1e9:.2f}GB model, "
          f"{fixed/1e9:.2f}GB fixed, "
          f"{bytes_per_pixel/1e3:.0f}KB/px (calibrated), "
          f"{available/1e9:.1f}GB for activations")
    print(f"  Pixel budget: {pixel_budget}")

    return pixel_budget
