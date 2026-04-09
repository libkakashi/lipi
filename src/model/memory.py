"""VRAM budget estimation for LipiMoEEncoder.

Estimates the maximum pixel budget (B*W) that fits in GPU memory
by running two calibration batches at different sizes and measuring
actual VRAM to derive per-pixel cost.
"""

import torch


def estimate_pixel_budget(model, vram_gb: float = 32) -> int:
    """Estimate max pixel budget (B*W) by measuring actual VRAM usage.

    Runs two calibration batches (small and large) to separate fixed
    overhead from per-pixel cost, then extrapolates to available VRAM.
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        print("  VRAM estimate: non-CUDA device, using default budget")
        return 50000

    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    fixed_model = model_bytes * 4  # params + grads + adam m + adam v
    num_groups = model.num_groups

    def _measure(B, W):
        """Run forward+backward and return peak activation bytes."""
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)

        imgs = torch.randn(B, 2, 32, W, device=device, dtype=torch.float32)
        gids = torch.randint(0, num_groups, (B,), device=device)
        sids = torch.zeros(B, dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(imgs, group_ids=gids, script_ids=sids)
            dummy_loss = out["logits"].sum() + out["group_logits"].sum()
        dummy_loss.backward()

        peak = torch.cuda.max_memory_allocated(device)
        model.zero_grad(set_to_none=True)
        del imgs, gids, sids, out, dummy_loss
        torch.cuda.empty_cache()
        return peak - baseline

    was_training = model.training
    model.train()
    try:
        # Two measurements to separate fixed overhead from per-pixel cost
        small_pixels = 32 * 128   # B=32, W=128
        large_pixels = 64 * 192   # B=64, W=192
        small_bytes = _measure(32, 128)
        large_bytes = _measure(64, 192)
    finally:
        if not was_training:
            model.eval()

    # Linear fit: bytes = fixed_activation + bytes_per_pixel * pixels
    bytes_per_pixel = (large_bytes - small_bytes) / (large_pixels - small_pixels)
    fixed_activation = small_bytes - bytes_per_pixel * small_pixels

    available = vram_gb * 1e9 - fixed_model - max(fixed_activation, 0)
    # 5% reserve — expandable_segments handles fragmentation well
    pixel_budget = int(available * 0.95 / bytes_per_pixel)

    print(f"  VRAM estimate: {model_bytes/1e9:.2f}GB model, "
          f"{fixed_model/1e9:.2f}GB fixed (optimizer), "
          f"{max(fixed_activation, 0)/1e6:.0f}MB fixed (activation)")
    print(f"  {bytes_per_pixel/1e3:.0f}KB/px (calibrated), "
          f"{available/1e9:.1f}GB available")
    print(f"  Pixel budget: {pixel_budget}")

    return pixel_budget
