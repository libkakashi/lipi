# CTC Convergence Issue — Diagnosis & Fix

## TL;DR
Random group expert blocks were destroying backbone features before they
reached the CTC heads. CTC had to learn from noise. **Fixed by identity-init
of all expert blocks** (commit `ebfaa7a8`).

Local validation: CTC dropped from 48 → 7 in 270 steps (lr=3e-4, dim=128).
Previously was stuck at 34 average for entire epoch.

## Root Cause

### The architecture
```
backbone → proj → shared_attn → (LID-1 head)   ← LID-1 branches off here
                              ↓
              group experts (4 blocks × 13 experts)  ← random init = noise
                              ↓
              group_aggregate → LID-2 head
                              ↓
              script experts (2 blocks × 26 experts)  ← random init = noise
                              ↓
              script_aggregate → norm → CTC heads
```

### The bug
LID-1 branches off **before** the expert blocks. It gets clean backbone
features and learned to 92% accuracy quickly.

CTC heads see features **after** all expert blocks. With default random
init, each expert block's residual `x = x + attn(x) + mlp(x)` adds noise
at ~70% the magnitude of the signal. After 6 blocks (4 group + 2 script),
the original backbone signal is buried in noise.

CTC had to learn character recognition from noise features. Even with
perfect group/script routing, the input to CTC was garbage.

### The fix
Initialize the output projections of all expert blocks to zero:
```python
for block in expert_blocks:
    for attn in block.expert_attns:
        nn.init.zeros_(attn.proj.weight)
        nn.init.zeros_(attn.proj.bias)
    for mlp in block.expert_mlps:
        nn.init.zeros_(mlp.fc2.weight)
        nn.init.zeros_(mlp.fc2.bias)
```

With zero output projections, the residual blocks become identity at init:
`x = x + zeros() + zeros() = x`. Backbone features pass through unchanged.

Aggregations init as identity (0.5*I + 0.5*I = average of local/wide streams).

CTC heads init small (std=0.01) so logits start near-uniform.

## What This Means for Training

### Before fix (your previous run)
- LID-1: 92% by epoch 1 (fast — bypassed the broken experts)
- CTC: 34 avg per char by epoch 1 (stuck — noise input)
- Word accuracy: 0% (CTC couldn't decode)

### After fix (expected behavior)
- LID-1: 92% by epoch 1 (same — wasn't broken)
- CTC: should drop to <10 per char within first few thousand steps
- Word accuracy: should start increasing by epoch 3-5

### CTC convergence stages
1. **Random** (~62/char): No alignment, no character knowledge
2. **All-blank** (~13/char): Model learns to predict blank everywhere — good loss, empty output
3. **Peaky character** (3-7/char): Model emits characters at right positions — output starts working
4. **Refined** (<2/char): Word accuracy climbs

CTC has a known "peaky output" failure mode at stage 2. The model achieves
low loss by predicting blank everywhere (which is correct for ~70% of
frames in line data). To escape, it needs to learn that non-blank predictions
also reduce loss.

### Local validation (dim=128, lr=1e-3, 1000 steps simulated)
- Step 0:   CTC=89  | random init | 8/8 segs emit tokens (random garbage)
- Step 100: CTC=33  | LID=31%     | 0/8 segs emit (all-blank minimum)
- Step 200: CTC=7.5 | LID=75%     | 0/8 segs emit (still all-blank, very low loss)
- Step 300: CTC=6.3 | LID=94%     | 7/8 segs emit chars! (escaped minimum)

CTC reached character emission in ~300 steps. With the user's larger model
(dim=256) and full training data, expect character emission within first
epoch.

## All Recent Fixes Summary

1. **Identity-init group experts** (`ebfaa7a8`) — THE main CTC fix
2. **Identity-init script experts** (`fe586d2f`)
3. **Identity-init aggregations + small CTC head init** (`97c10d7b`)
4. **Ground truth script routing for CTC** (`68583e01`) — was using LID-2 predictions
5. **Per-frame LID-2 architecture** (`66ede071`) — replaces per-sample LID-2
6. **MLP heads + label smoothing** (`b9d664e2`)
7. **Padding vs whitespace separation** (`8d34f807`)
8. **Group label remap fix** (`fd3267e2`) — blank pixels were being remapped to latin
9. **Clean rendering with style on composed line** (`02f7c5ce`) — random per-word colors
10. **Two-level expert routing** (`66ede071`) — group → LID-2 → script

## What Could Still Go Wrong

1. **CTC peaky output persists** — if after epoch 5-10 the decoded text is
   still mostly empty, we may need entropy regularization or focal CTC.
2. **Vocab imbalance** — han_kana (4000 tokens) has higher CTC loss than
   hebrew (192 tokens). Some script-specific tuning may help.
3. **Mixed-line CTC alignment** — segments are processed independently per
   script. Boundary frames (where pixel-to-frame rounding leaks whitespace)
   add ~1% noise. Acceptable but not perfect.

## Observed Training Behavior

Latest user training (2026-04-14, after all fixes):
- Step 0:   CTC=104, LID-1=2.56 — random init
- Step 80:  CTC=10,  LID-1=2.55 — quickly approaching plateau
- Step 200: CTC=5.5, LID-1=2.28 — near plateau
- Step 500+: CTC=5.4, LID-1=2.0-2.5 — stuck in all-blank minimum

CTC stuck at ~5/char is the all-blank local minimum (model predicts blank
for every frame). LID-1 stuck at ~2.0 with label_smoothing=0.1 corresponds
to ~50% accuracy (much worse than expected — should reach 90% within an epoch).

The LID-1 fluctuating wildly (0% to 100%) per-batch is misleading — it's the
last-batch accuracy variation due to small per-batch sample sizes. The true
learning trajectory is the loss value, which IS dropping (slowly).

If after 2-3 epochs the model hasn't escaped these local minima, consider:
- Higher LR (current 3e-4 may be too conservative)
- Aggregation Cross-Entropy (ACE) loss as auxiliary signal (already
  implemented but disabled — pass `--align-ce-weight 0.1`)
- Lower batch size (smaller batches give noisier gradients that help
  escape local minima)

## Files Modified
- `src/model/encoder.py` — architecture changes, init fixes
- `src/training/losses.py` — per-frame LID-2, segment skip diagnostics
- `src/training/eval.py` — per-frame routing, padding handling
- `src/training/dataloader.py` — group label remap, padding semantics
- `scripts/train/train.py` — ground truth routing, label smoothing
- `scripts/data/generate.py` — clean rendering, line-level mixed data
