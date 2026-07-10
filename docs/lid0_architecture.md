# LID-0: Hierarchical Super-Group Routing

## Motivation

Currently all 28 routed script heads share 4 SWA blocks before LID-1 classifies into 15 groups. The shared features are a compromise — optimized for no script family in particular. LID-1 must distinguish 15 groups using these generic features.

With LID-0, we split early: 2 shared SWA blocks → LID-0 routes to a super-group → 2 per-super-group SWA blocks → LID-1 routes within the super-group. Each super-group gets specialized features tuned for its script family before LID-1 even runs.

## Architecture

```
Input (B, 3, 32, W)
  → ConvStem → (B, 128, 8, W/2)
  → 2 shared SWA blocks (h=8)
  → Pool h=8→2, merge 128→256
  → LID-0: classify into 5 super-groups (+ blank)
  → Route frames to super-group SWA blocks
  → 2 SWA blocks per super-group (h=2, dim=256, 5 sets)
  → Pool h=2→1
  → LID-1: classify into group within super-group
  → Group expert blocks (1 local + 1 wide)
  → LID-2: classify into script within group (multi-script groups only)
  → Script expert blocks (1 local + 1 wide)
  → CTC heads
```

Total attention depth per frame: 2 shared + 2 super-group + 2 group expert + 2 script expert = 8 blocks (same as current, just reorganized).

## Super-Groups

Based on LID-1 feature similarity analysis from the trained model:

### 1. Alphabetic (5 groups, 7 scripts)
Scripts with discrete letters, left-to-right, no complex ligatures.
- latin, cyrillic_greek, caucasus (armenian, georgian)
- Internal LID-1 similarity: latin↔armenian 0.84, cyrillic↔greek 0.72

### 2. Semitic (2 groups, 2 scripts)
Right-to-left, cursive/connected, diacritical marks.
- arabic, hebrew
- Very distinct from all other families. Low similarity to everything (0.57 max).

### 3. CJK (3 groups, 4 routed heads)
Dense logographic/syllabic, square grid layout.
- han (han_sparse, han_dense), kana, korean
- Internal similarity: han↔kana 0.83, han↔korean 0.81, kana↔korean 0.73

### 4. Brahmic (7 groups, 17 scripts)
Derived from Brahmi script. Base consonant + vowel mark system. Includes South/SE Asian scripts that share this structural DNA.
- ne_indic (devanagari, gurmukhi), gujarati, bengali, odia
- dravidian_north (kannada, telugu, sinhala), dravidian_south (malayalam, tamil)
- thai_lao, burmese, khmer
- Internal: devanagari↔gurmukhi 0.86, thai↔lao 0.92, kannada↔telugu 0.83
- Note: SE Asian scripts (Thai, Lao, Burmese, Khmer) are historically Brahmic.
  Shared super-group SWA can learn the base+mark structure common to all.

### 5. Other (3 groups, 3 scripts)
Unique scripts with no close relatives.
- ethiopic, tibetan, emoji
- Low similarity to each other and everything else.
- Could also merge with Alphabetic (ethiopic is an abugida) or Brahmic (tibetan is Brahmic-derived). Grouping here avoids forcing them into a poor fit.

## Benefits

1. **Specialized features before LID-1.** Brahmic super-group SWA learns matra patterns, headline bars, conjunct structures. CJK SWA learns stroke density, radical patterns. LID-1 then classifies within a family using family-tuned features.

2. **LID-1 becomes easier.** Instead of 15-way classification on generic features, each LID-1 classifies 2-7 groups using specialized features. Brahmic LID-1 (hardest: 7 groups among similar scripts) gets Brahmic-tuned features.

3. **Gradual feature specialization.** Shared → super-group → group → script. Each level narrows the feature space. No single bottleneck where generic features must distinguish all 28 routed heads.

4. **Natural parameter scaling.** Super-group SWA blocks are shared within the family (not per-group), so cost is 5 × 2 blocks = 10 SWA blocks total vs current 4 shared. Net +6 SWA blocks, each ~1M params = ~6M extra. Modest.

## Parameter Impact

Current model: ~56M params
- 4 shared SWA: ~4M
- Group experts (15): ~18M
- Script experts (27): ~32M
- Heads/stems: ~2M

With LID-0:
- 2 shared SWA: ~2M
- 5 × 2 super-group SWA: ~10M
- LID-0 head: ~0.1M
- Group experts (15): ~18M (unchanged)
- Script experts (27): ~32M (unchanged)
- Total: ~62M (+6M, ~11% increase)

## Implementation Plan

### Phase 1: Architecture
- Add `SuperGroupSWA` module: `nn.ModuleList` of 5 `nn.ModuleList`s, each containing 2 SWA blocks.
- Add `lid0_head`: `nn.Sequential(Linear(dim, dim//2), GELU, Linear(dim//2, 6))` (5 super-groups + blank).
- Move `shared_b` blocks into per-super-group modules.
- Add `SUPER_GROUPS` and `GROUP_TO_SUPER_GROUP` mappings to `lid.py`.

### Phase 2: Forward Pass
- After `shared_a` + `merge_a`: run LID-0 to get per-frame super-group predictions.
- Route frames to super-group SWA blocks (same `_collect_segments` / `_scatter_segments` pattern as group experts).
- After super-group SWA + `merge_b`: run LID-1 within each super-group's groups.
- Rest of pipeline unchanged.

### Phase 3: Training
- Add `--lid0-weight` loss weight (default 1.0).
- During training, use GT super-group labels (derived from GT group labels via `GROUP_TO_SUPER_GROUP`).
- Ground truth super-group labels come from per-pixel group labels in training data (no new data generation needed — just map group→super-group at train time).
- Staged warm-start: freeze all, train LID-0 first, then unfreeze LID-1, then unfreeze all.

### Phase 4: Migration
- Existing `shared_b` weights can seed all 5 super-group SWA blocks (copy shared_b weights to each super-group — they start identical and specialize during training).
- LID-1 heads stay as-is (same group classification task, just with better features).
- Everything else unchanged.

## Open Questions

1. **Should LID-0 be per-frame or per-image?** Per-frame allows mixed super-groups in one image (e.g., CJK + Latin in a multilingual line). Per-image is simpler but can't handle mixed lines. Recommendation: per-frame, consistent with LID-1.

2. **Brahmic super-group size.** 7 groups / 17 scripts is large. Could split into "Indic" and "SE Asian" super-groups (4+3 groups). But the shared Brahmic structure (base+vowel mark) benefits from joint features. Keep as one for now, split later if LID-1 within Brahmic underperforms.

3. **"Other" super-group coherence.** Ethiopic, Tibetan, and Emoji have nothing in common. The super-group SWA blocks can't learn shared features. Alternative: merge ethiopic with Alphabetic (it's an abugida with letter-like structure), tibetan with Brahmic (historically derived), emoji stays solo. This eliminates the "Other" super-group.

4. **Drop path across super-group boundary.** Currently drop_path increases linearly across all blocks. With the super-group split, should drop_path continue across the boundary or reset? Recommendation: continue — the frame has still passed through N blocks total regardless of routing.
