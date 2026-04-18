# Lipi v5 MoE Encoder — Architecture Detail

## Model: 57.4M params, 27 scripts, 15 groups

```
INPUT: (B, 3, 32, W) RGB image
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ CONV STEM                                          0.1M params │
│  │                                                                │
│  │  Conv2d(3→64, k=3×3, stride=(2,1), pad=1)  → (B, 64, 16, W)  │
│  │  GroupNorm(1, 64) + GELU                                       │
│  │  Conv2d(64→128, k=3×3, stride=(2,2), pad=1)→ (B, 128, 8, W/2)│
│  │  GroupNorm(1, 128) + GELU                                      │
│  │                                                                │
│  │  RF: ~7px H × 5px W (tight for clean routing boundaries)      │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, 128, 8, W/2)  reshape → (B, 8·W/2, 128)
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ SHARED SWA-A: 2 blocks                             0.3M params │
│  │                                                                │
│  │  Block 0: SWABlock(dim=128, heads=2, window=8×16, shift=off)  │
│  │    │ LayerNorm → WindowedAttention → LayerScale → DropPath     │
│  │    │ + residual                                                │
│  │    │ LayerNorm → MLP(128→256→128) → LayerScale → DropPath     │
│  │    │ + residual                                                │
│  │  Block 1: SWABlock(dim=128, heads=2, window=8×16, shift=ON)   │
│  │                                                                │
│  │  Window covers full height (8) × 16 cols = ~2 characters      │
│  │  Relative position bias: (15×31) × 2 heads                    │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, 8·W/2, 128)
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ MERGE A: h=8 → h=4                                0.03M params│
│  │                                                                │
│  │  _patch_merge_h: concat 2 adjacent rows → Linear(256→128)     │
│  │  Gradual 2× vertical downsampling, preserves fine features.    │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, 4·W/2, 128)  h=4, w=W/2
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ SHARED SWA-B: 2 blocks                             0.3M params │
│  │                                                                │
│  │  Block 0: SWABlock(dim=128, heads=2, window=4×32, shift=off)  │
│  │  Block 1: SWABlock(dim=128, heads=2, window=4×32, shift=ON)   │
│  │                                                                │
│  │  Window: full h=4 × 32 cols = ~4 characters.                  │
│  │  Character-level context at intermediate vertical resolution.  │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, 4·W/2, 128)
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ MERGE B: h=4 → h=2                                0.07M params│
│  │                                                                │
│  │  _patch_merge_h: concat 2 adjacent rows → Linear(256→256)     │
│  │  dim: 128 → 256.                                              │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, 2·W/2, 256)  h=2, w=W/2
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ SHARED SWA-C: 2 blocks                             1.1M params │
│  │                                                                │
│  │  Block 0: SWABlock(dim=256, heads=4, window=2×64, shift=off)  │
│  │    │ LayerNorm → WindowedAttention(256, 4 heads)               │
│  │    │   QK-norm: LayerNorm(64) on q,k per head                 │
│  │    │   Rel pos bias: (3×127) × 4 heads                        │
│  │    │ → LayerScale → DropPath + residual                        │
│  │    │ LayerNorm → MLP(256→512→256) → LayerScale → DropPath     │
│  │    │ + residual                                                │
│  │  Block 1: same, shift=ON                                       │
│  │                                                                │
│  │  Window: full h=2 × 64 cols = ~7 characters.                  │
│  │  Wide multi-character context for script discrimination.       │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, 2·W/2, 256)
│
├────────────────────────────────┐
│ CTC PATH                      │ LID-1 PATH
│                                │
│                                ▼
│                    ┌──────────────────────────────────────────────┐
│                    │ LID-1 BRANCH                     0.5M params│
│                    │                                             │
│                    │  AdaptiveAvgPool2d h=2→1                    │
│                    │    (B, 2, W/2, 256) → (B, 1, W/2, 256)     │
│                    │    squeeze → (B, W/2, 256)                  │
│                    │                                             │
│                    │  lid1_attn: SWABlock                        │
│                    │    dim=256, heads=4, window=1×32            │
│                    │    shift=off, mlp_ratio=2, drop_path=0      │
│                    │    Zero-init (identity at start)            │
│                    │                                             │
│                    │  group_head: Sequential                     │
│                    │    Linear(256→128) + GELU + Linear(128→16)  │
│                    │    16 = 15 groups + 1 blank                 │
│                    │                                             │
│                    │  Output: (B, W/2, 16) per-frame group logits│
│                    │                                             │
│                    │  Training: CE loss vs GT group_labels       │
│                    │  Inference: argmax → per-frame group_id     │
│                    └──────────────────────────────────────────────┘
│                                │
│                                │ group_ids: (B, T)
│                                │ one of {0..14} per frame, or 15=blank
▼                                │
┌────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ MERGE C: h=2 → h=1                                0.1M params │
│  │                                                                │
│  │  Concat 2 rows → Linear(512 → 256)                            │
│  │  Output: (B, W/2, 256)  = (B, T, 256) where T = W/2          │
│  │                                                                │
│  │  CTC-path-only merge. LID-1 already pooled separately.        │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, T, 256)  T = W/2 frames
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ GROUP EXPERT BLOCKS: 15 groups                    17.9M params │
│  │                                                                │
│  │  For each group g in {0..14}:                                  │
│  │    mask_g = (group_ids == g)  → select frames for this group  │
│  │    if no frames → skip                                        │
│  │    batch_x = _collect_segments(x, mask_g)  → (N, max_len, 256)│
│  │                                                                │
│  │    ┌─── LOCAL STREAM ────────────────────────────────────┐     │
│  │    │ ExpertBlock(dim=256, heads=4, 15 experts)           │     │
│  │    │   window: h=1, w=16  (~2 chars local context)       │     │
│  │    │   expert_attns[g]: WindowedAttention (own weights)  │     │
│  │    │   expert_mlps[g]:  MLP(256→512→256) (own weights)   │     │
│  │    │   norm1, norm2, ls1, ls2: SHARED across experts     │     │
│  │    │   Identity-init: proj+fc2 zeroed at start           │     │
│  │    └─────────────────────────────────────────────────────┘     │
│  │    ┌─── WIDE STREAM ─────────────────────────────────────┐     │
│  │    │ ExpertBlock(dim=256, heads=4, 15 experts)           │     │
│  │    │   window: h=1, w=64  (~7 chars wide context)        │     │
│  │    │   Same structure, own expert weights                │     │
│  │    └─────────────────────────────────────────────────────┘     │
│  │                                                                │
│  │    concat(local, wide) → Linear(512→256) per group            │
│  │      = group_aggregates[g]                                     │
│  │      Init: 0.5·I | 0.5·I  (average of streams)               │
│  │                                                                │
│  │    _scatter_segments → write back to (B, T, 256)              │
│  │                                                                │
│  │  Per expert: 527K params (attn 264K + MLP 263K)               │
│  │  Per group:  527K × 2 blocks + 131K agg = 1.2M               │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, T, 256)  post-group-expert features
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ LID-2: Per-frame script classification             0.2M params │
│  │                                                                │
│  │  Only for multi-script groups (6 of 15 groups):               │
│  │                                                                │
│  │  lid2_heads = ModuleDict:                                      │
│  │    "1"  → Sequential(Linear(256→128), GELU, Linear(128→2))    │
│  │           cyrillic vs greek                                    │
│  │    "7"  → Sequential(..., Linear(128→5))                       │
│  │           devanagari/gurmukhi/gujarati/bengali/odia            │
│  │    "8"  → Sequential(..., Linear(128→3))                       │
│  │           kannada/telugu/sinhala                               │
│  │    "9"  → Sequential(..., Linear(128→2))                       │
│  │           malayalam/tamil                                      │
│  │    "10" → Sequential(..., Linear(128→4))                       │
│  │           thai/lao/burmese/khmer                               │
│  │    "12" → Sequential(..., Linear(128→2))                       │
│  │           armenian/georgian                                    │
│  │                                                                │
│  │  Single-script groups: script_id = 0 (trivial)                │
│  │  Flat script_id = group_offset + local_script_id              │
│  └────────────────────────────────────────────────────────────────┘
│                                │
│                                │ flat_script_ids: (B, T)
│                                │ one of {0..26} per frame
▼                                │
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ SCRIPT EXPERT BLOCKS: 27 scripts                  31.9M params │
│  │                                                                │
│  │  Same structure as group experts but with 27 experts:          │
│  │                                                                │
│  │  For each flat script s in {0..26}:                            │
│  │    mask_s = (flat_scripts == s)                                │
│  │    batch_x = _collect_segments(post_group_features, mask_s)    │
│  │                                                                │
│  │    LOCAL: ExpertBlock(dim=256, heads=4, w=16, 27 experts)     │
│  │    WIDE:  ExpertBlock(dim=256, heads=4, w=64, 27 experts)     │
│  │    concat → script_aggregates[s]: Linear(512→256)             │
│  │    scatter back                                                │
│  │                                                                │
│  │  Per script: 527K × 2 + 131K = 1.2M                          │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, T, 256)  post-script-expert features
│
│  ┌─────────────────────────────────────────────────────────────────┐
│  │ OUTPUT NORM + CTC HEADS                            5.0M params │
│  │                                                                │
│  │  LayerNorm(256)                                                │
│  │                                                                │
│  │  15 GroupCTCModules, per-script Linear(256→vocab_size):        │
│  │  ┌────────────────────────────────────────────────────────┐    │
│  │  │ G0  latin:           latin=797                        │    │
│  │  │ G1  cyrillic_greek:  cyrillic=364, greek=426          │    │
│  │  │ G2  arabic:          arabic=500                       │    │
│  │  │ G3  hebrew:          hebrew=192                       │    │
│  │  │ G4  han:             han=3811                         │    │
│  │  │ G5  kana:            kana=263                         │    │
│  │  │ G6  korean:          korean=1500                      │    │
│  │  │ G7  ne_indic:        deva=1000 gurm=600 guj=900      │    │
│  │  │                      beng=900 odia=850                │    │
│  │  │ G8  dravidian_north: kann=550 telu=950 sinh=500      │    │
│  │  │ G9  dravidian_south: mala=900 tamil=350              │    │
│  │  │ G10 se_asian:        thai=450 lao=500 burm=650       │    │
│  │  │                      khmer=950                        │    │
│  │  │ G11 emoji:           emoji=107                        │    │
│  │  │ G12 caucasus:        arme=150 geor=186               │    │
│  │  │ G13 ethiopic:        ethiopic=521                     │    │
│  │  │ G14 tibetan:         tibetan=550                      │    │
│  │  └────────────────────────────────────────────────────────┘    │
│  └────────────────────────────────────────────────────────────────┘
│
▼ (B, T, max_vocab) logits per frame
  CTC greedy decode: argmax → collapse repeats → remove blanks → token IDs
  Token IDs → decode_ids(ids, script_name) → Unicode text


TRAINING LOSSES:
  ┌─────────────────────────────────────────────────────────────────┐
  │  CTC loss:  per-segment, batched by (group, script)            │
  │  LID-1:     per-frame CrossEntropy vs GT group_labels          │
  │  LID-2:     per-frame CrossEntropy vs GT script_labels         │
  │             (multi-script groups only)                          │
  │                                                                │
  │  total = ctc_weight·CTC + lid1_weight·LID1 + lid2_weight·LID2 │
  │  Default weights: all 1.0                                      │
  └────────────────────────────────────────────────────────────────┘


DROP PATH SCHEDULE (linearly increasing 0 → 0.1):
  shared_a[0]:  0.000    shared_a[1]:  0.014
  shared_b[0]:  0.029    shared_b[1]:  0.043
  shared_c[0]:  0.057    shared_c[1]:  0.071
  group experts: 0.086   (local + wide share same rate)
  script experts: 0.100  (local + wide share same rate)


FRAME STRIDE:
  Input pixel W → stem stride (1,2) → W/2
  No further width downsampling after stem
  T = W/2 frames throughout attention + experts
  Each frame ≈ 2 input pixels wide


PARAMETER BREAKDOWN:
  ┌──────────────────────────┬──────────┬───────┐
  │ Component                │   Params │     % │
  ├──────────────────────────┼──────────┼───────┤
  │ ConvStem                 │    0.1M  │  0.1% │
  │ Shared SWA-A (2, h=8)   │    0.3M  │  0.5% │
  │ Merge A (h=8→4)         │    0.0M  │  0.1% │
  │ Shared SWA-B (2, h=4)   │    0.3M  │  0.5% │
  │ Merge B (h=4→2)         │    0.1M  │  0.1% │
  │ Shared SWA-C (2, h=2)   │    1.1M  │  1.9% │
  │ Merge C (h=2→1)         │    0.1M  │  0.2% │
  │ lid1_attn + group_head   │    0.5M  │  0.9% │
  │ Group experts (15×2)     │   15.8M  │ 27.5% │
  │ Group aggregates (15)    │    2.0M  │  3.5% │
  │ LID-2 heads (6)          │    0.2M  │  0.3% │
  │ Script experts (27×2)    │   28.5M  │ 49.6% │
  │ Script aggregates (27)   │    3.5M  │  6.1% │
  │ CTC heads (27 scripts)   │    5.0M  │  8.7% │
  ├──────────────────────────┼──────────┼───────┤
  │ TOTAL                    │   57.4M  │  100% │
  │ Shared (all scripts)     │    2.5M  │  4.4% │
  │ MoE (per-group/script)   │   54.9M  │ 95.6% │
  └──────────────────────────┴──────────┴───────┘
```
