# Lipi: Multilingual OCR via Mixture of Experts

> 27 routed script heads, 14 groups, 100+ languages. 80.5M params, ~1.7 GFLOPs/word.

---

## System Overview

Takes a cropped word image and outputs the text. Script identification and character recognition happen in a single forward pass through a Mixture of Experts architecture.

```
Word Image (32 x W x 3 RGB)
  -> ConvStem (stride-4 height, stride-2 width)  -> (8, W/2, 128)
  -> 6 shared SWA blocks (gradual h=8→4→2, progressively wider windows)
  -> LID-1 (15-group classifier + dedicated attn block)
  -> Group expert blocks (local w=16 + wide w=64, per-group)
  -> LID-2 (per-script classifier, multi-script groups only)
  -> Script expert blocks (local w=16 + wide w=64, per-script)
  -> Per-script CTC head (T=W/2)
  -> Output: decoded text
```

---

## Model Architecture (80.5M params)

### ConvStem (0.1M)

```
Input: (B, 3, 32, W) RGB
  Conv2d(3→64,  k=3x3, stride=(2,1), pad=1) + GN + GELU  -> (64, 16, W)
  Conv2d(64→128, k=3x3, stride=(2,2), pad=1) + GN + GELU -> (128, 8, W/2)

RF: ~7px H × 5px W (tight for clean routing boundaries)
```

### Shared Backbone: 6 SWA Blocks (2.4M)

Gradual vertical downsampling with progressively wider windows.

```
Shared SWA-A:  2 blocks  h=8, w=16, dim=128, mlp=4  (local features)
  merge_a:     h=8→4     Linear(256→128)
Shared SWA-B:  2 blocks  h=4, w=32, dim=128, mlp=4  (character context)
  merge_b:     h=4→2     Linear(256→256)
Shared SWA-C:  2 blocks  h=2, w=64, dim=256, mlp=4  (multi-char script context)
```

Each SWA block: LayerNorm → WindowedAttention (QK-norm, relative pos bias) → LayerScale → DropPath + residual → LayerNorm → MLP → LayerScale → DropPath + residual. Shifted windows on alternate blocks.

### LID-1: Script Group Classification (0.5M)

Branches off shared_c output before the CTC path merge.

```
AdaptiveAvgPool h=2→1                      pool vertical
lid1_attn: SWABlock(dim=256, w=32, h=1)   dedicated script-discrimination context
group_head: Linear(256→128) + GELU + Linear(128→15)   14 groups + blank

Output: per-frame group logits (B, T, 16)
```

lid1_attn is zero-initialized (identity at start) so LID-1 has its own capacity without affecting the CTC feature path.

### Patch Merge to 1D (CTC path)

```
merge_c: h=2→1, Linear(512→256)
Output: (B, T, 256) where T = W/2
```

### Group Expert Blocks: 15 experts (23.7M)

Per-group routing via LID-1 predictions (training: GT labels).

```
For each group g (0..14):
  local:  ExpertBlock(dim=256, heads=4, w=16, mlp=4, identity-init)
  wide:   ExpertBlock(dim=256, heads=4, w=64, mlp=4, identity-init)
  agg:    concat(local, wide) → Linear(512→256)  init: 0.5·I | 0.5·I

Per expert: 789K params (attn 264K + MLP 525K)
Per group:  789K × 2 blocks + 131K agg = 1.71M
```

Each ExpertBlock has N parallel attention+MLP experts sharing LayerNorm and LayerScale. Identity-initialized output projections so experts start as pass-through.

### LID-2: Per-Script Classification (0.2M)

Only for multi-script groups. Single-script groups skip LID-2.

```
6 heads (ModuleDict):
  group 1  (cyrillic_greek):   2-way  → cyrillic, greek
  group 7  (ne_indic):         5-way  → devanagari, gurmukhi, gujarati, bengali, odia
  group 8  (dravidian_north):  3-way  → kannada, telugu, sinhala
  group 9  (dravidian_south):  2-way  → malayalam, tamil
  group 10 (se_asian):         4-way  → thai, lao, burmese, khmer
  group 12 (caucasus):         2-way  → armenian, georgian
```

### Script Expert Blocks: 27 experts (42.6M)

Same structure as group experts but routed by flat script ID from LID-2.

```
script_local: ExpertBlock(dim=256, heads=4, w=16, mlp=4, 27 experts)
script_wide:  ExpertBlock(dim=256, heads=4, w=64, mlp=4, 27 experts)
Per-script aggregate: Linear(512→256)
```

### CTC Heads (5.0M)

```
LayerNorm(256) → per-script Linear(256→vocab_size)

CTC traversal: LTR scripts use frames left→right; Arabic/Hebrew reverse each
segment's frame slice so time follows logical reading order.
Greedy decode: argmax → collapse repeats → remove blanks → token IDs → text
```

---

## Groups and Scripts (14 groups, 27 routed script heads)

| # | Group | Scripts | Vocab | LID-2 |
|---|-------|---------|-------|-------|
| 0 | latin | latin | 797 | - |
| 1 | cyrillic_greek | cyrillic, greek | 364, 426 | yes |
| 2 | arabic | arabic | 500 | - |
| 3 | hebrew | hebrew | 192 | - |
| 4 | han | han_sparse, han_dense | 2153, 2018 | yes |
| 5 | kana | kana | 263 | - |
| 6 | korean | korean | 1500 | - |
| 7 | ne_indic | devanagari, gurmukhi, gujarati, bengali, odia | 1000, 600, 900, 900, 850 | yes |
| 8 | dravidian_north | kannada, telugu, sinhala | 550, 950, 500 | yes |
| 9 | dravidian_south | malayalam, tamil | 900, 350 | yes |
| 10 | se_asian | thai, lao, burmese, khmer | 450, 500, 650, 950 | yes |
| 11 | caucasus | armenian, georgian | 150, 186 | yes |
| 12 | ethiopic | ethiopic | 521 | - |
| 13 | tibetan | tibetan | 550 | - |

---

## Encoding

| Type | Scripts | Method |
|------|---------|--------|
| No-fusion | latin, cyrillic, greek, hebrew, armenian, georgian, ethiopic, kana | 1 char = 1 token |
| Fusion | devanagari, bengali, gujarati, gurmukhi, odia, kannada, telugu, malayalam, tamil, sinhala, thai, lao, burmese, khmer | base chars + virama-pair conjuncts |
| CJK | han_sparse, han_dense | outline-complexity routing; single-token frequent chars + ALT-slot visual similarity for rare chars |
| Korean | korean | jamo decomposition (onset + vowel + coda) |
| Arabic | arabic | base characters + frequent diacritic grapheme clusters; HarfBuzz shapes positional glyph forms |

ASCII punctuation and digits are in every codec's vocab but labeled as latin in training data so LID-1 learns to route them to the latin expert.

---

## Training

### Data Generation

Synthetic data from word lists + fonts. Per-line content plans have mixed-script support. Hanzi/Kanji/Hanja are split into maximal sparse/dense runs, and Japanese words additionally split at Kana boundaries via `split_by_script()`. ASCII characters in non-latin words become separate Latin segments.

### Loss Functions

```
loss = ctc_weight × CTC + lid1_weight × LID1 + lid2_weight × LID2

CTC:   per-segment, batched by (group, script)
LID-1: per-frame CrossEntropy vs GT group_labels
LID-2: per-frame CrossEntropy vs GT script_labels (multi-script groups only)
```

### Selective Freezing

```
--freeze-except lid          train LID-1 + LID-2 heads only
--freeze-except backbone     train shared blocks + stem only
--freeze-except ctc          train CTC heads only
--freeze-except experts      train expert blocks only
--freeze-except lid,ctc      comma-separated combinations
```

### Drop Path Schedule

Linearly increasing from 0 → 0.1 across all stages:
```
shared_a[0]: 0.000   shared_a[1]: 0.014
shared_b[0]: 0.029   shared_b[1]: 0.043
shared_c[0]: 0.057   shared_c[1]: 0.071
group experts: 0.086
script experts: 0.100
```

---

## Parameter Breakdown

| Component | Params | % |
|-----------|--------|---|
| ConvStem | 0.1M | 0.1% |
| Shared SWA-A (2, h=8, w=16, dim=128) | 0.4M | 0.5% |
| Shared SWA-B (2, h=4, w=32, dim=128) | 0.4M | 0.5% |
| Shared SWA-C (2, h=2, w=64, dim=256) | 1.6M | 2.0% |
| Merges (a + b + c) | 0.2M | 0.3% |
| lid1_attn + group_head | 0.8M | 1.0% |
| Group experts (14 × 2 blocks) | 23.7M | 29.4% |
| Group aggregates (14) | 2.0M | 2.4% |
| LID-2 heads (6) | 0.2M | 0.2% |
| Script experts (27 × 2 blocks) | 42.6M | 52.9% |
| Script aggregates (27) | 3.5M | 4.4% |
| CTC heads (27 routed scripts) | 5.0M | 6.2% |
| **Total** | **80.5M** | |
| Shared (all scripts) | 3.5M | 4.3% |
| MoE (per-group/script) | 77.0M | 95.7% |

Inference: only 1 group expert + 1 script expert fire per sample.
Active path is ~6.5M params, ~1.7 GFLOPs at W=128.

---

## Code Structure

```
src/
  taxonomy.py           SCRIPTS, GROUPS, mappings (data-only, torch-free)
  model/
    encoder.py          LipiMoEEncoder (ConvStem + shared SWA + MoE experts)
    blocks.py           DropPath, LayerScale, WindowedAttention, SWABlock,
                        ExpertBlock, CTCHead, GroupCTCModule, routing helpers
    lid.py              LIDCoarse module
  encoding/
    config.py           Codec definitions (NoFusion, Fusion, CJK, Korean)
    decompose.py        encode_text / decode_ids routing
    vocab.py            Per-script vocab building
    renderable.py       get_renderable_chars (codec introspection)
  data/
    fonts.py            Font discovery and script mapping (logic only)
    font_registry.py    FONT_TO_SCRIPTS / FONT_CATEGORIES data tables
    rendering.py        Word/char rendering
    word_lists.py       Word list loading
    script_detect.py    detect_script, split_by_script, cp predicates
    augmentation.py     Image augmentation ops
    color.py            Colorspace helpers
    text_renderer.py    Low-level PIL / FreeType rendering
  training/
    dataloader.py       MDS streaming dataset + tokenizer building
    losses.py           CTC + LID1 + LID2 losses (batched by group/script)
    routing.py          Frame-level label construction from segments
    eval.py             Per-group/per-script evaluation

scripts/
  data/                 Data prep
    generate.py         Synthetic data generation (split_by_script, ASCII→latin)
    convert_to_mds.py   Convert external .pt shards → MDS
    build_word_freq.py  Word-frequency table build
    cjk_visual_similarity.py  CJK codec visual-similarity table build
    setup/              Source fetchers + real-dataset conversion
  train/                Training
    train.py            Training loop (selective freeze, staged warm-start)
    train_lid0_probe.py Auxiliary LID probe training
  eval/                 Checkpoint evaluation
    eval.py             Per-group/per-script eval
    benchmark.py        Standard OCR benchmarks
  inference/            Deployable pipelines
    run_doctr.py        Full document OCR pipeline
  tools/                Analysis + one-off utilities
    count_flops.py      FLOP counter
    migrate_checkpoint.py  Checkpoint migration (13/14→15 groups)
    test_punct_routing.py  Punctuation routing analysis
    diagnose.py, merge_checkpoints.py, identify_checkpoint.py,
    check_fonts.py, check_renders.py
  experiments/          Experimental architectures
```
