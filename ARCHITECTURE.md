# Lipi: Multilingual OCR via Mixture of Experts

> 26 scripts, 13 groups, 100+ languages, ~95% of world's literate population.
> 759M total params, 65M active per sample. 32 MB inference on phone.

---

## System Overview

Takes a cropped word image and outputs the text. Script identification and character recognition happen in a single forward pass through a Mixture of Experts architecture.

```
Word Image (32 × W × 3)
  → Color Projection (RGB → L+a → 1ch)
  → ResNet Stem (stride 2×2, 64ch)       → (16 × W/2)
  → Shared SWA (12 blocks, dim=288)      ← universal visual features
  → LID-1 (13-group classifier)          ← which script family?
  → Expert SWA Stage 1 (12 blocks)       ← group-specific features (h=16, w=W/2)
  → Height Pool (16→4) + Width Pool (2×) → (h=4, w=W/4)
  → Expert SWA Stage 2 (8 blocks)        ← deep group-specific features
  → Height Pool (4→2), fold h into C
  → LID-2 (per-script classifier)        ← which exact script?
  → Per-Script CTC Head (T=W/4)          ← character sequence
  → Output: decoded text
```

---

## Groups and Scripts (13 groups, 26 scripts)

| # | Group | Scripts | LID-2 | Speakers |
|---|-------|---------|-------|----------|
| 0 | latin | latin | - | ~3B |
| 1 | cyrillic_greek | cyrillic, greek | yes | ~300M |
| 2 | arabic | arabic | - | ~500M |
| 3 | hebrew | hebrew | - | ~9M |
| 4 | sino_japanese | han_kana | - | ~1.4B |
| 5 | korean | korean | - | ~80M |
| 6 | ne_indic | devanagari, gurmukhi, gujarati, bengali, odia | yes | ~1B |
| 7 | south_indic | kannada, telugu, malayalam, tamil, sinhala | yes | ~300M |
| 8 | se_asian | thai, lao, burmese, khmer | yes | ~150M |
| 9 | emoji | emoji | - | universal |
| 10 | caucasus | armenian, georgian | yes | ~10M |
| 11 | ethiopic | ethiopic | - | ~57M |
| 12 | tibetan | tibetan | - | ~6M |

---

## Model Architecture

### Shared Path (always active, ~13M params)

```
ColorProjection:     Conv2d(2→32→16→1, 1×1)     641 params
ResNet Stem:         depth=3, 64ch, stride 4×    526K params
Shared SWA 4×4:      8 blocks, dim=288           8.0M params
Shared SWA 4×16:     4 blocks, dim=288           4.0M params
LID-1 Classifier:    attn pool (288→64→1) → MLP 288→576→288→13    354K params
```

### Expert Path (1 of 13 active, ~52M per group)

```
Expert SWA Stage 1:  12 FullyExpertSWABlock, dim=288   12.0M/group
Height Pool 8→4:     LearnedHeightPooling               13K
Channel Projection:  Linear(288→576)                    166K
Expert SWA Stage 2:  8 FullyExpertSWABlock, dim=576    31.9M/group
Height Pool 4→1:     LearnedHeightPooling               4K
LayerNorm:           dim=576                            1.2K
```

### CTC Heads (1 of 26 active, ~7M per script)

```
LID-2:      attention pool → MLP (multi-script groups only)
BiLSTM:     input=576, hidden=384, layers=2, bidirectional
Projection: Linear(768 → vocab_size)
```

### Per-Script Vocab Sizes

| Script | Tokens | Script | Tokens |
|--------|--------|--------|--------|
| latin | 797 | kannada | 146 |
| cyrillic | 361 | telugu | 155 |
| greek | 423 | malayalam | 173 |
| arabic | 491 | tamil | 127 |
| hebrew | 189 | sinhala | 147 |
| han_kana | 3213 | thai | 142 |
| korean | 373 | lao | 141 |
| devanagari | 216 | burmese | 215 |
| gurmukhi | 136 | khmer | 169 |
| gujarati | 146 | armenian | 147 |
| bengali | 154 | georgian | 183 |
| odia | 148 | ethiopic | 518 |
| emoji | 107 | tibetan | 267 |

### Parameter Summary

| Component | Total | Active/sample |
|-----------|-------|--------------|
| Shared | 12.9M | 12.9M |
| Expert Stage 1 (13 groups) | 155.7M | 12.0M |
| Expert Stage 2 (13 groups) | 414.6M | 31.9M |
| CTC Heads (26 scripts) | 175.8M | ~7M |
| **Total** | **759.2M** | **~65M (8%)** |

---

## Training

### Loss Functions

```
loss = CTC_loss + lid1_weight × LID1_loss + LID2_loss
```

- **CTC loss**: per-script vocab slicing before log_softmax. No zero-padding dilution.
- **LID-1 loss**: CrossEntropy on all samples. Weight=3 to overcome grad clipping.
- **LID-2 loss**: CrossEntropy on correctly LID-1-routed samples. Averaged across groups.

### Routing Masks

| Mask | Condition | Used for |
|------|-----------|----------|
| lid1_ok | pred_group == true_group | LID-2 loss |
| ctc_ok | lid1_ok AND pred_script == true_script AND valid lengths | CTC loss |

### Training Data

- 50K samples per script, balanced per group
- 30% clean renders, 70% augmented (24 augmentation ops)
- Single-char images at 25% of word budget (adaptive reps)
- Triple validation: font_covers_text + pre-aug ink + post-aug ink
- 173 fonts with weighted diversity (70% clean, 20% handwriting, 10% display)

### Decomposition

- **Han_kana**: CJK depth-2 IDS decomposition (21K chars → 1900 components)
- **Korean**: hybrid jamo (rare syllables → 67 jamo, top-250 common syllables kept whole)

### Key Hyperparameters

```
lr:           5e-4
optimizer:    AdamW (weight_decay=0.01)
scheduler:    linear warmup (1 epoch) + cosine decay
batch_size:   256-500
grad_clip:    max_norm=25
amp:          bf16 (model cast to bf16 natively)
checkpointing: all 32 SWA blocks (shared + expert)
```

---

## Inference

| Format | Full model | Active path | Per-group download |
|--------|-----------|-------------|-------------------|
| bf16 | 1.5 GB | 130 MB | 90 MB |
| FP4 | 380 MB | 32 MB | 25 MB |

### Deployment

1. Ship shared weights (13M, always loaded) + LID-1
2. User installs language packs (expert group + CTC heads, 25 MB each in FP4)
3. LID-1 routes → load correct expert group → decode
4. Top-K fallback: if LID-1 uncertain, try top-2 groups, pick best CTC confidence

---

## Code Structure

```
src/
  model/
    moe_encoder.py      LipiMoEEncoder (full model)
    lid.py              SCRIPTS, GROUPS, LIDCoarse
    attention.py        SWABlock, FullyExpertSWABlock
    stem.py             ResNetStem
    pooling.py          LearnedHeightPooling
    rope.py             RoPE2D
  data/
    vocab.py            Frozen vocab loading (hex files)
    decompose.py        CJK IDS + Korean jamo decomposition
    fonts.py            Font discovery, cmap validation
    rendering.py        Word/char rendering, ink detection
    word_lists.py       Word list loading
    color.py            L+a color projection
    augmentation.py     24 augmentation ops
    renderer.py         FreeType/PIL text rendering
  training/
    moe_losses.py       CTC, LID-1, LID-2 loss functions
    routing.py          Routing mask computation
    moe_data.py         Data loading, tokenization, encoding
    moe_eval.py         Per-group/per-script evaluation

scripts/
    train_moe.py        Training orchestration
    generate_data.py    Synthetic data generation
    setup_fonts.py      Font downloading (173 fonts)
    train_lid.py        LID-only training (legacy)

tests/
    test_moe_pipeline.py     83 tests: vocabs, tokenizers, model, losses, routing
    test_font_rendering.py   32 tests: ink, cmap, per-script rendering
```

---

## Results (18-script run, epoch 25)

| Script | LID-1 | Word | Char |
|--------|-------|------|------|
| latin | 94.0% | 75.7% | 91.4% |
| cyrillic | 92.7% | 67.4% | 80.0% |
| greek | 92.7% | 85.0% | 96.5% |
| arabic | 95.4% | 22.5% | 50.8% |
| hebrew | 96.5% | 66.8% | 79.4% |
| han_kana | 98.1% | 48.7% | 62.0% |
| korean | 97.0% | 80.5% | 90.7% |
| devanagari | 95.3% | 52.2% | 76.9% |
| gurmukhi | 95.3% | 77.3% | 89.9% |
| gujarati | 95.3% | 67.2% | 81.1% |
| bengali | 95.3% | 53.0% | 73.2% |
| kannada | 93.7% | 66.9% | 87.7% |
| telugu | 93.7% | 57.1% | 81.0% |
| malayalam | 93.7% | 71.5% | 88.0% |
| tamil | 93.7% | 85.5% | 94.6% |
| thai | 94.0% | 60.8% | 86.5% |
| lao | 94.0% | 59.2% | 85.7% |
| emoji | 99.5% | 99.5% | 99.5% |
| **Overall** | **95.5%** | **65.8%** | **82.6%** |

26-script run in progress. 8 new scripts (odia, sinhala, burmese, khmer, armenian, georgian, ethiopic, tibetan) training from scratch.
