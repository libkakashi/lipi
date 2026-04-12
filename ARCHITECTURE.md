# Lipi: Multilingual OCR via Mixture of Experts

> 26 scripts, 13 groups, 100+ languages.

---

## System Overview

Takes a cropped word image and outputs the text. Script identification and character recognition happen in a single forward pass through a Mixture of Experts architecture.

```
Word Image (32 x W x 3)
  -> RGB to L+a (preprocessing, 2ch)
  -> HGNetV2 backbone (pretrained, OCR strides) -> (h=1, w=W/2, 2048ch)
  -> Project 2048 -> dim                         -> (W/2, dim)
  -> LID-1 (13-group classifier)                 <- which script family?
  -> Expert global attention (1D, per-group)      <- script-specific refinement
  -> LID-2 (per-script classifier)               <- which exact script?
  -> Per-Script CTC Head (T=W/2)                  <- character sequence
  -> Output: decoded text
```

---

## Model Architecture

### Backbone: HGNetV2 (pretrained, shared)

```
Input: (B, 2, 32, W) L+a
  -> Conv2d(2, 3, 1x1)              adapt L+a to RGB for pretrained weights
  -> HGNetV2 with OCR strides       height-aggressive, width-conservative
     Stem:    stride (2,1), (2,1)    h=8, w=W
     Stage 0: HGBlocks               h=8, w=W, 192ch
     Stage 1: stride (2,2)           h=4, w=W/2, 512ch
     Stage 2: stride (2,1)           h=2, w=W/2, 1024ch
     Stage 3: stride (2,1)           h=1, w=W/2, 2048ch
  -> Output: (B, 2048, 1, W/2)
```

### Projection + LID-1

```
Project:         Linear(2048, dim)
LID-1:           AdaptiveAvgPool -> MLP -> 13 groups
```

### Expert Path (1 of 13 active per sample)

```
Expert blocks:   2x ExpertBlock (1D global attention + MLP, per-group)
                 dim, mlp_ratio=2, global attention (no windows)
LID-2:           Conv1d pool -> MLP (multi-script groups only)
CTC Head:        Linear(dim, vocab_size) per script
```

### Training-only: GTC Decoder (planned)

```
NRTR decoder:    2-layer transformer (cross-attention to encoder)
                 Guides CTC alignment during training, discarded at inference
```

---

## Groups and Scripts (13 groups, 26 scripts)

| # | Group | Scripts | LID-2 |
|---|-------|---------|-------|
| 0 | latin | latin | - |
| 1 | cyrillic_greek | cyrillic, greek | yes |
| 2 | arabic | arabic | - |
| 3 | hebrew | hebrew | - |
| 4 | sino_japanese | han_kana | - |
| 5 | korean | korean | - |
| 6 | ne_indic | devanagari, gurmukhi, gujarati, bengali, odia | yes |
| 7 | south_indic | kannada, telugu, malayalam, tamil, sinhala | yes |
| 8 | se_asian | thai, lao, burmese, khmer | yes |
| 9 | emoji | emoji | - |
| 10 | caucasus | armenian, georgian | yes |
| 11 | ethiopic | ethiopic | - |
| 12 | tibetan | tibetan | - |

---

## Training

### Loss Functions

```
loss = CTC_loss + lid1_weight x LID1_loss + LID2_loss
```

### Key Hyperparameters

```
lr:           3e-4 (shared), 1e-3 (experts)
optimizer:    AdamW (weight_decay=0.01)
scheduler:    cosine decay
amp:          bf16
```

---

## Code Structure

```
src/
  model/
    encoder.py          Re-exports from encoder_v3
    encoder_v3.py       LipiMoEEncoder (HGNetV2 + expert attention)
    memory.py           VRAM budget estimation
    lid.py              SCRIPTS, GROUPS, LIDCoarse
  encoding/
    decompose.py        CJK 13-symbol + Korean jamo decomposition
    encoding.py         Script-aware encoding pipeline
    tokenizer.py        Per-script tokenizers
    vocab.py            Frozen vocab loading
  data/
    augmentation.py     25 augmentation ops
    color.py            RGB to L+a conversion
    dataset.py          Dataset classes
    fonts.py            Font discovery
    rendering.py        Word/char rendering
  training/
    dataloader.py       Data loading and batching
    losses.py           CTC, LID-1, LID-2 losses
    routing.py          Routing mask computation
    eval.py             Per-group/per-script evaluation

scripts/
  train.py              Training
  eval.py               Standalone evaluation
  benchmark.py          Standard OCR benchmarks
  generate.py           Synthetic data generation
  run_doctr.py          Full document OCR pipeline
```
