# Lipi: Multilingual OCR via Mixture of Experts

> 27 routed script heads, 14 groups, 100+ languages. 171.9M stored
> parameters; about 26.4M parameters touched and 4.0 GFLOPs for a
> single-script Latin crop at `W=128`.

---

## System Overview

Lipi reads a cropped text-line image and predicts per-frame script routing
and text in one model forward. The visual trunk is shared. LID-1 selects a
script family, LID-2 selects a script within multi-script families, and the
MoE layers share attention while routing only their channel MLPs.

```text
Image (B, 3, 32, W)
  -> ConvStem + 3 ConvNeXt-A + 3 ConvNeXt-B
  -> 3 SWA-C blocks at (h=4, T=W/4, dim=256)
  -> merge h=4->2
  -> 3 SWA-D blocks at (h=2, T=W/4, dim=384)
  |    -> LID-1 branch: texture merge + private SWA + 14 groups / blank
  -> merge h=2->1
  -> 3 Group MoE layers, routed by LID-1
  |    -> LID-2 taps after group layer 2
  -> intermediate CTC (pass 1) + tied-weight posterior feedback
  -> 3 Script MoE layers, routed by LID-2
  -> final CTC (pass 2)
  -> decode contiguous routing runs
```

The current final CTC path has 12 sequential attention blocks:

```text
3 SWA-C + 3 SWA-D + 3 group MoE attention + 3 script MoE attention
```

The private LID-1 block is a thirteenth instantiated attention block, but it
is on a side branch and does not deepen the final CTC path. CTC heads are
linear projections and contain no attention.

---

## Model Architecture

### Convolutional Frontend (0.75M)

```text
Input                                              (B, 3,   32, W)
Conv 3x3, 3->64, stride=(2,2) + GN + GELU         (B, 64,  16, W/2)
Conv 3x3, 64->96, stride=(1,1) + GN + GELU        (B, 96,  16, W/2)
3 x ConvNeXt, dim=96, dw7x7, MLP ratio=4          (B, 96,  16, W/2)
BlurPool, stride=(2,1), 96->128                    (B, 128,  8, W/2)
3 x ConvNeXt, dim=128, dw7x7, MLP ratio=4         (B, 128,  8, W/2)
BlurPool, stride=(2,2), 128->192                   (B, 192,  4, W/4)
```

BlurPool applies a fixed depthwise `[1, 2, 1] x [1, 2, 1]` low-pass filter
before striding, followed by a `1x1` channel projection where required.
The six ConvNeXt blocks use LayerScale and stochastic depth but no
attention.

### Shared Attention Trunk (7.76M)

```text
Linear 192->256
3 x SWA-C: dim=256, heads=4, window=4x32          (B, 4*T, 256)
merge_cd: h=4->2, Linear(512->384)                (B, 2*T, 384)
3 x SWA-D: dim=384, heads=6, window=2x64          (B, 2*T, 384)
```

Each SWA block contains:

```text
LayerNorm -> windowed QK-normalized attention + relative-position bias
          -> LayerScale -> DropPath -> residual
LayerNorm -> MLP (ratio 4) -> LayerScale -> DropPath -> residual
```

Blocks alternate unshifted, shifted, unshifted windows. Width is already
downsampled to `T = W/4`, so all later routing and CTC operations use the
same frame resolution.

### LID-1: Script-Group Routing (2.20M)

LID-1 branches from the `h=2` SWA-D output before the main path is merged to
one row. It combines both vertical rows with a texture feature taken from
ConvB.

```text
ConvB texture: mean over h=8, average-pool width by 2     (B, T, 128)
lid1_merge: [row 0 | row 1 | texture], Linear(896->384)  (B, T, 384)
lid1_attn: SWABlock(dim=384, heads=6, window=1x32)
group_head: Linear(384->192) + GELU + Linear(192->15)

Output: group_logits (B, T, 15) = 14 groups + blank
```

The merge starts as an average of the two SWA-D rows, with the texture
columns zero-initialized. The private attention block starts as an identity
because its attention and MLP output projections are zero-initialized.

At training time, routing normally uses ground-truth frame labels, with a
scheduled fraction replaced by LID predictions. At inference, routing uses
the LID-1 argmax. Blank frames still receive shared attention and the shared
MLP but skip the routed group MLP.

### Main 2D-to-1D Merge (0.30M)

```text
merge_d1: h=2->1, Linear(768->384)
Output: (B, T, 384), T=W/4
```

It is initialized as the average of the two input rows. This path is
separate from `lid1_merge`, so the CTC feature merge and LID-1 feature merge
can specialize independently.

### Group MoE Stack (53.18M)

There are three sequential `MoELayer`s at dimension 384. Their attention is
shared by every frame; only the MLP branch is routed by the per-frame group
ID.

```text
Layer 1: shared attention w=16, unshifted
Layer 2: shared attention w=64, shifted
Layer 3: shared attention w=16, unshifted
```

Each layer contains:

```text
shared LayerNorm + shared windowed attention
shared LayerNorm
  -> one selected MLP from 14 routed group MLPs (ratio 4)
  -> one always-on shared MLP (ratio 2)
  -> sum, LayerScale, DropPath, residual
```

Each group layer stores about 17.73M parameters. The routed MLP output
projections start at zero and both residual branches use LayerScale
initialized to `1e-4`.

### LID-2: Within-Group Script Routing (0.69M)

LID-2 reads the group-stack features after layer 2, concatenated with the
same 128-dimensional ConvB texture tap. Group layer 3 continues refining
the main CTC feature stream after this branch point.

Only multi-script groups have LID-2 heads:

```text
cyrillic_greek:   2-way  cyrillic, greek
han:              2-way  han_sparse, han_dense
ne_indic:         5-way  devanagari, gurmukhi, gujarati, bengali, odia
dravidian_north:  3-way  kannada, telugu, sinhala
dravidian_south:  2-way  malayalam, tamil
se_asian:         4-way  thai, lao, burmese, khmer
caucasus:         2-way  armenian, georgian
```

Single-script groups assign local script ID 0 without a classifier.

### Intermediate CTC and Self-Conditioning

After all three group layers, the selected per-script CTC head makes a full
first prediction over every frame:

```text
inter_logits = CTC(LayerNorm(group_features))
posterior = softmax(inter_logits over the selected script vocabulary)
feedback = posterior @ CTC_head.weight
x = group_features + zero_init_LayerScale(feedback)
```

The projection is tied to the same CTC weights used for prediction, so the
feedback adds only a 384-parameter gate. The gate is initialized to zero,
which makes the feature path identical to the pre-self-conditioning model
at initialization. During training, `inter_logits` also receive the
auxiliary intermediate-CTC loss.

This is currently one-step self-conditioning: there are two full CTC
predictions and one feedback event. It is not an iterative diffusion loop.

### Script MoE Stack (99.26M)

The feedback-conditioned features pass through three sequential script
MoE layers, routed per frame by the flat 27-script ID:

```text
Layer 1: shared attention w=16, unshifted
Layer 2: shared attention w=64, shifted
Layer 3: shared attention w=16, unshifted
```

Each layer has one shared attention branch, 27 routed script MLPs at ratio
4, and one always-on shared MLP at ratio 2. One layer stores about 33.09M
parameters:

```text
shared attention branch:   0.593M
27 routed script MLPs:    31.902M
shared MLP branch:         0.592M
```

Attention weights are shared across scripts within a layer. The three
layers do not share weights with one another.

### Final CTC and Decoding (7.57M)

There are 27 per-script linear heads:

```text
LayerNorm(384) -> selected Linear(384->script_vocab_size)
```

Both CTC passes use the same heads. CTC loss and decoding operate on
contiguous routing runs rather than forcing one script label onto the whole
line. Greedy decoding performs argmax, repeat collapse, blank removal, and
codec decoding.

Image pixels and encoder frames are stored left-to-right. Arabic and Hebrew
segment frame slices are reversed before CTC loss and greedy decoding so
CTC time follows logical right-to-left reading order.

---

## Groups and Scripts

| # | Group | Scripts | Vocabulary sizes | LID-2 |
|---|-------|---------|------------------|-------|
| 0 | latin | latin | 797 | no |
| 1 | cyrillic_greek | cyrillic, greek | 364, 426 | yes |
| 2 | arabic | arabic | 500 | no |
| 3 | hebrew | hebrew | 192 | no |
| 4 | han | han_sparse, han_dense | 2153, 2018 | yes |
| 5 | kana | kana | 263 | no |
| 6 | korean | korean | 1500 | no |
| 7 | ne_indic | devanagari, gurmukhi, gujarati, bengali, odia | 1000, 600, 900, 900, 850 | yes |
| 8 | dravidian_north | kannada, telugu, sinhala | 550, 950, 500 | yes |
| 9 | dravidian_south | malayalam, tamil | 900, 350 | yes |
| 10 | se_asian | thai, lao, burmese, khmer | 450, 500, 650, 950 | yes |
| 11 | caucasus | armenian, georgian | 150, 186 | yes |
| 12 | ethiopic | ethiopic | 521 | no |
| 13 | tibetan | tibetan | 550 | no |

Emoji is not part of the taxonomy, model, codecs, routing, or data
generation.

---

## Encoding and Segmentation

| Type | Scripts | Method |
|------|---------|--------|
| No fusion | latin, cyrillic, greek, hebrew, armenian, georgian, ethiopic, kana | one Unicode character per token |
| Fusion | arabic, devanagari, gurmukhi, gujarati, bengali, odia, kannada, telugu, sinhala, malayalam, tamil, thai, lao, burmese, khmer, tibetan | grapheme clusters, learned frequent fusions, codepoint fallback |
| Han | han_sparse, han_dense | direct frequent characters plus ALT slots for rare visually similar characters |
| Korean | korean | onset, vowel, and coda Jamo decomposition |

Han characters, including Chinese Hanzi, Japanese Kanji, and Korean Hanja,
route to `han_sparse` or `han_dense`. The stable direct-character manifest
uses Noto Sans CJK outline complexity with a threshold of 60 recorded outline
operations. Rare ALT characters follow the route of their visual prototype,
with IDS component count as a fallback.

Mixed lines are split into maximal routing runs. Japanese text separates
Kana from Han and also separates sparse from dense Han. ASCII punctuation
and digits become Latin segments even inside non-Latin words. CJK
punctuation attaches to an adjacent Han route. Emoji codepoints are rejected
from generated and converted data.

---

## Training

### Data and Routing

Synthetic data is built from multilingual word lists and font registries,
with mixed-script content plans and per-segment pixel offsets. Real datasets
are converted to the same MDS format. Frame labels and CTC slices use the
model's fixed `W/4` time downsample.

Training normally routes group and script MLPs with ground truth. Scheduled
routing sampling replaces a per-frame fraction with LID predictions,
linearly increasing from 0 to `--route-sample-max` (default 0.25), while
loss targets remain ground truth.

### Losses

```text
loss = ctc_weight       * final_CTC
     + inter_ctc_weight * intermediate_CTC
     + lid1_weight      * LID1_CE
     + lid2_weight      * LID2_CE
     + consistency_weight * two_view_consistency
```

Defaults:

```text
ctc_weight=1.0
inter_ctc_weight=0.3
lid1_weight=1.0
lid2_weight=1.0
consistency_weight=0.0
```

Both CTC losses are computed per segment and batched by `(group, script)`.
LID-1 is per-frame cross-entropy over 14 groups plus blank. LID-2 is
per-frame cross-entropy within present multi-script groups with label
smoothing. Optional two-view consistency uses symmetric KL between aligned
augmented views for LID-1 and script-vocabulary-sliced CTC posteriors.

### Optimization

- AdamW parameters remain FP32; autocast supplies BF16/FP16 forward compute.
- EMA defaults to 0.999 and evaluation uses EMA weights when enabled.
- Drop path rises linearly from 0 to 0.1 across 18 stages: 3 ConvA,
  3 ConvB, 3 SWA-C, 3 SWA-D, 3 group MoE, and 3 script MoE layers.
- Routed expert output projections are zero-initialized; MoE residual
  LayerScale starts at `1e-4`.
- Script-balanced sampling mass is proportional to
  `vocab_size(script)^beta` with default `beta=0.5`.

### Selective Freezing

```text
--freeze-except lid          LID-1 and LID-2 branches
--freeze-except lid1         LID-1 merge, private attention, and head
--freeze-except lid2         LID-2 heads
--freeze-except backbone     stem, ConvNeXt, SWA-C/D, merges, final norm
--freeze-except ctc          per-script CTC heads
--freeze-except experts      group and script MoE stacks
```

Values can be comma-separated.

---

## Parameter and Compute Breakdown

| Component | Parameters | Share |
|-----------|-----------:|------:|
| Convolutional frontend | 0.75M | 0.4% |
| SWA-C/D and SWA-C input projection | 7.76M | 4.5% |
| Height merges | 0.49M | 0.3% |
| LID-1 branch | 2.20M | 1.3% |
| Group MoE stack, 3 layers | 53.18M | 30.9% |
| LID-2 heads | 0.69M | 0.4% |
| Script MoE stack, 3 layers | 99.26M | 57.7% |
| Final norm and feedback gate | 0.001M | <0.1% |
| CTC heads, 27 scripts | 7.57M | 4.4% |
| **Total stored** | **171.91M** | **100%** |

All expert banks are stored and optimized, but routing only executes the
MLPs needed by the frames in the batch. A single-script line touches about
26.4M unique parameters for Latin and 26.9M for Han. Mixed-script lines can
activate additional routed MLPs and CTC heads.

Measured with `torch.utils.flop_counter.FlopCounterMode`, batch size 1, and
ground-truth routing:

| Input | Active path |
|-------|------------:|
| Latin, `W=128`, `T=32` | 3.975 GFLOPs |
| Han sparse, `W=128`, `T=32` | 4.075 GFLOPs |
| Han dense, `W=128`, `T=32` | 4.065 GFLOPs |
| Latin, `W=256`, `T=64` | 7.258 GFLOPs |

The CTC head vocabulary causes the small per-script FLOP difference. Both
CTC passes are included.

---

## Code Structure

```text
src/
  taxonomy.py                  script/group IDs and mappings
  model/
    encoder.py                 LipiMoEEncoder and complete forward path
    blocks.py                  ConvNeXt, SWA, MoELayer, CTC heads, helpers
    memory.py                  memory instrumentation helpers
  encoding/
    config.py                  per-script codecs and vocabulary definitions
    decompose.py               encode_text, decode_ids, vocab dispatch
    direction.py               RTL CTC traversal helpers
    han_split.py               sparse/dense Han routing
    vocab.py                   fusion and frequency vocabulary construction
  data/
    font_registry.py           font-to-script registry
    fonts.py                   font discovery and filtering
    script_detect.py           Unicode detection and mixed-run splitting
    rendering.py               rendering orchestration
    text_renderer.py           low-level PIL/FreeType rendering
    word_lists.py              multilingual word-list loading
    augmentation.py            OCR image augmentation
  training/
    dataloader.py              MDS dataset, tokenizers, bucket sampling
    losses.py                  final/intermediate CTC, LID, consistency
    routing.py                 pixel-to-frame routing labels
    eval.py                    group/script/word/character evaluation
    checkpoint.py              checkpoint key normalization
    taxonomy_checkpoint.py     taxonomy-aware checkpoint migration
    ema.py                     model weight EMA

scripts/
  data/
    generate.py                synthetic mixed-script data generation
    convert_to_mds.py          external shard conversion
    build_han_split.py         outline-complexity Han manifest
    cjk_visual_similarity.py   Han ALT visual-prototype mapping
    setup/                     source fetchers and real-data conversion
  train/
    train.py                   training, resume, freezing, EMA, scheduling
    train_lid0_probe.py        auxiliary LID probe
  eval/
    eval.py                    validation-set evaluation
    benchmark.py               standard OCR benchmarks
  inference/
    run_doctr.py               document OCR pipeline
  tools/
    count_flops.py             active-path FLOP measurement
    diagnose.py                model/data diagnostics
```
