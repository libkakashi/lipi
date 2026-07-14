# Lipi: Multilingual OCR via Mixture of Experts

> 26 routed script heads, 14 groups, 100+ languages. 210.7M stored
> parameters (v6); about 29.9M parameters touched and 13.4 GMACs for a
> single-script Latin line at 64×512 (104.6 MMACs per output frame).

---

## System Overview

Lipi reads a cropped text-line image and predicts per-frame script routing
and text in one model forward. The visual trunk is shared. LID-1 selects a
script family, LID-2 selects a script within multi-script families, and the
MoE layers share attention while routing only their channel MLPs.

The v6 vertical axis is organized around one principle: **no script-agnostic
layer ever makes an irreversible vertical decision**. Height survives to
h=2 through the group stack; the final collapse to one row is performed by
the routed script's own attention query at the LID-2 boundary, and a
zero-init ReLook lets routed frames retrieve raw stroke rows afterward.

```text
Image (B, 3, 64, W)              (32px input runs the same graph at h/2)
  -> ConvStem (PixelUnshuffle, lossless) + 3 ConvNeXt-A + 3 ConvNeXt-B
  |    -> texture tap: 4 v-bands x 2 width sub-positions   (B, T, 1280)
  -> 3 SWA-C blocks at (h=8, T=W/4, dim=256)   [kept as the ReLook grid]
  -> AttentionReadout h=8->2 (learned queries + row embeddings)
  -> 3 SWA-D blocks at (h=2, T=W/4, dim=384)
  |    -> LID-1 branch: texture merge + private SWA + 14 groups / blank
  -> 3 Group MoE layers at h=2, routed by LID-1
  |    -> LID-2 taps after group layer 2 (reads both rows + texture)
  -> RoutedReadout h=2->1: per-script queries (script-conditioned collapse)
  -> intermediate CTC (pass 1) + tied-weight posterior feedback
  -> 4 Script MoE layers, routed by LID-2
  |    -> ReLook after layer 1: cross-attention into the SWA-C grid
  -> final CTC (pass 2), 2 emission slots per frame (length 2T)
  -> decode contiguous routing runs
```

The final CTC path has 13 sequential attention blocks:

```text
3 SWA-C + 3 SWA-D + 3 group MoE attention + 4 script MoE attention
```

plus three cross-attention readouts (AttentionReadout, RoutedReadout,
ReLook) with h-way key sets — per-column, so their cost is negligible. The
private LID-1 block is on a side branch and does not deepen the final CTC
path. CTC heads are linear projections and contain no attention.

---

## Model Architecture

### Convolutional Frontend (1.0M)

Shapes shown for the 64px config (`convb_ch=160, swac_in_ch=256`). The
32px config runs the identical graph at half the row counts.

```text
Input                                              (B, 3,   64, W)
PixelUnshuffle 2x2, 3->12                           (B, 12,  32, W/2)
Conv 3x3, 12->64, stride=1 + GN + GELU            (B, 64,  32, W/2)
Conv 3x3, 64->96, stride=1 + GN + GELU            (B, 96,  32, W/2)
3 x ConvNeXt, dim=96, dw7x7, MLP ratio=4          (B, 96,  32, W/2)
BlurPool, stride=(2,1), 96->160                    (B, 160, 16, W/2)
3 x ConvNeXt, dim=160, dw7x7, MLP ratio=4         (B, 160, 16, W/2)
BlurPool, stride=(2,2), 160->256                   (B, 256,  8, W/4)
```

The stem uses **PixelUnshuffle** rather than a strided conv: each 2×2 pixel
block is packed losslessly into channels, so the first downsample discards
no information and the following conv learns its own low-pass. This is the
only downsample that touches raw pixels, and it is where the top octave
(hairline strokes, i'jam dots, thin matras) that 64px input exists to
capture would otherwise be aliased.

BlurPool applies a fixed depthwise `[1, 2, 1] x [1, 2, 1]` low-pass filter
before the later strides, followed by a `1x1` channel projection. The six
ConvNeXt blocks use LayerScale and stochastic depth but no attention.

**Texture tap.** Before `blur_bc`, the ConvB feature map is pooled into 4
vertical bands and adjacent width columns are paired (the two width
sub-positions of each output frame), giving a `(B, T, 4·2·160=1280)`
tensor that feeds LID-1, LID-2, and the CTC feedback path. Unlike the old
full-height mean, this preserves both vertical position (which tier a mark
sits on) and sub-frame stroke order (for the two CTC emission slots).

### Shared Attention Trunk (8.1M)

```text
Linear 256->256
3 x SWA-C: dim=256, heads=4, window=8x16          (B, 8*T, 256)   [64px]
AttentionReadout: h=8->2, 256->384                (B, 2*T, 384)
3 x SWA-D: dim=384, heads=6, window=2x64          (B, 2*T, 384)
```

SWA-C runs on the full 8-row grid (window area held at 128 tokens: 8×16 at
64px, 4×32 at 32px), so the finest post-stride resolution gets real
attention before any vertical compression — there is no naked resolution
level. Its output is retained as the ReLook key/value grid.

Each SWA block contains:

```text
LayerNorm -> windowed QK-normalized attention + relative-position bias
          -> LayerScale -> DropPath -> residual
LayerNorm -> MLP (ratio 4) -> LayerScale -> DropPath -> residual
```

**AttentionReadout** replaces the old fixed pairwise linear merge. Two
learned queries per column attend over the column's 8 rows; keys carry a
learned row embedding so vertical *position*, not just content, survives
the collapse. Initialized to near-uniform attention with identity value/
output projections, so at step 0 it computes the row mean (matching the
old avg-init merge) and specializes from there. An attention collapse also
*finds* the text band, so baseline wander and loose crops degrade
gracefully. Width is downsampled to `T = W/4`, and the trunk now carries
`h=2` rows all the way to the routed collapse below.

### LID-1: Script-Group Routing (2.6M)

LID-1 branches from the `h=2` SWA-D output. It combines both vertical rows
with the texture tap.

```text
lid1_merge: [row 0 | row 1 | texture(1280)], Linear(2048->384)  (B, T, 384)
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

### Group MoE Stack at h=2 (53.18M)

There are three sequential `MoELayer`s at dimension 384, run on **both
vertical rows** (window height 2, per-column group IDs repeated across
rows). Their attention is shared by every frame; only the MLP branch is
routed by the per-frame group ID. Keeping `h=2` here means no
script-agnostic layer collapses the vertical axis — the rows survive until
script identity is known.

```text
Layer 1: shared attention 2x16, unshifted
Layer 2: shared attention 2x64, shifted
Layer 3: shared attention 2x16, unshifted
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

### LID-2: Within-Group Script Routing (2.4M)

LID-2 reads the group-stack features after layer 2 — **both rows** —
concatenated with the same texture tap. Group layer 3 continues refining
the main CTC feature stream after this branch point.

Only multi-script groups have LID-2 heads:

```text
cyrillic_greek:   2-way  cyrillic, greek
ne_indic:         5-way  devanagari, gurmukhi, gujarati, bengali, odia
dravidian_north:  3-way  kannada, telugu, sinhala
dravidian_south:  2-way  malayalam, tamil
se_asian:         4-way  thai, lao, burmese, khmer
caucasus:         2-way  armenian, georgian
```

Single-script groups assign local script ID 0 without a classifier.

### Script-Conditioned Vertical Collapse (0.46M)

`RoutedReadout` collapses the two rows to one frame at the LID-2 boundary —
the first and only script-conditioned vertical decision. Each flat script
ID owns a learned attention query (plus a shared default for blank/unrouted
frames); the routed script decides how its frames weight the two rows. A
Thai frame can weight the tone-mark tier while a Latin frame stays near
uniform, without any upstream layer having to hedge across scripts.

```text
RoutedReadout: h=2->1, per-script queries + row embeddings   (B, T, 384)
```

Initialized to the row mean (uniform attention, identity value/output), so
it matches the old avg-init `merge_d1` at step 0.

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

### Script MoE Stack (127.62M) and ReLook

The feedback-conditioned features pass through **four** sequential script
MoE layers at `h=1`, routed per frame by the flat 26-script ID:

```text
Layer 1: shared attention w=16, unshifted
  -> ReLook: cross-attention into the SWA-C grid
Layer 2: shared attention w=64, shifted
Layer 3: shared attention w=16, unshifted
Layer 4: shared attention w=64, shifted
```

Each layer has one shared attention branch, 26 routed script MLPs at ratio
4, and one always-on shared MLP at ratio 2. One layer stores about 31.9M
parameters. Attention weights are shared across scripts within a layer; the
four layers do not share weights with one another. The fourth layer is new
in v6 — routed capacity is the cheapest place to add parameters (only one
expert fires per frame, so `+33M` params costs `~+0.3 GMACs`).

**ReLook** (retrieval after routing). After the first script layer, each
frame — which by now knows what script it is reading — cross-attends back
into its column's 8 SWA-C rows (keys carry row embeddings). This lets a
routed frame retrieve near-raw stroke evidence on demand, so the shared
trunk no longer has to preserve every vertical detail losslessly. The
output projection is zero-initialized, so ReLook is an exact no-op at step
0 and grows only if useful.

### Final CTC and Decoding (14.87M)

There are 26 per-script linear heads, each emitting **2 tokens per frame**:

```text
LayerNorm(384) -> selected Linear(384 -> 2 * script_vocab_size)
              -> reshape to (T, 2, vocab), interleave to length 2T
```

Two emission slots per frame give adjacent same-token pairs (Arabic tooth
runs, Korean jamo within one syllable block) room for their CTC blank
separator — at one token per frame those segments fail the
`len(ids)+repeats` feasibility check and were silently dropped from the
loss. The self-conditioning feedback is position-aware: the tied head
weight is viewed `(2, vocab, dim)` and each emission slot's posterior is
fed back through its own block, adding no parameters.

Both CTC passes use the same heads. CTC loss and decoding operate on
contiguous routing runs rather than forcing one script label onto the whole
line. Greedy decoding performs argmax, repeat collapse, blank removal, and
codec decoding, at the 2T emission resolution.

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
| 4 | han | han | 3811 | no |
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
| Han | han | direct frequent characters plus ALT slots for rare visually similar characters |
| Korean | korean | onset, vowel, and coda Jamo decomposition |

Han characters, including Chinese Hanzi, Japanese Kanji, and Korean Hanja,
route to the single `han` head. The stable direct-character manifest covers
frequent characters, and rare ALT characters follow the route of their
visual prototype.

Mixed lines are split into maximal routing runs. Japanese text separates
Kana from Han. ASCII punctuation
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
- Drop path rises linearly from 0 to 0.1 across 19 stages: 3 ConvA,
  3 ConvB, 3 SWA-C, 3 SWA-D, 3 group MoE, and 4 script MoE layers.
- Routed expert output projections are zero-initialized; MoE residual
  LayerScale starts at `1e-4`.
- Script-balanced sampling mass is proportional to
  `vocab_size(script)^beta` with default `beta=0.5`.

### Selective Freezing

```text
--freeze-except lid          LID-1 and LID-2 branches
--freeze-except lid1         LID-1 merge, private attention, and head
--freeze-except lid2         LID-2 heads
--freeze-except backbone     stem, ConvNeXt, SWA-C/D, readouts, ReLook, norm
--freeze-except ctc          per-script CTC heads
--freeze-except experts      group and script MoE stacks
```

Values can be comma-separated.

---

## Parameter and Compute Breakdown

Measured for the 64px config (`dim=384, convb_ch=160, swac_in_ch=256`):

| Component | Parameters | Share |
|-----------|-----------:|------:|
| Convolutional frontend | 1.00M | 0.5% |
| SWA-C/D and input projection | 7.78M | 3.7% |
| AttentionReadout (h=8->2) | 0.33M | 0.2% |
| LID-1 branch | 2.64M | 1.3% |
| Group MoE stack, 3 layers | 53.18M | 25.2% |
| LID-2 heads | 2.36M | 1.1% |
| RoutedReadout (h=2->1) | 0.46M | 0.2% |
| Script MoE stack, 4 layers | 127.62M | 60.6% |
| ReLook | 0.50M | 0.2% |
| Final norm and feedback gate | 0.001M | <0.1% |
| CTC heads, 26 scripts (2 emission slots) | 14.87M | 7.1% |
| **Total stored** | **210.74M** | **100%** |

All expert banks are stored and optimized, but routing only executes the
MLPs needed by the frames in the batch. A single-script line touches about
29.9M unique parameters for Latin and 32.2M for Han. Mixed-script lines can
activate additional routed MLPs and CTC heads.

Measured with `torch.utils.flop_counter.FlopCounterMode`, batch size 1, and
ground-truth routing:

| Input | Active path |
|-------|------------:|
| Latin, 64×512, `T=128` | 13.39 GMACs |
| Han, 64×512, `T=128` | 14.28 GMACs |

The conv frontend is ~50% of the compute at ~4% of active params (a weight
in ConvA fires at 64 positions per output frame; a routed expert weight
fires at most once per frame and only when its script is present). The CTC
head vocabulary causes the small per-script difference. Both CTC passes and
both emission slots are included.

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
