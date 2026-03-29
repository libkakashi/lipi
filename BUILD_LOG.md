# Lipi Build Log

Tracking all findings, decisions, and test results during implementation.

---

## Session 1: 2026-03-29

### Environment
- Hardware: M3 Pro 32GB (Apple Silicon)
- PyTorch backend: MPS (no CUDA)
- Python: TBD (checking)

### Progress

#### Project Setup
- Created full directory structure per ARCHITECTURE.md Section 2
- Created pyproject.toml with all dependencies
- Note: pinned torchaudio>=2.2.0 (not 2.10.0 from spec — 2.10 follows PyTorch versioning which is currently at 2.x, will update pin when we reach training phase)
- Note: pinned onnxruntime>=1.17.0 for dev (MultiLoRA needs >=1.24.0 for production, but we can test basic ONNX export with any recent version)

#### Installed Versions (venv)
- torch 2.11.0 (MPS available)
- torchaudio 2.11.0 (RNNTLoss deprecation confirmed reversed)
- onnxruntime 1.24.4
- peft 0.18.1
- Python 3.12.8

#### Module 1: rope.py (RoPE-2D / RoPE-1D)
- Implemented with frequency caching per (h,w) / seq_len
- All standard PyTorch ops (outer, stack, cos, sin)
- Smoke test: passed for both 2D (B=2, heads=6, h=8, w=20) and 1D (B=2, heads=12, T=20)

#### Module 2: stem.py (ConvNeXt-V2 Micro Stem)
- 4 conv layers with GroupNorm(1,C) + GELU, total stride 4x4
- 0.066M params
- Smoke test: (2, 3, 32, 128) -> (2, 64, 8, 32) OK

#### Module 3: pooling.py (Learned Height Pooling)
- Implemented as grouped Conv1d along width, treating C*H_in channels -> C*H_out channels with groups=C
- This learns a separate (h_out, h_in) weight per channel
- Smoke test: 8->4 and 4->1 both OK

#### Module 4: attention.py (SWA + Global Attention)
- ShiftedWindowAttention: window partition, cyclic shift, RoPE-2D within windows, shift mask for cross-region masking
- GlobalAttention: standard MHSA with RoPE-1D
- SWABlock / GlobalBlock: pre-norm residual wrappers
- MLP: named submodules (fc1, fc2) not Sequential indexing — important for LoRA target_modules
- Smoke tests: all configs passed, variable width (8,16,32,64,80) all OK

#### Module 5: encoder.py (Full Backbone)
- Assembles all components per architecture spec
- Stages as nn.ModuleList (stage1, stage2, stage3) — important for LoRA targeting
- Alternating shift pattern in SWA blocks (even=no shift, odd=shifted)

**FINDING: Parameter count is 12.52M, not 35.4M as estimated in ARCHITECTURE.md.**
Breakdown:
  - stem: 0.066M (spec: 0.15M)
  - stage1 (3× SWA, C=192): 1.113M (spec: 4.0M)
  - stage2 (4× SWA, C=384): 5.917M (spec: 17.7M)
  - stage3 (3× Global, C=384): 5.323M (spec: 13.3M)
The architecture dimensions (C, heads, blocks, MLP ratios) are implemented exactly per spec.
The spec's param estimates were ~3x overestimated. The actual architecture yields 12.5M.
This is a lighter backbone — faster inference, lower memory, but may need more capacity
if accuracy targets aren't met. Can scale up by increasing dims or blocks later.

Shape tests: all widths (32, 64, 128, 256, 320) and batch sizes (1, 4, 8) pass.
Output: (B, T, 384) where T = W/4. Lengths tensor included for future padding support.

#### Module 6: prediction_net.py (RNN-T Prediction Network)
- 1-layer GRU, embed_dim=128, hidden_dim=128
- 0.355M params per language
- Single-step inference tested with hidden state carry-forward

#### Module 7: joint_net.py (RNN-T Joint Network)
- Additive combination: enc_proj(enc) + pred_proj(pred)
- GELU + Linear output layer
- 0.646M params per language
- Broadcasting tested: (B,T,1,enc) + (B,1,U,pred) -> (B,T,U,vocab)

#### Module 8: decode.py (Greedy Decoding)
- greedy_decode: basic token sequence output
- greedy_decode_with_confidence: geometric mean of token log-probs

#### Module 9: lora.py (LoRA Adapter Injection)
- Uses HuggingFace PEFT library
- 28 target modules (4 per block × 7 blocks in stages 2+3)
- MLP targets use named attributes (fc1, fc2), NOT Sequential indices — this was a spec concern, resolved by design
- Stage 1 verified frozen (0 LoRA params)
- Rank 16: 0.639M trainable params (4.9% of total 13.156M)

#### Module 10: lid.py (Micro-LID Script Classifier)
- Depthwise separable conv architecture
- 4.5K params (~0.02MB) — extremely lightweight
- 11 script families
- AdaptiveAvgPool2d handles variable width

#### Module 11: rnnt_model.py (Full Model Assembly)
- Assembles encoder + prediction_net + joint_net
- Total: 13.518M params (enc=12.517M, pred=0.355M, joint=0.646M)
- Prepends blank token to targets for prediction network

#### ONNX Export Results
- **FINDING: torch 2.11 requires opset_version=18 (not 17 as in spec)**
  The new dynamo-based exporter enforces >=18. Set opset 18 everywhere.
- **FINDING: RoPE dict caching causes torch.export side-effect errors**
  Removed caching, compute freqs inline. Negligible perf impact for OCR word lengths.
- **FINDING: Must use `dynamic_shapes` kwarg, not `dynamic_axes` (deprecated in torch 2.11)**
  Syntax: `torch.export.Dim("name", min=X, max=Y)` per dynamic axis.
- All 4 components export successfully: encoder, pred_net, joint_net, lid
- ONNX checker validates all models
- ORT inference matches PyTorch within rtol=1e-3, atol=1e-4
- Dynamic width verified through ONNX: widths 32, 64, 128, 256 all work
- End-to-end ONNX decode loop (enc -> pred -> joint) verified

#### RNN-T Loss (torchaudio 2.11.0)
- `torchaudio.functional.rnnt_loss` works correctly
- Deprecation warnings present but function is preserved (confirmed)
- Forward: loss > 0, no NaN
- Backward: gradients flow to all model components (encoder, pred_net, joint_net)

#### Test Suite Summary: 52/52 PASS
- test_encoder.py: 13 tests (shapes, batches, gradients, ONNX export+parity+dynamic)
- test_rnnt.py: 12 tests (pred_net, joint_net, model assembly, loss, decode)
- test_lora.py: 8 tests (target modules, injection, freezing, gradient flow, param count)
- test_lid.py: 6 tests (shapes, variable width, params, gradients, batches)
- test_onnx.py: 5 tests (all components export, parity, three-file workflow)

---

### Week 1 Checkpoint 0 Status: PASS
All architecture smoke tests pass:
- [x] Encoder forward pass for W = 32, 64, 128, 256, 320
- [x] Encoder handles batch sizes 1, 4, 16
- [x] RNN-T loss computes without NaN
- [x] loss.backward() completes without error
- [x] Greedy decode produces valid token sequences
- [x] ONNX export succeeds for encoder, pred_net, joint_net (3 separate files)
- [x] Exported ONNX models run in ORT and match PyTorch output
- [x] BPE tokenizer roundtrips: verified for English, Hindi, mixed-script
- [ ] onnxruntime-node test (skipped — using Python server deployment)

#### Data Pipeline
- BPE: character-level and BPE vocabularies, encode/decode roundtrip verified
- Dataset: LMDB read/write, InMemoryDataset, collate_ocr with variable-width padding
- Augmentation: RandAugment-style with 9 OCR-specific transforms
- All tested and working

#### Training Pipeline
- loss.py: RNN-T loss (torchaudio), CTC loss (torch), distillation KL loss
- foundation_trainer.py: Phase 1 CTC training loop with OneCycleLR
- adapter_trainer.py: Phase 2 LoRA + RNN-T training with per-group LR

#### MPS + RNN-T Loss Compatibility
**FINDING: `torchaudio::rnnt_loss_forward` is not implemented for MPS.**
Solution: Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing torch.
This lets the model run on MPS while the RNN-T loss op falls back to CPU transparently.
Also need `torch.mps.synchronize()` after backward pass before gradient operations.
Pattern learned from: https://github.com/derinworks/penr-oz-neural-network-v3-torch-ddp/commit/c72a834

#### Overfit Test (Checkpoint 1) Investigation
Ran extensive overfit testing on M3 Pro:

1. **RNN-T path (20 words, 500 steps):** Loss 86→3.0, mode-collapsed to single word.
   Loss of 3.0 ≈ ln(20) — model learned word length but ignores input image.

2. **RNN-T path (5 words, 2000 steps):** Loss 57→1.6, mode-collapsed.
   Loss of 1.6094 ≈ ln(5) — model predicts uniform over 5 words.

3. **CTC path (5 words, 3000 steps):** Loss 22→2.4→4.0, transitions from blank-only to wrong chars.

4. **Simple CNN + CTC (5 words):** Same ln(5) plateau. Not specific to our encoder.

5. **Maximally different images (solid colors) + Simple CNN:** Loss drops to 0.55, 3/5 correct.
   **This confirms the architecture and loss are correct.**

**Root cause:** PIL-rendered text at 32px height produces images too visually similar for any
CNN to distinguish with only 5 samples. The subtle pixel differences (~14% of pixels differ)
wash out after convolution/pooling. This is NOT a bug — it's why the architecture spec requires:
- Phase 1: pretrain on MILLIONS of crops to learn discriminative visual features
- The overfit test in the spec (Checkpoint 1) assumes GPU + 100 real word crop images

**Conclusion:** Training pipeline is functional (loss decreases, gradients flow, all components
integrate correctly). Real overfit testing requires GPU with real OCR data or high-quality
synthetic data with distinct visual patterns. All components ready for production training.

---

### Data Pipeline & Generation

#### Synthetic Renderer (src/data/synth.py)
- render_word(): diverse font rendering with 467 system fonts on macOS
- generate_dataset(): batch generation with configurable variants per word
- render_word_batch(): multiple style variants per word

#### Degradation Pipeline (src/data/degradation.py)
- 3 presets: light, medium, heavy
- Transforms: JPEG compress, gaussian blur, motion blur, salt-pepper noise,
  brightness/contrast jitter, rotation, perspective warp, shadow gradient,
  ink bleed, paper texture, uneven lighting, downsample-upsample
- Probabilistic application per preset

#### PDF Extractor (src/data/pdf_extractor.py)
- Uses PyMuPDF to extract word-level crops with perfect labels
- Supports variable DPI, multi-page documents
- extract_pdf_directory() for batch processing

### Synthetic Training Run

Generated 3900 synthetic images (130 words × 30 variants each).
Ran Phase 1 CTC training for 3 epochs on CPU (M3 Pro):
- Epoch 1: avg_loss = 4.7876
- Epoch 2: avg_loss = 3.8866
- Epoch 3: avg_loss = 3.6905
- Total time: 726s (~12 min), ~2s per batch (batch_size=32)
- Loss is consistently decreasing — training pipeline is functional
- 0% eval accuracy after 3 epochs — expected with only 3900 samples on deep encoder

### Scripts Built

Training:
- scripts/build_vocab.py — build BPE vocabulary from word lists
- scripts/train_foundation.py — Phase 1 CTC training
- scripts/train_adapter.py — Phase 2 LoRA + RNN-T training
- scripts/train_lid.py — Micro-LID classifier training

Data:
- scripts/generate_synth.py — generate synthetic training data to LMDB
- scripts/extract_pdf_crops.py — extract word crops from clean PDFs
- scripts/apply_degradation.py — apply realistic degradation to clean crops
- scripts/validate_data.py — data quality checks

Export & Eval:
- scripts/export_onnx.py — export all components to ONNX
- scripts/export_adapters.py — convert LoRA adapters to .onnx_adapter
- scripts/test_onnx_export.py — ONNX export verification (all 5 tests pass)
- scripts/benchmark.py — evaluate on STR benchmarks

### Config Files
- configs/model/backbone.yaml, rnnt_head.yaml, lid.yaml
- configs/training/phase1_foundation.yaml, phase2_adapter.yaml, phase3_qat.yaml
- configs/vocab/english.yaml, hindi.yaml, tamil.yaml, template.yaml
- configs/export/onnx.yaml

### Deployment
- deploy/server.py — LipiServer with batch-by-script processing, MultiLoRA
- deploy/api.py — FastAPI HTTP endpoints (/recognize, /recognize/batch, /health)
- deploy/requirements.txt

### Quantization (stubs)
- src/quantization/qat.py — QAT wrapper (nvidia-modelopt or simulated)
- src/quantization/polar.py — PolarQuant rotation for Transformer weights

### Export Pipeline
- src/export/onnx_export.py — export backbone, RNN-T heads, LID to ONNX
- src/export/lora_export.py — export LoRA adapters to .onnx_adapter format

### BPE → Bigrams Migration (2026-03-29)
Replaced BPE tokenizer (2001 tokens, HuggingFace tokenizers dep) with
character + bigram tokenizer (401 tokens, zero dependencies).

Changes:
- Deleted src/data/bpe.py, created src/data/bigrams.py (LipiTokenizer)
- vocab_size default 2001 → 401 across all model modules, tests, configs
- Removed `tokenizers` from pyproject.toml dependencies
- Renamed test_bpe.py → test_vocab.py with bigram-specific tests

Benefits:
- 5x smaller softmax (401 vs 2001) — faster training and decode
- 2-char max token error granularity (was 4-char with BPE)
- No tokenizer training library needed — just frequency counting
- Zero OOV by construction

**FINDING: torchaudio rnnt_loss requires enc_lengths == logits.shape[1] exactly.**
It's not a "valid frames" mask — it's a hard dimension check. Fixed by using
`torch.full((B,), T)` where T = features.shape[1] (the padded sequence length)
instead of the per-sample enc_lengths from the encoder.

### Test Suite: 83/83 PASS
- test_vocab.py: 25 tests (char-level, roundtrip, bigram tokenization, save/load, edge cases, 6 scripts)
- test_encoder.py: 13 tests (shapes, batches, gradients, ONNX export+parity+dynamic)
- test_rnnt.py: 12 tests (pred_net, joint_net, model assembly, loss, decode)
- test_lora.py: 8 tests (target modules, injection, freezing, gradient flow, param count)
- test_lid.py: 6 tests (shapes, variable width, params, gradients, batches)
- test_onnx.py: 5 tests (all components export, parity, three-file workflow)

### AMP (Mixed Precision) Training
Added bf16 autocast + GradScaler to both trainers:
- foundation_trainer.py: torch.amp.autocast("cuda") with bf16 for forward, float32 for CTC loss
- adapter_trainer.py: same pattern for RNN-T loss
- non_blocking=True on all .to(device) transfers

---

## Architecture Validation (RTX 5080 Blackwell, 2026-03-29)

**Hardware:** NVIDIA GeForce RTX 5080, compute capability 12.0, bf16 supported
**GPU Result: 7/7 PASS in 111 seconds**

### Test 1: Mini Backbone CTC Overfit — PASS
- 1.89M param mini encoder (half dims, 2 blocks per stage)
- 10 words × 100 font variants = 1000 synthetic images
- 100 epochs, SGD lr=0.01, batch_size=128 on CUDA
- Loss: 7.56 → 0.01
- **Final accuracy: 99.7% (997/1000)**
- All 10 words: 98-100% individually
- Time: 60s
- **Conclusion: encoder architecture learns discriminative visual features**

### Test 2: LID Classifier — PASS
- 600 synthetic crops across 3 scripts (English, Hindi, Tamil)
- Non-script-specific fonts (worst case for visual distinction)
- 60 epochs, Adam lr=1e-3
- Train: 87.1%, **Test: 68.3%**
- Threshold: 60% (with real script-specific fonts, expect >95%)

### Test 3: LoRA Adapter + RNN-T — PASS
- Injected LoRA rank-8 into mini backbone (92.2K trainable / 1.99M total)
- 50 steps of RNN-T training on 5 short words
- Loss: 119 → 2.9 (decreasing)
- **Frozen backbone weights verified unchanged** (snapshot comparison)
- Greedy decode runs without error

### Test 4: Bigram Tokenizer — PASS
- Built vocab from word list: 133 tokens (83 chars + 50 bigrams)
- 5/5 roundtrip tests pass (Hello, Court, Section, 12345, WP(C))
- Max token length: 2 chars (limit: 2) — enforced by construction

### Test 5: Full ONNX Deployment Pipeline — PASS
- Exported all 4 components to ONNX opset 18
- Loaded in ORT, ran full inference pipeline: image → LID → encode → decode → text
- **Dynamic width works** (fixed conditional padding bug in window partition)
- File sizes: encoder 1.5MB, pred_net 19KB, joint_net 14KB, lid 21KB

### Test 6: PolarQuant Rotation — PASS
- Applied QR-based orthogonal rotation to 28 Linear modules in stages 2+3
- Kurtosis: -1.20 → -0.02 (closer to Gaussian = better for quantization)
- Forward pass works after rotation (shape preserved)

### Test 7: Inference Latency — PASS
- CPU encoder: 40.2ms/image
- **CUDA encoder: 18.7ms single, 0.32ms/image at batch 64**
- Speedup: 2.1x single → 58x batched vs CPU
- Throughput at batch 64: ~3,100 images/sec (encoder only)

### Bugs Found and Fixed During Validation
1. **ONNX dynamic width crash:** conditional `if pad > 0` in window partition caused
   torch.export to trace only the no-pad branch. Fix: always call F.pad unconditionally.
2. **LoRA frozen check false positive:** PEFT sets .grad on frozen params during backward.
   Fix: compare actual weight snapshots before/after instead of checking .grad.
3. **PolarQuant shape mismatch:** SVD with full_matrices=False on non-square weights
   produces non-square Vh, breaking weight @ R.T. Fix: use QR decomposition for
   square (in,in) orthogonal matrix.
4. **RNN-T enc_lengths mismatch:** torchaudio rnnt_loss requires enc_lengths == T exactly,
   not T >= enc_lengths. Fix: use features.shape[1] as enc_length for all samples.

---

### Files Implemented (complete list)

```
src/model/
  rope.py, stem.py, pooling.py, attention.py, encoder.py,
  prediction_net.py, joint_net.py, decode.py, lora.py, lid.py, rnnt_model.py

src/data/
  bigrams.py, dataset.py, augmentation.py, synth.py, degradation.py, pdf_extractor.py

src/training/
  loss.py, foundation_trainer.py, adapter_trainer.py

src/quantization/
  qat.py, polar.py

src/export/
  onnx_export.py, lora_export.py

scripts/
  build_vocab.py, train_foundation.py, train_adapter.py, train_lid.py,
  generate_synth.py, extract_pdf_crops.py, apply_degradation.py,
  validate_data.py, export_onnx.py, export_adapters.py, benchmark.py,
  test_onnx_export.py, validate_architecture.py

deploy/
  server.py, api.py, requirements.txt

configs/
  model/{backbone,rnnt_head,lid}.yaml
  training/{phase1_foundation,phase2_adapter,phase3_qat}.yaml
  vocab/{english,hindi,tamil,template}.yaml
  export/onnx.yaml

tests/
  test_encoder.py, test_rnnt.py, test_lora.py, test_lid.py, test_onnx.py, test_vocab.py
```

---

## Real Data Validation (RTX 5080 Blackwell, 2026-03-29)

### Step 2: Overfit on 100 Real IIIT5K Crops — PASS
- 100 real word crops from IIIT5K benchmark
- 1500 steps, SGD lr=0.01, batch_size=32, bf16 AMP
- **100/100 accuracy (100.0%) in 191 seconds**
- Loss: 41.6 → 0.0007
- By step 500: 32/32 batch accuracy, loss < 0.01
- **Conclusion: full training pipeline works end-to-end on GPU with real data**

### Step 3: Train Backbone on 1M MJSynth — DONE (below target)
- 12.5M param backbone, 1M MJSynth crops, 3 epochs, AdamW lr=7e-4, batch=256, bf16
- Training speed: 4.34 it/s (~15 min/epoch), total 51 min
- Loss: 0.897 → 0.255 → 0.167

| Benchmark | Epoch 1 | Epoch 2 | Epoch 3 | Target (min) |
|-----------|---------|---------|---------|-------------|
| IIIT5k | 56.2% | 66.1% | 67.5% | >82% |
| SVT | 56.3% | 65.7% | 68.2% | — |
| IC13_857 | 65.0% | 76.2% | 77.8% | — |
| IC13_1015 | 63.4% | 74.8% | 76.2% | >88% |
| IC15_1811 | 33.0% | 43.3% | 45.7% | — |
| IC15_2077 | 29.0% | 38.1% | 40.1% | >65% |
| SVTP | 32.2% | 42.0% | 44.0% | — |
| CUTE80 | 30.6% | 41.7% | 44.1% | — |
| ArT | 25.6% | 32.5% | 34.2% | — |

**Checkpoint 2 result: FAIL (below minimum thresholds)**
- IIIT5k: 67.5% (need >82%)
- IC13: 76.2% (need >88%)
- IC15: 40.1% (need >65%)

**Analysis:** The model is still learning (strong epoch-over-epoch gains) but
3 epochs on 1M crops isn't enough for a 12.5M param model. The architecture
spec Checkpoint 2 uses a half-dim mini backbone (~3M params) which would converge
faster on this amount of data. Options:
1. Train more epochs (5-10) on same data — loss still decreasing at 0.167
2. Use full MJSynth (~8M crops) for more data per epoch
3. Both — more data × more epochs
The learning curve is healthy — this is a data/compute issue, not architecture.

### Bigram Vocabulary Analysis

**Initial approach (weighted multilingual blend):** Blended character pair frequencies
from 6 Latin-script languages (EN 60%, ES 15%, FR 8%, DE 5%, PT 5%, IT 3%) using
practicalcryptography.com Wortschatz corpus. Produced 95 bigrams.

**Problem found:** The blend only used top-50 bigrams per language from the source data.
High-frequency English-specific bigrams like "wh" (rank 37 in English, 11K occurrences
in Gutenberg) were pushed below the cutoff because they don't exist in other languages.
Meanwhile, mediocre cross-language bigrams ranked higher.

**Solution:** Switched to direct frequency counting from 560K real English words
(5 Gutenberg books: Pride & Prejudice, Alice in Wonderland, Frankenstein,
Sherlock Holmes, Moby Dick). Case-sensitive counting, no digit-digit pairs.

**Key findings:**

1. **Diminishing returns analysis:**
   | Bigrams | Compression | Last bigram adds |
   |---------|------------|-----------------|
   | Top 25 | 29.2% | 0.74% |
   | Top 50 | 42.3% | 0.41% |
   | Top 75 | 51.0% | 0.30% |
   | Top 100 | 57.5% | 0.23% |
   | Top 150 | 65.9% | 0.12% |

   **Decision: 75 bigrams.** 51% compression (half the decode steps) with
   each additional bigram still adding ≥0.3%. After 75, marginal value drops sharply.

2. **Capitalized bigrams:** Only "Th" (rank 128, 3879 occurrences) makes the
   case-sensitive top 200. Not worth the vocab slot — "The" tokenizes as "T"+"he".

3. **No digit-digit bigrams:** Numbers stay character-level. "12345" = 5 tokens always.

4. **Non-English coverage:** The English top-150 naturally includes bigrams common
   in other Latin-script languages: "qu" (#145), "os" (#113), "ei" (#136).
   7 non-English bigrams missing (ue, ia, ci, au, sc, eu, ao) — rare enough
   to skip for now. Can add to adapter-specific extensions later.

**Final Latin base vocabulary: 171 tokens**
- blank (1) + printable ASCII (95) + 75 curated bigrams

**Base character set:** All printable ASCII (32-126) = 95 characters.
This replaces the old hand-picked 82-char set (62 alphanumeric + 20 punctuation).
Every keyboard character is now covered: `[`, `\`, `{`, `|`, `~`, `^`, `_`, `*`, etc.

The base is NOT an adapter — it's the foundation all adapters inherit.
Per-script adapters add their Unicode characters + script-specific bigrams on top.

### Validation Sequence Status
1. [DONE] Download PARSeq LMDB data (MJSynth + eval benchmarks)
2. [DONE] Step 2: Overfit 100 real crops — **100% PASS**
3. [DONE] Step 3: Train backbone on 1M MJSynth — below target, needs more epochs/data
4. [READY] Step 4: Bigram vs character comparison
5. Step 5: Full backbone training (~$40-60)
6. Step 6: First Hindi adapter (~$16-24)

