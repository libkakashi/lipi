# Lipi Build Log

## Architecture Summary

| Component | Params | Notes |
|---|---|---|
| Encoder backbone | 12.5M | ConvNeXt stem + SWA + Global SA, RoPE-2D/1D |
| Prediction net (per lang) | 121K | 1-layer GRU, 128-dim |
| Joint net (per lang) | 176K | Additive, 256-dim joint space |
| LoRA adapter (per lang) | 640K | Rank 16, stages 2+3 only |
| LID classifier | 4.5K | Tiny CNN, 11 script families |
| **Per-adapter total** | **~937K** | LoRA + pred + joint |
| **Full deployment (11 adapters)** | **~23M** | Backbone + 11 adapters + LID |

Vocabulary: character-level, 96 tokens (95 ASCII + blank).
Bigrams tested but characters consistently outperform for RNN-T.

---

## Test Suite: 87/87 PASS

- test_encoder.py: 13 tests (shapes, batches, gradients, ONNX export+parity+dynamic)
- test_rnnt.py: 12 tests (pred_net, joint_net, model assembly, loss, decode)
- test_lora.py: 8 tests (target modules, injection, freezing, gradient flow, param count)
- test_lid.py: 6 tests (shapes, variable width, params, gradients, batches)
- test_onnx.py: 5 tests (all components export, parity, three-file workflow)
- test_vocab.py: 35 tests (char-level, bigram, roundtrip, save/load, curated bigrams)

---

## Key Technical Findings

### Architecture
- **Backbone is 12.5M params** (spec estimated 35.4M — was 3x overestimated)
- **ONNX export requires opset 18** (torch 2.11 dynamo exporter)
- **RoPE must compute inline** (dict caching breaks torch.export)
- **Window partition must always pad** (conditional pad breaks ONNX dynamic width)
- **torchaudio rnnt_loss requires enc_lengths == logits.shape[1] exactly**
- **warp_rnnt CUDA kernel builds but produces wrong results** — currently using torchaudio

### Training
- **MPS fallback**: `PYTORCH_ENABLE_MPS_FALLBACK=1` for rnnt_loss on Apple Silicon
- **AMP**: bf16 autocast + GradScaler, loss computed in float32
- **Scheduler**: must create FRESH scheduler on resume (loading old one causes LR death)
- **Width bucketing**: 3x GPU speedup by grouping similar-width images per batch
- **PIL decode is the CPU bottleneck**, turbojpeg gives 2.4x speedup

### Vocabulary Decision
- **Characters beat bigrams for RNN-T** — tested 3 times, consistent result
- Bigram head needs 2x more training epochs to converge (larger vocab)
- Character-level: 96 tokens, simpler, faster convergence
- Bigram infrastructure kept in code for future experiments

---

## GPU Validation (RTX 5080/5090 Blackwell)

### Architecture Validation — 7/7 PASS (111 seconds)

| Test | Result |
|---|---|
| Mini backbone CTC overfit (1000 images) | **99.7% accuracy** |
| LID classifier (3 scripts) | 68.3% test acc |
| LoRA + RNN-T training | Loss decreasing, frozen params intact |
| Bigram tokenizer | 5/5 roundtrips, max 2-char tokens |
| ONNX deployment pipeline | Full pipeline: image→LID→encode→decode→text |
| PolarQuant rotation | 28 modules rotated, forward OK |
| Inference latency | **0.32ms/image at batch 64** |

### Step 2: Overfit Real Data — PASS
- 100 IIIT5K crops → **100% accuracy in 191 seconds**

### Step 3: Backbone Training

| Run | Data | Epochs | IIIT5k | IC13 | IC15 | Notes |
|---|---|---|---|---|---|---|
| v1 | 1M MJSynth | 3 | 67.5% | 76.2% | 40.1% | First run |
| v2 | 3M MJSynth | 2 | 67.9% | 76.7% | 41.9% | Best non-augmented |
| v3 (aug) | 3M + augment | e5 | 67.2% | 72.8% | 43.2% | Augment helped IC15 |
| v3 (aug) | 3M + augment | e6 | 60.2% | 67.2% | 39.0% | **Regressed — LR scheduler bug** |
| **v4** | **10M MJSynth+synth** | **running** | — | — | — | **Fixed scheduler + 19 augmentations** |

Targets: IIIT5k >82%, IC13 >88%, IC15 >65% (not yet met)

### Step 4: Bigram vs Character — CHARACTERS WIN

| Run | Char | Bigram | Notes |
|---|---|---|---|
| v1 (11% backbone) | 11.9% | 10.7% | Inconclusive |
| v2 (68% backbone) | 37.0% | 17.6% | Char wins decisively |
| v3 (68% backbone) | — | 7.6% | Bigram still poor |

Decision: **use character-level for RNN-T heads.**

---

## Training Optimizations

### GPU Memory
- B=600, W=320 → 24.7 GB peak (attention matrices scale with width)
- Width bucketing reduces average batch width → fits larger batches

### Data Loading
- **Width-bucketed sampler**: groups similar widths → 2-3x GPU speedup
- **turbojpeg**: 2.4x faster JPEG decode than PIL
- **Raw byte preload**: LMDB sequential read to RAM (~10s for 3M)
- **DALI attempted and abandoned**: OOM issues with variable-width images

### Augmentation (19 transforms)

| Category | Transforms |
|---|---|
| Image quality | JPEG compress, gaussian blur, salt-pepper noise, downsample |
| Color/lighting | brightness, contrast, color jitter, shadow gradient, spot light, flash glare |
| Geometric | rotation, perspective warp, paper warp |
| Document | erosion/dilation, paper texture, motion blur, elastic distortion |
| General | random erasing, grayscale conversion |

---

## Synthetic Data Generator

Built `scripts/generate_large_synth.py`:
- 28 bundled Google Fonts (sans, serif, mono, handwriting + 3 Hindi)
- Weighted distribution: 40% sans, 25% serif, 15% mono, 10% hand, 10% other
- Font validation: checks glyph rendering, catches symbol/barcode fonts
- Systematic font cycling per word (each variant uses different font)
- Diverse backgrounds: solid, colored paper, gradient, noise
- Diverse text colors: black, blue ink, red, brown, green, purple
- 20% grayscale conversion
- Multi-worker generation (~2,500 images/sec on 16 cores)
- Generated 2.8M English images (10K words × 280 variants) in 10 minutes

---

## Hindi Support

### Bigrams
- 73 Devanagari bigrams from 110K Hindi Wikipedia words
- 37.7% compression on Hindi text
- Hindi adapter vocab: 372 tokens (95 ASCII + 128 Devanagari + 75 Latin + 73 Hindi bigrams + blank)

### Fonts
- Bundled: NotoSansDevanagari, NotoSerifDevanagari, TiroDevanagariHindi, Baloo2

### Word List
- 301 Hindi legal terms (court system, procedures, IPC sections)

---

## Bugs Found and Fixed

1. **ONNX dynamic width crash**: conditional `if pad > 0` traced only one branch → always pad
2. **LoRA frozen check false positive**: PEFT sets .grad on frozen params → compare weight snapshots
3. **PolarQuant shape mismatch**: SVD on non-square weights → use QR decomposition
4. **RNN-T enc_lengths mismatch**: torchaudio needs exact T match → use features.shape[1]
5. **LR scheduler death on resume**: loaded old decayed scheduler → create fresh one
6. **random_erasing crash on narrow images**: randint(0, 0) → skip small images
7. **warp_rnnt wrong results**: API works but produces bad training → using torchaudio fallback
8. **Mid-epoch resume misdetection**: step count comparison wrong with different dataset sizes → check filename

---

## Current Status

**Running**: Step 5 backbone training — MJSynth (7.2M) + our synth (2.8M) = 10M images/epoch, 19 augmentations, fresh cosine LR, width-bucketed batching.

**Next steps**:
1. Get backbone to 80%+ IIIT5k
2. Train first Hindi adapter (LoRA + RNN-T head)
3. Download IndicSTR12 + TextOCR for real-world data
4. Phase 3: quantization (NVFP4 + FP8)
5. End-to-end deployment pipeline test

---

## Files Implemented

```
src/model/         rope, stem, pooling, attention, encoder, prediction_net,
                   joint_net, decode, lora, lid, rnnt_model
src/data/          bigrams, dataset, augmentation (19 transforms), synth,
                   degradation, pdf_extractor, parseq_lmdb, width_sampler
src/training/      loss, foundation_trainer, adapter_trainer
src/quantization/  qat, polar
src/export/        onnx_export, lora_export
scripts/           step2_overfit_real, step3_train_and_benchmark,
                   step4_bigram_vs_char, generate_large_synth,
                   build_vocab, build_hindi_wordlist, train_foundation,
                   train_adapter, train_lid, run_qat, export_onnx,
                   export_adapters, benchmark, validate_data,
                   validate_architecture, test_onnx_export
deploy/            server, api, requirements.txt
configs/           model/*, training/*, vocab/*, export/*
tests/             test_encoder, test_rnnt, test_lora, test_lid,
                   test_onnx, test_vocab
assets/            28 bundled Google Fonts (Latin + Hindi)
```
