# Lipi: Multilingual OCR for Indian Scripts

> **Purpose:** Complete specification for building a multilingual OCR word-recognition system from scratch. Feed this to Claude Code in an empty repository. Every module, training phase, quantization step, and deployment detail is specified.

> **Version:** 3.0 — incorporates ONNX Runtime MultiLoRA (runtime adapter swapping), NVFP4→TensorRT pipeline, Pan-Indic foundation training, and all prior architectural refinements.

---

## 1. System Overview

### What This System Does

Takes cropped word images from a detection model and outputs the text in each crop. Supports English + 22 Indian regional languages via a shared vision backbone with swappable language-specific adapters.

### Architecture Summary

```
                    ┌──────────────────────────────────────────────────┐
                    │              INFERENCE PIPELINE                   │
                    │                                                  │
  word crop ───────►│  1. Micro-LID Router (1MB CNN, ~0.2ms)          │
  (32 × W × 3)     │     → identifies script (Latin/Devanagari/Tamil) │
                    │                                                  │
                    │  2. Adapter Swap (~0.1ms)                        │
                    │     → SetActiveAdapters([script_adapter])        │
                    │     → load script-specific RNN-T head            │
                    │                                                  │
                    │  3. Recognition (~3-4ms)                        │
                    │     → Frozen backbone + active LoRA adapter      │
                    │     → RNN-T decode with script-specific head     │
                    │     → Bigram detokenize → word string            │
                    │                                                  │
                    │  Total: ~3.5-5ms per word crop                   │
                    └──────────────────────────────────────────────────┘
```

### Key Design Decisions (final, do not change)

1. **Single frozen vision backbone** shared across all languages, with per-language LoRA adapters swapped at runtime via ONNX Runtime MultiLoRA API
2. **RNN-Transducer decoding** (not CTC) — provides implicit language modeling, handles variable-length token alignment naturally
3. **Character + bigram vocabulary** — ~400 tokens per script (individual characters + top 150 bigrams), 2-character max token length
4. **NVFP4 quantization** for backbone (with FP8 fallback for non-Blackwell), FP8 for GRU
5. **No spell correction** — model must handle legal names, case numbers, and domain-specific terminology without external correction
6. **Shifted Window Attention only** — no custom CUDA kernels, clean ONNX export guaranteed
7. **ONNX Runtime MultiLoRA** for adapter swapping at inference — NOT merged encoder copies

### Performance Targets

| Metric | Legacy CRNN | This System |
|---|---|---|
| Word accuracy (English avg) | 78.5% | 93-95% |
| Word accuracy (Hindi) | Poor | 90%+ |
| Latency per word | 6.3ms | 3.5-5ms |
| Memory (all 11 adapters loaded) | N/A | ~33MB |
| Memory (single script hot) | 15MB (FP16) | ~22MB (NVFP4+FP8) |
| Add new language | Retrain everything | Train 1.5MB adapter, zero backbone changes |

---

## 2. Repository Structure

```
lipi/
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── model/
│   │   ├── backbone.yaml              # Vision encoder config (shared)
│   │   ├── rnnt_head.yaml             # RNN-T head config (per-language template)
│   │   └── lid.yaml                   # Script classifier config
│   ├── training/
│   │   ├── phase1_foundation.yaml     # Pan-Indic backbone training
│   │   ├── phase2_adapter.yaml        # LoRA adapter training (per language)
│   │   └── phase3_qat.yaml           # Quantization-aware training
│   ├── vocab/
│   │   ├── english.yaml               # English/Latin vocab config
│   │   ├── hindi.yaml                 # Hindi + English bilingual vocab config
│   │   ├── tamil.yaml                 # Tamil + English bilingual vocab config
│   │   └── template.yaml             # Template for new languages
│   └── export/
│       └── onnx.yaml                  # ONNX export config
│
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── stem.py                    # ConvNeXt-V2 micro stem
│   │   ├── attention.py               # Shifted Window Attention + Global Attention
│   │   ├── rope.py                    # RoPE-2D and RoPE-1D
│   │   ├── pooling.py                 # Learned height pooling
│   │   ├── encoder.py                 # Lipi vision encoder (backbone)
│   │   ├── lora.py                    # LoRA adapter injection
│   │   ├── prediction_net.py          # RNN-T Prediction Network (GRU)
│   │   ├── joint_net.py               # RNN-T Joint Network
│   │   ├── rnnt_model.py              # Full model: encoder + RNN-T head
│   │   ├── decode.py                  # Greedy and beam search decoding
│   │   └── lid.py                     # Micro-LID script classifier
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # LMDB dataset loader
│   │   ├── augmentation.py            # Training augmentations
│   │   ├── bpe.py                     # Bigram vocabulary builder + tokenizer
│   │   ├── pdf_extractor.py           # Extract word crops + labels from clean PDFs
│   │   ├── degradation.py             # Synthetic noise/degradation pipeline
│   │   └── synth.py                   # trdg/SynthTiger wrapper for general synthetic data
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss.py                    # RNN-T loss + distillation loss
│   │   ├── foundation_trainer.py      # Phase 1: Pan-Indic backbone training
│   │   └── adapter_trainer.py         # Phase 2: LoRA adapter training
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── qat.py                     # Mixed-precision QAT (NVFP4 + FP8)
│   │   └── polar.py                   # PolarQuant rotation for Transformer weights
│   └── export/
│       ├── __init__.py
│       ├── onnx_export.py             # Export backbone + heads to ONNX
│       ├── lora_export.py             # Convert LoRA to .onnx_adapter via Olive
│       └── tensorrt_build.py          # Build TensorRT engines (optional, Blackwell)
│
├── scripts/
│   ├── build_vocab.py                 # Build bigram vocabulary for a script
│   ├── extract_pdf_crops.py           # Extract word crops + ground truth from clean PDFs
│   ├── generate_synth.py             # Generate trdg/SynthTiger synthetic images
│   ├── apply_degradation.py          # Apply noise/blur/warp to clean PDF crops
│   ├── distill_labels.py             # Run Qwen3-VL on real scanned crops ONLY (confidence-gated)
│   ├── validate_data.py              # Quality checks: run LID on synth data, verify encodings
│   ├── train_foundation.py           # Phase 1: train backbone
│   ├── train_adapter.py              # Phase 2: train LoRA adapter for a language
│   ├── train_lid.py                  # Train script classifier
│   ├── run_qat.py                    # Phase 3: quantization-aware training
│   ├── export_onnx.py                # Export to ONNX
│   ├── export_adapters.py            # Convert LoRA adapters to ONNX format
│   ├── benchmark.py                  # Evaluate on STR benchmarks
│   └── test_onnx_export.py           # ONNX export + MultiLoRA swap verification (run Week 1!)
│
├── deploy/
│   ├── server.py                      # LipiServer: high-throughput batch inference
│   ├── api.py                         # FastAPI wrapper for HTTP serving
│   └── requirements.txt               # onnxruntime-gpu, fastapi, uvicorn
│
├── tests/
│   ├── test_encoder.py
│   ├── test_rnnt.py
│   ├── test_lora.py
│   ├── test_vocab.py
│   ├── test_onnx.py                  # ONNX export + MultiLoRA swap test
│   └── test_lid.py
│
└── training_data/
    ├── vocabs/                        # Generated bigram vocabularies
    ├── word_lists/                    # Word corpora for bigram counting
    │   ├── english_100k.txt
    │   ├── hindi_legal.txt
    │   ├── tamil_legal.txt
    │   └── ...
    ├── pdfs/                          # Clean legal PDFs for extraction
    │   ├── english/
    │   ├── hindi/
    │   └── tamil/
    └── datasets/                      # LMDB training data (generated)
```

---

## 3. Dependencies

```toml
[project]
name = "lipi-ocr"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "torchaudio>=2.10.0",          # RNNTLoss preserved in 2.10+ (deprecation reversed)
    "lightning>=2.2.0",
    # "tokenizers" — NOT needed. Bigram tokenizer is 10 lines of Python, no external library.
    "lmdb>=1.4.0",
    "pillow>=10.0.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "onnx>=1.15.0",
    "onnxruntime>=1.24.0",
    "peft>=0.10.0",
]

[project.optional-dependencies]
train = ["wandb>=0.16.0", "timm>=0.9.0"]
synth = ["trdg>=1.8.0"]
distill = ["transformers>=4.57.0", "accelerate>=0.27.0"]
pdf = ["pymupdf>=1.24.0"]
export = ["olive-ai>=0.8.0"]
tensorrt = ["nvidia-modelopt>=0.20.0"]
```

**Key dependency notes:**
- **No `natten`** — all attention uses standard PyTorch ops
- **`peft`** — HuggingFace PEFT library for LoRA injection and management
- **`olive-ai`** — Microsoft Olive for converting LoRA adapters to `.onnx_adapter` format
- **`onnxruntime>=1.24.0`** — required for MultiLoRA: `ort.LoraAdapter.Load()` + `RunOptions.add_active_adapter()`
- **`nvidia-modelopt`** — NVIDIA TensorRT Model Optimizer for NVFP4 QAT (optional, for Blackwell deployment)

---

## 4. Vision Encoder (Backbone)

### Input Preprocessing

All word crops from the detection model are normalized before entering the encoder:

```python
def preprocess_crop(crop: np.ndarray, target_height: int = 32, 
                    max_width: int = 320) -> np.ndarray:
    """
    Normalize a word crop to fixed height with preserved aspect ratio.
    
    1. Resize height to target_height (32px), scale width proportionally
    2. If width > max_width, resize width to max_width (squash)
    3. Normalize pixel values to [-1, 1]
    4. Convert HWC → CHW
    
    Input:  (H, W, 3) uint8 from detection crop (any height/width)
    Output: (3, 32, W') float32, where W' = min(W * 32/H, max_width)
    """
    h, w = crop.shape[:2]
    new_w = int(w * target_height / h)
    new_w = min(new_w, max_width)
    
    resized = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    return normalized.transpose(2, 0, 1)  # CHW
```

**Why 32px height (not 48 or 64):** 32px is the standard for STR and sufficient for Latin and most Devanagari. For complex Dravidian scripts (Tamil, Malayalam) with stacked vowel signs, 32px may be tight after the stem's 4× stride-down to 8px feature height. **Test this in Checkpoint 2:** if Dravidian accuracy is poor, try 48px as an ablation. The only architectural change needed is updating `LearnedHeightPooling` h_in values (8→12 at 48px).

**Batching with variable width:** Within a batch, crops are padded to the max width using zeros (black padding on the right). The encoder's RoPE-2D handles variable widths naturally — no positional encoding artifacts from padding.

### Architecture

```
Input: 32 × W × 3 (variable width word crop, height-normalized)
│
├── Stage 0: ConvNeXt-V2 Micro Stem
│   4 conv layers (LayerNorm + GELU), stride 4×4 total
│   32 × W × 3 → 8 × W/4 × 64
│
├── Channel projection: Linear(64 → 192)
│
├── Stage 1: 3× Shifted Window Attention (window 4×4)
│   + RoPE-2D positional encoding
│   Local stroke-level features
│   8 × W/4 × 192
│   *** FROZEN during adapter training ***
│
├── Learned Height Pooling: 8 → 4
│
├── Channel projection: Linear(192 → 384)
│
├── Stage 2: 4× Shifted Window Attention (window 4×8)
│   + RoPE-2D
│   Character/ligature-level features
│   4 × W/4 × 384
│   *** LoRA ACTIVE (Rank 16) during adapter training ***
│
├── Learned Height Pooling: 4 → 1 (height fully collapsed)
│
├── Stage 3: 3× Global Self-Attention
│   + RoPE-1D (1D sequence after height collapse)
│   Word-level sequence features
│   1 × W/4 × 384
│   *** LoRA ACTIVE (Rank 16) during adapter training ***
│
└── LayerNorm → output: (B, T, 384) where T = W/4
```

### Stage Configuration

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| Attention type | Shifted Window (4×4) | Shifted Window (4×8) | Global |
| Channels | 192 | 384 | 384 |
| Heads | 6 | 12 | 12 |
| Head dim | 32 | 32 | 32 |
| Blocks | 3 | 4 | 3 |
| Position encoding | RoPE-2D | RoPE-2D | RoPE-1D |
| MLP ratio | 3× | 3× | 4× |
| LoRA during Phase 2 | Frozen | Active (rank 16) | Active (rank 16) |

All stages use standard PyTorch ops (einsum, reshape, softmax). No custom CUDA kernels. Exports cleanly to ONNX opset 17.

**ONNX export safety rule:** Test `torch.onnx.export` on every attention block in Week 1, immediately after implementation. If any op fails to trace, fix it before building the training pipeline.

### Quantization-Friendly Design Choices

| Choice | Why it helps FP4 |
|---|---|
| LayerNorm (not BatchNorm) | No running stats to quantize separately |
| Pre-norm (LN before attention) | Smoother weight distributions |
| GELU (not ReLU) | Smoother activations, fewer zeros |
| Head dim = 32 | Power-of-2 aligned for Blackwell tensor cores |
| RoPE (not learned pos embed) | No embedding table to quantize |
| GroupNorm in stem (acts as LayerNorm) | Consistent normalization approach |

### Parameter Count

| Component | Params (spec estimate) | Params (actual) |
|---|---|---|
| ConvNeXt-V2 stem + projection | 0.15M | 0.066M |
| Stage 1 (3× SWA 4×4, C=192) | 4.0M | 1.113M |
| Height pooling 1 + projection | 0.08M | ~0.08M |
| Stage 2 (4× SWA 4×8, C=384) | 17.7M | 5.917M |
| Height pooling 2 | 0.15M | ~0.15M |
| Stage 3 (3× Global SA, C=384) | 13.3M | 5.323M |
| **Backbone total** | **~35.4M** | **~12.5M** |

**NOTE:** The spec's original estimates were ~3x overestimated (likely double-counting
or wrong MLP expansion math). The architecture dimensions (C, heads, blocks, MLP ratios)
are implemented exactly per spec — the actual model is simply lighter than predicted.
This is good news: faster inference, lower memory, cheaper training. If accuracy targets
aren't met, scaling levers exist: increase C from 192/384 to 256/512, or add blocks.

---

## 5. LoRA Adapter System

### Why LoRA Placement Matters

- **Stage 1 (Frozen):** Learns edge detection and basic stroke junctions. These features are script-universal — a horizontal stroke is a horizontal stroke in Devanagari, Tamil, or Latin.
- **Stage 2 (LoRA Active, Rank 16):** Learns how strokes form characters and ligatures. Script-specific geometry is critical here — Devanagari's shirorekha (headline), Tamil's curves, Bengali's distinct conjuncts all require adaptation.
- **Stage 3 (LoRA Active, Rank 16):** Learns word-level structures and sequence patterns. Different scripts have different character frequencies and sequence statistics.

### LoRA Implementation

```python
# src/model/lora.py

import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType


def inject_lora(backbone: nn.Module, rank: int = 16, alpha: int = 32) -> nn.Module:
    """
    Inject LoRA adapters into Stage 2 and Stage 3 attention blocks.
    Stage 1 remains completely frozen.
    
    Uses HuggingFace PEFT for LoRA management, which integrates
    with Olive for .onnx_adapter export.
    
    IMPORTANT: target_modules paths must exactly match the named submodules
    in the encoder. If using nn.Sequential for the MLP, paths are indexed
    (e.g., "mlp.0", "mlp.2"). If using named submodules, use the actual
    attribute names (e.g., "mlp.fc1", "mlp.fc2").
    
    PEFT silently skips unmatched target_modules — it does NOT raise an error.
    Always verify after injection by checking trainable param count.
    """
    # STEP 1: Discover the actual module paths in the backbone
    # Run this BEFORE defining target_modules:
    #   for name, module in backbone.named_modules():
    #       if isinstance(module, nn.Linear):
    #           print(name)
    # Then use the exact printed paths below.
    
    # Define which modules get LoRA
    # These paths assume TransformerBlock uses:
    #   self.attn.qkv = nn.Linear(...)
    #   self.attn.proj = nn.Linear(...)
    #   self.mlp = nn.Sequential(nn.Linear(...), nn.GELU(), nn.Linear(...))
    # If you use named submodules instead of Sequential, update paths accordingly.
    target_modules = []
    
    # Stage 2: QKV projections, output projections, MLP layers
    for i in range(4):  # 4 blocks in stage 2
        target_modules.extend([
            f"stage2.{i}.attn.qkv",
            f"stage2.{i}.attn.proj",
            f"stage2.{i}.mlp.0",   # nn.Sequential index 0 = first Linear
            f"stage2.{i}.mlp.2",   # nn.Sequential index 2 = second Linear (after GELU at index 1)
        ])
    
    # Stage 3: same pattern
    for i in range(3):  # 3 blocks in stage 3
        target_modules.extend([
            f"stage3.{i}.attn.qkv",
            f"stage3.{i}.attn.proj",
            f"stage3.{i}.mlp.0",
            f"stage3.{i}.mlp.2",
        ])
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    
    model = get_peft_model(backbone, lora_config)
    
    # Freeze everything, then unfreeze LoRA params
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA injected: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total "
          f"({100 * trainable / total:.1f}%)")
    
    # CRITICAL: Verify LoRA actually matched modules
    # If trainable == 0, target_modules paths are wrong
    assert trainable > 0, (
        "LoRA injected 0 trainable params! target_modules paths don't match "
        "any modules in the backbone. Run `backbone.named_modules()` to find "
        "the correct paths."
    )
    
    return model
```

### Per-Language LoRA Size

With rank 16, LoRA on Stage 2 + Stage 3:
- Stage 2: 4 blocks × 4 modules × 2 × (384 × 16) params = ~0.79M
- Stage 3: 3 blocks × 4 modules × 2 × (384 × 16) params = ~0.59M
- **Total per language: ~1.4M params ≈ 1.5MB on disk (FP16)**

---

## 6. RNN-Transducer Decoding

### Components

Each language gets its own Prediction Network + Joint Network. These are fully swapped per language (not adapted via LoRA) because they're tiny enough to afford separate copies.

#### Prediction Network (per language)

```python
class PredictionNetwork(nn.Module):
    """
    1-layer GRU, 128-dim. Provides language-specific priors.
    Fully swapped per language (not LoRA adapted).
    
    QUANTIZATION: Stays at FP8 (E4M3), NOT NVFP4.
    GRU gating mechanisms (sigmoid gates) become too coarse at 4-bit.
    Hidden state quantization errors compound through time steps.
    Memory cost of FP8 vs FP4 for this module: ~50KB. Negligible.
    """
    def __init__(self, vocab_size: int = 401, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.out_dim = hidden_dim
```

Parameters per language: ~0.15M (embedding: 401×128 = 0.05M, GRU: 0.10M)

#### Joint Network (per language)

```python
class JointNetwork(nn.Module):
    """
    Combines encoder frame features with prediction context.
    Additive combination (not concatenation) — standard in RNN-T.
    """
    def __init__(self, enc_dim: int = 384, pred_dim: int = 128,
                 joint_dim: int = 256, vocab_size: int = 401):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.output = nn.Sequential(nn.GELU(), nn.Linear(joint_dim, vocab_size))
```

Parameters per language: ~0.26M

#### Total RNN-T Head Size Per Language

| Component | Params | Disk (FP16) |
|---|---|---|
| Prediction Network (GRU + embedding) | 0.15M | 0.30MB |
| Joint Network | 0.26M | 0.52MB |
| **Per-language total** | **0.41M** | **~0.82MB** |

### Loss Function

```python
import torchaudio

def rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0):
    """
    RNN-T loss wrapper.
    
    Uses torchaudio.functional.rnnt_loss:
    - CUDA-optimized forward-backward algorithm
    - fused_log_softmax=True for numerical stability
    
    DEPRECATION NOTE (March 2026):
    torchaudio.functional.rnnt_loss was deprecated in torchaudio 2.8
    and was originally slated for removal in 2.9. However, after
    community feedback, the PyTorch team REVERSED the decision —
    RNNTLoss (along with forced_align, lfilter, overdrive, CUCT)
    was preserved and will remain in torchaudio 2.10+.
    See: https://github.com/pytorch/audio/issues/3902
    
    torchaudio is now in maintenance mode (no new features, bug fixes
    only). There is NO torch.nn.functional.rnnt_loss in core PyTorch.
    
    This wrapper exists so we can swap implementations later (e.g.,
    if a core PyTorch RNN-T loss eventually ships, or if we need
    to fall back to warp-rnnt) without changing any training code.
    
    IMPORTANT FOR CLAUDE CODE: You will see deprecation warnings
    during training. These are safe to ignore — the function is
    preserved and works correctly. Suppress with:
        import warnings
        warnings.filterwarnings("ignore", message=".*rnnt_loss.*deprecated.*")
    
    NOT warp-transducer (version conflicts with modern PyTorch 2.x).
    NOT k2 (overkill heavyweight dependency for a single loss fn).
    """
    return torchaudio.functional.rnnt_loss(
        logits=logits,
        targets=targets.int(),
        logit_lengths=logit_lengths.int(),
        target_lengths=target_lengths.int(),
        blank=blank,
        reduction='mean',
        fused_log_softmax=True,
    )
```

---

## 7. Vocabulary System: Characters + Bigrams

### Design: Character-Level with Common Bigrams

Each script adapter uses a vocabulary of individual characters plus the ~150 most common character pairs (bigrams) for that script. This gives ~30% fewer decode steps than pure character-level, with almost zero added complexity.

**Why bigrams, not full BPE:**
- **2-char tokens max** — error granularity is at most 2 characters, not 3-4
- **No tokenizer training library needed** — just count character pair frequencies in a word list
- **No ambiguity** — greedy left-to-right matching always gives the same result
- **Adding a new script takes minutes** — count bigrams from any word list, done
- **~400 total tokens** — barely larger than pure character-level (~300), softmax is fast
- **RNN-T GRU handles the rest** — longer patterns like "tion" are learned by the Prediction Network dynamically

### Per-Script Vocabulary Layout

| Segment | Tokens | Purpose |
|---|---|---|
| `[0]` | 1 | ∅ blank (RNN-T) |
| `[1-~250]` | ~250 | Individual characters (Latin + regional script) |
| `[~251-~400]` | ~150 | Top bigrams for this script |
| **Total** | **~400** | |

Every individual character in the script is an atomic token that can never be broken down further. Bigrams are an acceleration layer on top — if a bigram isn't matched, the tokenizer falls back to individual characters. **Zero OOV risk by construction.**

### Building Bigram Vocabularies

```python
# src/data/bigrams.py

from collections import Counter


def build_bigram_vocab(word_lists: list[str], script_chars: set[str],
                       max_bigrams: int = 150) -> list[str]:
    """
    Count character bigrams across word lists, keep top N.
    
    No BPE training, no HuggingFace tokenizers library needed.
    Just frequency counting.
    
    Args:
        word_lists: paths to text files, one word per line
        script_chars: set of valid characters for this script
        max_bigrams: how many bigrams to keep
    
    Returns:
        list of bigram strings, ordered by frequency
    """
    bigram_counts = Counter()
    
    for path in word_lists:
        with open(path) as f:
            for word in f:
                word = word.strip()
                for i in range(len(word) - 1):
                    bigram = word[i:i+2]
                    # Only keep bigrams where both chars are in the script
                    if bigram[0] in script_chars and bigram[1] in script_chars:
                        bigram_counts[bigram] += 1
    
    return [bg for bg, _ in bigram_counts.most_common(max_bigrams)]


def tokenize(word: str, bigram_set: set[str]) -> list[str]:
    """
    Greedy left-to-right tokenization.
    Try to match bigram first, fall back to single character.
    
    No ambiguity, no merge rules, no edge cases.
    """
    tokens = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i:i+2] in bigram_set:
            tokens.append(word[i:i+2])
            i += 2
        else:
            tokens.append(word[i])
            i += 1
    return tokens


def detokenize(tokens: list[str]) -> str:
    """Trivial — just concatenate."""
    return "".join(tokens)
```

### Bigram Vocabulary Per Script

For the Latin script adapter, combine word lists from major Latin-script languages:

```python
# Latin adapter bigrams
latin_bigrams = build_bigram_vocab(
    word_lists=["english_100k.txt", "french_50k.txt", "german_50k.txt", "spanish_50k.txt"],
    script_chars=LATIN_CHARS,
    max_bigrams=150,
)
# Result: ["th", "he", "in", "er", "an", "re", "on", "en", "ti", "es", ...]
```

For Devanagari, combine Hindi + Marathi word lists:

```python
# Devanagari adapter bigrams
devanagari_bigrams = build_bigram_vocab(
    word_lists=["hindi_100k.txt", "marathi_50k.txt"],
    script_chars=DEVANAGARI_CHARS,
    max_bigrams=150,
)
# Result: common consonant+matra pairs, halant combinations, etc.
```

**Bigrams are stable across related languages.** The top 150 English bigrams overlap ~90% with French/German/Spanish because letter-pair frequencies are similar across Latin-script languages. A single Latin bigram set works well for all European languages.

### Example Tokenizations

```
"the"           → ["th", "e"]                    2 tokens (vs 3 chars)
"Section"       → ["Se", "ct", "io", "n"]        4 tokens (vs 7 chars)
"defendant"     → ["de", "fe", "nd", "an", "t"]  5 tokens (vs 9 chars)
"Venkataraman"  → ["Ve", "nk", "at", "ar", "am", "an"]  6 tokens (vs 12 chars)
"149"           → ["1", "4", "9"]                 3 tokens (vs 3 chars)
"WP(C)"         → ["W", "P", "(", "C", ")"]      5 tokens (vs 5 chars)
"भारतीय"         → [bigram or char tokens]         depends on bigram matches
```

Numbers and punctuation stay character-level (no useful bigrams). Common text words get ~30% fewer tokens. Legal names and unusual terms fall back to characters gracefully.

### Impact on Model Components

| Component | With full BPE (old) | With bigrams (new) |
|---|---|---|
| Vocab size per script | 2001 | ~400 |
| Joint Network output | Linear(256, 2001) = 0.51M | Linear(256, 400) = 0.10M |
| Prediction Net embedding | Embedding(2001, 128) = 0.26M | Embedding(400, 128) = 0.05M |
| Per-script head total | ~1.0M | ~0.42M |
| Softmax per decode step | 2001-way | 400-way |
| Avg decode steps per word | ~2.5 | ~3.5 |
| Tokenizer code | HuggingFace library | 10 lines of Python |
| Time to add new script | Days | Minutes |

---

## 8. Micro-LID Script Classifier

### Architecture

MobileNet-V4-Tiny (~1MB) that classifies word crops into script families.

```python
class MicroLID(nn.Module):
    """
    Tiny CNN that classifies which script a word crop is in.
    
    Input: 32 × W × 3 (same word crop as recognition model)
    Output: script_id (0-10 for 11 script families)
    
    11 classes (not 12): Bengali and Assamese share one adapter
    because both use Eastern Nagari with near-identical glyphs.
    
    Runs in ~0.2ms. Negligible overhead.
    """
    def __init__(self, num_scripts: int = 11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_scripts)
    
    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)
```

### Supported Script Families

```
0: Latin (English)
1: Devanagari (Hindi, Marathi, Sanskrit, Nepali)
2: Tamil
3: Telugu
4: Kannada
5: Eastern Nagari (Bengali, Assamese)
6: Odia
7: Gujarati
8: Gurmukhi (Punjabi)
9: Malayalam
10: Urdu (Nastaliq Perso-Arabic)
```

**11 adapters covering 15+ languages.** The sharing principle: merge when script glyphs are visually identical and only vocabulary differs. Separate when visual rendering differs, even if scripts are historically related.

**Why these share:**
- **Devanagari (slot 1):** Hindi, Marathi, Sanskrit, Nepali all use identical Devanagari glyphs. Only vocabulary differs — the vision model sees the same shapes. One adapter, one bigram vocabulary covering all four languages' common character pairs.
- **Eastern Nagari (slot 5):** Bengali and Assamese share ~95% identical glyphs. Only a handful of characters differ (e.g., Bengali র vs Assamese ৰ for "ra"). Combined bigram vocabulary covers both.

**Why these stay separate (despite historical relationships):**
- **Urdu vs Arabic:** Both use Perso-Arabic script, but Urdu in Indian legal documents uses Nastaliq style (cursive, diagonal baseline) while Arabic uses Naskh (horizontal, geometric). Visually very different to a vision model. Urdu also has extra characters (ٹ ڈ ڑ ں ے). Keep separate. Arabic would be a new adapter if needed later.
- **Telugu vs Kannada:** Historically related scripts with some shared glyph shapes, but Telugu's rounded forms vs Kannada's angular tops are visually distinct enough to warrant separate adapters.
- **Tamil vs Malayalam:** Both Dravidian languages, but the scripts look completely different visually. No sharing possible.

**Adding new languages in the future:**
- **Arabic (Naskh):** New adapter (~$24 to train). Does NOT share with Urdu.
- **Sindhi (modified Perso-Arabic):** New adapter, or test if Urdu adapter works (similar Nastaliq rendering).
- **Kashmiri:** Can be written in Perso-Arabic (new adapter) or Devanagari (reuse slot 1 with expanded bigram vocab).
- **Manipuri (Meitei script):** New adapter. Unique script, no sharing possible.
- **Bodo / Dogri / Konkani (Devanagari):** Reuse slot 1 Devanagari adapter. May need bigram vocabulary expansion.
- **Santali (Ol Chiki script):** New adapter. Unique script.

### Training

Train on script-labeled word crops. Can use synthetic data from trdg with script-specific fonts. Simple cross-entropy classification — no special tricks needed for a 12-class problem with distinct visual features.

---

## 9. Training Phases

### Phase 1: Pan-Indic Foundation (~5-7 days, 2× A100)

**Goal:** Train the backbone to learn universal visual features covering Latin, Devanagari, and Dravidian scripts. This becomes the frozen base for all future adapters.

**Data (4-source pipeline):**

**Source 1: Clean PDF extraction + degradation (primary, ~10-15M crops)**
The highest-quality training data. Labels are perfect by construction.
```
1. Collect clean legal PDFs (judgments, filings, acts) in English + Hindi + Tamil
2. Extract text programmatically (PyMuPDF/pdftotext) → perfect ground truth labels
3. Render each word as a clean crop using the PDF's native font/layout
4. Apply synthetic degradations to simulate real-world scanning:
   - JPEG compression (quality 30-90)
   - Gaussian blur (σ 0.5-2.0) and motion blur
   - Salt-and-pepper noise
   - Perspective warp (simulate phone camera angle, ±5°)
   - Shadow gradients and uneven lighting
   - Paper texture overlay
   - Low resolution downsampling (0.5x-0.8x then upsample)
   - Ink bleed / smudging simulation
   - Slight rotation (±3°)
5. Pair: (degraded image, perfect text label) → write to LMDB
```
Split: ~5M English + ~3M Hindi + ~2M Tamil crops (adjust based on PDF availability).

**Source 2: trdg/SynthTiger general synthetic (~5M crops)**
Adds font and style diversity beyond what's in the PDFs.
- Use Google Fonts (1500+ families) for Latin
- Use script-specific fonts (Mangal, Latha, Nirmala UI, etc.) for Indic
- Random backgrounds from COCO/OpenImages

**Source 3: Real-world STR datasets (~2-3M crops)**
For benchmark compatibility and real-world distribution coverage.
- IC13, IC15, CUTE80, SVT, SVTP, IIIT5K (standard STR benchmarks)
- ArT, COCO-Text, TextOCR, OpenVINO, Uber-Text

**Source 4: Qwen3-VL labeled real scans (~100-500K crops, confidence-gated)**
For the small set of actual scanned document images where you don't have ground truth.
Only use soft targets when teacher confidence > 0.9 to prevent label corruption.
This is a supplement, NOT the primary data source.

**Data validation (run before training):**
```bash
python scripts/validate_data.py --dataset training_data/datasets/
# Checks:
# - Run LID on all crops to verify script labels match
# - Verify bigram tokenization roundtrips for all labels
# - Flag crops where rendered text doesn't match extracted text (encoding issues)
# - Report font coverage statistics per script
```

**Head:** Character-level CTC with unified Unicode character set (~300 characters: Latin + Devanagari + Tamil blocks). This CTC head is temporary — used only for pre-training the backbone, discarded in Phase 2.

**Why CTC for Phase 1 (not RNN-T):** CTC is simpler and converges faster. We just need the backbone to learn visual features. The sophisticated RNN-T decoding comes in Phase 2 with language-specific heads.

```yaml
# configs/training/phase1_foundation.yaml
data:
  pdf_extracted: training_data/datasets/pdf_crops_degraded  # Primary: clean PDFs + degradation
  general_synth: training_data/datasets/trdg_synth          # Secondary: trdg/SynthTiger diversity
  real_data: training_data/datasets/real_mixed               # Tertiary: standard STR datasets
  vlm_labeled: training_data/datasets/vlm_scans              # Supplement: Qwen3-VL labeled real scans

training:
  optimizer: AdamW
  lr: 7.0e-4
  weight_decay: 0.01
  epochs: 20
  batch_size: 512
  precision: bf16
  scheduler: onecycle_swa  # SWA at 75%
  augmentation: randaugment_plus

head:
  type: ctc
  charset: unicode_latin_devanagari_tamil  # ~300 chars
  note: >
    This head is DISCARDED after Phase 1. Only include characters from scripts
    present in Phase 1 training data (Latin + Devanagari + Tamil). Do NOT include
    Telugu, Kannada, Bengali, etc. — those output dimensions would be dead weight
    with zero training signal, wasting softmax capacity. Each script block adds
    ~50-100 chars: Latin ~92, Devanagari ~110, Tamil ~80.
```

**Optional MAE pre-training:** Before Phase 1, optionally pre-train the encoder with a Masked Autoencoder objective (mask 75% of patches, reconstruct). This teaches the encoder about text visual patterns without needing labels. ~2 days on 1× A100.

### Phase 2: Adapter Factory (per language, ~1-2 days each, 1× A100)

**Goal:** For each language, freeze the backbone, inject LoRA into Stage 2 + Stage 3, attach a fresh RNN-T head with a ~400-token character+bigram vocabulary, and train only the adapter + head.

```
Phase 2 for Hindi:
1. Load frozen Phase 1 backbone
2. Inject LoRA (rank 16) into Stage 2 + Stage 3
3. Build Hindi+English bigram vocabulary (~400 tokens)
4. Attach fresh PredictionNetwork(vocab_size=401) + JointNetwork(vocab_size=401)
5. Train LoRA params + RNN-T head on Hindi+English data
```

**Data per language:**
- 5-8M language-specific crops from clean PDF extraction + degradation (perfect labels)
- 3-5M English PDF crops (maintain English performance in bilingual head)
- 2M trdg/SynthTiger crops for font diversity
- 50-100K real scanned crops labeled by Qwen3-VL (confidence > 0.9 only)

**Training config:**

```yaml
# configs/training/phase2_adapter.yaml
backbone:
  checkpoint: checkpoints/phase1_foundation.pt
  frozen: true

lora:
  rank: 16
  alpha: 32
  target_stages: [2, 3]  # Stage 1 stays frozen
  dropout: 0.05

rnnt_head:
  vocab_size: 401  # ~250 chars + ~150 bigrams + blank
  pred_embed_dim: 128
  pred_hidden_dim: 128
  joint_dim: 256

training:
  optimizer: AdamW
  # NOTE: Do NOT add frozen backbone params to the optimizer with lr=0.
  # That wastes GPU memory on optimizer state (momentum, variance) for params
  # that never update. Instead, only pass trainable params:
  #   optimizer = AdamW([
  #       {'params': lora_params, 'lr': 1e-3},
  #       {'params': head_params, 'lr': 1e-3},
  #   ], weight_decay=0.01)
  # Frozen backbone params are excluded entirely.
  lora_lr: 1.0e-3
  head_lr: 1.0e-3
  weight_decay: 0.01
  epochs: 15
  batch_size: 512
  precision: bf16
  
distillation:
  teacher: Qwen/Qwen3-VL-8B
  alpha: 0.5  # 0.5 hard loss + 0.5 soft loss
  confidence_gate: 0.9  # only use soft targets when teacher confidence > 0.9
  apply_to: real_scans_only  # NOT applied to PDF-extracted or synthetic data (those have perfect labels)
```

**Prediction Network is fully swapped, not adapted.** It's only 0.15M params — cheaper to have separate copies per language than to try to adapt a shared GRU.

### Phase 3: Quantization-Aware Training (~1 day)

**Goal:** Quantize the backbone to NVFP4, keep GRU at FP8, fine-tune to recover accuracy.

```python
# Quantization config: mixed precision
quant_config = {
    # Everything NVFP4 by default
    "*": {"weight": "nvfp4", "activation": "nvfp4"},
    
    # GRU stays at FP8 — gating mechanisms too coarse at 4-bit
    "prediction_net.gru": {"weight": "fp8_e4m3", "activation": "fp8_e4m3"},
    
    # Embedding tables at FP8 — lookup tables, no arithmetic concern
    "prediction_net.embedding": {"weight": "fp8_e4m3"},
}
```

**PolarQuant rotation:** Applied to Transformer QKV and MLP weight matrices ONLY (not GRU). Spreads outlier weights across dimensions for better NVFP4 fidelity.

**Hardware fallback:** On non-Blackwell GPUs (Ampere, Hopper), fall back to FP8 for the full model. NVFP4 requires compute capability 10.0+.

---

## 10. Deployment Architecture

### Design: Python Inference Server with MultiLoRA

**Deployment target:** Server-side, high concurrency, millions of documents. Python with ONNX Runtime — MultiLoRA via `ort.LoraAdapter.Load()` + `RunOptions.add_active_adapter()` works in Python bindings today. No TypeScript, no Node.js, no workarounds needed.

### Architecture: Batch-by-Script Processing

```
                    ┌─────────────────────────────────────────────┐
                    │            INFERENCE SERVER                  │
                    │                                             │
  Document pages ──►│  1. Detection model → word bounding boxes   │
  (batch of images) │                                             │
                    │  2. Crop all words from page                │
                    │                                             │
                    │  3. LID classify all crops (batched, ~0.2ms)│
                    │     → group crops by script_id              │
                    │                                             │
                    │  4. For each script group:                  │
                    │     a. SetActiveAdapters([script_adapter])  │
                    │     b. Batch encode all crops (GPU, batched)│
                    │     c. Batch RNN-T decode                   │
                    │     d. Bigram detokenize → word strings        │
                    │                                             │
                    │  5. Reassemble words into page order        │
                    │     → return (text, bbox, confidence) list  │
                    └─────────────────────────────────────────────┘
```

**Key throughput optimization:** Crops are grouped by script BEFORE recognition. A typical Indian legal document is 90% one script + 10% English. Instead of swapping adapters per-crop (which would mean thousands of swaps per document), you swap once for the dominant script, process all those crops in one batched pass, then swap to English and process the rest. Typically **1-2 adapter swaps per page, not per word.**

### File Layout

```
models/
├── lid.onnx                        # 1MB — script classifier
├── backbone.onnx                   # ~6MB — shared vision encoder (NVFP4 quantized, ~50MB FP32 unquantized)
├── adapters/
│   ├── hindi.onnx_adapter          # ~1.5MB — Hindi LoRA weights
│   ├── tamil.onnx_adapter          # ~1.5MB
│   ├── telugu.onnx_adapter
│   ├── kannada.onnx_adapter
│   ├── bn_as.onnx_adapter          # Eastern Nagari (Bengali + Assamese combined)
│   ├── gujarati.onnx_adapter
│   ├── odia.onnx_adapter
│   ├── punjabi.onnx_adapter
│   ├── malayalam.onnx_adapter
│   ├── urdu.onnx_adapter
│   └── english.onnx_adapter
├── heads/
│   ├── en/
│   │   ├── pred_net.onnx           # 0.7MB
│   │   └── joint_net.onnx          # 1.2MB
│   ├── hi/
│   │   ├── pred_net.onnx
│   │   └── joint_net.onnx
│   ├── bn_as/
│   │   ├── pred_net.onnx
│   │   └── joint_net.onnx
│   └── ...                         # one per adapter (11 total)
└── vocabs/
    ├── en_2k.json
    ├── hi_2k.json
    ├── bn_as_2k.json
    └── ...                         # one per adapter (11 total)

Total disk: ~6MB backbone (NVFP4) + 11 × ~1.5MB adapters + 11 × ~0.82MB heads + 11 × ~0.1MB vocabs
         = ~6 + 16.5 + 9.0 + 1.1 ≈ 33MB for 11 adapters (15+ languages)
```

### Export Pipeline

```python
# scripts/export_onnx.py — Export backbone

def export_backbone(model_path, output_path):
    """Export the frozen backbone (without LoRA) to ONNX."""
    backbone = load_backbone(model_path)
    backbone.eval()
    
    dummy = torch.rand(1, 3, 32, 128)
    torch.onnx.export(
        backbone, dummy, output_path,
        input_names=["image"],
        output_names=["features", "lengths"],
        dynamic_axes={"image": {0: "batch", 3: "width"},
                      "features": {0: "batch", 1: "seq_len"}},
        opset_version=17,
    )
```

```python
# scripts/export_adapters.py — Convert LoRA to .onnx_adapter

def export_adapter(base_model_path, adapter_path, output_path, language):
    """
    Convert a HuggingFace PEFT LoRA adapter to ONNX Runtime .onnx_adapter format.
    Uses Microsoft Olive toolchain.
    """
    import subprocess
    
    subprocess.run([
        "olive", "auto-opt",
        "-m", base_model_path,
        "--adapter_path", adapter_path,
        "-o", f"tmp/{language}_opt",
        "--device", "gpu",
        "--provider", "CUDAExecutionProvider",
    ], check=True)
    
    subprocess.run([
        "olive", "convert-adapters",
        "--adapter_path", adapter_path,
        "--output_path", output_path,
        "--dtype", "float16",
    ], check=True)
```

```python
# scripts/export_onnx.py — Export RNN-T heads (per language)

def export_rnnt_head(pred_net, joint_net, output_dir, language):
    """Export Prediction Network and Joint Network as separate ONNX models."""
    
    dummy_token = torch.tensor([[1]])
    dummy_hidden = torch.zeros(1, 1, 128)
    torch.onnx.export(
        pred_net, (dummy_token, dummy_hidden),
        f"{output_dir}/{language}/pred_net.onnx",
        input_names=["token", "hidden_in"],
        output_names=["output", "hidden_out"],
        dynamic_axes={"token": {0: "batch"}},
        opset_version=17,
    )
    
    dummy_enc = torch.rand(1, 1, 384)
    dummy_pred = torch.rand(1, 1, 128)
    torch.onnx.export(
        joint_net, (dummy_enc, dummy_pred),
        f"{output_dir}/{language}/joint_net.onnx",
        input_names=["enc_frame", "pred_out"],
        output_names=["logits"],
        opset_version=17,
    )
```

### Python Inference Server

```python
# deploy/server.py

import numpy as np
import onnxruntime as ort
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json


SCRIPT_MAP = {
    0: 'en', 1: 'hi', 2: 'ta', 3: 'te', 4: 'kn',
    5: 'bn_as', 6: 'or', 7: 'gu', 8: 'pa', 9: 'ml', 10: 'ur',
}


@dataclass
class WordResult:
    text: str
    confidence: float
    script_id: str
    bbox: tuple  # (x1, y1, x2, y2) from detection model


@dataclass
class LanguagePack:
    script_id: str
    adapter_path: str
    pred_net: ort.InferenceSession
    joint_net: ort.InferenceSession
    vocab: list[str]


class LipiServer:
    """
    High-throughput multilingual OCR inference server.
    
    Designed for:
    - Millions of documents
    - High concurrency (multiple worker processes per GPU)
    - Batch processing (32-128 crops at a time)
    - Minimal adapter swaps (group by script, swap per-group not per-crop)
    
    Uses ONNX Runtime MultiLoRA: one backbone in GPU memory,
    adapters swapped via SetActiveAdapters() — a pointer operation.
    """
    
    def __init__(self, model_dir: str, batch_size: int = 64):
        self.model_dir = Path(model_dir)
        self.batch_size = batch_size
        
        # Session options for high throughput
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 4
        
        providers = [
            ('CUDAExecutionProvider', {'device_id': 0}),
            'CPUExecutionProvider',
        ]
        
        # Load LID classifier
        self.lid = ort.InferenceSession(
            str(self.model_dir / "lid.onnx"), sess_opts, providers=providers
        )
        
        # Load shared backbone (ONE model, adapters swapped at runtime)
        self.backbone = ort.InferenceSession(
            str(self.model_dir / "backbone.onnx"), sess_opts, providers=providers
        )
        
        # Load all language packs (pre-load adapters into memory)
        self.packs: dict[str, LanguagePack] = {}
        self.adapters: dict[str, ort.LoraAdapter] = {}
        
        for script_id in SCRIPT_MAP.values():
            adapter_path = str(self.model_dir / f"adapters/{script_id}.onnx_adapter")
            
            # Pre-load adapter into memory (fast swap later)
            self.adapters[script_id] = ort.LoraAdapter.Load(adapter_path)
            
            pack = LanguagePack(
                script_id=script_id,
                adapter_path=adapter_path,
                pred_net=ort.InferenceSession(
                    str(self.model_dir / f"heads/{script_id}/pred_net.onnx"),
                    sess_opts, providers=providers
                ),
                joint_net=ort.InferenceSession(
                    str(self.model_dir / f"heads/{script_id}/joint_net.onnx"),
                    sess_opts, providers=providers
                ),
                vocab=json.loads(
                    (self.model_dir / f"vocabs/{script_id}_2k.json").read_text()
                ),
            )
            self.packs[script_id] = pack
    
    def process_page(self, word_crops: list[np.ndarray], 
                     bboxes: list[tuple]) -> list[WordResult]:
        """
        Process all word crops from a single page.
        
        Args:
            word_crops: list of (3, 32, W) float32 arrays, one per detected word
            bboxes: list of (x1, y1, x2, y2) tuples from detection model
        
        Returns:
            list of WordResult in the same order as input
        """
        n = len(word_crops)
        if n == 0:
            return []
        
        # Step 1: Classify all crops by script (batched LID)
        script_ids = self._classify_scripts(word_crops)
        
        # Step 2: Group crop indices by script
        script_groups: dict[str, list[int]] = defaultdict(list)
        for i, sid in enumerate(script_ids):
            script_groups[sid].append(i)
        
        # Step 3: Process each script group (1 adapter swap per group)
        results = [None] * n
        for script_id, indices in script_groups.items():
            group_crops = [word_crops[i] for i in indices]
            group_texts = self._recognize_batch(script_id, group_crops)
            
            for idx, (text, confidence) in zip(indices, group_texts):
                results[idx] = WordResult(
                    text=text,
                    confidence=confidence,
                    script_id=script_id,
                    bbox=bboxes[idx],
                )
        
        return results
    
    def _classify_scripts(self, crops: list[np.ndarray]) -> list[str]:
        """Batch LID classification. Returns script_id per crop."""
        # Pad all crops to same width for batching
        max_w = max(c.shape[2] for c in crops)
        batch = np.zeros((len(crops), 3, 32, max_w), dtype=np.float32)
        for i, c in enumerate(crops):
            batch[i, :, :, :c.shape[2]] = c
        
        logits = self.lid.run(None, {"image": batch})[0]  # (N, num_scripts)
        return [SCRIPT_MAP[int(np.argmax(logits[i]))] for i in range(len(crops))]
    
    def _recognize_batch(self, script_id: str, 
                         crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """
        Recognize a batch of crops that all share the same script.
        Swaps adapter once, then processes all crops.
        """
        pack = self.packs[script_id]
        adapter = self.adapters[script_id]
        
        results = []
        
        # Process in batches of self.batch_size
        for start in range(0, len(crops), self.batch_size):
            batch_crops = crops[start:start + self.batch_size]
            
            # Pad to same width within this batch
            max_w = max(c.shape[2] for c in batch_crops)
            batch = np.zeros((len(batch_crops), 3, 32, max_w), dtype=np.float32)
            for i, c in enumerate(batch_crops):
                batch[i, :, :, :c.shape[2]] = c
            
            # Set active adapter via RunOptions (per-run, not per-session)
            # This is the correct ORT 1.24+ API for MultiLoRA
            run_options = ort.RunOptions()
            run_options.add_active_adapter(adapter)
            
            # Batched encoder forward pass (GPU, fast)
            enc_out, enc_lengths = self.backbone.run(
                None, {"image": batch}, run_options=run_options
            )
            
            # RNN-T decode per sample (sequential within batch)
            # TODO: Implement batched greedy decode with padding for higher throughput.
            # For now, sequential decode per sample. At batch_size=64, this is
            # ~64 × 3ms = ~192ms per batch. Batched decode would be ~20-30ms.
            for i in range(len(batch_crops)):
                tokens, confidence = self._greedy_decode(
                    pack, enc_out[i:i+1], int(enc_lengths[i])
                )
                text = "".join(pack.vocab[t] for t in tokens if t < len(pack.vocab))
                results.append((text, confidence))
        
        return results
    
    def _greedy_decode(self, pack: LanguagePack, enc_out: np.ndarray,
                       T: int) -> tuple[list[int], float]:
        """
        Greedy RNN-T decode for a single sample.
        
        Returns (token_ids, confidence).
        Confidence = geometric mean of token log-probabilities from Joint Network.
        """
        tokens = []
        log_probs = []
        t = 0
        hidden = np.zeros((1, 1, 128), dtype=np.float32)
        prev_token = np.array([[0]], dtype=np.int64)
        
        while t < T and len(tokens) < 25:
            pred_out, hidden = pack.pred_net.run(
                None, {"token": prev_token, "hidden_in": hidden}
            )
            
            enc_frame = enc_out[:, t:t+1, :]  # (1, 1, 384)
            logits = pack.joint_net.run(
                None, {"enc_frame": enc_frame, "pred_out": pred_out}
            )[0]  # (1, 1, 1, vocab_size)
            
            logits = logits.squeeze()  # (vocab_size,)
            
            # Softmax for confidence
            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()
            
            pred_id = int(np.argmax(probs))
            
            if pred_id == 0:
                t += 1  # blank → next frame
            else:
                tokens.append(pred_id)
                log_probs.append(np.log(probs[pred_id] + 1e-10))
                prev_token = np.array([[pred_id]], dtype=np.int64)
        
        # Confidence = geometric mean of token probabilities
        if log_probs:
            confidence = float(np.exp(np.mean(log_probs)))
        else:
            confidence = 0.0
        
        return tokens, confidence


# === Example: FastAPI wrapper for HTTP serving ===

# from fastapi import FastAPI, UploadFile
# import uvicorn
#
# app = FastAPI()
# lipi = LipiServer("./models", batch_size=64)
#
# @app.post("/recognize")
# async def recognize(images: list[UploadFile]):
#     crops = [preprocess(await img.read()) for img in images]
#     bboxes = [(0,0,0,0)] * len(crops)  # placeholder, normally from detection
#     results = lipi.process_page(crops, bboxes)
#     return [{"text": r.text, "confidence": r.confidence, "script": r.script_id} for r in results]
#
# uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

### Throughput Considerations

**RNN-T decode is the throughput bottleneck.** The encoder runs beautifully batched on GPU — 64 crops in one matmul. But greedy decode is sequential per-sample because each sample advances through frames at different rates (different words have different lengths and blank patterns).

**Options for scaling throughput:**

| Approach | Throughput | Complexity | When to implement |
|---|---|---|---|
| Sequential decode (current) | ~200 crops/sec per GPU | Simple | v1 (start here) |
| Batched decode with padding | ~500-1000 crops/sec per GPU | Medium | v2 (when throughput matters) |
| CTC fast path + RNN-T rescore | ~1000-2000 crops/sec per GPU | High | v3 (if needed) |

**Batched decode with padding:** Pad all sequences to max frame length, run all samples in lockstep, mask out finished sequences. Wastes some compute on padding but keeps everything on GPU. This is what NVIDIA Riva does in production.

**CTC fast path + RNN-T rescore:** Add a parallel CTC head (trivially batchable, no sequential decode). Use CTC for the ~90% of crops where CTC confidence is high. Only run RNN-T on the ~10% of uncertain crops. Gets CTC throughput (~2000+ crops/sec) with RNN-T accuracy on hard cases. Implement this as a v3 optimization if throughput becomes a bottleneck.

**Multi-GPU scaling:** Run one LipiServer instance per GPU. Use a process-level load balancer (or CUDA MPS for shared GPU). Each GPU holds the full backbone + all adapters (~33MB) — easily fits even on 8GB cards. Linear scaling: 2 GPUs = 2× throughput.

**Worker concurrency:** For CPU-heavy preprocessing (image decoding, padding, collation), use Python multiprocessing with `torch.multiprocessing` or a WSGI/ASGI server with multiple workers. The GPU inference is the bottleneck, so CPU workers should feed the GPU fast enough to keep it saturated.

---

## 11. Validation Checkpoints

**Every checkpoint is a STOP gate.** Do not proceed past a checkpoint until it passes. Each checkpoint has specific pass/fail criteria, estimated cost, and remediation steps if it fails.

### Checkpoint 0: Smoke Test (Week 1, cost: $0)

**Run on:** Local machine or free Colab GPU. No training data needed.

**Tests:**
- [ ] Encoder forward pass produces correct shapes for W = 32, 64, 128, 256, 320
- [ ] Encoder handles batch sizes 1, 4, 16
- [ ] RNN-T loss computes without NaN for random inputs
- [ ] Loss.backward() completes without error
- [ ] Greedy decode produces valid token sequences (all IDs in [1, vocab_size))
- [ ] Bigram tokenizer roundtrips: `detokenize(tokenize(text)) == text` for 1000 test words
- [ ] ONNX export succeeds for encoder, pred_net, joint_net (3 separate files)
- [ ] Exported ONNX models run in onnxruntime and produce same outputs as PyTorch (within float tolerance)
- [ ] ONNX Runtime Python MultiLoRA API works: `adapter = ort.LoraAdapter.Load(path)` then `run_options.add_active_adapter(adapter)` runs without error

**Pass criteria:** All tests pass. Zero exceptions.

**If it fails:** Fix before spending any money. Common issues: attention mask bugs in SWA, ONNX trace failure on dynamic shapes, RoPE computation errors.

---

### Checkpoint 1: Overfit Test (Week 2, cost: ~$1)

**Run on:** GPU instance, ~1-2 hours.

**Setup:** Take 100 word crop images with known labels. Train for 1000 steps with high learning rate.

**Tests:**
- [ ] Training loss drops to near zero (<0.1)
- [ ] Model achieves 100% word accuracy on the 100 training samples
- [ ] Greedy decode outputs match ground truth exactly
- [ ] GRU hidden state values are reasonable (not exploding or collapsing to zero)
- [ ] GPU memory usage is within expected bounds (~2-4GB for batch size 16)

**Pass criteria:** 100% accuracy on memorized samples, loss < 0.1.

**If it fails:**
- Loss doesn't decrease → check gradient flow. Is the GRU receiving gradients? Are LoRA params (if active) getting updated?
- Loss decreases but decode is wrong → check tokenization. Are labels encoded correctly? Is the blank token (ID 0) handled properly in RNN-T decode?
- OOM → reduce batch size or check for memory leaks in the RNN-T (T, U) lattice computation

---

### Checkpoint 2: Mini Backbone Validation (Week 3, cost: ~$8)

**Run on:** GPU instance, ~12 hours.

**Setup:** Train a SMALL backbone (half channel dims: C=96/192/192, ~9M params) on 1M synthetic English crops for 3 epochs with character-level CTC.

**DATA NOTE:** Use publicly available datasets for this checkpoint — do NOT wait for the custom PDF pipeline. Download the MJSynth LMDB archive from the PARSeq dataset links (see https://github.com/baudm/parseq/blob/main/Datasets.md). Take a 1M sample subset. The IIIT5K, SVT, IC13, IC15 test splits are also available from the same source. This checkpoint validates the architecture, not the data pipeline.

**Tests:**
- [ ] Training loss curve is smooth and decreasing (no spikes or plateaus)
- [ ] Evaluate on IIIT5K test set → measure word accuracy
- [ ] Evaluate on IC13 test set → measure word accuracy
- [ ] Evaluate on IC15 test set → measure word accuracy
- [ ] Compare learned height pooling vs hardcoded stride-2 pooling (ablation)
- [ ] Profile: measure actual inference time per image (ms)

**Pass criteria:**

| Benchmark | Minimum | Good | Excellent |
|---|---|---|---|
| IIIT5K | >82% | >86% | >89% |
| IC13 | >88% | >92% | >95% |
| IC15 | >65% | >72% | >78% |
| Inference time | <15ms | <10ms | <7ms |

**If minimum not met:**
- Below minimum on all benchmarks → architecture issue. Check attention is computing correctly (visualize attention maps). Try removing RoPE (use learned position embeddings instead) as a diagnostic.
- Below minimum on IC15 only → augmentation issue. IC15 has blur/noise. Add stronger augmentation.
- Inference too slow → profile per-stage. If Stage 2 dominates, reduce blocks from 4 to 3.

**Decision point:** If minimums are met, proceed to full-scale backbone. If not, debug before spending $95 on Phase 1.

---

### Checkpoint 3: Bigram Vocabulary Validation (Week 3-4, cost: ~$5)

**Run on:** GPU instance, ~8 hours.

**Setup:** Using the mini backbone from Checkpoint 2, train TWO RNN-T heads on the same data:
- Head A: Pure character-level vocabulary (~300 tokens)
- Head B: Character + bigram vocabulary (~400 tokens)

Train each for 2 epochs on the same 1M English crops (same MJSynth subset from Checkpoint 2).

**Tests:**
- [ ] Compare word accuracy on IIIT5K, IC13, IC15
- [ ] Compare average tokens per word (bigrams should be ~30% fewer)
- [ ] Compare training convergence speed (loss at epoch 1)
- [ ] Verify bigram tokenization doesn't introduce errors on unusual words
- [ ] Verify numbers/punctuation work correctly (these stay character-level)

**Pass criteria:** Bigram accuracy ≥ pure character accuracy.

**If bigrams are worse (>0.5% lower accuracy):**
- Check bigram set: are the top 150 bigrams reasonable? Print them and inspect.
- Try fewer bigrams (top 50 instead of 150)
- If still worse, drop bigrams — use pure character-level. Simpler is fine.

**Decision point:** This confirms whether bigrams help or are neutral. If neutral, keep them for the ~30% decode speedup. If harmful, drop them.

---

### Checkpoint 4: PDF Extraction Pipeline Validation (Week 4, cost: ~$0)

**Run on:** CPU. No GPU needed.

**Tests:**
- [ ] Extract 1000 word crops from 10 clean legal PDFs
- [ ] Verify extracted text matches visual content (manually spot-check 50 crops)
- [ ] Check Indian script extraction: Devanagari, Tamil rendering correct (no mojibake)
- [ ] Apply degradation pipeline → verify degraded images look like realistic scans
- [ ] Run LID classifier on extracted crops → verify script labels are correct
- [ ] Tokenize all extracted labels with bigram tokenizer → verify no encoding failures
- [ ] Verify LMDB write/read roundtrip preserves images and labels

**Pass criteria:** >98% of extracted crops have correct, readable labels. Zero mojibake in Indian scripts.

**If it fails:**
- Mojibake in Indian scripts → PDF uses custom font encoding. Try `pdftotext` as fallback, or extract from a different PDF source.
- Wrong word boundaries → adjust PyMuPDF word extraction parameters (blocks vs words vs spans).
- Degradation looks unrealistic → tune degradation parameters. Compare visually to actual scanned documents.

---

### Checkpoint 5: Full Backbone Evaluation (Week 7, cost: ~$95 cumulative for backbone training)

**This is the most expensive checkpoint. The backbone cannot be cheaply re-done.**

**Setup:** Full Phase 1 backbone (12.5M params) trained on the complete dataset (10-15M PDF crops + 5M synthetic + 2-3M real) for 20 epochs with CTC head.

**Tests:**
- [ ] Evaluate on ALL standard STR benchmarks with character-level CTC

| Benchmark | Minimum | Good | Target |
|---|---|---|---|
| IIIT5K | >90% | >93% | >95% |
| SVT | >88% | >91% | >93% |
| IC13 | >94% | >96% | >97% |
| IC15 | >78% | >83% | >87% |
| SVTP | >80% | >85% | >88% |
| CUTE80 | >80% | >85% | >89% |
| **Average** | **>85%** | **>89%** | **>92%** |

- [ ] Evaluate on Hindi crops (if available): >80% with CTC
- [ ] Evaluate on Tamil crops (if available): >75% with CTC
- [ ] Check that English accuracy is NOT degraded by multilingual training (compare to English-only mini backbone scaled up)
- [ ] Profile inference speed: should be 4-8ms for encoder alone

**Pass criteria:** Average >85% on standard benchmarks. English not degraded by >2% vs English-only training.

**If minimum not met:**
- Below 85% average → check training logs for learning rate issues, data loading bugs, augmentation too aggressive
- English degraded by >2% → multilingual data ratio is wrong. Try 60/20/20 split instead of 50/25/25
- Hindi/Tamil very poor (<70%) → not enough Indic training data, or PDF extraction produced bad labels

**If it fails badly (<80% average):** This is a $95 loss. Debug thoroughly using the mini backbone (Checkpoint 2) before retraining. Common causes: bad data (run validate_data.py again), learning rate too high/low, augmentation broken.

**What you can iterate WITHOUT retraining the backbone (all cheap, $5-25 each):**
- LoRA rank (8 vs 16 vs 32)
- RNN-T head dimensions
- Bigram vocabulary composition
- Augmentation strategy for adapter training
- Learning rate and scheduler for adapter training

---

### Checkpoint 6: First Language Adapter (Week 8, cost: ~$24)

**Setup:** Train Hindi LoRA adapter + RNN-T head on frozen backbone.

**Tests:**
- [ ] Hindi word accuracy on Hindi benchmark crops
- [ ] English word accuracy through Hindi adapter (bilingual test)
- [ ] Mixed-script test: "Section 149 of भारतीय दंड संहिता" → correct output
- [ ] Compare LoRA rank 16 vs rank 8 (quick ablation, ~$12 extra)
- [ ] Verify frozen Stage 1 isn't hurting: temporarily unfreeze and compare (quick ablation)

**Pass criteria:**

| Test | Minimum | Target |
|---|---|---|
| Hindi accuracy | >85% | >90% |
| English through Hindi adapter | >90% | >93% |
| Mixed-script accuracy | >80% | >88% |

**If Hindi accuracy < 85%:**
- Try LoRA rank 32 instead of 16 (more capacity)
- Try unfreezing Stage 1 (costs backbone retrain if it helps significantly)
- Check training data: are Hindi PDF labels correct? Run manual inspection.
- Try 48px input height instead of 32px (better for complex Devanagari conjuncts)

**If English degrades through Hindi adapter (< 88%):**
- Increase English data ratio in adapter training (60% English, 40% Hindi)
- Check that English tokens aren't being corrupted by Hindi vocabulary

---

### Checkpoint 7: Quantization Validation (Week 9, cost: ~$16)

**Setup:** Run QAT on the backbone + Hindi adapter. Compare pre/post quantization.

**Tests:**
- [ ] Accuracy drop from NVFP4 quantization (pre vs post QAT)
- [ ] GRU at FP8: check hidden state stability on long words (>10 characters)
- [ ] Profile inference speed on target hardware
- [ ] Export quantized model to ONNX → verify it loads and runs
- [ ] Compare NVFP4 vs FP8-only (for non-Blackwell fallback)

**Pass criteria:**

| Metric | Maximum allowed |
|---|---|
| Accuracy drop from quantization | <1.5% (after QAT) |
| Accuracy drop (FP8 fallback) | <0.5% |
| Inference speed (Blackwell) | <6ms encoder |
| Inference speed (Ampere/Hopper FP8) | <10ms encoder |

**If accuracy drops > 2%:**
- Run more QAT epochs (5 instead of 3)
- Increase QAT learning rate slightly (2e-5 instead of 1e-5)
- Try keeping more layers at FP8 instead of NVFP4 (attention projections first)
- Check PolarQuant rotation: is it being applied correctly to QKV matrices?

---

### Checkpoint 8: End-to-End Pipeline (Week 10, cost: ~$0)

**Setup:** Full pipeline running: LID → adapter swap → recognition → bigram decode.

**Tests:**
- [ ] Process 100 word crops through the full Python LipiServer pipeline
- [ ] Verify LID correctly identifies script for each crop
- [ ] Verify adapter swapping works (process Hindi crop, then English crop, then Hindi again)
- [ ] Measure end-to-end latency (LID + swap + recognition + decode)
- [ ] Confidence scores are reasonable (not all 1.0 — should vary with image quality)
- [ ] Process a full page: detection model → crop all words → recognize each → reconstruct text

**Pass criteria:**

| Metric | Target |
|---|---|
| End-to-end latency per word | <12ms |
| LID accuracy | >97% |
| Adapter swap overhead | <0.5ms |
| Full page (50 words) processing | <600ms |

**If it fails:** This is usually an integration bug, not a model quality issue. Debug the Python server code, ONNX session management, or adapter loading path.

---

### Checkpoint Summary (copy this and track progress)

```
CHECKPOINT 0 (Week 1, $0):     [ ] PASS / [ ] FAIL — Architecture smoke test
CHECKPOINT 1 (Week 2, $1):     [ ] PASS / [ ] FAIL — Overfit test
CHECKPOINT 2 (Week 3, $8):     [ ] PASS / [ ] FAIL — Mini backbone accuracy
CHECKPOINT 3 (Week 3-4, $5):   [ ] PASS / [ ] FAIL — BPE vs character decision
CHECKPOINT 4 (Week 4, $0):     [ ] PASS / [ ] FAIL — PDF extraction pipeline
CHECKPOINT 5 (Week 7, $95):    [ ] PASS / [ ] FAIL — Full backbone accuracy
CHECKPOINT 6 (Week 8, $24):    [ ] PASS / [ ] FAIL — First adapter (Hindi)
CHECKPOINT 7 (Week 9, $16):    [ ] PASS / [ ] FAIL — Quantization holds accuracy
CHECKPOINT 8 (Week 10, $0):    [ ] PASS / [ ] FAIL — End-to-end pipeline

Total budget if everything passes first try: ~$150
Total budget with one backbone retry:       ~$250
```

---

## 12. Implementation Order

This is the exact order Claude Code should build things:

### Week 1: Core Architecture + ONNX Verification

1. `src/model/rope.py` — RoPE-2D and RoPE-1D (no dependencies, pure math)
2. `src/model/stem.py` — ConvNeXt-V2 micro stem (no dependencies)
3. `src/model/pooling.py` — Learned height pooling (no dependencies)
4. `src/model/attention.py` — Shifted Window Attention + Global Attention (depends on rope.py)
5. `src/model/encoder.py` — Full Lipi encoder (depends on all above)
6. `tests/test_encoder.py` — Run: verify shapes for W = 32, 64, 128, 256, 320
7. **`scripts/test_onnx_export.py` — RUN IMMEDIATELY. If export fails, fix attention.py NOW.**

### Week 2: RNN-T + LoRA + LID

8. `src/model/prediction_net.py` — GRU prediction network
9. `src/model/joint_net.py` — Joint network
10. `src/model/rnnt_model.py` — Full model assembly
11. `src/model/decode.py` — Greedy decoding
12. `src/model/lora.py` — LoRA injection using PEFT
13. `src/model/lid.py` — Micro-LID script classifier
14. `tests/test_rnnt.py` — Verify forward pass + `torchaudio.functional.rnnt_loss` + decode. **Note:** Deprecation warnings from torchaudio are expected and safe to ignore (RNNTLoss was preserved after deprecation reversal).
15. `tests/test_lora.py` — Verify LoRA injects correctly, only target modules are trainable
16. **Test ONNX export of FULL model (encoder + pred_net + joint_net as 3 separate files)**

### Week 3: Data Pipeline (Core)

17. `src/data/vocab.py` — Bigram vocabulary builder and tokenizer
18. `scripts/build_vocab.py` — Build vocabularies for English, Hindi, Tamil
19. `src/data/dataset.py` — LMDB dataset loader with collation
20. `src/data/augmentation.py` — RandAugment + custom augmentations
21. `src/data/pdf_extractor.py` — Extract word crops + ground truth labels from clean PDFs using PyMuPDF
22. `src/data/degradation.py` — Synthetic degradation pipeline (blur, noise, warp, shadow, compression)
23. `tests/test_vocab.py` — Verify roundtrip tokenize/detokenize, bigram matching

### Week 4: Data Generation + Validation

24. `scripts/extract_pdf_crops.py` — Run PDF extraction on legal PDF corpus → clean crops with perfect labels
25. `scripts/apply_degradation.py` — Apply degradation pipeline to clean PDF crops → training-ready LMDB
26. `src/data/synth.py` — trdg/SynthTiger wrapper for supplementary synthetic data
27. `scripts/generate_synth.py` — Generate 5M general synthetic crops for font/style diversity
28. `scripts/validate_data.py` — Quality checks: run LID on all crops, verify encodings, check tokenization roundtrips
29. `scripts/distill_labels.py` — Run Qwen3-VL on real scanned crops ONLY (confidence-gated > 0.9)
30. `src/training/loss.py` — RNN-T loss wrapper + distillation KL loss

### Week 5-7: Training

31. `src/training/foundation_trainer.py` — Phase 1 training loop
32. `scripts/train_foundation.py` — Phase 1: Pan-Indic backbone (~5-7 days)
33. `src/training/adapter_trainer.py` — Phase 2 training loop (freeze backbone, train LoRA + head)
34. `scripts/train_adapter.py` — Phase 2: Train Hindi adapter, then Tamil, then English
35. `scripts/train_lid.py` — Train script classifier

### Week 8: Quantization

36. `src/quantization/polar.py` — PolarQuant rotation
37. `src/quantization/qat.py` — Mixed-precision QAT (NVFP4 backbone + FP8 GRU)
38. `scripts/run_qat.py` — Run QAT fine-tuning

### Week 9: Export + Deploy

39. `src/export/onnx_export.py` — Export backbone + heads to ONNX
40. `src/export/lora_export.py` — Convert LoRA adapters to .onnx_adapter via Olive
41. `scripts/export_onnx.py` — Run full export
42. `scripts/export_adapters.py` — Export all language adapters
43. `tests/test_onnx.py` — Verify MultiLoRA adapter swapping works in ONNX Runtime

### Week 10: Python Server Deployment

44. `deploy/server.py` — LipiServer class with batch-by-script processing, MultiLoRA adapter swapping, greedy decode with confidence scoring
45. `deploy/api.py` — FastAPI wrapper for HTTP serving (POST /recognize endpoint)
46. `deploy/requirements.txt` — onnxruntime-gpu, fastapi, uvicorn
47. End-to-end integration test: detection → crop → LID → adapt → recognize → output

### Week 11-12: Benchmark + Ship

48. `scripts/benchmark.py` — Evaluate on STR benchmarks (IIIT5K, SVT, IC13, IC15, SVTP, CUTE80)
49. Domain-specific evaluation on Indian legal documents
50. Load test: measure throughput at batch_size=64 on target GPU, verify crops/sec targets
51. Fine-tune any underperforming adapters
52. Ship

---

## 13. Summary

| | CRNN (legacy) | VIPTR-v1-T (2024) | Lipi Regional Expert (proposed) |
|---|---|---|---|
| Encoder | VGG16 CNN | CNN + Pyramid Attn | ConvNeXt + SWA pyramid |
| Decoder | BiLSTM + CTC | None (CTC head) | RNN-T (GRU + Joint) |
| Vocabulary | 94 chars | 94 chars | ~400 (chars + bigrams per script) |
| Params (backbone) | 15M FP16 | 5.9M FP16 | 12.5M BF16 → ~6MB NVFP4 |
| Params (per language) | N/A | N/A | ~1.05M (0.64M LoRA + 0.41M head) |
| Languages | 1 | 1 | 15+ via 11 adapters (add new in ~1 day) |
| Position encoding | Learned (fixed) | Learned (fixed) | RoPE-2D (variable width) |
| Language modeling | None | None | Implicit (GRU prediction net) |
| Quantization | None | None | NVFP4 + FP8 mixed |
| Adapter swapping | N/A | N/A | ONNX Runtime MultiLoRA (runtime) |
| Training data | MJSynth (2014) | MJSynth (2016) | PDF extraction + synth + targeted VLM |
| Spell correction needed | Yes | Yes | No |
| Accuracy (English avg) | 78.5% | 91.4% | 93-95% |
| Accuracy (Hindi) | Poor | N/A | 90%+ |
| Speed per word | 6.3ms | 4.2ms | 3.5-5ms |
| Total deployment (11 adapters, 15+ langs) | N/A | N/A | ~33MB |