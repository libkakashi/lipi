# Lipi

Multilingual OCR model supporting 27 scripts and 100+ languages.

80.5M total parameters, ~6.5M active per sample (one group + one script
expert) via Mixture of Experts routing — ~1.7 GFLOPs/word.
Covers ~95% of the world's literate population.

## Quickstart

```bash
# Setup fonts and word lists
python scripts/setup/fonts.py
python scripts/setup/word_lists.py

# Generate synthetic training data (writes MDS shards to data/shards/{train,val})
python scripts/generate.py --scripts all --samples-per-script 50000 --out data/shards

# Train
python scripts/train.py --data data/shards --epochs 30
```

Per-script vocabularies are derived at runtime from `src/encoding/config.py`
(no separate vocab-build step). The fusion/CJK corpora that feed those codecs
are prebuilt in `training_data/corpora/` (regenerate with
`scripts/build_word_freq.py` and `scripts/cjk_visual_similarity.py`).

## Project Structure

```
src/
  model/          Encoder (ConvStem + shared SWA + MoE experts), LID tables
  encoding/       Codecs (config), decomposition, vocab building
  data/           Augmentation, color, fonts, rendering, word lists
  training/       Dataloader (MDS streaming), losses, eval, routing

scripts/
  train.py        Main training script
  generate.py     Synthetic data generation (MDS output)
  convert_to_mds.py  Convert external .pt shards → MDS
  eval.py / benchmark.py  Checkpoint evaluation and OCR benchmarks
  setup/          One-time setup (fonts, datasets, word lists)
  tools/          Debugging and diagnostic utilities

tests/            Encoding, rendering, CJK, model-smoke, and LID tests
```

## Architecture

A shared SWA backbone identifies the script group (LID-1) and routes to a
group-specific expert encoder; multi-script groups then run a per-script
classifier (LID-2) and route to a script-specific expert. Per-script CTC
heads decode the text — all in a single forward pass.

- 15 expert groups, 27 scripts
- CJK/Korean/Arabic use symbol decomposition + fusion tokens
- Brahmic scripts use base chars + virama-pair conjuncts (fusion)
- All other scripts use direct character tokens
- CTC decoding throughout

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

## Scripts Supported

Latin, Cyrillic, Greek, Arabic, Hebrew, Han, Kana, Korean,
Devanagari, Gurmukhi, Gujarati, Bengali, Odia, Kannada, Telugu,
Malayalam, Tamil, Sinhala, Thai, Lao, Burmese, Khmer, Armenian,
Georgian, Ethiopic, Tibetan, Emoji.

## License

Proprietary.
