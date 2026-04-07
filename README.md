# Lipi

Multilingual OCR model supporting 26 scripts and 100+ languages.

759M total parameters, 65M active per sample via Mixture of Experts routing.
Covers ~95% of the world's literate population.

## Quickstart

```bash
# Setup fonts and word lists
python scripts/setup/fonts.py
python scripts/setup/word_lists.py

# Build vocabs
python scripts/build_vocab.py

# Generate training data
python scripts/generate.py --scripts all --samples-per-script 50000

# Train
python scripts/train.py --epochs 30
```

## Project Structure

```
src/
  model/          Encoder, attention, stem, pooling, RoPE, LID
  encoding/       Decomposition, tokenizers, vocab, frozen vocabs
  data/           Augmentation, color, dataset, fonts, rendering, word lists
  training/       Dataloader, losses, eval, routing

scripts/
  train.py        Main training script
  generate.py     Synthetic data generation
  build_vocab.py  Vocab/encoding builder
  setup/          One-time setup (fonts, datasets, word lists)
  tools/          Debugging and diagnostic utilities

tests/            Pipeline, rendering, CJK, and LID tests
```

## Architecture

Shared SWA encoder identifies the script group (LID-1), then routes to
a group-specific expert encoder. Per-script CTC heads decode the text.

- 13 expert groups, 26 scripts
- CJK/Korean/Arabic use symbol decomposition + BPE
- All other scripts use direct character tokens
- CTC decoding throughout

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

## Scripts Supported

Latin, Cyrillic, Greek, Arabic, Hebrew, CJK (Han + Kana), Korean,
Devanagari, Gurmukhi, Gujarati, Bengali, Odia, Kannada, Telugu,
Malayalam, Tamil, Sinhala, Thai, Lao, Burmese, Khmer, Armenian,
Georgian, Ethiopic, Tibetan, Emoji.

## License

Proprietary.
