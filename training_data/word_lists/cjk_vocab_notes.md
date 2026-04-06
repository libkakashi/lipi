# CJK Vocabulary Analysis

Source: 13-symbol arbitrary encoding with word-level BPE

## Coverage

- CJK Unified (U+4E00-9FFF): 20,992 assigned characters
- CJK Ext-A (U+3400-4DBF): 6,592 assigned characters
- Total target: 27,584 characters

## Encoding System

13 arbitrary symbols (PUA codepoints U+E000-E00C) + 1 SEP token (U+2E3B).

All 27,584 CJK chars are ranked by real-world frequency (from wordfreq library,
saved at `training_data/corpora/chinese_word_freq.tsv`). Character frequency is
derived from the word corpus by summing word frequencies weighted by char occurrence.

### Code Assignment (vary-first enumeration)

Leftmost position varies fastest:
- 1 symbol + SEP = 2 tokens: top 13 chars
- 2 symbols + SEP = 3 tokens: next 169 chars (13^2)
- 3 symbols + SEP = 4 tokens: next 2,197 chars (13^3)
- 4 symbols + SEP = 5 tokens: remaining ~25,205 chars (13^4)

Vary-first means 3-symbol codes are: [0,0,0], [1,0,0], [2,0,0], ..., [12,0,0],
[0,1,0], [1,1,0], ..., giving leftmost symbols the most diversity (good for BPE).

### Word-Level BPE

2,500 BPE merges run on word-level sequences (not per-char). This allows merges
to cross character boundaries within words, capturing common multi-char patterns.

BPE merged tokens use PUA codepoints U+E00D onwards.

SEP gets fully absorbed by BPE after ~500 merges — it becomes a dead token.

At 2,500 BPE merges: ~0.893 tokens/char (frequency-weighted).

## Vocab Composition

Base symbols: 13 (U+E000-E00C)
SEP: 1 (U+2E3B)
BPE merges: 2,500 (U+E00D+)
Kana: ~179 (hiragana + katakana)
Punctuation/ASCII/fullwidth: ~130
BLANK (CTC): 1

Total: ~2,824 tokens

## Why 13 Symbols?

13 symbols gives a good balance:
- 13^1 = 13 (top chars at 2 tokens)
- 13^2 = 169 (common chars at 3 tokens)
- 13^3 = 2,197 (mid-frequency chars at 4 tokens)
- 13^4 = 28,561 (capacity for all 27,584 chars)

With fewer symbols, more chars would need 4-symbol codes.
With more symbols, the base vocab is larger but codes are shorter.
13 is near-optimal for minimizing total tokens at 2,500 BPE merges.

## Advantages Over IDS Decomposition

1. **No external database**: No dependency on cjkvi-ids or any decomposition DB
2. **Zero collisions by construction**: Every char gets a unique code
3. **Simpler reconstruction**: SEP always marks char boundary, no greedy matching
4. **Better BPE compression**: Word-level BPE crosses char boundaries
5. **Deterministic**: No collision resolution levels (bag/bag_ops/ordered/full)
6. **Lower token count**: ~0.89 tok/char vs ~1.10 tok/char with IDS

## Verification

- 0 reconstruction collisions across all 27,584 characters (by construction)
- Full round-trip: decompose -> encode -> decode -> reconstruct = original
- All code tokens present in frozen vocab
