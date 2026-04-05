# CJK Vocabulary Analysis

Source: cjkvi-ids database (recursive IDS decomposition)

## Coverage

- CJK Unified (U+4E00-9FFF): 20,992 assigned characters
- CJK Ext-A (U+3400-4DBF): 6,592 assigned characters
- Total target: 27,584 characters

## Token Counts

- Unique atoms (leaf components): 336 (+51 collision overrides = 387)
- IDS spatial operators (only needed for 532 chars, but in vocab): 12
- CJK punctuation: 33
- Separator: 1

## Base Vocab: 382 tokens (336 atoms + 12 operators + 33 punctuation + 1 separator)

Separator token needed between characters to avoid segmentation ambiguity
(48,333 prefix collisions exist without it).

## Chosen Vocab: 1500 total (709 base + 790 BPE merges + 1 BLANK)

Base: 179 kana + 387 atoms + 12 IDS operators + 1 SEP + 130 punct/ASCII = 709
BPE merges: 790 (PUA codepoints U+E000-U+E315)
BLANK (CTC): 1

Freq-weighted ~1.4 tokens/CJK char (excluding separator).
84% of real text is single-token characters.

## Atom Breakdown

- Atoms that are CJK Unified chars: 228 (of which 257 undecomposed target chars)
- Atoms that are Ext-A chars: 6
- Atoms external to target range: 59
  - CJK Radicals Supplement: 4
  - CJK Strokes: 5
  - CJK Compatibility Ideograph: 1
  - Non-BMP stroke fragments (Ext-B+): 35
  - Placeholders (circled numbers, katakana, etc.): 14
- Collision overrides (frequent char in IDS-duplicate pairs): 51

## Undecomposed Target Characters: 277

- Used as components in other chars: 242 (genuinely atomic)
- Not used anywhere (likely database gaps): 35

## IDS Operators Used (12 of 16)

⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻

## CJK Punctuation: 33

、。〈〉《》「」『』【】〜︰！（），－／：；？［］｜～·—…""※

## Operator Redundancy Analysis (fully recursive, leaf atoms)

Considering fully-recursed leaf atoms only:

- Atomic (1 token, no decomposition):              277 chars
- Atoms only (unordered, no operators needed):   25,606 chars (93.8% of composable)
- Atoms + operators (both unordered):             1,148 chars  (4.2%)
- Atom order only (no operators needed):            391 chars  (1.4%)
- Full sequence (order + operators):                 60 chars  (0.2%)
- Unresolvable (IDS database duplicates):           102 chars  (0.4%)

## Encoding Strategy: Flat atoms + separator

We use flat encoding: decompose to leaf atoms, output between SEP tokens.
Decoding: collect atoms between two SEPs, look up which character has that bag.

Why flat wins for CTC:
- Shorter sequences than recursive
- Order doesn't matter for 93.8% of chars -> more valid CTC paths -> easier training
- Same atom tokens as recursive, just without 116,112 redundant operator tokens

For the 1.9% ambiguous chars (532), operators appear between SEPs to disambiguate.

Source: Jun Da's Modern Chinese Character Frequency List (~193.5M token corpus).

## BPE Merges: Vocab Size vs Tokens/Char

Starting from the optimized 382-token base vocab, BPE merges on frequency-weighted
sequences greedily combine the most impactful adjacent pairs. Each merge accounts
for interactions with prior merges (no double-counting).

All token counts and percentages EXCLUDE the implicit separator token.

| Vocab | Merges | Avg tok/char | % 1-tok | % 2-tok | % 3-tok | % >3-tok |
|-------|--------|--------------|---------|---------|---------|----------|
|   382 |      0 |         3.36 |   14.0% |   21.1% |   27.0% |    37.9% |
|   383 |     +1 |         3.24 |   14.1% |   21.5% |   28.3% |    36.1% |
|   387 |     +5 |         3.07 |   14.1% |   27.0% |   25.5% |    33.4% |
|   392 |    +10 |         2.90 |   18.2% |   24.8% |   27.7% |    29.3% |
|   402 |    +20 |         2.75 |   20.7% |   26.3% |   28.6% |    24.4% |
|   432 |    +50 |         2.49 |   28.1% |   25.1% |   28.6% |    18.3% |
|   482 |   +100 |         2.25 |   34.7% |   27.4% |   23.9% |    14.1% |
|   532 |   +150 |         2.09 |   40.6% |   27.1% |   21.5% |    10.9% |
|   582 |   +200 |         1.97 |   46.2% |   25.9% |   18.0% |     9.9% |
|   632 |   +250 |         1.87 |   51.4% |   23.2% |   16.5% |     8.9% |
|   682 |   +300 |         1.79 |   55.3% |   21.9% |   14.6% |     8.1% |
|   782 |   +400 |         1.67 |   61.3% |   19.9% |   12.2% |     6.5% |
|   882 |   +500 |         1.57 |   66.3% |   17.6% |   10.7% |     5.4% |
|   982 |   +600 |         1.50 |   70.2% |   16.0% |    9.3% |     4.4% |
|  1082 |   +700 |         1.44 |   74.0% |   13.9% |    8.3% |     3.9% |
|  1182 |   +800 |         1.39 |   76.8% |   12.7% |    7.1% |     3.4% |
|  1282 |   +900 |         1.34 |   79.5% |   10.9% |    6.6% |     3.0% |
|  1382 |  +1000 |         1.31 |   81.5% |   10.1% |    5.7% |     2.7% |
|  1500 |  +1118 |         1.27 |   83.8% |    8.8% |    5.2% |     2.2% |

Diminishing returns: first 100 merges save 1.11 tokens/char, next 100 save 0.28.

## Optimal Vocab Size Analysis

Character accuracy = Σ freq(char) × p^n, where p = per-token accuracy, n = tokens for char.
More merges reduces n but increases vocab (which may reduce p).

Model: p(vocab) = base_p × (382/vocab)^alpha, where alpha = vocab size penalty.

| alpha | Meaning                | Optimal merges | Optimal vocab |
|-------|------------------------|----------------|---------------|
| 0.00  | Vocab size is free     | +1118          | 1500          |
| 0.02  | Very mild penalty      | +250 to +1118  | 632-1500      |
| 0.05  | Moderate penalty       | +5 to +20      | 387-402       |
| 0.10  | Strong penalty         | +1 to +10      | 383-392       |

Chosen alpha: ~0.02. Rationale:
- Old working vocab was 2,017 tokens — we're going DOWN to 632 (3x smaller)
- CTC head is a simple linear projection; 382→632 classes is trivial
- BPE merges are high-frequency patterns the model sees constantly
- Standard Chinese OCR uses 6,000-8,000 classes without issue

At alpha=0.02, 632 is comfortably in the optimal zone.

## Collision Overrides

51 character pairs have identical decompositions in the IDS database (e.g., 土/士 both
decompose to ⿱十一). For each pair, the more frequent character gets a dedicated
single token; the less frequent keeps the shared decomposition.

## Verification

- 0 reconstruction collisions across all 27,584 characters
- han_kana.txt word list: 20,756 CJK chars, all covered
- Full round-trip: decompose -> encode -> decode -> reconstruct = original
