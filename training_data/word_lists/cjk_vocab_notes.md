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

## Base Vocab: 400 CJK tokens (387 atoms + 12 operators + 1 separator)

### SEP / Prefix Collision Analysis (trie-based, on base sequences pre-BPE)

2,868 chars (10.4%) have token sequences that are prefixes of other characters.
These get SEP appended to their base sequences BEFORE BPE runs.
By frequency: 45.1% of text needs SEP at the base level.

BPE then naturally merges high-frequency (token, SEP) pairs into single tokens,
effectively creating "standalone" tokens as a side effect. No manual dual-identity
mechanism needed — BPE optimally allocates merges between content compression
and SEP elimination based on frequency.

## Chosen Vocab: ~2,103 total (712 base + 1,390 BPE merges + 1 BLANK)

Base: 179 kana + 387 atoms + 3 kana-replacement atoms + 12 IDS operators + 1 SEP + 130 punct/ASCII = 712
BPE merges: 1,390 (SEP-aware, PUA codepoints U+E000+)
BLANK (CTC): 1

Verified against Jun Da frequency corpus (193.5M tokens):

| Metric                      | Value |
|------------------------------|-------|
| Avg tokens/char              |  1.27 |
| Median tokens/char           |     1 |
| % 1-token (freq-weighted)   | 84.0% |
| % 2-token                   |  8.2% |
| % 3-token                   |  5.5% |
| % >3-token                  |  2.3% |
| Bare SEP remaining          |   401 |

### SEP-aware BPE: why it works

The key insight: include SEP in sequences BEFORE running BPE, not after.

1. Start with base atom decompositions (sorted, optimized, no BPE)
2. Find prefix collisions on base sequences → append SEP to those chars
3. Run BPE on the SEP-augmented sequences

BPE naturally merges (atom, SEP) for high-frequency single-token chars that
need disambiguation. Each such merge helps every character containing that
atom+SEP pair, making it far more efficient than dedicated standalone tokens.

At 1,500 merges, BPE merges away SEP for the vast majority of collisions.

### Comparison: SEP-aware BPE vs alternatives (at equal vocab budget)

| Budget | Method A (no-SEP BPE + dual-identity) | Method B (SEP-aware BPE) |
|--------|---------------------------------------|--------------------------|
|    790 | 1.91 avg, 56.5% 1-tok                | 1.51 avg, 70.8% 1-tok   |
|  1,000 | 1.64 avg, 67.2% 1-tok                | 1.40 avg, 76.4% 1-tok   |
|  1,267 | 1.45 avg, 76.4% 1-tok                | 1.31 avg, 82.0% 1-tok   |
|  1,500 | 1.34 avg, 82.0% 1-tok                | 1.24 avg, 85.5% 1-tok   |
|  2,000 | 1.20 avg, 89.3% 1-tok                | 1.15 avg, 90.8% 1-tok   |

SEP-aware BPE wins at every budget. Dual-identity wastes vocab slots on
per-character standalone tokens; BPE merges amortize across many characters.

### Greedy vs optimal BPE

Stochastic beam search (top-k sampling with temperature) found <0.03% improvement
over greedy. Greedy BPE is near-optimal — it captures 83.5% of maximum possible
token savings using only 3% of the merges needed to make everything single-token.

## Atom Breakdown

- Atoms that are CJK Unified chars: 228 (of which 257 undecomposed target chars)
- Atoms that are Ext-A chars: 6
- Atoms external to target range: 59
  - CJK Radicals Supplement: 4
  - CJK Strokes: 5
  - CJK Compatibility Ideograph: 1
  - Non-BMP stroke fragments (Ext-B+): 35
  - Placeholders (circled numbers, etc.): 14
- Collision overrides (frequent char in IDS-duplicate pairs): 51
- Kana atom replacements: 3 (コ→U+F000, ス→U+F001, ユ→U+F002)

### Kana Atom Replacements

The IDS database uses 3 katakana as shape placeholders for CJK components:
- コ (U+30B3) → U+F000: right-angle enclosure component (317 chars)
- ス (U+30B9) → U+F001: diagonal stroke pair (8 chars)
- ユ (U+30E6) → U+F002: horizontal hook component (27 chars)

Replaced at build time so kana codepoints never appear in CJK decompositions.
This avoids ambiguity with standalone kana in mixed Japanese/Chinese text.

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

## Encoding Strategy: Flat atoms + SEP-aware BPE

We use flat encoding: decompose to leaf atoms, then run SEP-aware BPE.

Steps:
1. Decompose each CJK char to its minimal atom sequence (bag/bag_ops/ordered/full)
2. Build trie of all sequences, find prefix collisions
3. Append SEP token to chars whose sequence is a prefix of another's
4. Run greedy BPE on the SEP-augmented sequences

BPE handles both content compression and separator elimination in one pass.
High-frequency (atom, SEP) pairs get merged into single tokens automatically.

Decoding:
- Single tokens (including BPE-merged atom+SEP): look up directly
- Multi-token sequences: collect tokens until SEP or non-CJK token, look up
- Non-CJK tokens (kana, punctuation) pass through directly

Why flat wins for CTC:
- Shorter sequences than recursive
- Order doesn't matter for 93.8% of chars -> more valid CTC paths -> easier training
- Same atom tokens as recursive, just without 116,112 redundant operator tokens

For the 1.9% ambiguous chars (532), operators appear in the sequence to disambiguate.

Source: Jun Da's Modern Chinese Character Frequency List (~193.5M token corpus).

## BPE Merges: Vocab Size vs Tokens/Char (SEP-aware)

Starting from the 400-token CJK base vocab with SEP pre-appended to prefix-collision
chars, BPE merges on frequency-weighted sequences greedily combine the most impactful
adjacent pairs. SEP tokens participate in merges naturally.

| BPE merges | Avg tok/char | Median | % 1-tok | % 2-tok | % 3-tok | % >3-tok |
|------------|--------------|--------|---------|---------|---------|----------|
|          0 |         3.81 |      3 |   10.2% |   14.6% |   24.7% |    50.5% |
|        790 |         1.51 |      1 |   70.8% |    ... |    ... |    ...  |
|      1,000 |         1.40 |      1 |   76.4% |    ... |    ... |    ...  |
|      1,267 |         1.31 |      1 |   82.0% |   8.9% |   6.5% |     2.6% |
|      1,390 |         1.27 |      1 |   84.0% |   8.2% |   5.5% |     2.3% |
|      1,500 |         1.24 |      1 |   85.5% |   7.7% |   4.8% |     2.0% |
|      2,000 |         1.15 |      1 |   90.8% |    ... |    ... |    ...  |

## Optimal Vocab Size Analysis

Character accuracy = Σ freq(char) × p^n, where p = per-token accuracy, n = tokens for char.
More merges reduces n but increases vocab (which may reduce p).

Model: p(vocab) = base_p × (400/vocab)^alpha, where alpha = vocab size penalty.

Chosen: 1,390 BPE merges (2,100 total vocab). Rationale:
- Standard Chinese OCR uses 6,000-8,000 classes without issue
- CTC head is a simple linear projection; 2,100 classes is trivial
- 84% of real text is single-token — effectively direct classification
- Diminishing returns beyond ~1,400 merges

## Collision Overrides

51 character pairs have identical decompositions in the IDS database (e.g., 土/士 both
decompose to ⿱十一). For each pair, the more frequent character gets a dedicated
single token; the less frequent keeps the shared decomposition.

## Verification

- 0 reconstruction collisions across all 27,584 characters
- han_kana.txt word list: 20,756 CJK chars, all covered
- Full round-trip: decompose -> encode -> decode -> reconstruct = original
