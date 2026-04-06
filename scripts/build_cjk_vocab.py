"""
Build CJK vocabulary with 13-symbol arbitrary encoding + word-level BPE.

Pipeline:
  1. Load word frequency corpus (295K CJK words from wordfreq)
  2. Derive char frequency from word frequencies
  3. Rank all 27,584 CJK chars by frequency
  4. Assign vary-first codes using 13 PUA symbols (U+E000-E00C)
  5. Add SEP (U+2E3B) after every char's code
  6. Run word-level BPE (2,500 merges) on word sequences
  7. Write outputs

Encoding:
  - 13 base symbols (U+E000-E00C) + SEP (U+2E3B)
  - Vary-first enumeration: leftmost position varies fastest
  - Top 13 chars: 1 symbol + SEP = 2 tokens
  - Next 169 chars: 2 symbols + SEP = 3 tokens
  - Next 2,197 chars: 3 symbols + SEP = 4 tokens
  - Remaining ~25,205 chars: 4 symbols + SEP = 5 tokens
  - Word-level BPE merges can cross character boundaries

Outputs:
  - training_data/word_lists/cjk_char_codes.tsv
  - training_data/word_lists/bpe_merges.tsv
  - src/data/frozen_vocabs/han_kana_vocab.txt

Usage:
    python -m scripts.build_cjk_vocab
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORD_FREQ_PATH = PROJECT_ROOT / "training_data" / "corpora" / "chinese_word_freq.tsv"
CHAR_CODES_OUT = PROJECT_ROOT / "training_data" / "word_lists" / "cjk_char_codes.tsv"
BPE_OUT = PROJECT_ROOT / "training_data" / "word_lists" / "bpe_merges.tsv"
VOCAB_OUT = PROJECT_ROOT / "src" / "data" / "frozen_vocabs" / "han_kana_vocab.txt"

CJK_UNIFIED_START = 0x4E00
CJK_UNIFIED_END = 0x9FFF
CJK_EXT_A_START = 0x3400
CJK_EXT_A_END = 0x4DBF

N_SYMBOLS = 13  # base encoding symbols
SEP_CODEPOINT = 0x2E3B
SEP_CHAR = chr(SEP_CODEPOINT)

# Base symbols: PUA codepoints U+E000-E00C
BASE_SYMBOLS = [chr(0xE000 + i) for i in range(N_SYMBOLS)]

# BPE merged tokens start after base symbols
BPE_PUA_START = 0xE000 + N_SYMBOLS  # U+E00D

NUM_BPE_MERGES = 2500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_cjk(cp: int) -> bool:
    return (CJK_UNIFIED_START <= cp <= CJK_UNIFIED_END or
            CJK_EXT_A_START <= cp <= CJK_EXT_A_END)


def all_cjk_chars() -> list[str]:
    """All 27,584 CJK chars in range order."""
    chars: list[str] = []
    for cp in range(CJK_EXT_A_START, CJK_EXT_A_END + 1):
        chars.append(chr(cp))
    for cp in range(CJK_UNIFIED_START, CJK_UNIFIED_END + 1):
        chars.append(chr(cp))
    return chars


def gen_vary_first():
    """Generate vary-first code tuples: leftmost position varies fastest.

    Yields:
        (0,), (1,), ..., (12,),
        (0,0), (1,0), ..., (12,0), (0,1), (1,1), ..., (12,12),
        (0,0,0), (1,0,0), ..., (12,12,12),
        (0,0,0,0), ...
    """
    for d0 in range(N_SYMBOLS):
        yield (d0,)
    for d1 in range(N_SYMBOLS):
        for d0 in range(N_SYMBOLS):
            yield (d0, d1)
    for d2 in range(N_SYMBOLS):
        for d1 in range(N_SYMBOLS):
            for d0 in range(N_SYMBOLS):
                yield (d0, d1, d2)
    for d3 in range(N_SYMBOLS):
        for d2 in range(N_SYMBOLS):
            for d1 in range(N_SYMBOLS):
                for d0 in range(N_SYMBOLS):
                    yield (d0, d1, d2, d3)


# ---------------------------------------------------------------------------
# Fast BPE (adapted from compare_cjk_bpe.py)
# ---------------------------------------------------------------------------

def fast_bpe(
    seqs: list[list[str]],
    freqs: list[int],
    num_merges: int,
    nchars: list[int] | None = None,
) -> tuple[list[tuple[str, str, str]], list[list[str]]]:
    """Run frequency-weighted word-level BPE.

    Args:
        seqs: list of token sequences (one per word)
        freqs: frequency of each word
        num_merges: number of BPE merges to perform
        nchars: number of CJK chars per word (for stats)

    Returns:
        (merges, final_sequences)
    """
    seqs = [list(s) for s in seqs]
    n = len(seqs)
    pua_next = BPE_PUA_START

    # Build pair counts and pair→sequence index
    pc: Counter[tuple[str, str]] = Counter()
    p2s: dict[tuple[str, str], set[int]] = defaultdict(set)
    for sid in range(n):
        f = freqs[sid]
        s = seqs[sid]
        for i in range(len(s) - 1):
            p = (s[i], s[i + 1])
            pc[p] += f
            p2s[p].add(sid)

    merges: list[tuple[str, str, str]] = []
    t0 = time.time()

    for step in range(num_merges):
        if not pc:
            print(f"  No more pairs at step {step}")
            break

        bp = pc.most_common(1)[0][0]
        if pc[bp] <= 0:
            break

        a, b = bp
        mg = chr(pua_next)
        pua_next += 1
        merges.append((a, b, mg))

        affected = list(p2s.pop(bp, set()))
        del pc[bp]

        for sid in affected:
            s = seqs[sid]
            f = freqs[sid]
            ns: list[str] = []
            i = 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    # Decrement old left pair
                    if ns:
                        lp = (ns[-1], a)
                        pc[lp] -= f
                        if pc[lp] <= 0:
                            del pc[lp]
                        p2s[lp].discard(sid)
                    # Decrement old right pair
                    if i + 2 < len(s):
                        rp = (b, s[i + 2])
                        pc[rp] -= f
                        if pc[rp] <= 0:
                            del pc[rp]
                        p2s[rp].discard(sid)
                    ns.append(mg)
                    # Increment new left pair
                    if len(ns) >= 2:
                        lp = (ns[-2], mg)
                        pc[lp] += f
                        p2s[lp].add(sid)
                    # Increment new right pair
                    if i + 2 < len(s):
                        rp = (mg, s[i + 2])
                        pc[rp] += f
                        p2s[rp].add(sid)
                    i += 2
                else:
                    ns.append(s[i])
                    i += 1
            seqs[sid] = ns

        m = step + 1
        if m <= 5 or m % 500 == 0:
            # Compute stats
            if nchars:
                tf = tt = tc = 0
                for sid in range(n):
                    f = freqs[sid]
                    tf += f
                    tt += f * len(seqs[sid])
                    tc += f * nchars[sid]
                tpc = tt / tc if tc else 0
                has_sep = SEP_CHAR in bp
                sep_tag = " [SEP-merge]" if has_sep else ""
                print(f"  Merge {m}: tok/char={tpc:.4f} ({time.time()-t0:.1f}s){sep_tag}")

    return merges, seqs


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build():
    print("=== Building CJK vocab: 13-symbol encoding + word-level BPE ===")

    # --- Step 1: Load word frequency corpus ---
    print(f"\n[1/7] Loading word frequency from {WORD_FREQ_PATH}...")
    if not WORD_FREQ_PATH.exists():
        print(f"  ERROR: {WORD_FREQ_PATH} not found.")
        sys.exit(1)

    word_freq: dict[str, int] = {}
    for line in WORD_FREQ_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("word"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            w, f = parts[0], int(parts[1])
            # Only keep words that are pure CJK
            if f > 0 and all(is_cjk(ord(c)) for c in w):
                word_freq[w] = f
    # Ensure all single CJK chars are in corpus
    for c in all_cjk_chars():
        if c not in word_freq:
            word_freq[c] = 1
    print(f"  {len(word_freq)} words loaded")

    # --- Step 2: Derive char frequency from word corpus ---
    print("\n[2/7] Deriving character frequency from word corpus...")
    char_freq: Counter[str] = Counter()
    for w, f in word_freq.items():
        for c in w:
            if is_cjk(ord(c)):
                char_freq[c] += f
    print(f"  {len(char_freq)} chars with frequency data")

    # --- Step 3: Rank all CJK chars by frequency ---
    print("\n[3/7] Ranking all 27,584 CJK chars by frequency...")
    chars = all_cjk_chars()
    # Sort by frequency descending, then by codepoint for stability
    chars_ranked = sorted(chars, key=lambda c: (-char_freq.get(c, 0), ord(c)))
    print(f"  Top 5: {' '.join(chars_ranked[:5])}")
    print(f"  Frequency coverage: top 13 = {sum(char_freq.get(c,0) for c in chars_ranked[:13])} / "
          f"{sum(char_freq.values())}")

    # --- Step 4: Assign vary-first codes ---
    print(f"\n[4/7] Assigning codes with {N_SYMBOLS} symbols (U+E000-E00C)...")
    gen = gen_vary_first()
    char_to_code: dict[str, list[str]] = {}
    code_to_char: dict[tuple[str, ...], str] = {}

    tier_counts = Counter()
    for c in chars_ranked:
        code_tuple = next(gen)
        code = [BASE_SYMBOLS[d] for d in code_tuple] + [SEP_CHAR]
        char_to_code[c] = code
        code_to_char[tuple(code)] = c
        tier_counts[len(code_tuple)] += 1

    print(f"  Tier distribution:")
    print(f"    1 symbol + SEP = 2 tokens: {tier_counts[1]} chars")
    print(f"    2 symbols + SEP = 3 tokens: {tier_counts[2]} chars")
    print(f"    3 symbols + SEP = 4 tokens: {tier_counts[3]} chars")
    print(f"    4 symbols + SEP = 5 tokens: {tier_counts[4]} chars")
    assert len(char_to_code) == len(chars), (
        f"Code assignment mismatch: {len(char_to_code)} vs {len(chars)}")
    assert len(code_to_char) == len(chars), "Duplicate codes found!"

    # --- Step 5: Build word-level sequences ---
    print("\n[5/7] Building word sequences for BPE...")
    w_seqs: list[list[str]] = []
    w_freqs: list[int] = []
    w_nchars: list[int] = []
    for w, f in word_freq.items():
        seq: list[str] = []
        ok = True
        for ch in w:
            if ch in char_to_code:
                seq.extend(char_to_code[ch])
            else:
                ok = False
                break
        if ok and seq:
            w_seqs.append(seq)
            w_freqs.append(f)
            w_nchars.append(len(w))
    print(f"  {len(w_seqs)} word sequences")

    # Pre-BPE stats
    total_freq = sum(w_freqs)
    total_tokens = sum(f * len(s) for f, s in zip(w_freqs, w_seqs))
    total_chars = sum(f * nc for f, nc in zip(w_freqs, w_nchars))
    print(f"  Pre-BPE tokens/char: {total_tokens / total_chars:.4f}")

    # --- Step 6: Run word-level BPE ---
    print(f"\n[6/7] Running word-level BPE ({NUM_BPE_MERGES} merges)...")
    merges, final_seqs = fast_bpe(w_seqs, w_freqs, NUM_BPE_MERGES, w_nchars)
    print(f"  {len(merges)} merges applied")

    # Post-BPE stats
    total_tokens = sum(f * len(s) for f, s in zip(w_freqs, final_seqs))
    print(f"  Post-BPE tokens/char: {total_tokens / total_chars:.4f}")

    # Check SEP absorption
    bare_sep = sum(
        1 for s in final_seqs for t in s if t == SEP_CHAR
    )
    total_seps = sum(nc for nc in w_nchars)
    print(f"  Bare SEP remaining: {bare_sep} / {total_seps} "
          f"({100*bare_sep/total_seps:.1f}%)")

    # Apply merges to per-char codes for the char_codes table
    char_codes_post_bpe: dict[str, list[str]] = {}
    for c in chars:
        seq = list(char_to_code[c])
        for a, b, mg in merges:
            ns: list[str] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    ns.append(mg)
                    i += 2
                else:
                    ns.append(seq[i])
                    i += 1
            seq = ns
        char_codes_post_bpe[c] = seq

    # --- Step 7: Write outputs ---
    print(f"\n[7/7] Writing outputs...")

    # 7a. cjk_char_codes.tsv
    print(f"  {CHAR_CODES_OUT}...")
    CHAR_CODES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAR_CODES_OUT, "w", encoding="utf-8") as f:
        f.write("character\tcode_hex\n")
        for c in chars:
            code = char_codes_post_bpe[c]
            hex_str = " ".join(f"{ord(t):04X}" for t in code)
            f.write(f"{c}\t{hex_str}\n")
    print(f"    {len(chars)} entries")

    # 7b. bpe_merges.tsv
    print(f"  {BPE_OUT}...")
    with open(BPE_OUT, "w", encoding="utf-8") as f:
        f.write("index\ttoken_a\ttoken_b\tmerged\n")
        for idx, (a, b, m) in enumerate(merges):
            a_hex = f"{ord(a):04X}" if len(a) == 1 else a
            b_hex = f"{ord(b):04X}" if len(b) == 1 else b
            m_hex = f"{ord(m):04X}"
            f.write(f"{idx}\t{a_hex}\t{b_hex}\t{m_hex}\n")
    print(f"    {len(merges)} merges")

    # 7c. han_kana_vocab.txt
    print(f"  {VOCAB_OUT}...")

    vocab_cps: list[int] = []

    # Preserve existing non-CJK-component tokens from current vocab
    existing_non_cjk: list[int] = []
    if VOCAB_OUT.exists():
        for line in VOCAB_OUT.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cp = int(line, 16)
            is_cjk_component = (
                is_cjk(cp) or
                (cp >= 0x20000) or                   # Supplementary CJK
                (0x2E80 <= cp <= 0x2EFF) or           # CJK Radicals Supplement
                (0x2FF0 <= cp <= 0x2FFB) or           # IDS operators
                (0xFA00 <= cp <= 0xFA6D) or           # CJK Compat Ideographs
                (0xE000 <= cp <= 0xF8FF) or           # PUA (old BPE + symbols)
                (0x31C0 <= cp <= 0x31EF) or           # CJK Strokes
                (0xF000 <= cp <= 0xF002) or           # Old kana atom replacements
                cp == SEP_CODEPOINT
            )
            if not is_cjk_component:
                existing_non_cjk.append(cp)

    vocab_cps.extend(existing_non_cjk)

    # 13 base symbols (U+E000-E00C)
    for i in range(N_SYMBOLS):
        vocab_cps.append(0xE000 + i)

    # SEP token
    vocab_cps.append(SEP_CODEPOINT)

    # BPE merged tokens
    for _, _, merged_char in merges:
        vocab_cps.append(ord(merged_char))

    # Deduplicate and sort
    vocab_cps = sorted(set(vocab_cps))

    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        for cp in vocab_cps:
            f.write(f"{cp:04X}\n")

    # Report vocab composition
    n_kana = sum(1 for cp in vocab_cps
                 if (0x3041 <= cp <= 0x3096) or (0x30A1 <= cp <= 0x30FA) or
                 cp in (0x30A0, 0x30FB, 0x30FC))
    n_base = sum(1 for cp in vocab_cps if 0xE000 <= cp < 0xE000 + N_SYMBOLS)
    n_bpe = sum(1 for cp in vocab_cps
                if BPE_PUA_START <= cp < BPE_PUA_START + len(merges))
    n_sep = 1
    n_other = len(vocab_cps) - n_kana - n_base - n_bpe - n_sep

    print(f"\n  Total vocab: {len(vocab_cps)} tokens (+1 BLANK = {len(vocab_cps) + 1})")
    print(f"    Kana: {n_kana}")
    print(f"    Base symbols (13): {n_base}")
    print(f"    SEP: {n_sep}")
    print(f"    BPE merges: {n_bpe}")
    print(f"    Other (punct/ASCII/fullwidth): {n_other}")

    # --- Verification ---
    print("\n=== Verification ===")

    # Round-trip check: all chars have unique codes
    code_map: dict[tuple[str, ...], str] = {}
    collisions = 0
    for c in chars:
        key = tuple(char_codes_post_bpe[c])
        if key in code_map:
            collisions += 1
            print(f"  COLLISION: {c} and {code_map[key]} -> {key}")
        else:
            code_map[key] = c

    print(f"  Unique codes: {len(code_map)} / {len(chars)}")
    if collisions:
        print(f"  WARNING: {collisions} collisions!")
    else:
        print(f"  Zero collisions — round-trip clean")

    # Verify all tokens in char codes are in vocab
    vocab_chars = {chr(cp) for cp in vocab_cps}
    missing_tokens: set[str] = set()
    for c, code in char_codes_post_bpe.items():
        for t in code:
            if t not in vocab_chars:
                missing_tokens.add(t)

    if missing_tokens:
        print(f"  WARNING: {len(missing_tokens)} tokens in codes not in vocab!")
        for t in sorted(missing_tokens, key=ord):
            print(f"    U+{ord(t):04X}")
    else:
        print(f"  All code tokens present in vocab")

    # Sample codes
    test_chars = "的一想国人"
    print(f"\n  Sample codes:")
    for ch in test_chars:
        if ch in char_codes_post_bpe:
            code = char_codes_post_bpe[ch]
            code_repr = " ".join(f"U+{ord(t):04X}" for t in code)
            print(f"    {ch}: {len(code)} tok [{code_repr}]")

    print("\nDone!")


if __name__ == "__main__":
    build()
