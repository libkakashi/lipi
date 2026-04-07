"""
Unified arbitrary encoding builder for any script with a large character set.

Algorithm:
  1. Count unique characters in the script's ranges
  2. Find minimum N such that N + N² + N³ + N⁴ >= char_count
  3. Assign N base symbols (PUA codepoints)
  4. Rank all chars by real-world frequency (highest first)
  5. Assign vary-first codes + SEP: top N get 1 symbol + SEP, next N² get 2, etc.
  6. Run word-level BPE on a word frequency corpus
  7. Output: char->code mapping TSV, BPE merges TSV, frozen vocab file

Usage:
    python -m scripts.build_encoding --script han_kana
    python -m scripts.build_encoding --script korean
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoding.encoding import (
    SCRIPT_CONFIG, SEP_CODEPOINT, SEP_CHAR,
    find_min_n, all_chars_in_ranges, is_in_ranges, gen_vary_first, fast_bpe,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add word_freq_path and vocab_name to configs (build-script-only fields)
_BUILD_CONFIG = {
    "han_kana": {
        "word_freq_path": "training_data/corpora/han_kana_word_freq.tsv",
        "vocab_name": "han_kana",
    },
    "korean": {
        "word_freq_path": "training_data/corpora/korean_word_freq.tsv",
        "vocab_name": "korean",
    },
    "arabic": {
        "word_freq_path": "training_data/corpora/arabic_word_freq.tsv",
        "vocab_name": "arabic",
    },
}



# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build(script_name: str):
    if script_name not in SCRIPT_CONFIG:
        print(f"Unknown script: {script_name}")
        print(f"Available: {', '.join(SCRIPT_CONFIG.keys())}")
        sys.exit(1)

    cfg = SCRIPT_CONFIG[script_name]
    bcfg = _BUILD_CONFIG.get(script_name, {})
    char_ranges = cfg["char_ranges"]
    word_freq_path = PROJECT_ROOT / bcfg.get("word_freq_path", f"training_data/corpora/{script_name}_word_freq.tsv")
    num_bpe_merges = cfg["num_bpe_merges"]
    pua_base = cfg["pua_base"]
    vocab_name = bcfg.get("vocab_name", script_name)

    char_codes_out = PROJECT_ROOT / "training_data" / "word_lists" / cfg["char_codes_filename"]
    bpe_out = PROJECT_ROOT / "training_data" / "word_lists" / cfg["bpe_merges_filename"]
    vocab_out = PROJECT_ROOT / "src" / "data" / "frozen_vocabs" / f"{vocab_name}_vocab.txt"

    print(f"=== Building {script_name} vocab: arbitrary encoding + word-level BPE ===")

    # --- Step 1: Enumerate all chars ---
    chars = all_chars_in_ranges(char_ranges, cfg.get("extra_chars", []))
    char_count = len(chars)
    print(f"\n[1/7] {char_count} characters in ranges")

    # --- Step 2: Find minimum N ---
    n_symbols = find_min_n(char_count)
    capacity = n_symbols + n_symbols**2 + n_symbols**3 + n_symbols**4
    print(f"\n[2/7] N = {n_symbols} (capacity {capacity} >= {char_count})")

    base_symbols = [chr(pua_base + i) for i in range(n_symbols)]
    bpe_pua_start = pua_base + n_symbols

    # --- Step 3: Load word frequency corpus ---
    print(f"\n[3/7] Loading word frequency from {word_freq_path}...")
    if not word_freq_path.exists():
        print(f"  ERROR: {word_freq_path} not found.")
        sys.exit(1)

    word_freq: dict[str, int] = {}
    for line in word_freq_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("word"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            w, f = parts[0], int(parts[1])
            if f > 0 and all(is_in_ranges(ord(c), char_ranges) for c in w):
                word_freq[w] = f

    # Ensure all single chars in the range are in the corpus
    for c in chars:
        if c not in word_freq:
            word_freq[c] = 1
    print(f"  {len(word_freq)} words loaded")

    # --- Step 4: Derive char frequency and rank ---
    print("\n[4/7] Deriving character frequency and ranking...")
    char_freq: Counter[str] = Counter()
    for w, f in word_freq.items():
        for c in w:
            if is_in_ranges(ord(c), char_ranges):
                char_freq[c] += f

    chars_ranked = sorted(chars, key=lambda c: (-char_freq.get(c, 0), ord(c)))
    print(f"  {len(char_freq)} chars with frequency data")
    print(f"  Top 5: {' '.join(chars_ranked[:5])}")
    top_n_freq = sum(char_freq.get(c, 0) for c in chars_ranked[:n_symbols])
    total_freq = sum(char_freq.values())
    print(f"  Frequency coverage: top {n_symbols} = {top_n_freq} / {total_freq}")

    # --- Step 5: Assign vary-first codes ---
    print(f"\n[5/7] Assigning codes with {n_symbols} symbols (U+{pua_base:04X}-U+{pua_base+n_symbols-1:04X})...")
    gen = gen_vary_first(n_symbols)
    char_to_code: dict[str, list[str]] = {}
    code_to_char: dict[tuple[str, ...], str] = {}

    tier_counts = Counter()
    for c in chars_ranked:
        code_tuple = next(gen)
        code = [base_symbols[d] for d in code_tuple] + [SEP_CHAR]
        char_to_code[c] = code
        code_to_char[tuple(code)] = c
        tier_counts[len(code_tuple)] += 1

    print(f"  Tier distribution:")
    for tier in sorted(tier_counts.keys()):
        print(f"    {tier} symbol(s) + SEP = {tier+1} tokens: {tier_counts[tier]} chars")
    assert len(char_to_code) == char_count, (
        f"Code assignment mismatch: {len(char_to_code)} vs {char_count}")
    assert len(code_to_char) == char_count, "Duplicate codes found!"

    # --- Step 6: Build word sequences and run BPE ---
    print(f"\n[6/7] Building word sequences and running BPE ({num_bpe_merges} merges)...")
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

    total_freq_w = sum(w_freqs)
    total_tokens = sum(f * len(s) for f, s in zip(w_freqs, w_seqs))
    total_chars_w = sum(f * nc for f, nc in zip(w_freqs, w_nchars))
    print(f"  Pre-BPE tokens/char: {total_tokens / total_chars_w:.4f}")

    merges, final_seqs = fast_bpe(w_seqs, w_freqs, num_bpe_merges, bpe_pua_start, w_nchars)
    print(f"  {len(merges)} merges applied")

    total_tokens = sum(f * len(s) for f, s in zip(w_freqs, final_seqs))
    print(f"  Post-BPE tokens/char: {total_tokens / total_chars_w:.4f}")

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

    # 7a. char_codes.tsv
    print(f"  {char_codes_out}...")
    char_codes_out.parent.mkdir(parents=True, exist_ok=True)
    with open(char_codes_out, "w", encoding="utf-8") as f:
        f.write("character\tcode_hex\trank\tfreq\n")
        for rank, c in enumerate(chars_ranked):
            code = char_codes_post_bpe[c]
            hex_str = " ".join(f"{ord(t):04X}" for t in code)
            freq = char_freq.get(c, 0)
            f.write(f"{c}\t{hex_str}\t{rank}\t{freq}\n")
    print(f"    {len(chars)} entries")

    # 7b. bpe_merges.tsv
    print(f"  {bpe_out}...")
    with open(bpe_out, "w", encoding="utf-8") as f:
        f.write("index\ttoken_a\ttoken_b\tmerged\n")
        for idx, (a, b, m) in enumerate(merges):
            a_hex = f"{ord(a):04X}"
            b_hex = f"{ord(b):04X}"
            m_hex = f"{ord(m):04X}"
            f.write(f"{idx}\t{a_hex}\t{b_hex}\t{m_hex}\n")
    print(f"    {len(merges)} merges")

    # 7c. vocab file — contains ONLY encoding tokens (no legacy)
    print(f"  {vocab_out}...")
    vocab_cps: list[int] = []

    # Base symbols
    for i in range(n_symbols):
        vocab_cps.append(pua_base + i)

    # SEP token
    vocab_cps.append(SEP_CODEPOINT)

    # BPE merged tokens
    for _, _, merged_char in merges:
        vocab_cps.append(ord(merged_char))

    # Deduplicate and sort
    vocab_cps = sorted(set(vocab_cps))

    with open(vocab_out, "w", encoding="utf-8") as f:
        for cp in vocab_cps:
            f.write(f"{cp:04X}\n")

    print(f"    {len(vocab_cps)} tokens total")

    # --- Verification ---
    print("\n=== Verification ===")

    # Round-trip: unique codes
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
        print(f"  Zero collisions -- round-trip clean")

    # All tokens in vocab
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
    print(f"\n  Sample codes (top 5 by frequency):")
    for ch in chars_ranked[:5]:
        code = char_codes_post_bpe[ch]
        code_repr = " ".join(f"U+{ord(t):04X}" for t in code)
        print(f"    {ch} (U+{ord(ch):04X}): {len(code)} tok [{code_repr}]")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build arbitrary encoding for a script")
    parser.add_argument("--script", required=True, choices=list(SCRIPT_CONFIG.keys()),
                        help="Script to build encoding for")
    args = parser.parse_args()
    build(args.script)
