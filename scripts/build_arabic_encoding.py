"""
Build Arabic vocabulary with N-symbol arbitrary encoding + word-level BPE.

Same approach as CJK and Korean: assign each Arabic character a short code
from N base symbols, ranked by frequency. Then run word-level BPE to merge
common patterns across character boundaries.

Pipeline:
  1. Load Arabic word frequency corpus (1.4M words from Leipzig)
  2. Collect all Arabic-script characters that appear in the corpus
  3. Rank by frequency, assign vary-first codes with N=9 symbols
  4. Add SEP (U+2E3B) after every char's code
  5. Run word-level BPE (2,500 merges) on word sequences
  6. Write outputs

Encoding:
  - 9 base symbols (U+EB00-EB08) + SEP (U+2E3B)
  - Top 9 chars: 1 symbol + SEP = 2 tokens
  - Next 81 chars: 2 symbols + SEP = 3 tokens
  - Next 729 chars: 3 symbols + SEP = 4 tokens  (490 chars fit here)
  - Word-level BPE merges cross character boundaries

Outputs:
  - training_data/word_lists/arabic_char_codes.tsv
  - training_data/word_lists/arabic_arb_bpe_merges.tsv
  - src/data/frozen_vocabs/arabic_vocab.txt

Usage:
    python -m scripts.build_arabic_encoding
"""

from __future__ import annotations

import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORD_FREQ_PATH = PROJECT_ROOT / "training_data" / "corpora" / "arabic_word_freq.tsv"
CHAR_CODES_OUT = PROJECT_ROOT / "training_data" / "word_lists" / "arabic_char_codes.tsv"
BPE_OUT = PROJECT_ROOT / "training_data" / "word_lists" / "arabic_arb_bpe_merges.tsv"
VOCAB_OUT = PROJECT_ROOT / "src" / "data" / "frozen_vocabs" / "arabic_vocab.txt"

N_SYMBOLS = 9
SEP_CODEPOINT = 0x2E3B
SEP_CHAR = chr(SEP_CODEPOINT)

# Base symbols: PUA U+EB00-EB08 (distinct from CJK E000+, Korean EA00+, old Arabic F100+)
PUA_BASE = 0xEB00
BASE_SYMBOLS = [chr(PUA_BASE + i) for i in range(N_SYMBOLS)]

# BPE merged tokens start after base symbols
BPE_PUA_START = PUA_BASE + N_SYMBOLS  # U+EB09

NUM_BPE_MERGES = 2500

# Arabic Unicode ranges (base + extensions, no Presentation Forms-B)
ARABIC_RANGES = [
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x0870, 0x089F),   # Arabic Extended-B
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_arabic_script(cp: int) -> bool:
    """Check if codepoint is in Arabic script ranges."""
    for start, end in ARABIC_RANGES:
        if start <= cp <= end:
            return True
    return False


def is_arabic_char(ch: str) -> bool:
    """Check if character is Arabic script or shared (digits, punctuation, space)."""
    cp = ord(ch)
    if is_arabic_script(cp):
        return True
    # Shared: ASCII printable
    if 0x0020 <= cp <= 0x007E:
        return True
    # Common punctuation/symbols
    cat = unicodedata.category(ch)
    if cat.startswith('Z') or cat.startswith('P') or cat == 'Nd':
        return True
    return False


def gen_vary_first():
    """Generate vary-first code tuples: leftmost position varies fastest."""
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
# Fast BPE (same algorithm as build_cjk_vocab.py)
# ---------------------------------------------------------------------------

def fast_bpe(
    seqs: list[list[str]],
    freqs: list[int],
    num_merges: int,
    nchars: list[int] | None = None,
) -> tuple[list[tuple[str, str, str]], list[list[str]]]:
    """Run frequency-weighted word-level BPE."""
    seqs = [list(s) for s in seqs]
    n = len(seqs)
    pua_next = BPE_PUA_START

    # Build pair counts
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
                    if ns:
                        lp = (ns[-1], a)
                        pc[lp] -= f
                        if pc[lp] <= 0:
                            del pc[lp]
                        p2s[lp].discard(sid)
                    if i + 2 < len(s):
                        rp = (b, s[i + 2])
                        pc[rp] -= f
                        if pc[rp] <= 0:
                            del pc[rp]
                        p2s[rp].discard(sid)
                    ns.append(mg)
                    if len(ns) >= 2:
                        lp = (ns[-2], mg)
                        pc[lp] += f
                        p2s[lp].add(sid)
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
# Main
# ---------------------------------------------------------------------------

def build():
    print("=== Building Arabic vocab: 9-symbol encoding + word-level BPE ===")

    # --- Step 1: Load word frequencies ---
    print(f"\n[1/7] Loading word frequencies from {WORD_FREQ_PATH}...")
    if not WORD_FREQ_PATH.exists():
        print(f"  ERROR: {WORD_FREQ_PATH} not found.")
        print("  Run: python -m scripts.build_arabic_vocab --download")
        sys.exit(1)

    word_freq: dict[str, int] = {}
    for line in WORD_FREQ_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            w = parts[0].strip()
            try:
                f = int(parts[1])
            except ValueError:
                continue
            if f > 0 and w:
                word_freq[w] = f
    print(f"  {len(word_freq)} words loaded")

    # --- Step 2: Collect all characters from corpus ---
    print("\n[2/7] Collecting characters from corpus...")
    char_freq: Counter[str] = Counter()
    for w, f in word_freq.items():
        for ch in w:
            char_freq[ch] += f

    # Include all chars from word lists too
    wl_dir = PROJECT_ROOT / "training_data" / "word_lists"
    for wl_file in ["arabic.txt", "persian.txt", "urdu.txt"]:
        wl_path = wl_dir / wl_file
        if wl_path.exists():
            for line in wl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                for ch in line.strip():
                    if ch not in char_freq:
                        char_freq[ch] = 1

    # Filter to Arabic-usable characters (Arabic script + shared ASCII/punctuation)
    all_chars = sorted(
        [ch for ch in char_freq if is_arabic_char(ch)],
        key=lambda c: (-char_freq.get(c, 0), ord(c))
    )
    print(f"  {len(all_chars)} unique characters")
    print(f"  Top 10: {' '.join(all_chars[:10])}")

    # Capacity check
    capacity = N_SYMBOLS + N_SYMBOLS**2 + N_SYMBOLS**3
    print(f"  Encoding capacity (up to 3 symbols): {capacity}")
    if len(all_chars) > capacity:
        overflow = len(all_chars) - capacity
        print(f"  WARNING: {overflow} chars need 4 symbols")
    else:
        print(f"  All chars fit in ≤3 symbols + SEP")

    # --- Step 3: Assign vary-first codes ---
    print(f"\n[3/7] Assigning codes with {N_SYMBOLS} symbols...")
    gen = gen_vary_first()
    char_to_code: dict[str, list[str]] = {}

    tier_counts = Counter()
    for ch in all_chars:
        code_tuple = next(gen)
        code = [BASE_SYMBOLS[d] for d in code_tuple] + [SEP_CHAR]
        char_to_code[ch] = code
        tier_counts[len(code_tuple)] += 1

    print(f"  Tier distribution:")
    for tier in sorted(tier_counts):
        print(f"    {tier} symbol{'s' if tier > 1 else ''} + SEP = {tier+1} tokens: "
              f"{tier_counts[tier]} chars")

    # --- Step 4: Build word-level sequences ---
    print("\n[4/7] Building word sequences for BPE...")
    w_seqs: list[list[str]] = []
    w_freqs: list[int] = []
    w_nchars: list[int] = []
    skipped = 0

    for w, f in word_freq.items():
        seq: list[str] = []
        nchars = 0
        ok = True
        for ch in w:
            if ch in char_to_code:
                seq.extend(char_to_code[ch])
                nchars += 1
            else:
                ok = False
                break
        if ok and seq:
            w_seqs.append(seq)
            w_freqs.append(f)
            w_nchars.append(nchars)
        else:
            skipped += 1

    print(f"  {len(w_seqs)} word sequences ({skipped} skipped)")

    # Pre-BPE stats
    total_freq = sum(w_freqs)
    total_tokens = sum(f * len(s) for f, s in zip(w_freqs, w_seqs))
    total_chars = sum(f * nc for f, nc in zip(w_freqs, w_nchars))
    print(f"  Pre-BPE tokens/char: {total_tokens / total_chars:.4f}")

    # --- Step 5: Run word-level BPE ---
    print(f"\n[5/7] Running word-level BPE ({NUM_BPE_MERGES} merges)...")
    merges, final_seqs = fast_bpe(w_seqs, w_freqs, NUM_BPE_MERGES, w_nchars)
    print(f"  {len(merges)} merges applied")

    # Post-BPE stats
    total_tokens = sum(f * len(s) for f, s in zip(w_freqs, final_seqs))
    print(f"  Post-BPE tokens/char: {total_tokens / total_chars:.4f}")

    # --- Step 6: Apply merges to per-char codes ---
    print("\n[6/7] Applying BPE to per-char codes...")
    char_codes_post_bpe: dict[str, list[str]] = {}
    for ch in all_chars:
        seq = list(char_to_code[ch])
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
        char_codes_post_bpe[ch] = seq

    # Count how many chars got compressed to 1 token (full char merged)
    single_token = sum(1 for seq in char_codes_post_bpe.values() if len(seq) == 1)
    print(f"  {single_token} chars encoded as single token (after BPE)")

    # --- Step 7: Write outputs ---
    print(f"\n[7/7] Writing outputs...")

    # 7a. arabic_char_codes.tsv
    print(f"  {CHAR_CODES_OUT}...")
    CHAR_CODES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAR_CODES_OUT, "w", encoding="utf-8") as f:
        f.write("character\tcode_hex\trank\tfreq\n")
        for rank, ch in enumerate(all_chars):
            code = char_codes_post_bpe[ch]
            hex_str = " ".join(f"{ord(t):04X}" for t in code)
            f.write(f"{ch}\t{hex_str}\t{rank}\t{char_freq.get(ch, 0)}\n")
    print(f"    {len(all_chars)} entries")

    # 7b. arabic_arb_bpe_merges.tsv
    print(f"  {BPE_OUT}...")
    with open(BPE_OUT, "w", encoding="utf-8") as f:
        f.write("index\ttoken_a\ttoken_b\tmerged\n")
        for idx, (a, b, m) in enumerate(merges):
            a_hex = f"{ord(a):04X}"
            b_hex = f"{ord(b):04X}"
            m_hex = f"{ord(m):04X}"
            f.write(f"{idx}\t{a_hex}\t{b_hex}\t{m_hex}\n")
    print(f"    {len(merges)} merges")

    # 7c. arabic_vocab.txt — all tokens that appear in the encoding
    print(f"  {VOCAB_OUT}...")
    vocab_cps: set[int] = set()

    # Base symbols
    for sym in BASE_SYMBOLS:
        vocab_cps.add(ord(sym))

    # SEP
    vocab_cps.add(SEP_CODEPOINT)

    # All tokens from char codes (includes BPE merged tokens)
    for code in char_codes_post_bpe.values():
        for t in code:
            vocab_cps.add(ord(t))

    # All BPE merge results
    for _, _, m in merges:
        vocab_cps.add(ord(m))

    vocab_sorted = sorted(vocab_cps)
    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        for cp in vocab_sorted:
            f.write(f"{cp:04X}\n")
    print(f"    {len(vocab_sorted)} vocab entries")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"  Characters encoded: {len(all_chars)}")
    print(f"  Base symbols: {N_SYMBOLS} (U+{PUA_BASE:04X}-{PUA_BASE+N_SYMBOLS-1:04X})")
    print(f"  BPE merges: {len(merges)}")
    print(f"  Final vocab size: {len(vocab_sorted)}")
    print(f"  Pre-BPE tokens/char: {sum(f * len(s) for f, s in zip(w_freqs, [char_to_code[ch] for ch in all_chars[:1]])) if False else 'see above'}")
    print(f"  Post-BPE tokens/char: {total_tokens / total_chars:.4f}")


if __name__ == "__main__":
    build()
