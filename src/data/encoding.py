"""
Arbitrary N-symbol encoding for large character sets.

Core algorithm:
  1. Find minimum N base symbols needed: N + N² + N³ + N⁴ >= char_count
  2. Rank chars by frequency, assign vary-first codes + SEP
  3. Run word-level BPE on word corpus
  4. Result: each char → short PUA token sequence

Used by CJK (N=13), Korean (N=11), Arabic (N=9).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

# SEP token: U+2E3B THREE-EM DASH
SEP_CODEPOINT = 0x2E3B
SEP_CHAR = chr(SEP_CODEPOINT)

# Script configurations
SCRIPT_CONFIG = {
    "han_kana": {
        "char_ranges": [
            (0x3400, 0x4DBF),   # CJK Extension A
            (0x4E00, 0x9FFF),   # CJK Unified Ideographs
            (0x3040, 0x309F),   # Hiragana
            (0x30A0, 0x30FF),   # Katakana
            (0x3000, 0x303F),   # CJK Symbols and Punctuation
            (0xFF01, 0xFF5E),   # Fullwidth ASCII variants
            (0xFF61, 0xFF9F),   # Halfwidth Katakana
        ],
        "extra_chars": list("0123456789(),.!?:;-/'\"% "),
        "num_bpe_merges": 2500,
        "pua_base": 0xE000,
        "bpe_merges_filename": "bpe_merges.tsv",
        "char_codes_filename": "cjk_char_codes.tsv",
    },
    "korean": {
        "char_ranges": [(0xAC00, 0xD7A3)],
        "extra_chars": [],
        "num_bpe_merges": 2500,
        "pua_base": 0xEA00,
        "bpe_merges_filename": "korean_bpe_merges.tsv",
        "char_codes_filename": "korean_char_codes.tsv",
    },
    "arabic": {
        "char_ranges": [
            (0x0600, 0x06FF),   # Arabic
            (0x0750, 0x077F),   # Arabic Supplement
            (0x0870, 0x089F),   # Arabic Extended-B
            (0x08A0, 0x08FF),   # Arabic Extended-A
            (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
        ],
        "extra_chars": [],
        "num_bpe_merges": 2500,
        "pua_base": 0xEB00,
        "bpe_merges_filename": "arabic_arb_bpe_merges.tsv",
        "char_codes_filename": "arabic_char_codes.tsv",
    },
}


def find_min_n(char_count: int) -> int:
    """Find minimum N such that N + N² + N³ + N⁴ >= char_count."""
    for n in range(2, 200):
        if n + n**2 + n**3 + n**4 >= char_count:
            return n
    raise ValueError(f"Cannot find N for {char_count} chars")


def all_chars_in_ranges(ranges: list[tuple[int, int]],
                        extra_chars: list[str] | None = None) -> list[str]:
    """All chars in the given codepoint ranges + extras."""
    chars: list[str] = []
    for start, end in ranges:
        for cp in range(start, end + 1):
            chars.append(chr(cp))
    if extra_chars:
        for c in extra_chars:
            if c not in chars:
                chars.append(c)
    return chars


def is_in_ranges(cp: int, ranges: list[tuple[int, int]]) -> bool:
    """Check if a codepoint is in any of the given ranges."""
    return any(start <= cp <= end for start, end in ranges)


def gen_vary_first(n: int):
    """Generate vary-first code tuples: leftmost position varies fastest.

    Yields (0,), (1,), ..., (N-1,), (0,0), (1,0), ..., (0,0,0,0), ...
    """
    for d0 in range(n):
        yield (d0,)
    for d1 in range(n):
        for d0 in range(n):
            yield (d0, d1)
    for d2 in range(n):
        for d1 in range(n):
            for d0 in range(n):
                yield (d0, d1, d2)
    for d3 in range(n):
        for d2 in range(n):
            for d1 in range(n):
                for d0 in range(n):
                    yield (d0, d1, d2, d3)


def fast_bpe(
    seqs: list[list[str]],
    freqs: list[int],
    num_merges: int,
    pua_start: int,
    nchars: list[int] | None = None,
) -> tuple[list[tuple[str, str, str]], list[list[str]]]:
    """Run frequency-weighted word-level BPE with O(affected) per merge.

    Args:
        seqs: list of token sequences (one per word)
        freqs: frequency of each word
        num_merges: number of BPE merges to perform
        pua_start: PUA codepoint for first BPE merge token
        nchars: number of script chars per word (for stats)

    Returns:
        (merges, final_sequences)
    """
    seqs = [list(s) for s in seqs]
    n = len(seqs)
    pua_next = pua_start

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
