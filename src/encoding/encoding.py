"""
Arbitrary N-symbol encoding for large character sets.

Core algorithm:
  1. Find minimum N base symbols needed: N + N² + N³ + N⁴ >= char_count
  2. Rank chars by frequency, assign vary-first codes + SEP
  3. Run word-level BPE on a word frequency corpus
  4. Prune dead tokens (tokens that never appear after decomposing the corpus)
  5. If pruned tokens exist, run more BPE merges to fill back to target vocab size
  6. Repeat prune+refill until vocab is stable with zero dead tokens
  7. Result: each char → short PUA token sequence, vocab size 2500-2600

Used by CJK (N=13), Korean (N=11), Arabic (N=9).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

# SEP token: U+2E3B THREE-EM DASH
SEP_CODEPOINT = 0x2E3B
SEP_CHAR = chr(SEP_CODEPOINT)

# Target vocab range after pruning+refilling
TARGET_VOCAB_MIN = 2500
TARGET_VOCAB_MAX = 2600

# Common characters included in every script's encoding
COMMON_EXTRA_CHARS = list("0123456789(),.!?:;-/'\"% _~")

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
        "extra_chars": COMMON_EXTRA_CHARS,
        "num_bpe_merges": 2500,
        "pua_base": 0xE000,
        "bpe_merges_filename": "bpe_merges.tsv",
        "char_codes_filename": "cjk_char_codes.tsv",
    },
    "korean": {
        "char_ranges": [
            (0xAC00, 0xD7A3),   # Hangul Syllables
            (0x3131, 0x318E),   # Hangul Compatibility Jamo (standalone ㄱㄴㄷ)
            (0x3000, 0x303F),   # CJK Symbols and Punctuation (《》「」)
            (0x2018, 0x201F),   # Smart quotes (''""‟)
        ],
        "extra_chars": COMMON_EXTRA_CHARS,
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
        "extra_chars": COMMON_EXTRA_CHARS,
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


# ---------------------------------------------------------------------------
# Build configuration per script
# ---------------------------------------------------------------------------

# Corpus and word list sources for each script
BUILD_SOURCES = {
    "han_kana": {
        "corpus": "han_kana_word_freq.tsv",
        "word_lists": ["chinese.txt", "japanese.txt"],
        "group": "sino_japanese",
    },
    "korean": {
        "corpus": "korean_word_freq.tsv",
        "word_lists": ["korean.txt"],
        "group": "korean",
    },
    "arabic": {
        "corpus": "arabic_word_freq.tsv",
        "word_lists": ["arabic.txt", "persian.txt", "urdu.txt"],
        "group": "arabic",
    },
}


# ---------------------------------------------------------------------------
# Build pipeline helpers
# ---------------------------------------------------------------------------

def _load_word_freq(script_name: str, char_ranges: list[tuple[int, int]],
                    all_chars: list[str], project_root: Path,
                    ) -> tuple[dict[str, int], Counter[str]]:
    """Load word frequency corpus and derive char frequencies."""
    sources = BUILD_SOURCES[script_name]
    corpus_path = project_root / "training_data" / "corpora" / sources["corpus"]
    wl_dir = project_root / "training_data" / "word_lists"

    word_freq: dict[str, int] = {}

    # Primary: corpus with frequencies
    if corpus_path.exists():
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("word"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                w, f = parts[0], int(parts[1])
                if f > 0 and all(is_in_ranges(ord(c), char_ranges) for c in w):
                    word_freq[w] = f

    # Supplement: word lists (freq=1 for words not in corpus)
    for wl_name in sources["word_lists"]:
        wl_path = wl_dir / wl_name
        if wl_path.exists():
            for line in wl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if w and w not in word_freq:
                    if all(is_in_ranges(ord(c), char_ranges) for c in w):
                        word_freq[w] = 1

    # Ensure all single chars are in corpus
    for c in all_chars:
        if c not in word_freq:
            word_freq[c] = 1

    # Derive char frequencies
    char_freq: Counter[str] = Counter()
    for w, f in word_freq.items():
        for c in w:
            if is_in_ranges(ord(c), char_ranges):
                char_freq[c] += f

    return word_freq, char_freq


def _assign_codes(chars_ranked: list[str], n_symbols: int,
                  base_symbols: list[str]) -> dict[str, list[str]]:
    """Assign vary-first codes + SEP to ranked chars."""
    gen = gen_vary_first(n_symbols)
    char_to_code: dict[str, list[str]] = {}
    for c in chars_ranked:
        code_tuple = next(gen)
        code = [base_symbols[d] for d in code_tuple] + [SEP_CHAR]
        char_to_code[c] = code
    return char_to_code


def _build_word_seqs(word_freq: dict[str, int],
                     char_to_code: dict[str, list[str]],
                     ) -> tuple[list[list[str]], list[int], list[int]]:
    """Build BPE input sequences from word corpus."""
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
    return w_seqs, w_freqs, w_nchars


def _apply_merges_to_char_codes(
    chars: list[str],
    char_to_code: dict[str, list[str]],
    merges: list[tuple[str, str, str]],
) -> dict[str, list[str]]:
    """Apply BPE merges to per-char codes."""
    result: dict[str, list[str]] = {}
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
        result[c] = seq
    return result


def _collect_used_tokens(word_freq: dict[str, int],
                         char_codes_post_bpe: dict[str, list[str]],
                         merges: list[tuple[str, str, str]],
                         ) -> set[str]:
    """Decompose all words and collect tokens that actually appear.

    Simulates the full decompose+BPE pipeline without needing the
    decompose module (avoids circular imports).
    """
    # Build merge lookup for priority-based BPE
    lookup: dict[tuple[str, str], tuple[str, int]] = {}
    for priority, (a, b, merged) in enumerate(merges):
        pair = (a, b)
        if pair not in lookup:
            lookup[pair] = (merged, priority)

    def apply_bpe(parts: list[str]) -> list[str]:
        if len(parts) <= 1:
            return parts
        while True:
            best_priority = len(merges)
            best_pos = -1
            best_merged = ""
            for i in range(len(parts) - 1):
                entry = lookup.get((parts[i], parts[i + 1]))
                if entry and entry[1] < best_priority:
                    best_merged, best_priority = entry
                    best_pos = i
            if best_pos < 0:
                break
            a, b = parts[best_pos], parts[best_pos + 1]
            new_parts: list[str] = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == a and parts[i + 1] == b:
                    new_parts.append(best_merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    used: set[str] = set()
    for w in word_freq:
        # Per-char lookup
        parts: list[str] = []
        for ch in w:
            tokens = char_codes_post_bpe.get(ch)
            if tokens:
                parts.extend(tokens)
            else:
                parts.append(ch)
        # Apply cross-char BPE
        parts = apply_bpe(parts)
        used.update(parts)

    return used


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build_encoding(
    script_name: str,
    project_root: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Build complete encoding for a script: codes + BPE + prune + refill.

    The prune+refill loop ensures no dead tokens in the final vocab:
      1. Run N BPE merges
      2. Decompose entire corpus, find which tokens actually appear
      3. Count dead tokens (in vocab but never used)
      4. If dead > 0: run more BPE merges to compensate, repeat from step 2
      5. Stop when vocab is in [TARGET_VOCAB_MIN, TARGET_VOCAB_MAX] with 0 dead

    Returns dict with all build artifacts and stats.
    """
    if script_name not in SCRIPT_CONFIG:
        raise ValueError(f"Unknown script: {script_name}. "
                         f"Available: {', '.join(SCRIPT_CONFIG.keys())}")

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent

    cfg = SCRIPT_CONFIG[script_name]
    char_ranges = cfg["char_ranges"]
    pua_base = cfg["pua_base"]
    initial_bpe_merges = cfg["num_bpe_merges"]

    char_codes_path = project_root / "training_data" / "word_lists" / cfg["char_codes_filename"]
    bpe_path = project_root / "training_data" / "word_lists" / cfg["bpe_merges_filename"]
    vocab_path = project_root / "src" / "encoding" / "frozen_vocabs" / f"{script_name}_vocab.txt"

    def log(msg: str):
        if verbose:
            print(msg)

    log(f"=== Building {script_name} encoding ===")

    # Step 1: Enumerate chars
    all_chars = all_chars_in_ranges(char_ranges, cfg.get("extra_chars"))
    char_count = len(all_chars)
    n_symbols = find_min_n(char_count)
    base_symbols = [chr(pua_base + i) for i in range(n_symbols)]
    bpe_pua_start = pua_base + n_symbols

    log(f"  {char_count} chars, N={n_symbols} symbols")

    # Step 2: Load corpus
    word_freq, char_freq = _load_word_freq(
        script_name, char_ranges, all_chars, project_root)
    log(f"  {len(word_freq)} words loaded")

    # Step 3: Rank chars and assign codes
    chars_ranked = sorted(all_chars, key=lambda c: (-char_freq.get(c, 0), ord(c)))
    char_to_code = _assign_codes(chars_ranked, n_symbols, base_symbols)

    # Step 4: Build word sequences
    w_seqs, w_freqs, w_nchars = _build_word_seqs(word_freq, char_to_code)
    total_chars_w = sum(f * nc for f, nc in zip(w_freqs, w_nchars))
    total_tokens_pre = sum(f * len(s) for f, s in zip(w_freqs, w_seqs))
    log(f"  Pre-BPE tokens/char: {total_tokens_pre / total_chars_w:.4f}")

    # Step 5: BPE with iterative sizing
    #
    # Strategy: estimate the dead token ratio from a first pass, then compute
    # how many total merges we need so that (total_merges - dead) lands in
    # [TARGET_VOCAB_MIN, TARGET_VOCAB_MAX]. One calibration round is enough.
    num_merges = initial_bpe_merges

    log(f"\n  [Pass 1] Running {num_merges} BPE merges (calibration)...")
    merges, final_seqs = fast_bpe(
        w_seqs, w_freqs, num_merges, bpe_pua_start,
        w_nchars if verbose else None)

    total_tokens_post = sum(f * len(s) for f, s in zip(w_freqs, final_seqs))
    log(f"  Post-BPE tokens/char: {total_tokens_post / total_chars_w:.4f}")

    char_codes_post_bpe = _apply_merges_to_char_codes(
        all_chars, char_to_code, merges)

    # Count dead tokens to estimate the dead ratio
    vocab_cps: set[int] = set()
    for i in range(n_symbols):
        vocab_cps.add(pua_base + i)
    vocab_cps.add(SEP_CODEPOINT)
    for _, _, m in merges:
        vocab_cps.add(ord(m))

    used_tokens = _collect_used_tokens(word_freq, char_codes_post_bpe, merges)
    used_cps = {ord(t) for t in used_tokens if 0x2E00 <= ord(t) <= 0xFFFF}
    dead_cps = vocab_cps - used_cps
    n_dead = len(dead_cps)
    n_live = len(vocab_cps) - n_dead

    log(f"  Calibration: {len(vocab_cps)} total, {n_dead} dead, {n_live} live")

    # If we already have enough live tokens, we're done — just prune
    # Otherwise, estimate needed merges: if dead_ratio = dead/total,
    # we need total_merges such that total * (1 - dead_ratio) ≈ target
    if n_live < TARGET_VOCAB_MIN:
        dead_ratio = n_dead / len(vocab_cps) if vocab_cps else 0.2
        # Solve: (n_symbols + 1 + needed_merges) * (1 - dead_ratio) = target
        fixed_tokens = n_symbols + 1  # base symbols + SEP
        target_mid = (TARGET_VOCAB_MIN + TARGET_VOCAB_MAX) // 2
        needed_total = int(target_mid / max(1 - dead_ratio, 0.5))
        needed_merges = needed_total - fixed_tokens

        log(f"  Dead ratio: {dead_ratio:.2%}. "
            f"Need ~{needed_merges} merges for ~{target_mid} live tokens")

        # Re-run BPE with adjusted merge count
        w_seqs, w_freqs, w_nchars = _build_word_seqs(word_freq, char_to_code)
        log(f"\n  [Pass 2] Running {needed_merges} BPE merges...")
        merges, final_seqs = fast_bpe(
            w_seqs, w_freqs, needed_merges, bpe_pua_start,
            w_nchars if verbose else None)

        total_tokens_post = sum(f * len(s) for f, s in zip(w_freqs, final_seqs))
        log(f"  Post-BPE tokens/char: {total_tokens_post / total_chars_w:.4f}")

        char_codes_post_bpe = _apply_merges_to_char_codes(
            all_chars, char_to_code, merges)

        vocab_cps = set()
        for i in range(n_symbols):
            vocab_cps.add(pua_base + i)
        vocab_cps.add(SEP_CODEPOINT)
        for _, _, m in merges:
            vocab_cps.add(ord(m))

        used_tokens = _collect_used_tokens(word_freq, char_codes_post_bpe, merges)
        used_cps = {ord(t) for t in used_tokens if 0x2E00 <= ord(t) <= 0xFFFF}
        dead_cps = vocab_cps - used_cps
        n_dead = len(dead_cps)
        n_live = len(vocab_cps) - n_dead
        log(f"  Result: {len(vocab_cps)} total, {n_dead} dead, {n_live} live")

    # Prune dead tokens from final vocab
    final_vocab_cps = sorted(cp for cp in vocab_cps if cp in used_cps)

    # Write outputs
    log(f"\n  Writing outputs...")

    # char_codes.tsv
    char_codes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(char_codes_path, "w", encoding="utf-8") as f:
        f.write("character\tcode_hex\trank\tfreq\n")
        for rank, c in enumerate(chars_ranked):
            code = char_codes_post_bpe[c]
            hex_str = " ".join(f"{ord(t):04X}" for t in code)
            freq = char_freq.get(c, 0)
            f.write(f"{c}\t{hex_str}\t{rank}\t{freq}\n")

    # bpe_merges.tsv
    with open(bpe_path, "w", encoding="utf-8") as f:
        f.write("index\ttoken_a\ttoken_b\tmerged\n")
        for idx, (a, b, m) in enumerate(merges):
            f.write(f"{idx}\t{ord(a):04X}\t{ord(b):04X}\t{ord(m):04X}\n")

    # vocab.txt (pruned — only live tokens)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        for cp in final_vocab_cps:
            f.write(f"{cp:04X}\n")

    log(f"  {char_codes_path.name}: {len(all_chars)} entries")
    log(f"  {bpe_path.name}: {len(merges)} merges")
    log(f"  {vocab_path.name}: {len(final_vocab_cps)} tokens (0 dead)")

    stats = {
        "script": script_name,
        "n_symbols": n_symbols,
        "char_count": char_count,
        "n_merges": len(merges),
        "vocab_size": len(final_vocab_cps),
        "pruned": len(dead_cps),
        "tokens_per_char": total_tokens_post / total_chars_w,
    }
    log(f"\n  Final: {stats['vocab_size']} vocab, "
        f"{stats['tokens_per_char']:.4f} tok/char, "
        f"{stats['pruned']} pruned")

    return stats
