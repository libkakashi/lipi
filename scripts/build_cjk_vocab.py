"""
Build CJK vocabulary with atom-based decomposition + BPE merges.

Reads the cjkvi-ids master file and Chinese character frequency data,
computes optimal minimal representations for all 27,584 CJK characters
(Unified + Ext-A), then applies 250 frequency-weighted BPE merges.

Outputs:
  - training_data/word_lists/cjk_decomposition.tsv  (with BPE tokens)
  - training_data/word_lists/bpe_merges.tsv          (250 merges in order)
  - src/data/frozen_vocabs/han_kana_vocab.txt         (new frozen vocab)

Usage:
    python -m scripts.build_cjk_vocab
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDS_PATH = Path("/tmp/ids.txt")
FREQ_PATH = Path("/tmp/chinese_char_freq.tsv")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DECOMP_OUT = PROJECT_ROOT / "training_data" / "word_lists" / "cjk_decomposition.tsv"
BPE_OUT = PROJECT_ROOT / "training_data" / "word_lists" / "bpe_merges.tsv"
VOCAB_OUT = PROJECT_ROOT / "src" / "data" / "frozen_vocabs" / "han_kana_vocab.txt"

CJK_UNIFIED_START = 0x4E00
CJK_UNIFIED_END = 0x9FFF
CJK_EXT_A_START = 0x3400
CJK_EXT_A_END = 0x4DBF

IDS_OPERATORS = set("⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻")
OP_ARITY = {
    "⿰": 2, "⿱": 2, "⿴": 2, "⿵": 2, "⿶": 2,
    "⿷": 2, "⿸": 2, "⿹": 2, "⿺": 2, "⿻": 2,
    "⿲": 3, "⿳": 3,
}

# SEP token: U+2E3B THREE-EM DASH (unused in CJK text)
SEP_CODEPOINT = 0x2E3B
SEP_CHAR = chr(SEP_CODEPOINT)

# BPE merged tokens go into PUA starting at U+E000
PUA_START = 0xE000

NUM_BPE_MERGES = 790

# CJK punctuation tokens (kept in vocab alongside kana, atoms, etc.)
CJK_PUNCTUATION = "ー々、。「」『』（）！？・…〜【】《》〔〕〈〉〝〞〟〆゠︰"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_cjk(cp: int) -> bool:
    return (CJK_UNIFIED_START <= cp <= CJK_UNIFIED_END or
            CJK_EXT_A_START <= cp <= CJK_EXT_A_END)


def parse_ids_file(path: Path) -> dict[str, list[str]]:
    """Parse cjkvi-ids file. Returns char -> list of decomposition strings.

    Parses ALL entries (not just CJK range) because non-CJK entries
    (radicals, strokes) are needed for recursive leaf decomposition.
    For chars with multiple decompositions (variant tags like [GJ]),
    keeps all of them in order.
    """
    char_decomps: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("#") or line.startswith(";") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        char = parts[1]
        if not char or len(char) != 1:
            continue
        for col in parts[2:]:
            decomp = col.split("[")[0].strip()
            if decomp and "&" not in decomp:
                char_decomps[char].append(decomp)
    return dict(char_decomps)


def decompose_to_atoms(
    char: str,
    ids_db: dict[str, list[str]],
    visited: set[str] | None = None,
) -> tuple[list[str], list[str]] | None:
    """Recursively decompose a CJK char to leaf atoms using first decomposition.

    Returns (full_sequence_with_ops, atoms_only) or None if char is atomic.
    Always uses the first available decomposition to get consistent 336 atoms.
    """
    if visited is None:
        visited = set()
    if char in visited:
        return None
    visited = visited | {char}

    if char not in ids_db:
        return None

    # Use FIRST decomposition only for consistency
    decomp_str = ids_db[char][0]
    tokens = [ch for ch in decomp_str if ch.strip() and ord(ch) >= 0x20]
    if not tokens or tokens == [char]:
        return None

    full_seq = []
    atoms = []
    for t in tokens:
        if t in IDS_OPERATORS:
            full_seq.append(t)
        elif t == char:
            # Self-reference: treat as atomic
            full_seq.append(t)
            atoms.append(t)
        else:
            sub = decompose_to_atoms(t, ids_db, visited)
            if sub is not None:
                sub_full, sub_atoms = sub
                full_seq.extend(sub_full)
                atoms.extend(sub_atoms)
            else:
                full_seq.append(t)
                atoms.append(t)

    return (full_seq, atoms) if full_seq else None


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build():
    print("=== Building CJK vocab with atom decomposition + BPE ===")

    # --- Step 1: Parse IDS database ---
    print(f"Reading IDS database from {IDS_PATH}...")
    ids_db = parse_ids_file(IDS_PATH)
    print(f"  {len(ids_db)} CJK chars have decompositions")

    # --- Step 2: Read character frequencies ---
    print(f"Reading frequency data from {FREQ_PATH}...")
    char_freq: dict[str, int] = {}
    for line in FREQ_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("rank"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            char_freq[parts[1]] = int(parts[2])
    print(f"  {len(char_freq)} chars with frequency data")
    default_freq = 1

    # --- Step 3: Decompose all chars to atoms ---
    print("Decomposing all CJK chars to leaf atoms (using first decomposition)...")

    all_cjk_chars = []
    for cp in range(CJK_EXT_A_START, CJK_EXT_A_END + 1):
        all_cjk_chars.append(chr(cp))
    for cp in range(CJK_UNIFIED_START, CJK_UNIFIED_END + 1):
        all_cjk_chars.append(chr(cp))

    # char -> (full_seq_with_ops, atoms_only)
    char_full_decomp: dict[str, tuple[list[str], list[str]]] = {}
    all_atoms: set[str] = set()

    for char in all_cjk_chars:
        result = decompose_to_atoms(char, ids_db)
        if result is None:
            # Atomic
            char_full_decomp[char] = ([char], [char])
            all_atoms.add(char)
        else:
            full_seq, atoms = result
            char_full_decomp[char] = (full_seq, atoms)
            all_atoms.update(atoms)

    print(f"  {len(all_atoms)} unique leaf atoms")
    print(f"  {len(char_full_decomp)} chars decomposed")

    # --- Step 4: Assign representation levels ---
    print("Assigning optimal representation levels...")

    # For each char, determine the minimal representation that is unambiguous
    # Level 1: bag (sorted atoms) -- cheapest
    # Level 2: bag_ops (sorted atoms + sorted ops) -- still orderless
    # Level 3: ordered (atoms in sequence order, no ops) -- order matters
    # Level 4: full (complete sequence with ops) -- most expensive
    # Level 5: unresolved (still ambiguous even with full sequence)

    # char -> (level, token_list)
    char_decomp: dict[str, tuple[str, list[str]]] = {}

    # Start with bag level for everyone
    for char, (full_seq, atoms) in char_full_decomp.items():
        char_decomp[char] = ("bag", sorted(atoms))

    # Find bag-level collisions
    def find_collisions(decomps: dict[str, tuple[str, list[str]]]) -> dict[tuple, list[str]]:
        key_to_chars: dict[tuple, list[str]] = defaultdict(list)
        for char, (level, tokens) in decomps.items():
            key_to_chars[tuple(tokens)].append(char)
        return {k: v for k, v in key_to_chars.items() if len(v) > 1}

    collisions = find_collisions(char_decomp)
    ambiguous_chars = set()
    for chars in collisions.values():
        ambiguous_chars.update(chars)
    print(f"  Bag-level collisions: {len(ambiguous_chars)} chars in {len(collisions)} groups")

    # Promote colliding chars to bag_ops
    for chars in collisions.values():
        for char in chars:
            full_seq, atoms = char_full_decomp[char]
            ops = [t for t in full_seq if t in IDS_OPERATORS]
            if ops:
                char_decomp[char] = ("bag_ops", sorted(atoms) + sorted(ops))
            # If no ops (e.g., atomic), leave as bag -- will be caught later

    # Check bag_ops collisions
    collisions = find_collisions(char_decomp)
    still_ambiguous = set()
    for chars in collisions.values():
        still_ambiguous.update(chars)

    if still_ambiguous:
        # Promote to ordered
        for chars in collisions.values():
            for char in chars:
                full_seq, atoms = char_full_decomp[char]
                ordered_atoms = [t for t in full_seq if t not in IDS_OPERATORS]
                char_decomp[char] = ("ordered", ordered_atoms)

        # Check ordered collisions
        collisions = find_collisions(char_decomp)
        still_ambiguous2 = set()
        for chars in collisions.values():
            still_ambiguous2.update(chars)

        if still_ambiguous2:
            # Promote to full
            for chars in collisions.values():
                for char in chars:
                    full_seq, _ = char_full_decomp[char]
                    char_decomp[char] = ("full", full_seq)

            # Check full collisions
            collisions = find_collisions(char_decomp)
            final_ambiguous = set()
            for chars in collisions.values():
                final_ambiguous.update(chars)

            if final_ambiguous:
                # For each collision group, the most frequent char becomes
                # a single atomic token (fastest); the less frequent chars
                # keep their full decomposition.
                for chars in collisions.values():
                    chars_sorted = sorted(
                        chars, key=lambda c: -char_freq.get(c, 0))
                    # Most frequent becomes atomic (single token = fastest)
                    char_decomp[chars_sorted[0]] = (
                        "atomic_override", [chars_sorted[0]])
                    all_atoms.add(chars_sorted[0])
                    # Least frequent keep their full decomposition
                    # (already set at "full" level from the promotion above)

    # Count levels
    level_counts = Counter(level for level, _ in char_decomp.values())
    total = sum(level_counts.values())
    print(f"  Level distribution:")
    for level in ["bag", "bag_ops", "ordered", "full", "unresolved"]:
        count = level_counts.get(level, 0)
        if count:
            print(f"    {level}: {count} ({100*count/total:.1f}%)")

    # --- Step 5: BPE merges ---
    print(f"\nRunning {NUM_BPE_MERGES} frequency-weighted BPE merges...")

    # Working sequences for BPE (copy of current decomp tokens)
    char_sequences: dict[str, list[str]] = {}
    for char, (level, tokens) in char_decomp.items():
        char_sequences[char] = list(tokens)

    merges: list[tuple[str, str, str]] = []
    pua_next = PUA_START

    for merge_idx in range(NUM_BPE_MERGES):
        # Count adjacent pairs weighted by character frequency
        pair_freq: Counter[tuple[str, str]] = Counter()
        for char, seq in char_sequences.items():
            freq = char_freq.get(char, default_freq)
            for i in range(len(seq) - 1):
                pair_freq[(seq[i], seq[i + 1])] += freq

        if not pair_freq:
            print(f"  No more pairs at step {merge_idx}")
            break

        best_pair, best_count = pair_freq.most_common(1)[0]
        merged_char = chr(pua_next)
        pua_next += 1
        merges.append((best_pair[0], best_pair[1], merged_char))

        if merge_idx < 5 or merge_idx % 50 == 0:
            def _repr(t):
                return f"U+{ord(t):04X}" if len(t) == 1 else t
            print(f"  Merge {merge_idx}: {_repr(best_pair[0])} + {_repr(best_pair[1])}"
                  f" -> U+{ord(merged_char):04X} (freq={best_count})")

        # Apply merge to all sequences
        for char in char_sequences:
            seq = char_sequences[char]
            new_seq = []
            i = 0
            while i < len(seq):
                if (i + 1 < len(seq) and
                        seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]):
                    new_seq.append(merged_char)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            char_sequences[char] = new_seq

    print(f"  {len(merges)} merges applied")
    total_tokens = sum(len(seq) for seq in char_sequences.values())
    print(f"  Average tokens/char: {total_tokens / len(char_sequences):.2f}")

    # Update decomp with post-BPE sequences
    for char in char_decomp:
        level, _ = char_decomp[char]
        char_decomp[char] = (level, char_sequences[char])

    # --- Step 6: Write outputs ---

    # 6a. cjk_decomposition.tsv
    print(f"\nWriting {DECOMP_OUT}...")
    DECOMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DECOMP_OUT, "w", encoding="utf-8") as f:
        f.write("character\tlevel\tsequence\n")
        for char in all_cjk_chars:
            level, tokens = char_decomp[char]
            seq_str = " ".join(tokens)
            f.write(f"{char}\t{level}\t{seq_str}\n")
    print(f"  {len(char_decomp)} entries written")

    # 6b. bpe_merges.tsv
    print(f"Writing {BPE_OUT}...")
    with open(BPE_OUT, "w", encoding="utf-8") as f:
        f.write("index\ttoken_a\ttoken_b\tmerged\n")
        for idx, (a, b, m) in enumerate(merges):
            a_hex = f"{ord(a):04X}" if len(a) == 1 else a
            b_hex = f"{ord(b):04X}" if len(b) == 1 else b
            m_hex = f"{ord(m):04X}"
            f.write(f"{idx}\t{a_hex}\t{b_hex}\t{m_hex}\n")
    print(f"  {len(merges)} merges written")

    # 6c. han_kana_vocab.txt
    print(f"Writing {VOCAB_OUT}...")

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
                (CJK_UNIFIED_START <= cp <= CJK_UNIFIED_END) or
                (CJK_EXT_A_START <= cp <= CJK_EXT_A_END) or
                (cp >= 0x20000) or
                (0x2E80 <= cp <= 0x2EFF) or  # CJK Radicals Supplement
                (0x2FF0 <= cp <= 0x2FFB) or  # IDS operators
                (0xFA00 <= cp <= 0xFA6D) or  # CJK Compat Ideographs
                (PUA_START <= cp <= 0xF8FF)  # PUA (old BPE tokens)
            )
            if not is_cjk_component:
                existing_non_cjk.append(cp)

    vocab_cps.extend(existing_non_cjk)

    # Atoms (sorted by codepoint)
    atom_cps = sorted(ord(a) for a in all_atoms)
    vocab_cps.extend(atom_cps)

    # IDS operators
    ops_cps = sorted(ord(op) for op in IDS_OPERATORS)
    vocab_cps.extend(ops_cps)

    # SEP token
    vocab_cps.append(SEP_CODEPOINT)

    # BPE merged tokens (PUA)
    for _, _, merged_char in merges:
        vocab_cps.append(ord(merged_char))

    # Deduplicate and sort by codepoint (test expects sorted after blank)
    vocab_cps = sorted(set(vocab_cps))

    with open(VOCAB_OUT, "w", encoding="utf-8") as f:
        for cp in vocab_cps:
            f.write(f"{cp:04X}\n")

    # Report
    n_kana = sum(1 for cp in vocab_cps
                 if (0x3041 <= cp <= 0x3096) or (0x30A1 <= cp <= 0x30FA) or
                 cp in (0x30A0, 0x30FB, 0x30FC))
    n_atoms_in_vocab = sum(1 for cp in vocab_cps
                           if chr(cp) in all_atoms)
    n_ops = sum(1 for cp in vocab_cps if 0x2FF0 <= cp <= 0x2FFB)
    n_bpe = sum(1 for cp in vocab_cps if PUA_START <= cp < PUA_START + len(merges))
    n_sep = 1
    n_other = len(vocab_cps) - n_kana - n_atoms_in_vocab - n_ops - n_bpe - n_sep

    print(f"  Total vocab: {len(vocab_cps)} tokens")
    print(f"    Kana: {n_kana}")
    print(f"    Atoms: {n_atoms_in_vocab}")
    print(f"    IDS operators: {n_ops}")
    print(f"    SEP: {n_sep}")
    print(f"    BPE merges: {n_bpe}")
    print(f"    Other (punct/ASCII/fullwidth): {n_other}")

    # --- Verification ---
    print("\n=== Verification ===")
    test_chars = "的想国一"
    for ch in test_chars:
        if ch in char_decomp:
            level, tokens = char_decomp[ch]
            tokens_repr = " ".join(f"U+{ord(t):04X}" for t in tokens)
            print(f"  {ch} (U+{ord(ch):04X}): level={level}, tokens=[{tokens_repr}]")

    # Check round-trip feasibility
    # Build reconstruction map
    token_to_char: dict[tuple, str] = {}
    collisions_found = 0
    for char, (level, tokens) in char_decomp.items():
        key = tuple(tokens)
        if key in token_to_char:
            collisions_found += 1
        else:
            token_to_char[key] = char

    print(f"\n  Reconstruction map: {len(token_to_char)} unique keys")
    if collisions_found:
        print(f"  WARNING: {collisions_found} reconstruction collisions")
    else:
        print(f"  No reconstruction collisions -- round-trip is clean")

    print("\nDone!")


if __name__ == "__main__":
    build()
