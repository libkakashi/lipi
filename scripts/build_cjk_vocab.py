"""
Build CJK vocabulary with atom-based decomposition + SEP-aware BPE.

Pipeline:
  1. Parse cjkvi-ids and decompose all 27,584 CJK chars to leaf atoms
  2. Assign minimal unambiguous representations (bag → bag_ops → ordered → full)
  3. Find prefix collisions on base sequences, append SEP to those chars
  4. Run 1,390 frequency-weighted BPE merges on SEP-augmented sequences
     (BPE naturally merges high-frequency atom+SEP pairs into single tokens)

Outputs:
  - training_data/word_lists/cjk_decomposition.tsv
  - training_data/word_lists/bpe_merges.tsv
  - src/data/frozen_vocabs/han_kana_vocab.txt

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

# SEP token: U+2E3B THREE-EM DASH (unused in CJK text)
SEP_CODEPOINT = 0x2E3B
SEP_CHAR = chr(SEP_CODEPOINT)

# BPE merged tokens go into PUA starting at U+E000
PUA_START = 0xE000

NUM_BPE_MERGES = 1390

# Kana atom replacements: the IDS database uses 3 katakana as shape
# placeholders (コ ス ユ). We replace them with PUA tokens so kana
# codepoints never appear in CJK decompositions — avoids ambiguity
# with standalone kana in Japanese text.
KANA_ATOM_REPLACEMENTS = {
    "\u30B3": "\uF000",  # コ -> ATOM_KO (right-angle enclosure)
    "\u30B9": "\uF001",  # ス -> ATOM_SU (diagonal stroke pair)
    "\u30E6": "\uF002",  # ユ -> ATOM_YU (horizontal hook)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_cjk(cp: int) -> bool:
    return (CJK_UNIFIED_START <= cp <= CJK_UNIFIED_END or
            CJK_EXT_A_START <= cp <= CJK_EXT_A_END)


def parse_ids_file(path: Path) -> dict[str, list[str]]:
    """Parse cjkvi-ids file → char → list of decomposition strings.

    Parses ALL entries (not just CJK range) because non-CJK entries
    (radicals, strokes) are needed for recursive leaf decomposition.
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
    """Recursively decompose a CJK char to leaf atoms.

    Returns (full_sequence_with_ops, atoms_only) or None if atomic.
    Always uses the first available decomposition for consistency.
    """
    if visited is None:
        visited = set()
    if char in visited:
        return None
    visited = visited | {char}

    if char not in ids_db:
        return None

    decomp_str = ids_db[char][0]
    tokens = [ch for ch in decomp_str if ch.strip() and ord(ch) >= 0x20]
    if not tokens or tokens == [char]:
        return None

    full_seq: list[str] = []
    atoms: list[str] = []
    for t in tokens:
        if t in IDS_OPERATORS:
            full_seq.append(t)
        elif t == char:
            full_seq.append(t)
            atoms.append(t)
        else:
            sub = decompose_to_atoms(t, ids_db, visited)
            if sub is not None:
                full_seq.extend(sub[0])
                atoms.extend(sub[1])
            else:
                full_seq.append(t)
                atoms.append(t)

    return (full_seq, atoms) if full_seq else None


def find_collisions(
    decomps: dict[str, tuple[str, list[str]]],
) -> dict[tuple[str, ...], list[str]]:
    """Find groups of chars with identical token sequences."""
    key_to_chars: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for char, (_, tokens) in decomps.items():
        key_to_chars[tuple(tokens)].append(char)
    return {k: v for k, v in key_to_chars.items() if len(v) > 1}


def find_prefix_collisions(sequences: dict[str, list[str]]) -> set[str]:
    """Find chars whose token sequence is a prefix of another char's sequence.

    Uses a trie for O(total_tokens) performance.
    """
    # Build trie
    trie: dict = {}
    END = object()
    for seq in sequences.values():
        node = trie
        for t in seq:
            node = node.setdefault(t, {})
        node[END] = True

    # Check each char: is its sequence a prefix of a longer one?
    needs_sep: set[str] = set()
    for char, seq in sequences.items():
        node = trie
        for t in seq:
            node = node[t]
        # If node has children beyond END, this sequence is a prefix of another
        if len(node) > 1 or (len(node) == 1 and END not in node):
            needs_sep.add(char)
    return needs_sep


def apply_bpe_merges(
    sequences: dict[str, list[str]],
    char_freq: dict[str, int],
    num_merges: int,
    default_freq: int = 1,
) -> tuple[dict[str, list[str]], list[tuple[str, str, str]]]:
    """Run frequency-weighted BPE on sequences.

    Returns (updated_sequences, list_of_merges).
    Each merge is (token_a, token_b, merged_token).
    """
    seqs = {char: list(seq) for char, seq in sequences.items()}
    merges: list[tuple[str, str, str]] = []
    pua_next = PUA_START

    for merge_idx in range(num_merges):
        # Count adjacent pairs weighted by character frequency
        pair_freq: Counter[tuple[str, str]] = Counter()
        for char, seq in seqs.items():
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

        if merge_idx < 5 or merge_idx % 100 == 0:
            has_sep = SEP_CHAR in best_pair
            sep_tag = " [SEP-merge]" if has_sep else ""
            print(f"  Merge {merge_idx}: freq={best_count}{sep_tag}")

        # Apply merge to all sequences
        for char in seqs:
            seq = seqs[char]
            new_seq: list[str] = []
            i = 0
            while i < len(seq):
                if (i + 1 < len(seq) and
                        seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]):
                    new_seq.append(merged_char)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seqs[char] = new_seq

    return seqs, merges


# ---------------------------------------------------------------------------
# Main build pipeline
# ---------------------------------------------------------------------------

def build():
    print("=== Building CJK vocab: atom decomposition + SEP-aware BPE ===")

    # --- Step 1: Parse IDS database ---
    print(f"\n[1/6] Reading IDS database from {IDS_PATH}...")
    if not IDS_PATH.exists():
        print(f"  ERROR: {IDS_PATH} not found. Download from cjkvi-ids.")
        sys.exit(1)
    ids_db = parse_ids_file(IDS_PATH)
    print(f"  {len(ids_db)} entries parsed")

    # --- Step 2: Read character frequencies ---
    print(f"\n[2/6] Reading frequency data from {FREQ_PATH}...")
    if not FREQ_PATH.exists():
        print(f"  ERROR: {FREQ_PATH} not found.")
        sys.exit(1)
    char_freq: dict[str, int] = {}
    for line in FREQ_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("rank"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            char_freq[parts[1]] = int(parts[2])
    print(f"  {len(char_freq)} chars with frequency data")

    # --- Step 3: Decompose all chars to atoms ---
    print("\n[3/6] Decomposing all CJK chars to leaf atoms...")

    all_cjk_chars: list[str] = []
    for cp in range(CJK_EXT_A_START, CJK_EXT_A_END + 1):
        all_cjk_chars.append(chr(cp))
    for cp in range(CJK_UNIFIED_START, CJK_UNIFIED_END + 1):
        all_cjk_chars.append(chr(cp))

    char_full_decomp: dict[str, tuple[list[str], list[str]]] = {}
    all_atoms: set[str] = set()

    for char in all_cjk_chars:
        result = decompose_to_atoms(char, ids_db)
        if result is None:
            char_full_decomp[char] = ([char], [char])
            all_atoms.add(char)
        else:
            full_seq, atoms = result
            char_full_decomp[char] = (full_seq, atoms)
            all_atoms.update(atoms)

    print(f"  {len(all_atoms)} unique leaf atoms")
    print(f"  {len(char_full_decomp)} chars decomposed")

    # Replace kana atoms with PUA tokens to avoid kana/CJK ambiguity
    replaced_count = 0
    for char in char_full_decomp:
        full_seq, atoms = char_full_decomp[char]
        new_full = [KANA_ATOM_REPLACEMENTS.get(t, t) for t in full_seq]
        new_atoms = [KANA_ATOM_REPLACEMENTS.get(t, t) for t in atoms]
        if new_full != full_seq:
            char_full_decomp[char] = (new_full, new_atoms)
            replaced_count += 1
    if replaced_count:
        # Update atom set
        for old, new in KANA_ATOM_REPLACEMENTS.items():
            if old in all_atoms:
                all_atoms.discard(old)
                all_atoms.add(new)
        print(f"  Replaced kana atoms (コ→F000, ス→F001, ユ→F002) in {replaced_count} chars")

    # --- Step 4: Assign representation levels ---
    print("\n[4/6] Assigning minimal unambiguous representations...")

    # Level progression: bag → bag_ops → ordered → full → atomic_override
    char_decomp: dict[str, tuple[str, list[str]]] = {}
    for char, (_, atoms) in char_full_decomp.items():
        char_decomp[char] = ("bag", sorted(atoms))

    # Promote through levels until no collisions remain
    collisions = find_collisions(char_decomp)
    if collisions:
        for chars in collisions.values():
            for char in chars:
                full_seq, atoms = char_full_decomp[char]
                ops = [t for t in full_seq if t in IDS_OPERATORS]
                if ops:
                    char_decomp[char] = ("bag_ops", sorted(atoms) + sorted(ops))

    collisions = find_collisions(char_decomp)
    if collisions:
        for chars in collisions.values():
            for char in chars:
                full_seq, _ = char_full_decomp[char]
                ordered = [t for t in full_seq if t not in IDS_OPERATORS]
                char_decomp[char] = ("ordered", ordered)

    collisions = find_collisions(char_decomp)
    if collisions:
        for chars in collisions.values():
            for char in chars:
                full_seq, _ = char_full_decomp[char]
                char_decomp[char] = ("full", full_seq)

    collisions = find_collisions(char_decomp)
    if collisions:
        # Collision overrides: most frequent char becomes atomic token
        for chars in collisions.values():
            chars_sorted = sorted(chars, key=lambda c: -char_freq.get(c, 0))
            char_decomp[chars_sorted[0]] = ("atomic_override", [chars_sorted[0]])
            all_atoms.add(chars_sorted[0])

    level_counts = Counter(level for level, _ in char_decomp.values())
    print(f"  Level distribution:")
    for level in ["bag", "bag_ops", "ordered", "full", "atomic_override"]:
        count = level_counts.get(level, 0)
        if count:
            print(f"    {level}: {count}")

    # Verify zero collisions
    collisions = find_collisions(char_decomp)
    assert not collisions, f"Still have {len(collisions)} collision groups!"

    # --- Step 5: SEP-aware BPE ---
    print(f"\n[5/6] SEP-aware BPE ({NUM_BPE_MERGES} merges)...")

    # 5a. Find chars that need SEP
    base_sequences = {char: list(tokens) for char, (_, tokens) in char_decomp.items()}

    # Prefix collisions: char's sequence is a prefix of another's
    needs_sep = find_prefix_collisions(base_sequences)

    print(f"  Prefix collisions: {len(needs_sep)} chars need SEP "
          f"({len(needs_sep) * 100 / len(base_sequences):.1f}%)")

    # 5b. Append SEP to chars that need it
    sep_sequences: dict[str, list[str]] = {}
    for char, seq in base_sequences.items():
        if char in needs_sep:
            sep_sequences[char] = seq + [SEP_CHAR]
        else:
            sep_sequences[char] = list(seq)

    # 5c. Run BPE on SEP-augmented sequences
    bpe_sequences, merges = apply_bpe_merges(
        sep_sequences, char_freq, NUM_BPE_MERGES,
    )
    print(f"  {len(merges)} merges applied")

    # Stats
    total_freq = sum(char_freq.get(c, 0) for c in bpe_sequences)
    total_tokens = sum(
        char_freq.get(c, 0) * len(seq)
        for c, seq in bpe_sequences.items()
        if char_freq.get(c, 0) > 0
    )
    print(f"  Freq-weighted avg tokens/char: {total_tokens / total_freq:.4f}")

    bare_sep = sum(1 for c in needs_sep if SEP_CHAR in bpe_sequences[c])
    merged_sep = len(needs_sep) - bare_sep
    print(f"  SEP merged by BPE: {merged_sep}/{len(needs_sep)}")
    print(f"  Bare SEP remaining: {bare_sep}")

    # Update decomp with post-BPE sequences (keep original level labels)
    for char in char_decomp:
        level, _ = char_decomp[char]
        char_decomp[char] = (level, bpe_sequences[char])

    # --- Step 6: Write outputs ---
    print(f"\n[6/6] Writing outputs...")

    # 6a. cjk_decomposition.tsv
    print(f"  {DECOMP_OUT}...")
    DECOMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DECOMP_OUT, "w", encoding="utf-8") as f:
        f.write("character\tlevel\tsequence\n")
        for char in all_cjk_chars:
            level, tokens = char_decomp[char]
            seq_str = " ".join(tokens)
            f.write(f"{char}\t{level}\t{seq_str}\n")
    print(f"    {len(char_decomp)} entries")

    # 6b. bpe_merges.tsv
    print(f"  {BPE_OUT}...")
    with open(BPE_OUT, "w", encoding="utf-8") as f:
        f.write("index\ttoken_a\ttoken_b\tmerged\n")
        for idx, (a, b, m) in enumerate(merges):
            a_hex = f"{ord(a):04X}" if len(a) == 1 else a
            b_hex = f"{ord(b):04X}" if len(b) == 1 else b
            m_hex = f"{ord(m):04X}"
            f.write(f"{idx}\t{a_hex}\t{b_hex}\t{m_hex}\n")
    print(f"    {len(merges)} merges")

    # 6c. han_kana_vocab.txt
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
                (cp >= 0x20000) or
                (0x2E80 <= cp <= 0x2EFF) or   # CJK Radicals Supplement
                (0x2FF0 <= cp <= 0x2FFB) or   # IDS operators
                (0xFA00 <= cp <= 0xFA6D) or   # CJK Compat Ideographs
                (PUA_START <= cp <= 0xF8FF) or  # PUA (old BPE tokens)
                (0x31C0 <= cp <= 0x31EF) or   # CJK Strokes
                cp == SEP_CODEPOINT
            )
            if not is_cjk_component:
                existing_non_cjk.append(cp)

    vocab_cps.extend(existing_non_cjk)

    # Atoms
    atom_cps = sorted(ord(a) for a in all_atoms)
    vocab_cps.extend(atom_cps)

    # IDS operators
    vocab_cps.extend(sorted(ord(op) for op in IDS_OPERATORS))

    # SEP token
    vocab_cps.append(SEP_CODEPOINT)

    # BPE merged tokens (PUA)
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
    n_atoms = sum(1 for cp in vocab_cps if chr(cp) in all_atoms)
    n_ops = sum(1 for cp in vocab_cps if 0x2FF0 <= cp <= 0x2FFB)
    n_bpe = sum(1 for cp in vocab_cps if PUA_START <= cp < PUA_START + len(merges))
    n_sep = 1
    n_other = len(vocab_cps) - n_kana - n_atoms - n_ops - n_bpe - n_sep

    print(f"\n  Total vocab: {len(vocab_cps)} tokens (+1 BLANK = {len(vocab_cps) + 1})")
    print(f"    Kana: {n_kana}")
    print(f"    Atoms: {n_atoms}")
    print(f"    IDS operators: {n_ops}")
    print(f"    SEP: {n_sep}")
    print(f"    BPE merges: {n_bpe}")
    print(f"    Other (punct/ASCII/fullwidth): {n_other}")

    # --- Verification ---
    print("\n=== Verification ===")

    # Round-trip check
    token_to_char: dict[tuple[str, ...], str] = {}
    collisions_found = 0
    for char, (_, tokens) in char_decomp.items():
        key = tuple(tokens)
        if key in token_to_char:
            collisions_found += 1
            print(f"  COLLISION: {char} and {token_to_char[key]} -> {key}")
        else:
            token_to_char[key] = char

    print(f"  Reconstruction map: {len(token_to_char)} unique keys")
    if collisions_found:
        print(f"  WARNING: {collisions_found} reconstruction collisions!")
    else:
        print(f"  Zero reconstruction collisions — round-trip clean")

    # Verify all tokens in decompositions are in vocab
    vocab_chars = {chr(cp) for cp in vocab_cps}
    missing_tokens: set[str] = set()
    for _, tokens in char_decomp.values():
        for t in tokens:
            if t not in vocab_chars:
                missing_tokens.add(t)

    if missing_tokens:
        print(f"  WARNING: {len(missing_tokens)} tokens in decompositions not in vocab!")
        for t in sorted(missing_tokens, key=ord):
            print(f"    U+{ord(t):04X}")
    else:
        print(f"  All decomposition tokens present in vocab")

    # Sample decompositions
    test_chars = "的一想国人"
    print(f"\n  Sample decompositions:")
    for ch in test_chars:
        if ch in char_decomp:
            level, tokens = char_decomp[ch]
            tokens_repr = " ".join(f"U+{ord(t):04X}" for t in tokens)
            has_sep = SEP_CHAR in tokens
            print(f"    {ch}: level={level}, {len(tokens)} tok, "
                  f"SEP={'yes' if has_sep else 'no'} [{tokens_repr}]")

    print("\nDone!")


if __name__ == "__main__":
    build()
