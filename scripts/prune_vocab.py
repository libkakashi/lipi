"""
Prune dead tokens from a script's frozen vocab.

Standard post-build step: decompose the full word corpus, collect all tokens
that actually appear in the output, remove everything else from the vocab.
This prevents wasted CTC head capacity on tokens the model will never see.

Handles: SEP tokens absorbed by BPE, intermediate BPE merges that got
merged away, and base symbols that never surface after BPE.

Usage:
    python -m scripts.prune_vocab arabic
    python -m scripts.prune_vocab han_kana
    python -m scripts.prune_vocab korean
    python -m scripts.prune_vocab --all
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_DIR = PROJECT_ROOT / "src" / "encoding" / "frozen_vocabs"
WORD_LIST_DIR = PROJECT_ROOT / "training_data" / "word_lists"
CORPUS_DIR = PROJECT_ROOT / "training_data" / "corpora"

# Which word sources to use per script for token collection
SCRIPT_SOURCES = {
    "arabic": {
        "corpus": "arabic_word_freq.tsv",
        "word_lists": ["arabic.txt", "persian.txt", "urdu.txt"],
        "group": "arabic",
    },
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
}


def collect_used_tokens(script: str) -> set[str]:
    """Decompose all words for a script and collect tokens that appear."""
    from src.data.decompose import decompose_text, _word_cache
    _word_cache.clear()

    config = SCRIPT_SOURCES[script]
    group = config["group"]

    words: set[str] = set()

    # Load from corpus (has frequencies, most comprehensive)
    corpus_path = CORPUS_DIR / config["corpus"]
    if corpus_path.exists():
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if parts:
                w = parts[0].strip()
                if w:
                    words.add(w)

    # Load from word lists
    for wl_name in config["word_lists"]:
        wl_path = WORD_LIST_DIR / wl_name
        if wl_path.exists():
            for line in wl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if w:
                    words.add(w)

    print(f"  {len(words)} unique words to decompose")

    used: set[str] = set()
    total_in = 0
    total_out = 0
    t0 = time.time()

    for w in words:
        d = decompose_text(w, group)
        total_in += len(w)
        total_out += len(d)
        for ch in d:
            used.add(ch)

    elapsed = time.time() - t0
    print(f"  Decomposed in {elapsed:.1f}s")
    print(f"  Avg tokens/char: {total_out / max(total_in, 1):.4f}")
    print(f"  Unique tokens used: {len(used)}")

    return used


def prune_vocab(script: str, dry_run: bool = False) -> dict:
    """Prune dead tokens from a script's vocab.

    Returns stats dict with before/after counts.
    """
    vocab_path = VOCAB_DIR / f"{script}_vocab.txt"
    if not vocab_path.exists():
        print(f"  ERROR: {vocab_path} not found")
        return {}

    # Load current vocab
    current_cps: list[int] = []
    for line in vocab_path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if line:
            current_cps.append(int(line, 16))

    current_tokens = {chr(cp) for cp in current_cps}
    print(f"  Current vocab: {len(current_cps)} tokens")

    # Collect used tokens
    used = collect_used_tokens(script)

    # Find dead tokens (in vocab but never used)
    dead = current_tokens - used
    # Find tokens used but not in vocab (shouldn't happen, but check)
    missing = used - current_tokens
    # Filter missing to only PUA/encoding tokens (regular ASCII chars aren't in vocab)
    missing_encoding = {t for t in missing if ord(t) >= 0x2E00}

    print(f"\n  Dead tokens (in vocab, never used): {len(dead)}")
    if missing_encoding:
        print(f"  WARNING: {len(missing_encoding)} used tokens NOT in vocab!")

    # Categorize dead tokens
    SEP_CP = 0x2E3B
    dead_sep = sum(1 for t in dead if ord(t) == SEP_CP)
    dead_base = sum(1 for t in dead if 0xE000 <= ord(t) < 0xF000 and t not in used)
    dead_bpe = len(dead) - dead_sep - dead_base
    print(f"    SEP: {dead_sep}")
    print(f"    Base symbols: {dead_base}")
    print(f"    BPE merges: {dead_bpe}")

    # Prune
    pruned_cps = sorted(cp for cp in current_cps if chr(cp) in used or chr(cp) in missing_encoding)

    stats = {
        "before": len(current_cps),
        "after": len(pruned_cps),
        "dead": len(dead),
        "dead_sep": dead_sep,
        "dead_base": dead_base,
        "dead_bpe": dead_bpe,
    }

    if not dry_run and dead:
        vocab_path.write_text(
            "\n".join(f"{cp:04X}" for cp in pruned_cps) + "\n",
            encoding="utf-8"
        )
        print(f"\n  Wrote {len(pruned_cps)} tokens to {vocab_path.name}")
        print(f"  Removed {len(dead)} dead tokens ({len(current_cps)} -> {len(pruned_cps)})")
    elif dead:
        print(f"\n  [DRY RUN] Would remove {len(dead)} tokens ({len(current_cps)} -> {len(pruned_cps)})")
    else:
        print(f"\n  No dead tokens found — vocab is clean")

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.prune_vocab <script|--all> [--dry-run]")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv
    target = sys.argv[1]

    scripts = list(SCRIPT_SOURCES.keys()) if target == "--all" else [target]

    for script in scripts:
        if script not in SCRIPT_SOURCES:
            print(f"Unknown script: {script}")
            print(f"Available: {', '.join(SCRIPT_SOURCES.keys())}")
            sys.exit(1)

        print(f"\n{'='*50}")
        print(f"Pruning {script} vocab...")
        print(f"{'='*50}")
        stats = prune_vocab(script, dry_run=dry_run)
        if stats:
            print(f"\n  Summary: {stats['before']} -> {stats['after']} "
                  f"(-{stats['dead']} dead)")


if __name__ == "__main__":
    main()
