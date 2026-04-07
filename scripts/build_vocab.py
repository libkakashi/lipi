"""
Build arbitrary encoding for a script.

Thin CLI wrapper around src.encoding.encoding.build_encoding().

Usage:
    python -m scripts.build_vocab --script han_kana
    python -m scripts.build_vocab --script korean
    python -m scripts.build_vocab --script arabic
    python -m scripts.build_vocab --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encoding.encoding import SCRIPT_CONFIG, build_encoding


def main():
    parser = argparse.ArgumentParser(description="Build arbitrary encoding for a script")
    parser.add_argument("--script", choices=list(SCRIPT_CONFIG.keys()),
                        help="Script to build encoding for")
    parser.add_argument("--all", action="store_true",
                        help="Build encodings for all scripts")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    if not args.script and not args.all:
        parser.error("Either --script or --all is required")

    scripts = list(SCRIPT_CONFIG.keys()) if args.all else [args.script]

    for script in scripts:
        stats = build_encoding(script, verbose=not args.quiet)
        print(f"\n{'='*50}")
        print(f"{script}: {stats['vocab_size']} vocab, "
              f"{stats['tokens_per_char']:.4f} tok/char, "
              f"0 dead tokens")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
