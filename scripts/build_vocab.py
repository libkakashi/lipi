#!/usr/bin/env python3
"""
Build vocabulary for a language.

Usage:
    python scripts/build_vocab.py --config configs/vocab/hindi.yaml
    python scripts/build_vocab.py --script_id en --word_list training_data/word_lists/english_100k.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bigrams import LipiTokenizer, SCRIPT_CHARSETS


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--script_id", type=str, help="Script ID (e.g., hi, ta)")
    parser.add_argument("--word_list", type=str, help="Word list file path")
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--max_token_length", type=int, default=4)
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--char_level", action="store_true", help="Character-level only (no bigrams)")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        script_id = cfg["script_id"]
        vocab_size = cfg.get("vocab_size", 2000)
        max_token_length = cfg.get("max_token_length", 4)
        output_path = cfg.get("output", f"training_data/vocabs/{script_id}_2k.json")
        word_lists = cfg.get("word_lists", [cfg.get("word_list", "")])
    else:
        script_id = args.script_id
        vocab_size = args.vocab_size
        max_token_length = args.max_token_length
        output_path = args.output or f"training_data/vocabs/{script_id}_2k.json"
        word_lists = [args.word_list] if args.word_list else []

    if not script_id:
        parser.error("Must specify --script_id or --config")

    print(f"Building vocabulary for: {script_id}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Max token length: {max_token_length}")
    print(f"  Script chars: {len(SCRIPT_CHARSETS.get(script_id, []))}")

    if args.char_level or not word_lists or not any(Path(w).exists() for w in word_lists):
        print("  Mode: character-level (no bigram merges)")
        tok = LipiTokenizer.build_character_level(script_id)
    else:
        print(f"  Mode: bigrams from word lists: {word_lists}")
        tok = LipiTokenizer.build_for_script(
            script_id,
            word_lists=word_lists,
            max_bigrams=vocab_size,
        )

    # Save
    tok.save(output_path)
    print(f"\nSaved vocabulary to: {output_path}")
    print(f"  Total tokens: {tok.vocab_size}")
    print(f"  Blank ID: {tok.blank_id} ('{tok.vocab[0]}')")

    # Quick test
    test_words = ["Hello", "World", "12345", "Section"]
    print("\n  Roundtrip test:")
    for word in test_words:
        ids = tok.encode(word)
        decoded = tok.decode(ids)
        status = "OK" if decoded == word else f"FAIL ({decoded})"
        print(f"    '{word}' -> {ids[:5]}... -> '{decoded}' {status}")


if __name__ == "__main__":
    main()
