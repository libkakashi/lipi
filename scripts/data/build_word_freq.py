"""
Build word frequency and grapheme cluster frequency tables from Wikipedia.

For each language, streams Wikipedia articles, extracts words in the target
script, counts word frequencies and grapheme cluster (fusion) frequencies.

Outputs per script:
  training_data/corpora/{script}_word_freq.tsv    — word\tcount
  training_data/corpora/{script}_fusions.tsv       — cluster\tcount\tlen

Usage:
  python -m scripts.build_word_freq                    # all fusion scripts
  python -m scripts.build_word_freq --scripts thai lao  # specific scripts
  python -m scripts.build_word_freq --max-articles 5000 # fewer articles
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import regex

# Script → Wikipedia language codes + Unicode ranges for filtering
SCRIPT_LANGUAGES: dict[str, dict] = {
    "arabic": {
        "langs": ["ar", "fa", "ur"],
        "ranges": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x0870, 0x089F),
                   (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    },
    "devanagari": {
        "langs": ["hi", "mr", "ne", "sa"],
        "ranges": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    },
    "gurmukhi": {
        "langs": ["pa"],
        "ranges": [(0x0A00, 0x0A7F)],
    },
    "gujarati": {
        "langs": ["gu"],
        "ranges": [(0x0A80, 0x0AFF)],
    },
    "bengali": {
        "langs": ["bn", "as"],
        "ranges": [(0x0980, 0x09FF)],
    },
    "odia": {
        "langs": ["or"],
        "ranges": [(0x0B00, 0x0B7F)],
    },
    "kannada": {
        "langs": ["kn"],
        "ranges": [(0x0C80, 0x0CFF)],
    },
    "telugu": {
        "langs": ["te"],
        "ranges": [(0x0C00, 0x0C7F)],
    },
    "malayalam": {
        "langs": ["ml"],
        "ranges": [(0x0D00, 0x0D7F)],
    },
    "tamil": {
        "langs": ["ta"],
        "ranges": [(0x0B80, 0x0BFF)],
    },
    "sinhala": {
        "langs": ["si"],
        "ranges": [(0x0D80, 0x0DFF)],
    },
    "thai": {
        "langs": ["th"],
        "ranges": [(0x0E00, 0x0E7F)],
    },
    "lao": {
        "langs": ["lo"],
        "ranges": [(0x0E80, 0x0EFF)],
    },
    "burmese": {
        "langs": ["my"],
        "ranges": [(0x1000, 0x109F)],
    },
    "khmer": {
        "langs": ["km"],
        "ranges": [(0x1780, 0x17FF)],
    },
    "tibetan": {
        "langs": ["bo"],
        "ranges": [(0x0F00, 0x0FFF)],
    },
}

# Wikipedia dataset configs available on HuggingFace
# Format: wikimedia/wikipedia, config = "20231101.{lang}"
WIKI_DATE = "20231101"

OUTPUT_DIR = Path("training_data/corpora")
WORD_LIST_DIR = Path("training_data/word_lists")


def _in_script(char: str, ranges: list[tuple[int, int]]) -> bool:
    cp = ord(char)
    return any(s <= cp <= e for s, e in ranges)


def _is_script_word(word: str, ranges: list[tuple[int, int]]) -> bool:
    """Check if word contains at least one script char and no foreign script chars."""
    has_script = False
    for c in word:
        if _in_script(c, ranges):
            has_script = True
        elif c.isalpha():
            # Foreign script letter — reject
            return False
    return has_script


def _extract_words_from_text(text: str, ranges: list[tuple[int, int]]) -> list[str]:
    """Split text into words, keep only those in target script."""
    # Split on whitespace and common punctuation
    raw_words = re.split(r'[\s\u200b\u200c\u200d\u00a0]+', text)
    words = []
    for w in raw_words:
        # Strip leading/trailing punctuation
        w = w.strip('.,;:!?()[]{}"""\'\'«»—–-/\\|@#$%^&*+=<>~`')
        if len(w) < 1 or len(w) > 50:
            continue
        if _is_script_word(w, ranges):
            words.append(w)
    return words


def build_for_script(
    script: str,
    max_articles: int = 20000,
    include_word_lists: bool = True,
) -> tuple[Counter, Counter]:
    """Build word freq and fusion freq for a script from Wikipedia + word lists.

    Returns (word_freq, cluster_freq).
    """
    from datasets import load_dataset

    config = SCRIPT_LANGUAGES[script]
    ranges = config["ranges"]
    word_freq: Counter = Counter()
    cluster_freq: Counter = Counter()

    # Stream Wikipedia articles for each language
    for lang in config["langs"]:
        wiki_config = f"{WIKI_DATE}.{lang}"
        print(f"  Streaming {lang} Wikipedia ({wiki_config})...", flush=True)
        try:
            ds = load_dataset(
                "wikimedia/wikipedia", wiki_config,
                split="train", streaming=True,
            )
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        n_articles = 0
        n_words = 0
        for article in ds:
            text = article.get("text", "")
            words = _extract_words_from_text(text, ranges)
            for w in words:
                word_freq[w] += 1
                clusters = regex.findall(r'\X', w)
                for c in clusters:
                    cluster_freq[c] += 1
            n_words += len(words)
            n_articles += 1
            if n_articles % 5000 == 0:
                print(f"    {n_articles:,} articles, {n_words:,} words, "
                      f"{len(word_freq):,} unique words, "
                      f"{len(cluster_freq):,} unique clusters", flush=True)
            if n_articles >= max_articles:
                break

        print(f"    Done: {n_articles:,} articles, {n_words:,} words")

    # Also include existing word lists for extra coverage
    if include_word_lists:
        wl_files = {
            "arabic": ["arabic.txt", "persian.txt", "urdu.txt"],
            "devanagari": ["devanagari.txt", "marathi.txt"],
            "gurmukhi": ["gurmukhi.txt"],
            "gujarati": ["gujarati.txt"],
            "bengali": ["bengali.txt"],
            "odia": ["odia.txt"],
            "kannada": ["kannada.txt"],
            "telugu": ["telugu.txt"],
            "malayalam": ["malayalam.txt"],
            "tamil": ["tamil.txt"],
            "sinhala": ["sinhala.txt"],
            "thai": ["thai.txt"],
            "lao": ["lao.txt"],
            "burmese": ["burmese.txt"],
            "khmer": ["khmer.txt"],
            "tibetan": ["tibetan.txt"],
        }
        for fname in wl_files.get(script, []):
            path = WORD_LIST_DIR / fname
            if not path.exists():
                continue
            print(f"  Adding word list: {fname}")
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if w and _is_script_word(w, ranges):
                    word_freq[w] += 1
                    clusters = regex.findall(r'\X', w)
                    for c in clusters:
                        cluster_freq[c] += 1

    return word_freq, cluster_freq


def save_results(
    script: str,
    word_freq: Counter,
    cluster_freq: Counter,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Word frequencies
    word_path = OUTPUT_DIR / f"{script}_word_freq.tsv"
    with open(word_path, "w", encoding="utf-8") as f:
        for word, count in word_freq.most_common():
            f.write(f"{word}\t{count}\n")

    # Cluster/fusion frequencies
    fusion_path = OUTPUT_DIR / f"{script}_fusions.tsv"
    single_count = 0
    multi_count = 0
    with open(fusion_path, "w", encoding="utf-8") as f:
        for cluster, count in cluster_freq.most_common():
            cp_len = len(cluster)
            f.write(f"{cluster}\t{count}\t{cp_len}\n")
            if cp_len == 1:
                single_count += 1
            else:
                multi_count += 1

    print(f"  Saved: {word_path.name} ({len(word_freq):,} words)")
    print(f"  Saved: {fusion_path.name} ({single_count} single + {multi_count} fusions)")

    # Update word list: merge Wikipedia words into existing word list
    _update_word_list(script, word_freq)


# Map scripts to their primary word list file
_SCRIPT_WORD_LIST = {
    "arabic": "arabic.txt",
    "devanagari": "devanagari.txt",
    "gurmukhi": "gurmukhi.txt",
    "gujarati": "gujarati.txt",
    "bengali": "bengali.txt",
    "odia": "odia.txt",
    "kannada": "kannada.txt",
    "telugu": "telugu.txt",
    "malayalam": "malayalam.txt",
    "tamil": "tamil.txt",
    "sinhala": "sinhala.txt",
    "thai": "thai.txt",
    "lao": "lao.txt",
    "burmese": "burmese.txt",
    "khmer": "khmer.txt",
    "tibetan": "tibetan.txt",
}


def _update_word_list(script: str, word_freq: Counter):
    """Merge Wikipedia words into the script's word list, sorted by frequency."""
    fname = _SCRIPT_WORD_LIST.get(script)
    if not fname:
        return

    path = WORD_LIST_DIR / fname
    existing = set()
    if path.exists():
        existing = {
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        }

    # All words from Wikipedia + existing, sorted by frequency (desc)
    all_words = set(word_freq.keys()) | existing
    sorted_words = sorted(all_words, key=lambda w: word_freq.get(w, 0), reverse=True)

    path.write_text("\n".join(sorted_words) + "\n", encoding="utf-8")
    new_count = len(sorted_words) - len(existing)
    print(f"  Updated: {fname} ({len(existing):,} → {len(sorted_words):,}, +{new_count:,} new words)")


def main():
    parser = argparse.ArgumentParser(description="Build word/fusion frequencies from Wikipedia")
    parser.add_argument("--scripts", nargs="+", default=list(SCRIPT_LANGUAGES.keys()),
                        help="Scripts to process (default: all)")
    parser.add_argument("--max-articles", type=int, default=20000,
                        help="Max Wikipedia articles per language (default: 20000)")
    parser.add_argument("--no-word-lists", action="store_true",
                        help="Skip existing word lists, Wikipedia only")
    args = parser.parse_args()

    for script in args.scripts:
        if script not in SCRIPT_LANGUAGES:
            print(f"Unknown script: {script}, skipping")
            continue

        print(f"\n=== {script} ===")
        word_freq, cluster_freq = build_for_script(
            script,
            max_articles=args.max_articles,
            include_word_lists=not args.no_word_lists,
        )

        if not word_freq:
            print(f"  No data collected for {script}, skipping")
            continue

        save_results(script, word_freq, cluster_freq)

        # Quick summary
        multi = {c: n for c, n in cluster_freq.items() if len(c) > 1}
        single = {c: n for c, n in cluster_freq.items() if len(c) == 1}
        total_occ = sum(cluster_freq.values())
        multi_occ = sum(multi.values())
        print(f"  Summary: {len(word_freq):,} unique words, "
              f"{len(single)} base chars, {len(multi):,} fusions, "
              f"fusion occurrence rate: {100*multi_occ/total_occ:.1f}%")


if __name__ == "__main__":
    main()
