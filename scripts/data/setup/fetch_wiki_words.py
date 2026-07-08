#!/usr/bin/env python3
"""
Aggressively fetch words from Wikipedia + Wiktionary for all languages.

Fetches hundreds of articles per language to build comprehensive word lists.
Runs sequentially per language but fetches in large batches.

Usage:
    python scripts/fetch_wiki_words.py                    # all langs, 200 pages each
    python scripts/fetch_wiki_words.py --pages 500        # more pages
    python scripts/fetch_wiki_words.py --lang ta,th,lo    # specific languages
    python scripts/fetch_wiki_words.py --pages 500 --lang zh,ja,ko  # CJK heavy
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

WORD_LIST_DIR = Path(__file__).parent.parent.parent / "training_data" / "word_lists"

# Wikipedia language code -> word list file(s)
WIKI_LANGS = {
    # Latin script
    "en": ["english_common.txt"],
    "fr": ["french.txt"],
    "de": ["german.txt"],
    "es": ["spanish.txt"],
    "tr": ["turkish.txt"],
    "vi": ["vietnamese.txt"],
    "pt": ["portuguese.txt"],
    "it": ["italian.txt"],
    "pl": ["polish.txt"],
    "nl": ["dutch.txt"],
    "ro": ["romanian.txt"],
    "cs": ["czech.txt"],
    "hu": ["hungarian.txt"],
    "sv": ["swedish.txt"],
    "no": ["norwegian.txt"],
    "da": ["danish.txt"],
    "fi": ["finnish.txt"],
    "hr": ["croatian.txt"],
    "id": ["indonesian.txt"],
    "ms": ["malay.txt"],
    "sw": ["swahili.txt"],
    # Cyrillic
    "ru": ["cyrillic.txt"],
    "uk": ["ukrainian.txt"],
    "bg": ["cyrillic.txt"],      # Bulgarian
    "sr": ["cyrillic.txt"],      # Serbian
    "mk": ["cyrillic.txt"],      # Macedonian
    "be": ["cyrillic.txt"],      # Belarusian
    "kk": ["cyrillic.txt"],      # Kazakh
    # Greek
    "el": ["greek.txt"],
    # Arabic script
    "ar": ["arabic.txt"],
    "fa": ["persian.txt"],
    "ur": ["urdu.txt"],
    # Hebrew
    "he": ["hebrew.txt"],
    # CJK
    "zh": ["chinese.txt"],
    "ja": ["japanese.txt"],
    # Korean
    "ko": ["korean.txt"],
    # Indic
    "hi": ["devanagari.txt"],
    "mr": ["marathi.txt"],
    "ne": ["devanagari.txt"],    # Nepali (Devanagari)
    "sa": ["devanagari.txt"],    # Sanskrit (Devanagari)
    "bn": ["bengali.txt"],
    "as": ["bengali.txt"],       # Assamese (Bengali script)
    "gu": ["gujarati.txt"],
    "pa": ["gurmukhi.txt"],
    "ta": ["tamil.txt"],
    "te": ["telugu.txt"],
    "kn": ["kannada.txt"],
    "ml": ["malayalam.txt"],
    # SE Asian
    "th": ["thai.txt"],
    "lo": ["lao.txt"],
}

# Skip English — already have 560K words
SKIP_LARGE = {"en"}


def _api_get(url: str, params: dict, retries: int = 3) -> dict | None:
    """Make a Wikipedia API request with retries."""
    query = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(query, headers={
                "User-Agent": "LipiOCR/1.0 (OCR training data; contact: lipi@example.com)"
            })
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
            else:
                return None
    return None


def fetch_random_articles(wiki_lang: str, count: int = 200) -> list[str]:
    """Fetch text from random Wikipedia articles in batches of 20."""
    texts = []
    url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
    batch_size = 20  # API max for extracts
    fetched_ids = set()

    batches = (count + batch_size - 1) // batch_size
    for i in range(batches):
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": "0",
            "grnlimit": str(batch_size),
            "prop": "extracts",
            "explaintext": "true",
            "exlimit": str(batch_size),
        }
        data = _api_get(url, params)
        if not data:
            continue

        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid in fetched_ids:
                continue
            fetched_ids.add(pid)
            text = page.get("extract", "")
            if text and len(text) > 100:
                texts.append(text)

        # Be polite — small delay between batches
        if i < batches - 1:
            time.sleep(0.3)

    return texts


def fetch_category_words(wiki_lang: str, categories: list[str]) -> list[str]:
    """Fetch page titles from Wikipedia categories (often good word sources)."""
    url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
    titles = []

    for cat in categories:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": cat,
            "cmlimit": "500",
            "cmtype": "page",
        }
        data = _api_get(url, params)
        if data:
            members = data.get("query", {}).get("categorymembers", [])
            for m in members:
                titles.append(m.get("title", ""))

    return titles


def fetch_allpages_batch(wiki_lang: str, start_from: str = "", count: int = 500) -> list[str]:
    """Fetch page titles using allpages API — gets systematic coverage."""
    url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
    titles = []
    ap_from = start_from

    while len(titles) < count:
        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",
            "aplimit": "500",
            "apnamespace": "0",
            "apfilterredir": "nonredirects",
        }
        if ap_from:
            params["apfrom"] = ap_from

        data = _api_get(url, params)
        if not data:
            break

        pages = data.get("query", {}).get("allpages", [])
        if not pages:
            break

        for p in pages:
            titles.append(p.get("title", ""))

        # Check for continuation
        cont = data.get("continue", {})
        if "apcontinue" in cont:
            ap_from = cont["apcontinue"]
        else:
            break

        time.sleep(0.3)

    return titles[:count]


def extract_words(texts: list[str], min_len: int = 2, max_len: int = 15) -> set[str]:
    """Extract unique words from text."""
    words = set()
    # Split on whitespace, digits, and common punctuation
    splitter = re.compile(r'[\s\d\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+=<>\|~`…–—•·«»""''\u200b\u200c\u200d\ufeff]+')

    for text in texts:
        tokens = splitter.split(text)
        for token in tokens:
            token = token.strip('.,;:!?-()[]{}"\'/\\«»""''…–—•·')
            if min_len <= len(token) <= max_len:
                # Skip if starts with digit or is all-ASCII (for non-Latin scripts)
                if token and not token[0].isdigit():
                    words.add(token)
    return words


def load_existing_words(filepath: Path) -> set[str]:
    """Load existing words from file."""
    if not filepath.exists():
        return set()
    words = set()
    for line in filepath.read_text(encoding="utf-8", errors="ignore").splitlines():
        w = line.strip()
        if w:
            words.add(w)
    return words


def count_unique_chars(filepath: Path) -> int:
    """Count unique non-ASCII characters in a file."""
    if not filepath.exists():
        return 0
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    return len(set(c for c in text if not c.isspace() and ord(c) > 127))


def process_language(wiki_lang: str, target_files: list[str], pages: int) -> dict:
    """Process one language: fetch articles + titles, extract words, append to files."""
    result = {"lang": wiki_lang, "files": {}}

    # 1. Fetch random articles (main source of diverse text)
    print(f"  [{wiki_lang}] Fetching {pages} random articles...")
    texts = fetch_random_articles(wiki_lang, count=pages)
    print(f"  [{wiki_lang}] Got {len(texts)} articles")

    # 2. Fetch page titles (article titles are often single words/phrases)
    print(f"  [{wiki_lang}] Fetching page titles...")
    titles = fetch_allpages_batch(wiki_lang, count=2000)
    # Titles are also words
    texts.append("\n".join(titles))
    print(f"  [{wiki_lang}] Got {len(titles)} page titles")

    # 3. Extract all unique words
    all_words = extract_words(texts)
    print(f"  [{wiki_lang}] Extracted {len(all_words)} unique words")

    # 4. Append to target files
    for filename in target_files:
        filepath = WORD_LIST_DIR / filename
        existing = load_existing_words(filepath)
        added = all_words - existing

        if not added:
            result["files"][filename] = {"added": 0, "total": len(existing)}
            continue

        with open(filepath, "a", encoding="utf-8") as f:
            for word in sorted(added):
                f.write(word + "\n")

        total = len(existing) + len(added)
        result["files"][filename] = {"added": len(added), "total": total}
        print(f"  [{wiki_lang}] {filename}: +{len(added)} words (total: {total})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch words from Wikipedia (aggressive)")
    parser.add_argument("--pages", type=int, default=200,
                        help="Random articles per language (default: 200)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Comma-separated language codes (default: all)")
    args = parser.parse_args()

    if args.lang:
        langs = [l.strip() for l in args.lang.split(",")]
        wiki_langs = {k: v for k, v in WIKI_LANGS.items() if k in langs}
    else:
        wiki_langs = {k: v for k, v in WIKI_LANGS.items() if k not in SKIP_LARGE}

    total_start = time.time()
    print(f"Fetching from {len(wiki_langs)} Wikipedia languages, {args.pages} articles each\n")

    for wiki_lang, target_files in wiki_langs.items():
        print(f"\n{'='*60}")
        print(f"  {wiki_lang}.wikipedia.org -> {', '.join(target_files)}")
        print(f"{'='*60}")
        try:
            process_language(wiki_lang, target_files, args.pages)
        except Exception as e:
            print(f"  [{wiki_lang}] ERROR: {e}")
            continue

    elapsed = time.time() - total_start

    # Final summary
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    print(f"{'File':<28} {'Words':>8} {'Unique chars':>13}")
    print("-" * 52)
    for filepath in sorted(WORD_LIST_DIR.glob("*.txt")):
        lines = sum(1 for l in filepath.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip())
        chars = count_unique_chars(filepath)
        if lines > 100:
            print(f"  {filepath.name:<26} {lines:>7} {chars:>10}")


if __name__ == "__main__":
    main()
