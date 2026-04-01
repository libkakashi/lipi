#!/usr/bin/env python3
"""
Fetch words from Project Gutenberg books in various languages.
Gutenberg has free ebooks in 60+ languages — great source of diverse vocabulary.
"""

import json
import re
import time
import urllib.request
from pathlib import Path

WORD_LIST_DIR = Path(__file__).parent.parent / "training_data" / "word_lists"

# Gutenberg language codes -> our word list files
GUTENBERG_LANGS = {
    "fr": "french.txt",
    "de": "german.txt",
    "es": "spanish.txt",
    "pt": "portuguese.txt",
    "it": "italian.txt",
    "nl": "dutch.txt",
    "fi": "finnish.txt",
    "hu": "hungarian.txt",
    "pl": "polish.txt",
    "cs": "czech.txt",
    "da": "danish.txt",
    "sv": "swedish.txt",
    "no": "norwegian.txt",
    "el": "greek.txt",
    "ru": "cyrillic.txt",
    "zh": "cjk.txt",
    "ja": "japanese.txt",
    "tl": "tagalog.txt",
}


def fetch_gutenberg_catalog(lang: str, limit: int = 50) -> list[str]:
    """Get book IDs from Gutenberg for a language."""
    url = f"https://gutendex.com/books/?languages={lang}&mime_type=text%2Fplain&page=1"
    book_urls = []
    pages = 0
    while url and pages < 5 and len(book_urls) < limit:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LipiOCR/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for book in data.get("results", []):
                    formats = book.get("formats", {})
                    for fmt_key, fmt_url in formats.items():
                        if "text/plain" in fmt_key and fmt_url.endswith(".txt"):
                            book_urls.append(fmt_url)
                            break
                url = data.get("next")
                pages += 1
        except Exception:
            break
        time.sleep(0.5)
    return book_urls[:limit]


def fetch_book_text(url: str) -> str:
    """Download a Gutenberg book's plain text."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LipiOCR/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_words(text: str, min_len: int = 2, max_len: int = 15) -> set[str]:
    """Extract unique words from text."""
    words = set()
    splitter = re.compile(r'[\s\d\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+=<>\|~`…–—•·«»""''\u200b-\u200f\ufeff\u00a0]+')
    for token in splitter.split(text):
        token = token.strip('.,;:!?-()[]{}"\'/\\«»""''…–—•·_')
        if min_len <= len(token) <= max_len and token and not token[0].isdigit():
            words.add(token)
    return words


def main():
    for lang, filename in GUTENBERG_LANGS.items():
        filepath = WORD_LIST_DIR / filename
        existing = set()
        if filepath.exists():
            existing = set(filepath.read_text(encoding="utf-8", errors="ignore").splitlines())

        print(f"\n{'='*50}")
        print(f"Gutenberg: {lang} -> {filename}")

        book_urls = fetch_gutenberg_catalog(lang, limit=30)
        print(f"  Found {len(book_urls)} books")

        all_words = set()
        for i, url in enumerate(book_urls[:20]):
            text = fetch_book_text(url)
            if text:
                words = extract_words(text)
                all_words |= words
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i+1} books, {len(all_words)} unique words")
            time.sleep(0.3)

        new_words = all_words - existing
        if new_words:
            with open(filepath, "a", encoding="utf-8") as f:
                for w in sorted(new_words):
                    f.write(w + "\n")
            print(f"  +{len(new_words)} new words (total: {len(existing) + len(new_words)})")
        else:
            print(f"  No new words")


if __name__ == "__main__":
    main()
