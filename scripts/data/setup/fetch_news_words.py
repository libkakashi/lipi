#!/usr/bin/env python3
"""
Fetch words from news RSS feeds and headlines in multiple languages.
News text has natural frequency distribution — common chars appear often.

Sources:
  - Google News RSS (available in many languages)
  - NHK (Japanese)
  - Various public RSS feeds
"""

import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

WORD_LIST_DIR = Path(__file__).parent.parent.parent / "training_data" / "word_lists"

# Google News RSS feeds by language
GOOGLE_NEWS = {
    "zh-CN": "chinese.txt",      # Chinese
    "ja": "japanese.txt",         # Japanese
    "ko": "korean.txt",          # Korean
    "hi": "devanagari.txt",      # Hindi
    "bn": "bengali.txt",         # Bengali
    "ta": "tamil.txt",           # Tamil
    "te": "telugu.txt",          # Telugu
    "ml": "malayalam.txt",       # Malayalam
    "kn": "kannada.txt",         # Kannada
    "gu": "gujarati.txt",        # Gujarati
    "mr": "marathi.txt",         # Marathi
    "pa": "gurmukhi.txt",        # Punjabi
    "ar": "arabic.txt",          # Arabic
    "fa": "persian.txt",         # Persian
    "ur": "urdu.txt",            # Urdu
    "he": "hebrew.txt",          # Hebrew
    "el": "greek.txt",           # Greek
    "ru": "cyrillic.txt",        # Russian
    "uk": "ukrainian.txt",       # Ukrainian
    "th": "thai.txt",            # Thai
    "vi": "vietnamese.txt",      # Vietnamese
    "tr": "turkish.txt",         # Turkish
    "fr": "french.txt",
    "de": "german.txt",
    "es": "spanish.txt",
    "pt-BR": "portuguese.txt",
    "it": "italian.txt",
    "pl": "polish.txt",
    "nl": "dutch.txt",
    "cs": "czech.txt",
    "ro": "romanian.txt",
    "hu": "hungarian.txt",
    "sv": "swedish.txt",
    "no": "norwegian.txt",
    "da": "danish.txt",
    "fi": "finnish.txt",
    "id": "indonesian.txt",
}

# Additional RSS feeds (public news sources)
EXTRA_FEEDS = {
    "japanese.txt": [
        "https://www3.nhk.or.jp/rss/news/cat0.xml",  # NHK Japanese
    ],
    "chinese.txt": [
        "http://www.people.com.cn/rss/politics.xml",   # People's Daily Chinese
    ],
    "korean.txt": [
        "https://www.chosun.com/arc/outboundfeeds/rss/?outputType=xml",  # Chosun
    ],
    "arabic.txt": [
        "https://www.aljazeera.net/aljazeerarss/a7029c45-23d3-4571-a7f6-418f8ced24c3/73d0e1b4-532f-45ef-b135-bfdff8b8cab9",
    ],
    "devanagari.txt": [
        "https://www.bbc.com/hindi/index.xml",  # BBC Hindi
    ],
    "thai.txt": [
        "https://www.thairath.co.th/rss",  # Thai Rath
    ],
}


def fetch_rss(url: str) -> list[str]:
    """Fetch and parse RSS feed, extract text from titles and descriptions."""
    texts = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LipiOCR/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
        root = ET.fromstring(content)

        # Try Atom and RSS formats
        for item in root.iter():
            if item.tag.endswith(('title', 'description', 'summary', 'content')):
                text = item.text
                if text and len(text) > 5:
                    # Strip HTML tags
                    text = re.sub(r'<[^>]+>', '', text)
                    texts.append(text)
    except Exception:
        pass
    return texts


def fetch_google_news(lang: str, num_pages: int = 5) -> list[str]:
    """Fetch Google News headlines for a language."""
    texts = []
    # Google News search in target language
    topics = ["news", "politics", "sports", "technology", "business", "health", "entertainment"]

    for topic in topics:
        url = f"https://news.google.com/rss/search?q={topic}&hl={lang}&gl={lang.split('-')[0].upper()}&ceid={lang.split('-')[0].upper()}:{lang}"
        feed_texts = fetch_rss(url)
        texts.extend(feed_texts)
        time.sleep(0.5)

    return texts


def extract_words(texts: list[str], min_len: int = 2, max_len: int = 15) -> set[str]:
    """Extract unique words from text."""
    words = set()
    splitter = re.compile(
        r'[\s\d\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+=<>\|~`…–—•·«»""''\u200b-\u200f\ufeff\u00a0]+')
    for text in texts:
        for token in splitter.split(text):
            token = token.strip('.,;:!?-()[]{}"\'/\\«»""''…–—•·')
            if min_len <= len(token) <= max_len and token and not token[0].isdigit():
                words.add(token)
    return words


def main():
    print("=== Fetching news words ===\n")

    # Google News
    for lang, filename in GOOGLE_NEWS.items():
        filepath = WORD_LIST_DIR / filename
        existing = set()
        if filepath.exists():
            existing = set(l.strip() for l in filepath.read_text(
                encoding="utf-8", errors="ignore").splitlines() if l.strip())

        print(f"  Google News {lang} -> {filename}...", end=" ", flush=True)
        texts = fetch_google_news(lang)
        words = extract_words(texts)
        new = words - existing

        if new:
            with open(filepath, "a", encoding="utf-8") as f:
                for w in sorted(new):
                    f.write(w + "\n")
            print(f"+{len(new)} words ({len(existing)+len(new)} total)")
        else:
            print("no new words")
        time.sleep(1)

    # Extra RSS feeds
    print("\n=== Extra RSS feeds ===\n")
    for filename, feeds in EXTRA_FEEDS.items():
        filepath = WORD_LIST_DIR / filename
        existing = set()
        if filepath.exists():
            existing = set(l.strip() for l in filepath.read_text(
                encoding="utf-8", errors="ignore").splitlines() if l.strip())

        all_words = set()
        for feed_url in feeds:
            texts = fetch_rss(feed_url)
            all_words |= extract_words(texts)
            time.sleep(0.5)

        new = all_words - existing
        if new:
            with open(filepath, "a", encoding="utf-8") as f:
                for w in sorted(new):
                    f.write(w + "\n")
            print(f"  {filename}: +{len(new)} from RSS feeds")
        else:
            print(f"  {filename}: no new words from RSS")

    print("\nDone.")


if __name__ == "__main__":
    main()
