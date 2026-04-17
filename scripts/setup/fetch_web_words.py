#!/usr/bin/env python3
"""
Fetch words from web text sources with natural frequency distributions.

Sources:
  - Leipzig Corpora Collection (wortschatz.uni-leipzig.de)
    Pre-built frequency word lists for 250+ languages
  - OPUS parallel corpus word lists
"""

import gzip
import io
import re
import time
import urllib.request
from pathlib import Path

WORD_LIST_DIR = Path(__file__).parent.parent / "training_data" / "word_lists"

# Leipzig Corpora Collection: pre-built word lists
# Format: http://pcai056.informatik.uni-leipzig.de/downloads/corpora/{lang}_wikipedia_2021_10K-words.txt
# These are frequency-sorted word lists from Wikipedia + news + web
LEIPZIG_LANGS = {
    "zho_simpl": "chinese.txt",      # Chinese Simplified
    "jpn": "japanese.txt",            # Japanese
    "kor": "korean.txt",             # Korean
    "hin": "devanagari.txt",         # Hindi
    "ben": "bengali.txt",            # Bengali
    "tam": "tamil.txt",              # Tamil
    "tel": "telugu.txt",             # Telugu
    "mal": "malayalam.txt",          # Malayalam
    "kan": "kannada.txt",            # Kannada
    "guj": "gujarati.txt",           # Gujarati
    "mar": "marathi.txt",            # Marathi
    "pan": "gurmukhi.txt",           # Punjabi
    "ara": "arabic.txt",             # Arabic
    "fas": "persian.txt",            # Persian
    "urd": "urdu.txt",               # Urdu
    "heb": "hebrew.txt",             # Hebrew
    "ell": "greek.txt",              # Greek
    "rus": "cyrillic.txt",           # Russian
    "ukr": "ukrainian.txt",          # Ukrainian
    "tha": "thai.txt",               # Thai
    "lao": "lao.txt",                # Lao
    "vie": "vietnamese.txt",         # Vietnamese
    "tur": "turkish.txt",            # Turkish
    "fra": "french.txt",
    "deu": "german.txt",
    "spa": "spanish.txt",
    "por": "portuguese.txt",
    "ita": "italian.txt",
    "pol": "polish.txt",
    "nld": "dutch.txt",
    "ces": "czech.txt",
    "ron": "romanian.txt",
    "hun": "hungarian.txt",
    "swe": "swedish.txt",
    "nor": "norwegian.txt",
    "dan": "danish.txt",
    "fin": "finnish.txt",
    "hrv": "croatian.txt",
    "ind": "indonesian.txt",
    "msa": "malay.txt",
    "swa": "swahili.txt",
}


def fetch_url(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LipiOCR/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def fetch_leipzig_words(lang_code: str) -> list[str]:
    """Try to fetch word frequency lists from Leipzig Corpora."""
    words = []

    # Try different corpus types and years
    for corpus_type in ["wikipedia", "news", "web"]:
        for year in ["2021", "2020", "2019", "2018"]:
            for size in ["100K", "30K", "10K"]:
                url = f"https://downloads.wortschatz-leipzig.de/corpora/{lang_code}_{corpus_type}_{year}_{size}.tar.gz"
                # These are tar.gz files — too complex to parse inline
                # Instead try the simpler txt format
                pass

    # Fallback: try fetching from their API
    url = f"https://corpora.uni-leipzig.de/en/res?corpusId={lang_code}_wikipedia_2021&word=&pos=&filter=words"
    text = fetch_url(url)
    if text:
        # Extract words from HTML response
        for match in re.findall(r'>([^<]{2,15})</td>', text):
            match = match.strip()
            if match and not match[0].isdigit():
                words.append(match)

    return words


def fetch_tatoeba_sentences(lang: str, limit: int = 500) -> list[str]:
    """Fetch example sentences from Tatoeba (CC-licensed sentence database)."""
    # Tatoeba has an API for fetching sentences by language
    url = f"https://tatoeba.org/en/api_v0/search?from={lang}&to=&query=&page=1"
    texts = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LipiOCR/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for result in data.get("results", []):
                text = result.get("text", "")
                if text:
                    texts.append(text)
    except Exception:
        pass
    return texts


# Tatoeba language codes
import json
TATOEBA_LANGS = {
    "cmn": "chinese.txt",      # Chinese
    "jpn": "japanese.txt",
    "kor": "korean.txt",
    "hin": "devanagari.txt",
    "ben": "bengali.txt",
    "tam": "tamil.txt",
    "tel": "telugu.txt",
    "ara": "arabic.txt",
    "fas": "persian.txt",
    "heb": "hebrew.txt",
    "ell": "greek.txt",
    "rus": "cyrillic.txt",
    "ukr": "ukrainian.txt",
    "tha": "thai.txt",
    "vie": "vietnamese.txt",
    "tur": "turkish.txt",
    "fra": "french.txt",
    "deu": "german.txt",
    "spa": "spanish.txt",
    "por": "portuguese.txt",
    "ita": "italian.txt",
    "pol": "polish.txt",
}


def extract_words(texts: list[str]) -> set[str]:
    words = set()
    splitter = re.compile(
        r'[\s\d\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+=<>\|~`…–—•·«»""''\u200b-\u200f\ufeff\u00a0]+')
    for text in texts:
        for token in splitter.split(text):
            token = token.strip('.,;:!?-()[]{}"\'/\\«»""''…–—•·')
            if 2 <= len(token) <= 15 and token and not token[0].isdigit():
                words.add(token)
    return words


def main():
    print("=== Fetching from Tatoeba ===\n")
    for lang, filename in TATOEBA_LANGS.items():
        filepath = WORD_LIST_DIR / filename
        existing = set()
        if filepath.exists():
            existing = set(l.strip() for l in filepath.read_text(
                encoding="utf-8", errors="ignore").splitlines() if l.strip())

        texts = fetch_tatoeba_sentences(lang)
        if texts:
            words = extract_words(texts)
            new = words - existing
            if new:
                with open(filepath, "a", encoding="utf-8") as f:
                    for w in sorted(new):
                        f.write(w + "\n")
                print(f"  {lang:>5} -> {filename:25s} +{len(new)} words from {len(texts)} sentences")
        time.sleep(0.5)

    print("\nDone.")


if __name__ == "__main__":
    main()
