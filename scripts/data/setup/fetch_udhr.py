#!/usr/bin/env python3
"""
Fetch words from the Universal Declaration of Human Rights (UDHR).
Available in 500+ languages at unicode.org/udhr.
"""

import re
import time
import urllib.request
from pathlib import Path

WORD_LIST_DIR = Path(__file__).parent.parent.parent / "training_data" / "word_lists"

# UDHR translation codes -> our word list files
UDHR_LANGS = {
    # Latin
    "eng": "english_common.txt",
    "fra": "french.txt",
    "deu": "german.txt",
    "spa": "spanish.txt",
    "por": "portuguese.txt",
    "ita": "italian.txt",
    "nld": "dutch.txt",
    "pol": "polish.txt",
    "ron": "romanian.txt",
    "ces": "czech.txt",
    "hun": "hungarian.txt",
    "fin": "finnish.txt",
    "swe": "swedish.txt",
    "dan": "danish.txt",
    "nor": "norwegian.txt",
    "hrv": "croatian.txt",
    "slk": "czech.txt",  # Slovak -> Czech file (similar diacritics)
    "slv": "croatian.txt",  # Slovenian
    "tur": "turkish.txt",
    "vie": "vietnamese.txt",
    "ind": "indonesian.txt",
    "msa": "malay.txt",
    "swa": "swahili.txt",
    "cat": "spanish.txt",  # Catalan diacritics similar
    "gle": "irish.txt",  # Irish
    "cym": "welsh.txt",  # Welsh
    "isl": "icelandic.txt",  # Icelandic
    "lav": "latvian.txt",
    "lit": "lithuanian.txt",
    "est": "estonian.txt",
    "afr": "dutch.txt",  # Afrikaans -> Dutch
    "tgl": "indonesian.txt",  # Tagalog
    # Cyrillic
    "rus": "cyrillic.txt",
    "ukr": "ukrainian.txt",
    "bul": "cyrillic.txt",
    "srp": "cyrillic.txt",
    "mkd": "cyrillic.txt",
    "bel": "cyrillic.txt",
    "kaz": "cyrillic.txt",
    "mon": "cyrillic.txt",
    "kir": "cyrillic.txt",
    # Greek
    "ell": "greek.txt",
    # Arabic script
    "arb": "arabic.txt",
    "pes": "persian.txt",
    "urd": "urdu.txt",
    "pus": "arabic.txt",  # Pashto
    # Hebrew
    "heb": "hebrew.txt",
    # Devanagari
    "hin": "devanagari.txt",
    "mar": "marathi.txt",
    "nep": "devanagari.txt",
    "san": "devanagari.txt",
    # Bengali
    "ben": "bengali.txt",
    # Gurmukhi
    "pan": "gurmukhi.txt",
    # Gujarati
    "guj": "gujarati.txt",
    # Tamil
    "tam": "tamil.txt",
    # Telugu
    "tel": "telugu.txt",
    # Kannada
    "kan": "kannada.txt",
    # Malayalam
    "mal": "malayalam.txt",
    # Thai
    "tha": "thai.txt",
    # Lao
    "lao": "lao.txt",
    # CJK
    "cmn_hans": "cjk.txt",
    "cmn_hant": "cjk.txt",
    "jpn": "japanese.txt",
    # Korean
    "kor": "korean.txt",
}

BASE_URL = "https://unicode.org/udhr/d"


def fetch_udhr(lang_code: str) -> str:
    """Fetch UDHR text for a language."""
    url = f"{BASE_URL}/udhr_{lang_code}.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LipiOCR/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_words(text: str) -> set[str]:
    words = set()
    splitter = re.compile(r'[\s\d\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+=<>\|~`…–—•·«»""''\u200b-\u200f\ufeff\u00a0]+')
    for token in splitter.split(text):
        token = token.strip('.,;:!?-()[]{}"\'/\\«»""''…–—•·_')
        if 2 <= len(token) <= 15 and token and not token[0].isdigit():
            words.add(token)
    return words


def main():
    for lang_code, filename in UDHR_LANGS.items():
        filepath = WORD_LIST_DIR / filename
        existing = set()
        if filepath.exists():
            existing = set(l.strip() for l in filepath.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip())

        text = fetch_udhr(lang_code)
        if not text:
            continue

        words = extract_words(text)
        new_words = words - existing

        if new_words:
            # Create file if doesn't exist
            with open(filepath, "a", encoding="utf-8") as f:
                for w in sorted(new_words):
                    f.write(w + "\n")
            print(f"  {lang_code:>10} -> {filename:25s} +{len(new_words):>5} words")
        time.sleep(0.2)

    print("\nDone.")


if __name__ == "__main__":
    main()
