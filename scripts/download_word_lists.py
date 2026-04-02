#!/usr/bin/env python3
"""
Download word lists for new scripts from Wiktionary frequency lists.

Uses Wiktionary's frequency lists which are compiled from large corpora.
Falls back to Wikipedia title extraction for scripts without Wiktionary lists.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

WORD_LIST_DIR = Path(__file__).parent.parent / "training_data" / "word_lists"


def is_in_script(text: str, ranges: list[tuple[int, int]]) -> bool:
    """Check if text contains only characters from the given Unicode ranges + shared."""
    for ch in text:
        cp = ord(ch)
        # Allow shared chars (digits, punctuation, spaces)
        if 0x0020 <= cp <= 0x007E:
            continue
        cat = unicodedata.category(ch)
        if cat.startswith('Z') or cat.startswith('P') or cat == 'Nd':
            continue
        # Check script ranges
        in_range = False
        for start, end in ranges:
            if start <= cp <= end:
                in_range = True
                break
        if not in_range:
            return False
    return True


def has_script_char(text: str, ranges: list[tuple[int, int]]) -> bool:
    """Check if text has at least one character from the script ranges."""
    for ch in text:
        cp = ord(ch)
        for start, end in ranges:
            if start <= cp <= end:
                return True
    return False


def fetch_url(url: str) -> str:
    """Fetch URL content."""
    req = Request(url, headers={"User-Agent": "LipiOCR/1.0 (word list download)"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def download_wiktionary_frequency(lang_code: str, script_ranges: list[tuple[int, int]],
                                   min_words: int = 5000) -> list[str]:
    """Try to download frequency list from Wiktionary."""
    url = f"https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/{lang_code}"
    try:
        html = fetch_url(url)
        # Extract words from wiki markup
        words = set()
        for match in re.findall(r'\[\[([^\]|]+)', html):
            w = match.strip()
            if 2 <= len(w) <= 15 and has_script_char(w, script_ranges) and is_in_script(w, script_ranges):
                words.add(w)
        return sorted(words)
    except Exception as e:
        print(f"  Wiktionary failed for {lang_code}: {e}")
        return []


def download_wikipedia_titles(lang_code: str, script_ranges: list[tuple[int, int]],
                               max_pages: int = 50000) -> list[str]:
    """Download Wikipedia article titles as word source."""
    from urllib.parse import quote
    url = (f"https://{lang_code}.wikipedia.org/w/api.php?"
           f"action=query&list=allpages&aplimit=500&format=json")

    words = set()
    apcontinue = ""
    pages_fetched = 0

    while pages_fetched < max_pages:
        fetch_url_str = url + (f"&apcontinue={quote(apcontinue)}" if apcontinue else "")
        try:
            data = json.loads(fetch_url(fetch_url_str))
        except Exception as e:
            print(f"  Wikipedia API error: {e}")
            break

        for page in data.get("query", {}).get("allpages", []):
            title = page["title"]
            # Split title into words
            for w in re.split(r'[\s\-_/()]+', title):
                w = w.strip()
                if 2 <= len(w) <= 15 and has_script_char(w, script_ranges) and is_in_script(w, script_ranges):
                    words.add(w)

        pages_fetched += 500
        if "continue" in data and "apcontinue" in data["continue"]:
            apcontinue = data["continue"]["apcontinue"]
        else:
            break

        if pages_fetched % 5000 == 0:
            print(f"    {pages_fetched} pages, {len(words)} words so far...")

    return sorted(words)


# Script definitions
SCRIPTS = {
    "odia": {
        "ranges": [(0x0B00, 0x0B7F)],
        "wiki_lang": "or",
        "wiktionary_lang": "Oriya",
    },
    "burmese": {
        "ranges": [(0x1000, 0x109F)],
        "wiki_lang": "my",
        "wiktionary_lang": "Burmese",
    },
    "khmer": {
        "ranges": [(0x1780, 0x17FF)],
        "wiki_lang": "km",
        "wiktionary_lang": "Khmer",
    },
    "sinhala": {
        "ranges": [(0x0D80, 0x0DFF)],
        "wiki_lang": "si",
        "wiktionary_lang": "Sinhalese",
    },
    "ethiopic": {
        "ranges": [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF)],
        "wiki_lang": "am",
        "wiktionary_lang": "Amharic",
    },
    "armenian": {
        "ranges": [(0x0530, 0x058F)],
        "wiki_lang": "hy",
        "wiktionary_lang": "Armenian",
    },
    "georgian": {
        "ranges": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
        "wiki_lang": "ka",
        "wiktionary_lang": "Georgian",
    },
    "tibetan": {
        "ranges": [(0x0F00, 0x0FFF)],
        "wiki_lang": "bo",
        "wiktionary_lang": "Tibetan",
    },
}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    scripts_to_process = SCRIPTS if target == "all" else {target: SCRIPTS[target]}

    for script_name, config in scripts_to_process.items():
        output_path = WORD_LIST_DIR / f"{script_name}.txt"
        existing = 0
        if output_path.exists():
            existing = len([l for l in output_path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()])

        print(f"\n{'='*50}")
        print(f"{script_name}: {existing} existing words")

        if existing >= 5000:
            print(f"  Already have enough words, skipping")
            continue

        ranges = config["ranges"]
        all_words = set()

        # Load existing words
        if output_path.exists():
            for line in output_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if w:
                    all_words.add(w)

        # Try Wikipedia titles
        print(f"  Downloading Wikipedia titles ({config['wiki_lang']})...")
        wiki_words = download_wikipedia_titles(config["wiki_lang"], ranges)
        print(f"  Got {len(wiki_words)} words from Wikipedia")
        all_words.update(wiki_words)

        # Filter: only words with script chars, no foreign chars, length 2-15
        clean = set()
        for w in all_words:
            if 2 <= len(w) <= 15 and has_script_char(w, ranges) and is_in_script(w, ranges):
                # No unassigned Unicode
                if not any(unicodedata.category(ch) == 'Cn' for ch in w):
                    clean.add(w)

        # Save
        sorted_words = sorted(clean)
        output_path.write_text("\n".join(sorted_words) + "\n", encoding="utf-8")
        print(f"  Saved {len(sorted_words)} clean words to {output_path.name}")


if __name__ == "__main__":
    main()
