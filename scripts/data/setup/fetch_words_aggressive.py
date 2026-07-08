#!/usr/bin/env python3
"""
Aggressively build word lists from multiple sources:
  1. Wikipedia random articles (bulk)
  2. Wikipedia AllPages (systematic title crawl)
  3. Wiktionary category dumps (actual dictionary words)

Designed to run for a long time and get as many words as possible.

Usage:
    python scripts/fetch_words_aggressive.py
    python scripts/fetch_words_aggressive.py --lang kn,ml,lo,th --wiki-pages 1000
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

WORD_LIST_DIR = Path(__file__).parent.parent.parent / "training_data" / "word_lists"

# lang -> (target files, wiktionary category patterns)
LANG_CONFIG = {
    # Non-Latin scripts (priority — these need the most help)
    "ar": {"files": ["arabic.txt"], "wikt_cats": ["Arabic_lemmas", "Arabic_nouns", "Arabic_verbs", "Arabic_adjectives"]},
    "fa": {"files": ["persian.txt"], "wikt_cats": ["Persian_lemmas", "Persian_nouns", "Persian_verbs"]},
    "ur": {"files": ["urdu.txt"], "wikt_cats": ["Urdu_lemmas", "Urdu_nouns", "Urdu_verbs"]},
    "he": {"files": ["hebrew.txt"], "wikt_cats": ["Hebrew_lemmas", "Hebrew_nouns", "Hebrew_verbs"]},
    "el": {"files": ["greek.txt"], "wikt_cats": ["Greek_lemmas", "Greek_nouns", "Greek_verbs"]},
    "zh": {"files": ["chinese.txt"], "wikt_cats": ["Mandarin_lemmas", "Chinese_lemmas"]},
    "ja": {"files": ["japanese.txt"], "wikt_cats": ["Japanese_lemmas", "Japanese_nouns", "Japanese_verbs"]},
    "ko": {"files": ["korean.txt"], "wikt_cats": ["Korean_lemmas", "Korean_nouns", "Korean_verbs"]},
    "hi": {"files": ["devanagari.txt"], "wikt_cats": ["Hindi_lemmas", "Hindi_nouns", "Hindi_verbs"]},
    "mr": {"files": ["marathi.txt"], "wikt_cats": ["Marathi_lemmas", "Marathi_nouns"]},
    "ne": {"files": ["devanagari.txt"], "wikt_cats": ["Nepali_lemmas"]},
    "sa": {"files": ["devanagari.txt"], "wikt_cats": ["Sanskrit_lemmas"]},
    "bn": {"files": ["bengali.txt"], "wikt_cats": ["Bengali_lemmas", "Bengali_nouns", "Bengali_verbs"]},
    "gu": {"files": ["gujarati.txt"], "wikt_cats": ["Gujarati_lemmas", "Gujarati_nouns"]},
    "pa": {"files": ["gurmukhi.txt"], "wikt_cats": ["Punjabi_lemmas", "Punjabi_nouns"]},
    "ta": {"files": ["tamil.txt"], "wikt_cats": ["Tamil_lemmas", "Tamil_nouns", "Tamil_verbs"]},
    "te": {"files": ["telugu.txt"], "wikt_cats": ["Telugu_lemmas", "Telugu_nouns"]},
    "kn": {"files": ["kannada.txt"], "wikt_cats": ["Kannada_lemmas", "Kannada_nouns"]},
    "ml": {"files": ["malayalam.txt"], "wikt_cats": ["Malayalam_lemmas", "Malayalam_nouns"]},
    "th": {"files": ["thai.txt"], "wikt_cats": ["Thai_lemmas", "Thai_nouns", "Thai_verbs"]},
    "lo": {"files": ["lao.txt"], "wikt_cats": ["Lao_lemmas", "Lao_nouns"]},
    # Cyrillic
    "ru": {"files": ["cyrillic.txt"], "wikt_cats": ["Russian_lemmas", "Russian_nouns", "Russian_verbs"]},
    "uk": {"files": ["ukrainian.txt"], "wikt_cats": ["Ukrainian_lemmas", "Ukrainian_nouns"]},
    "bg": {"files": ["cyrillic.txt"], "wikt_cats": ["Bulgarian_lemmas"]},
    "sr": {"files": ["cyrillic.txt"], "wikt_cats": ["Serbian_lemmas"]},
    "mk": {"files": ["cyrillic.txt"], "wikt_cats": ["Macedonian_lemmas"]},
    "be": {"files": ["cyrillic.txt"], "wikt_cats": ["Belarusian_lemmas"]},
    # Latin (supplemental)
    "fr": {"files": ["french.txt"], "wikt_cats": ["French_lemmas"]},
    "de": {"files": ["german.txt"], "wikt_cats": ["German_lemmas"]},
    "es": {"files": ["spanish.txt"], "wikt_cats": ["Spanish_lemmas"]},
    "tr": {"files": ["turkish.txt"], "wikt_cats": ["Turkish_lemmas"]},
    "vi": {"files": ["vietnamese.txt"], "wikt_cats": ["Vietnamese_lemmas"]},
    "pt": {"files": ["portuguese.txt"], "wikt_cats": ["Portuguese_lemmas"]},
    "it": {"files": ["italian.txt"], "wikt_cats": ["Italian_lemmas"]},
    "pl": {"files": ["polish.txt"], "wikt_cats": ["Polish_lemmas"]},
    "nl": {"files": ["dutch.txt"], "wikt_cats": ["Dutch_lemmas"]},
    "ro": {"files": ["romanian.txt"], "wikt_cats": ["Romanian_lemmas"]},
    "cs": {"files": ["czech.txt"], "wikt_cats": ["Czech_lemmas"]},
    "hu": {"files": ["hungarian.txt"], "wikt_cats": ["Hungarian_lemmas"]},
    "sv": {"files": ["swedish.txt"], "wikt_cats": ["Swedish_lemmas"]},
    "da": {"files": ["danish.txt"], "wikt_cats": ["Danish_lemmas"]},
    "fi": {"files": ["finnish.txt"], "wikt_cats": ["Finnish_lemmas"]},
    "hr": {"files": ["croatian.txt"], "wikt_cats": ["Croatian_lemmas"]},
    "no": {"files": ["norwegian.txt"], "wikt_cats": ["Norwegian_Bokmål_lemmas"]},
    "id": {"files": ["indonesian.txt"], "wikt_cats": ["Indonesian_lemmas"]},
    "ms": {"files": ["malay.txt"], "wikt_cats": ["Malay_lemmas"]},
    "sw": {"files": ["swahili.txt"], "wikt_cats": ["Swahili_lemmas"]},
    # Additional Latin-script languages
    "af": {"files": ["afrikaans.txt"], "wikt_cats": ["Afrikaans_lemmas"]},
    "sq": {"files": ["albanian.txt"], "wikt_cats": ["Albanian_lemmas"]},
    "eu": {"files": ["basque.txt"], "wikt_cats": ["Basque_lemmas"]},
    "ca": {"files": ["catalan.txt"], "wikt_cats": ["Catalan_lemmas"]},
    "et": {"files": ["estonian.txt"], "wikt_cats": ["Estonian_lemmas"]},
    "gl": {"files": ["galician.txt"], "wikt_cats": ["Galician_lemmas"]},
    "is": {"files": ["icelandic.txt"], "wikt_cats": ["Icelandic_lemmas"]},
    "lv": {"files": ["latvian.txt"], "wikt_cats": ["Latvian_lemmas"]},
    "lt": {"files": ["lithuanian.txt"], "wikt_cats": ["Lithuanian_lemmas"]},
    "mt": {"files": ["maltese.txt"], "wikt_cats": ["Maltese_lemmas"]},
    "sk": {"files": ["slovak.txt"], "wikt_cats": ["Slovak_lemmas"]},
    "sl": {"files": ["slovenian.txt"], "wikt_cats": ["Slovenian_lemmas"]},
    "cy": {"files": ["welsh.txt"], "wikt_cats": ["Welsh_lemmas"]},
    "ga": {"files": ["irish.txt"], "wikt_cats": ["Irish_lemmas"]},
    "tl": {"files": ["tagalog.txt"], "wikt_cats": ["Tagalog_lemmas"]},
    # Additional Cyrillic
    "mn": {"files": ["cyrillic.txt"], "wikt_cats": ["Mongolian_lemmas"]},
    "ky": {"files": ["cyrillic.txt"], "wikt_cats": ["Kyrgyz_lemmas"]},
    "tg": {"files": ["cyrillic.txt"], "wikt_cats": ["Tajik_lemmas"]},
    # Additional Arabic script
    "ps": {"files": ["arabic.txt"], "wikt_cats": ["Pashto_lemmas"]},
    "ku": {"files": ["arabic.txt"], "wikt_cats": ["Kurdish_lemmas"]},
}


def _api_get(url: str, params: dict, retries: int = 3) -> dict | None:
    query = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(query, headers={
                "User-Agent": "LipiOCR/1.0 (OCR training data collection)"
            })
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
    return None


def fetch_wiki_articles(lang: str, count: int) -> list[str]:
    """Fetch random Wikipedia articles."""
    texts = []
    url = f"https://{lang}.wikipedia.org/w/api.php"
    batch_size = 20

    for i in range(0, count, batch_size):
        params = {
            "action": "query", "format": "json",
            "generator": "random", "grnnamespace": "0",
            "grnlimit": str(min(batch_size, count - len(texts))),
            "prop": "extracts", "explaintext": "true",
            "exlimit": str(batch_size),
        }
        data = _api_get(url, params)
        if data:
            for page in data.get("query", {}).get("pages", {}).values():
                text = page.get("extract", "")
                if text and len(text) > 50:
                    texts.append(text)
        time.sleep(0.2)

        if (i // batch_size) % 10 == 9:
            print(f"    ... {len(texts)} articles fetched")

    return texts


def fetch_wiki_allpages(lang: str, count: int = 5000) -> list[str]:
    """Fetch page titles systematically via allpages API."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    titles = []
    ap_from = ""

    while len(titles) < count:
        params = {
            "action": "query", "format": "json",
            "list": "allpages", "aplimit": "500",
            "apnamespace": "0", "apfilterredir": "nonredirects",
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

        cont = data.get("continue", {})
        if "apcontinue" in cont:
            ap_from = cont["apcontinue"]
        else:
            break
        time.sleep(0.2)

    return titles[:count]


def fetch_wiktionary_words(lang: str, categories: list[str], max_per_cat: int = 5000) -> set[str]:
    """Fetch word entries from English Wiktionary categories.

    Wiktionary organizes words by language categories like 'Category:French_nouns'.
    These are actual dictionary headwords — perfect for word lists.
    """
    words = set()
    url = "https://en.wiktionary.org/w/api.php"

    for cat in categories:
        cm_continue = ""
        fetched = 0

        while fetched < max_per_cat:
            params = {
                "action": "query", "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{cat}",
                "cmlimit": "500",
                "cmtype": "page",
                "cmnamespace": "0",
            }
            if cm_continue:
                params["cmcontinue"] = cm_continue

            data = _api_get(url, params)
            if not data:
                break

            members = data.get("query", {}).get("categorymembers", [])
            if not members:
                break

            for m in members:
                title = m.get("title", "")
                if title and 2 <= len(title) <= 15:
                    words.add(title)
                fetched += 1

            cont = data.get("continue", {})
            if "cmcontinue" in cont:
                cm_continue = cont["cmcontinue"]
            else:
                break
            time.sleep(0.2)

        if fetched > 0:
            print(f"    Wiktionary {cat}: {fetched} entries")

    return words


def extract_words(texts: list[str], min_len: int = 2, max_len: int = 15) -> set[str]:
    """Extract unique words from text."""
    words = set()
    splitter = re.compile(
        r'[\s\d\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*\+=<>\|~`…–—•·«»""''\u200b-\u200f\ufeff\u00a0]+'
    )
    for text in texts:
        for token in splitter.split(text):
            token = token.strip('.,;:!?-()[]{}"\'/\\«»""''…–—•·')
            if min_len <= len(token) <= max_len and token and not token[0].isdigit():
                words.add(token)
    return words


def load_existing(filepath: Path) -> set[str]:
    if not filepath.exists():
        return set()
    return set(l.strip() for l in filepath.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip())


def append_words(filepath: Path, new_words: set[str]) -> int:
    """Append new words to file, return count added."""
    existing = load_existing(filepath)
    added = new_words - existing
    if added:
        with open(filepath, "a", encoding="utf-8") as f:
            for w in sorted(added):
                f.write(w + "\n")
    return len(added)


def process_lang(lang: str, config: dict, wiki_pages: int, wikt_limit: int):
    """Process one language from all sources."""
    all_words = set()

    # 1. Wikipedia random articles
    print(f"  [1/3] Wikipedia articles ({wiki_pages} pages)...")
    texts = fetch_wiki_articles(lang, wiki_pages)
    wiki_words = extract_words(texts)
    all_words |= wiki_words
    print(f"    -> {len(wiki_words)} unique words from {len(texts)} articles")

    # 2. Wikipedia page titles
    print(f"  [2/3] Wikipedia page titles...")
    titles = fetch_wiki_allpages(lang, count=5000)
    title_words = extract_words(titles)
    all_words |= title_words
    print(f"    -> {len(title_words)} unique words from {len(titles)} titles")

    # 3. Wiktionary categories
    cats = config.get("wikt_cats", [])
    if cats:
        print(f"  [3/3] Wiktionary ({len(cats)} categories)...")
        wikt_words = fetch_wiktionary_words(lang, cats, max_per_cat=wikt_limit)
        all_words |= wikt_words
        print(f"    -> {len(wikt_words)} unique words from Wiktionary")

    # Write to target files
    for filename in config["files"]:
        filepath = WORD_LIST_DIR / filename
        added = append_words(filepath, all_words)
        total = len(load_existing(filepath))
        chars = len(set(c for c in filepath.read_text(encoding="utf-8", errors="ignore")
                       if not c.isspace() and ord(c) > 127))
        print(f"  => {filename}: +{added} words (total: {total}, {chars} unique chars)")


def main():
    parser = argparse.ArgumentParser(description="Aggressively fetch words from Wikipedia + Wiktionary")
    parser.add_argument("--wiki-pages", type=int, default=500,
                        help="Wikipedia articles per language (default: 500)")
    parser.add_argument("--wikt-limit", type=int, default=5000,
                        help="Max entries per Wiktionary category (default: 5000)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Comma-separated lang codes (default: all)")
    parser.add_argument("--skip-latin", action="store_true",
                        help="Skip Latin-script languages (already large)")
    args = parser.parse_args()

    if args.lang:
        langs = {l.strip(): LANG_CONFIG[l.strip()] for l in args.lang.split(",") if l.strip() in LANG_CONFIG}
    else:
        langs = LANG_CONFIG.copy()

    # Skip English — already huge
    langs.pop("en", None)

    if args.skip_latin:
        latin_files = {"french.txt", "german.txt", "spanish.txt", "italian.txt",
                       "portuguese.txt", "polish.txt", "dutch.txt", "romanian.txt",
                       "czech.txt", "hungarian.txt", "swedish.txt", "norwegian.txt",
                       "danish.txt", "finnish.txt", "croatian.txt", "indonesian.txt",
                       "malay.txt", "swahili.txt"}
        langs = {k: v for k, v in langs.items() if not set(v["files"]) & latin_files}

    total_start = time.time()
    n = len(langs)
    print(f"Processing {n} languages, {args.wiki_pages} wiki articles + "
          f"{args.wikt_limit} wiktionary entries per category\n")

    for i, (lang, config) in enumerate(langs.items(), 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{n}] {lang} -> {', '.join(config['files'])}")
        print(f"{'='*60}")
        try:
            process_lang(lang, config, args.wiki_pages, args.wikt_limit)
        except Exception as e:
            print(f"  ERROR: {e}")

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    # Final summary
    print(f"{'File':<28} {'Words':>8} {'Chars':>7}")
    print("-" * 46)
    for filepath in sorted(WORD_LIST_DIR.glob("*.txt")):
        if filepath.name in ('english_100k.txt', 'english_legal.txt', 'hindi_legal.txt',
                              'burmese.txt', 'khmer.txt', 'odia.txt'):
            continue
        lines = sum(1 for l in filepath.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip())
        chars = len(set(c for c in filepath.read_text(encoding="utf-8", errors="ignore")
                       if not c.isspace() and ord(c) > 127))
        if lines > 50:
            print(f"  {filepath.name:<26} {lines:>7} {chars:>6}")


if __name__ == "__main__":
    main()
