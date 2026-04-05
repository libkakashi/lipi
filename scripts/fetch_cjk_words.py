#!/usr/bin/env python3
"""
Build a comprehensive CJK word list for OCR training.

Sources:
  1. CC-CEDICT dictionary (~120K entries, simplified + traditional Chinese)
  2. Wikipedia titles (zh, zh-yue/Cantonese, zh-classical, ja)
  3. Wiktionary CJK entries
  4. Unihan readings database (Mandarin, Cantonese, Japanese, Korean readings)
  5. Synthetic pairs for rare characters with zero natural coverage

Goal: every CJK char (U+4E00-9FFF, U+3400-4DBF) appears in training data.

Usage:
    python -m scripts.fetch_cjk_words                    # full pipeline
    python -m scripts.fetch_cjk_words --skip-download    # reuse cached files
    python -m scripts.fetch_cjk_words --stats-only       # just print coverage
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

WORD_LIST_DIR = Path(__file__).resolve().parent.parent / "training_data" / "word_lists"
CACHE_DIR = WORD_LIST_DIR / "raw_chinese"
TARGET_FILE = WORD_LIST_DIR / "han_kana.txt"
JAPANESE_FILE = WORD_LIST_DIR / "japanese.txt"

CJK_UNIFIED = range(0x4E00, 0x9FFF + 1)
CJK_EXT_A = range(0x3400, 0x4DBF + 1)
ALL_CJK = set(chr(cp) for cp in CJK_UNIFIED) | set(chr(cp) for cp in CJK_EXT_A)

MIN_WORD_LEN = 2
MAX_WORD_LEN = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF


def is_kana(ch: str) -> bool:
    cp = ord(ch)
    return 0x3040 <= cp <= 0x30FF


def has_cjk(text: str) -> bool:
    return any(is_cjk(c) for c in text)


def cjk_chars_in(text: str) -> set[str]:
    return {c for c in text if is_cjk(c)}


def clean_word(w: str) -> str:
    """Strip surrounding punctuation/whitespace, keep CJK/kana core."""
    return w.strip().strip(".,;:!?-()[]{}\"'/\\«»\u201c\u201d\u2018\u2019\u2026\u2013\u2014\u2022\u00b7")


def valid_word(w: str) -> bool:
    """Accept words that are 2-15 chars and contain CJK or kana."""
    if not (MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN):
        return False
    return has_cjk(w) or any(is_kana(c) for c in w)


def fetch_url(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "LipiOCR/1.0 (OCR training data)"
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed to fetch {url}: {e}")
                return b""
    return b""


def wiki_api_get(wiki_lang: str, params: dict) -> dict | None:
    url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
    params["format"] = "json"
    query = url + "?" + urllib.parse.urlencode(params)
    data = fetch_url(query)
    if data:
        return json.loads(data.decode("utf-8"))
    return None


import urllib.parse


# ---------------------------------------------------------------------------
# Source 1: CC-CEDICT
# ---------------------------------------------------------------------------

def fetch_cedict() -> list[str]:
    """Download and parse CC-CEDICT. Returns Chinese words (simplified + traditional)."""
    cache = CACHE_DIR / "cedict_ts.u8"
    if not cache.exists():
        print("  Downloading CC-CEDICT...")
        url = "https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz"
        data = fetch_url(url, timeout=60)
        if not data:
            print("  CC-CEDICT download failed")
            return []
        text = gzip.decompress(data).decode("utf-8")
        cache.write_text(text, encoding="utf-8")
        print(f"  Cached to {cache}")
    else:
        text = cache.read_text(encoding="utf-8")

    words: list[str] = []
    # Format: 繁體 简体 [pin1 yin1] /definition/
    pattern = re.compile(r"^(\S+)\s+(\S+)\s+\[")
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = pattern.match(line)
        if m:
            traditional, simplified = m.group(1), m.group(2)
            if has_cjk(traditional):
                words.append(traditional)
            if has_cjk(simplified) and simplified != traditional:
                words.append(simplified)

    print(f"  CC-CEDICT: {len(words)} entries")
    return words


# ---------------------------------------------------------------------------
# Source 2: Wikipedia titles (bulk)
# ---------------------------------------------------------------------------

def fetch_wiki_titles(wiki_lang: str, max_titles: int = 100_000) -> list[str]:
    """Fetch page titles from Wikipedia using allpages API."""
    print(f"  Fetching {wiki_lang}.wikipedia.org titles (up to {max_titles:,})...")
    titles: list[str] = []
    ap_from = ""

    while len(titles) < max_titles:
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": "500",
            "apnamespace": "0",
            "apfilterredir": "nonredirects",
        }
        if ap_from:
            params["apfrom"] = ap_from

        data = wiki_api_get(wiki_lang, params)
        if not data:
            break

        pages = data.get("query", {}).get("allpages", [])
        if not pages:
            break

        for p in pages:
            title = p.get("title", "")
            if title:
                titles.append(title)

        cont = data.get("continue", {})
        if "apcontinue" in cont:
            ap_from = cont["apcontinue"]
        else:
            break

        time.sleep(0.2)

        if len(titles) % 10_000 < 500:
            print(f"    ...{len(titles):,} titles so far")

    print(f"  {wiki_lang}: {len(titles):,} titles fetched")
    return titles


def fetch_all_wiki_titles() -> list[str]:
    """Fetch titles from Chinese + Cantonese + Classical Chinese + Japanese Wikipedia."""
    all_words: list[str] = []

    for wiki_lang, max_t in [
        ("zh", 100_000),       # Chinese Wikipedia (~1.3M articles)
        ("zh-yue", 50_000),    # Cantonese Wikipedia (~130K articles)
        ("ja", 100_000),       # Japanese Wikipedia (~1.4M articles)
    ]:
        cache = CACHE_DIR / f"wiki_{wiki_lang}_titles.txt"
        if cache.exists():
            titles = cache.read_text(encoding="utf-8").splitlines()
            print(f"  {wiki_lang}: {len(titles):,} titles (cached)")
        else:
            titles = fetch_wiki_titles(wiki_lang, max_titles=max_t)
            cache.write_text("\n".join(titles), encoding="utf-8")

        # Extract words from titles (split on common separators)
        for title in titles:
            # Many wiki titles are "Foo (disambiguation)" — strip parens
            title = re.sub(r"\s*\(.*?\)\s*", "", title)
            # Split on common title separators
            parts = re.split(r"[·\-—–/、,\s]+", title)
            for part in parts:
                part = clean_word(part)
                if valid_word(part):
                    all_words.append(part)
            # Also keep full title if valid
            title = clean_word(title)
            if valid_word(title):
                all_words.append(title)

    return all_words


# ---------------------------------------------------------------------------
# Source 3: Wiktionary CJK entries
# ---------------------------------------------------------------------------

def fetch_wiktionary_cjk() -> list[str]:
    """Fetch CJK words from Wiktionary category pages."""
    words: list[str] = []

    categories = [
        "Category:Mandarin_lemmas",
        "Category:Chinese_lemmas",
        "Category:Cantonese_lemmas",
        "Category:Classical_Chinese_lemmas",
        "Category:Japanese_lemmas",
        "Category:Min_Nan_lemmas",
        "Category:Hakka_lemmas",
        "Category:Wu_Chinese_lemmas",
    ]

    for cat in categories:
        print(f"  Wiktionary: {cat}...")
        cm_continue = ""
        cat_words = 0

        while True:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmlimit": "500",
                "cmtype": "page",
            }
            if cm_continue:
                params["cmcontinue"] = cm_continue

            url = "https://en.wiktionary.org/w/api.php"
            query = url + "?" + urllib.parse.urlencode(params)
            data = fetch_url(query)
            if not data:
                break

            result = json.loads(data.decode("utf-8"))
            members = result.get("query", {}).get("categorymembers", [])

            for m in members:
                title = m.get("title", "")
                title = clean_word(title)
                if valid_word(title):
                    words.append(title)
                    cat_words += 1

            cont = result.get("continue", {})
            if "cmcontinue" in cont:
                cm_continue = cont["cmcontinue"]
            else:
                break

            time.sleep(0.3)

            # Cap per category to avoid spending too long on huge ones
            if cat_words >= 50_000:
                break

        print(f"    {cat_words:,} words")

    return words


# ---------------------------------------------------------------------------
# Source 4: Unihan readings (characters as single-char "words")
# ---------------------------------------------------------------------------

def fetch_unihan_readings() -> list[str]:
    """Download Unihan readings database for character-level coverage.

    Extracts Mandarin, Cantonese, Japanese on/kun readings to generate
    natural character+reading pairs.
    """
    cache = CACHE_DIR / "Unihan_Readings.txt"
    if not cache.exists():
        print("  Downloading Unihan readings...")
        url = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
        data = fetch_url(url, timeout=60)
        if not data:
            print("  Unihan download failed")
            return []

        import zipfile, io
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if "Readings" in name:
                    content = zf.read(name).decode("utf-8")
                    cache.write_text(content, encoding="utf-8")
                    break

    if not cache.exists():
        return []

    text = cache.read_text(encoding="utf-8")

    # Parse: U+XXXX\tkFieldName\tvalue
    char_readings: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        cp_str, field, value = parts[0], parts[1], parts[2]
        if field in ("kMandarin", "kCantonese", "kJapaneseOn", "kJapaneseKun"):
            try:
                cp = int(cp_str.replace("U+", ""), 16)
                char = chr(cp)
                if is_cjk(char):
                    if char not in char_readings:
                        char_readings[char] = []
                    char_readings[char].append(value)
            except (ValueError, OverflowError):
                pass

    print(f"  Unihan: {len(char_readings):,} characters with readings")
    # We don't generate words from readings here — just track which chars
    # have readings (natural chars vs truly obscure ones)
    return list(char_readings.keys())


# ---------------------------------------------------------------------------
# Source 5: Synthetic pairs for uncovered characters
# ---------------------------------------------------------------------------

def generate_rare_char_words(
    covered_chars: set[str],
    all_chars: set[str],
    existing_words: set[str],
    common_chars: list[str],
) -> list[str]:
    """Generate 2-char words for CJK characters not yet in any word list.

    Strategy: pair each rare char with common characters to create plausible-
    looking words. This ensures the model sees every character during training,
    even the most obscure Ext-A ones.

    Each rare char gets paired with 5 different common chars in both positions
    (rare+common and common+rare) = 10 synthetic words per rare char.
    """
    uncovered = all_chars - covered_chars
    if not uncovered:
        print("  All CJK characters already covered!")
        return []

    print(f"  {len(uncovered):,} chars need synthetic coverage")

    # Use top-200 most common CJK chars as pairing partners
    partners = [c for c in common_chars if c in covered_chars][:200]
    if not partners:
        partners = list(covered_chars)[:200]

    words: list[str] = []
    for char in sorted(uncovered):
        # 5 pairs in each position
        for i in range(min(5, len(partners))):
            p = partners[i % len(partners)]
            w1 = char + p
            w2 = p + char
            if w1 not in existing_words:
                words.append(w1)
            if w2 not in existing_words:
                words.append(w2)

    print(f"  Generated {len(words):,} synthetic words for {len(uncovered):,} rare chars")
    return words


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def print_coverage(words: set[str], label: str = ""):
    """Print CJK character coverage statistics."""
    covered = set()
    for w in words:
        covered.update(cjk_chars_in(w))

    unified_covered = sum(1 for c in covered if 0x4E00 <= ord(c) <= 0x9FFF)
    ext_a_covered = sum(1 for c in covered if 0x3400 <= ord(c) <= 0x4DBF)

    print(f"\n  {label}Character coverage:")
    print(f"    CJK Unified: {unified_covered:,} / 20,992 ({unified_covered*100/20992:.1f}%)")
    print(f"    CJK Ext-A:   {ext_a_covered:,} / 6,592  ({ext_a_covered*100/6592:.1f}%)")
    print(f"    Total:        {len(covered):,} / 27,584 ({len(covered)*100/27584:.1f}%)")
    print(f"    Uncovered:    {len(ALL_CJK) - len(covered):,}")

    # Per-char frequency in word list
    char_freq = Counter()
    for w in words:
        for c in w:
            if is_cjk(c):
                char_freq[c] += 1

    if char_freq:
        freqs = sorted(char_freq.values())
        print(f"    Min appearances: {freqs[0]}")
        print(f"    Median appearances: {freqs[len(freqs)//2]}")
        print(f"    Chars appearing once: {sum(1 for f in freqs if f == 1):,}")
        print(f"    Chars appearing 5+: {sum(1 for f in freqs if f >= 5):,}")

    return covered


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build comprehensive CJK word list")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, use cached files only")
    parser.add_argument("--stats-only", action="store_true",
                        help="Just print coverage stats for existing word list")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing words
    existing_words: set[str] = set()
    for path in [TARGET_FILE, JAPANESE_FILE]:
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if w:
                    existing_words.add(w)

    print(f"Existing words: {len(existing_words):,}")
    covered_before = print_coverage(existing_words, "BEFORE — ")

    if args.stats_only:
        return

    # Collect new words from all sources
    new_words: set[str] = set()

    # Source 1: CC-CEDICT
    print(f"\n{'='*60}")
    print("Source 1: CC-CEDICT dictionary")
    print(f"{'='*60}")
    if not args.skip_download or (CACHE_DIR / "cedict_ts.u8").exists():
        for w in fetch_cedict():
            w = clean_word(w)
            if valid_word(w) and w not in existing_words:
                new_words.add(w)
        print(f"  New words from CEDICT: {len(new_words):,}")

    # Source 2: Wikipedia titles
    print(f"\n{'='*60}")
    print("Source 2: Wikipedia titles (zh, zh-yue, ja)")
    print(f"{'='*60}")
    if not args.skip_download or any(
        (CACHE_DIR / f"wiki_{lang}_titles.txt").exists()
        for lang in ["zh", "zh-yue", "ja"]
    ):
        before = len(new_words)
        for w in fetch_all_wiki_titles():
            if w not in existing_words:
                new_words.add(w)
        print(f"  New words from Wikipedia: {len(new_words) - before:,}")

    # Source 3: Wiktionary
    print(f"\n{'='*60}")
    print("Source 3: Wiktionary CJK categories")
    print(f"{'='*60}")
    if not args.skip_download:
        before = len(new_words)
        for w in fetch_wiktionary_cjk():
            if w not in existing_words:
                new_words.add(w)
        print(f"  New words from Wiktionary: {len(new_words) - before:,}")

    # Source 4: Unihan (just for tracking coverage)
    print(f"\n{'='*60}")
    print("Source 4: Unihan readings")
    print(f"{'='*60}")
    if not args.skip_download or (CACHE_DIR / "Unihan_Readings.txt").exists():
        unihan_chars = fetch_unihan_readings()

    # Check coverage after natural sources
    all_natural = existing_words | new_words
    covered_natural = print_coverage(all_natural, "AFTER NATURAL SOURCES — ")

    # Source 5: Synthetic pairs for remaining uncovered chars
    print(f"\n{'='*60}")
    print("Source 5: Synthetic words for uncovered characters")
    print(f"{'='*60}")

    # Build frequency-sorted common char list from existing data
    char_freq = Counter()
    for w in all_natural:
        for c in w:
            if is_cjk(c):
                char_freq[c] += 1
    common_chars = [c for c, _ in char_freq.most_common()]

    synthetic = generate_rare_char_words(
        covered_natural, ALL_CJK, all_natural, common_chars)
    for w in synthetic:
        new_words.add(w)

    # Final coverage
    all_words = existing_words | new_words
    print_coverage(all_words, "FINAL — ")

    # Write new words to han_kana.txt
    # Separate Japanese-only words into japanese.txt
    han_kana_new: list[str] = []
    japanese_new: list[str] = []

    for w in sorted(new_words):
        has_c = any(is_cjk(c) for c in w)
        has_k = any(is_kana(c) for c in w)
        if has_k and not has_c:
            japanese_new.append(w)
        else:
            han_kana_new.append(w)

    if han_kana_new:
        print(f"\nAppending {len(han_kana_new):,} words to {TARGET_FILE.name}")
        with open(TARGET_FILE, "a", encoding="utf-8") as f:
            for w in han_kana_new:
                f.write(w + "\n")

    if japanese_new:
        print(f"Appending {len(japanese_new):,} words to {JAPANESE_FILE.name}")
        with open(JAPANESE_FILE, "a", encoding="utf-8") as f:
            for w in japanese_new:
                f.write(w + "\n")

    total_added = len(han_kana_new) + len(japanese_new)
    print(f"\nTotal: +{total_added:,} new words")

    # Final line counts
    for path in [TARGET_FILE, JAPANESE_FILE]:
        if path.exists():
            lines = sum(1 for _ in open(path, encoding="utf-8"))
            print(f"  {path.name}: {lines:,} words")


if __name__ == "__main__":
    main()
