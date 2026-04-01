#!/usr/bin/env python3
"""
Fill character coverage gaps in word lists.

For each script, identifies missing Unicode code points and generates
synthetic entries (character combinations) to ensure full coverage.

- Indic scripts: generates consonant+vowel sign combinations
- CJK: adds missing common characters as 2-char pairs
- Korean: generates syllable blocks
- Arabic/Hebrew: adds missing forms in word context
- Thai/Lao: generates consonant+vowel+tone combinations

Usage:
    python scripts/fill_char_coverage.py
    python scripts/fill_char_coverage.py --dry-run  # just report gaps
"""

import argparse
import random
from pathlib import Path

WORD_LIST_DIR = Path(__file__).parent.parent / "training_data" / "word_lists"


def get_existing_chars(filepath: Path, ranges: list[tuple[int, int]]) -> set[int]:
    """Get code points from file that fall within given Unicode ranges."""
    if not filepath.exists():
        return set()
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    chars = set()
    for c in text:
        cp = ord(c)
        for start, end in ranges:
            if start <= cp <= end:
                chars.add(cp)
                break
    return chars


def get_assigned_codepoints(ranges: list[tuple[int, int]]) -> set[int]:
    """Get all printable, assigned code points in ranges."""
    assigned = set()
    for start, end in ranges:
        for cp in range(start, end + 1):
            try:
                c = chr(cp)
                if c.isprintable() and not c.isspace():
                    assigned.add(cp)
            except (ValueError, OverflowError):
                pass
    return assigned


def append_words(filepath: Path, words: list[str]):
    """Append words to file."""
    existing = set()
    if filepath.exists():
        existing = set(filepath.read_text(encoding="utf-8", errors="ignore").splitlines())
    new_words = [w for w in words if w not in existing and len(w) >= 2]
    if new_words:
        with open(filepath, "a", encoding="utf-8") as f:
            for w in new_words:
                f.write(w + "\n")
    return len(new_words)


# ---- Indic script generators ----

def generate_indic_combos(base_consonants: list[int], vowel_signs: list[int],
                          virama: int, extra_chars: list[int]) -> list[str]:
    """Generate consonant+vowel combinations for Indic scripts."""
    words = []
    # Each consonant with each vowel sign
    for c in base_consonants:
        for v in vowel_signs:
            words.append(chr(c) + chr(v))
    # Consonant clusters with virama
    for c1 in base_consonants[:10]:  # first 10 consonants
        for c2 in base_consonants[:5]:
            words.append(chr(c1) + chr(virama) + chr(c2))
    # Extra characters in context (between two common consonants)
    for cp in extra_chars:
        c = chr(cp)
        # Put it between common characters so it renders properly
        if base_consonants:
            words.append(chr(base_consonants[0]) + c)
            words.append(c + chr(base_consonants[0]))
            words.append(chr(base_consonants[0]) + c + chr(base_consonants[1]) if len(base_consonants) > 1 else c)
    return words


def fill_devanagari() -> list[str]:
    consonants = list(range(0x0915, 0x0940))  # ka to ha
    vowels_indep = list(range(0x0904, 0x0915))  # independent vowels
    vowel_signs = list(range(0x093E, 0x094D))  # dependent vowel signs
    virama = 0x094D
    # Nukta, anusvara, visarga, chandrabindu, etc.
    extras = [0x0901, 0x0902, 0x0903, 0x093C, 0x093D, 0x0950,  # OM
              0x0958, 0x0959, 0x095A, 0x095B, 0x095C, 0x095D, 0x095E, 0x095F,  # nukta forms
              0x0960, 0x0961, 0x0962, 0x0963,  # vocalic vowels
              0x0966, 0x0967, 0x0968, 0x0969, 0x096A, 0x096B, 0x096C, 0x096D, 0x096E, 0x096F,  # digits
              0x0970, 0x0971, 0x0972, 0x097B, 0x097C, 0x097D, 0x097E, 0x097F]
    words = generate_indic_combos(consonants, vowel_signs, virama, extras)
    # Add independent vowels as pairs
    for i in range(0, len(vowels_indep) - 1, 1):
        words.append(chr(vowels_indep[i]) + chr(consonants[0]))
    return words


def fill_bengali() -> list[str]:
    consonants = list(range(0x0995, 0x09B0)) + [0x09B2, 0x09B6, 0x09B7, 0x09B8, 0x09B9]
    vowel_signs = list(range(0x09BE, 0x09CD))
    virama = 0x09CD
    extras = [0x0981, 0x0982, 0x0983, 0x09BC, 0x09BD, 0x09CE,  # khanda ta
              0x09DC, 0x09DD, 0x09DF,  # ya-phalaa, etc.
              0x09E0, 0x09E1, 0x09E2, 0x09E3,  # vocalic
              0x09E6, 0x09E7, 0x09E8, 0x09E9, 0x09EA, 0x09EB, 0x09EC, 0x09ED, 0x09EE, 0x09EF,
              0x09F0, 0x09F1, 0x09F2, 0x09F3, 0x09F4, 0x09F5, 0x09F6, 0x09F7, 0x09F8, 0x09F9, 0x09FA, 0x09FB]
    return generate_indic_combos(consonants, vowel_signs, virama, extras)


def fill_tamil() -> list[str]:
    consonants = [0x0B95, 0x0B99, 0x0B9A, 0x0B9C, 0x0B9E, 0x0B9F,
                  0x0BA3, 0x0BA4, 0x0BA8, 0x0BA9, 0x0BAA, 0x0BAE,
                  0x0BAF, 0x0BB0, 0x0BB1, 0x0BB2, 0x0BB3, 0x0BB4,
                  0x0BB5, 0x0BB6, 0x0BB7, 0x0BB8, 0x0BB9]
    vowel_signs = list(range(0x0BBE, 0x0BCD))
    virama = 0x0BCD
    extras = [0x0B82, 0x0B83,  # anusvara, visarga
              0x0B85, 0x0B86, 0x0B87, 0x0B88, 0x0B89, 0x0B8A, 0x0B8E, 0x0B8F, 0x0B90,
              0x0B92, 0x0B93, 0x0B94,  # independent vowels
              0x0BD0,  # OM
              0x0BD7,  # au length mark
              0x0BE6, 0x0BE7, 0x0BE8, 0x0BE9, 0x0BEA, 0x0BEB, 0x0BEC, 0x0BED, 0x0BEE, 0x0BEF,  # digits
              0x0BF0, 0x0BF1, 0x0BF2, 0x0BF3, 0x0BF4, 0x0BF5, 0x0BF6, 0x0BF7, 0x0BF8, 0x0BF9]
    return generate_indic_combos(consonants, vowel_signs, virama, extras)


def fill_telugu() -> list[str]:
    consonants = list(range(0x0C15, 0x0C3A))
    vowel_signs = list(range(0x0C3E, 0x0C4D))
    virama = 0x0C4D
    extras = [0x0C01, 0x0C02, 0x0C03, 0x0C3C, 0x0C3D,
              0x0C58, 0x0C59, 0x0C5A,
              0x0C60, 0x0C61, 0x0C62, 0x0C63,
              0x0C66, 0x0C67, 0x0C68, 0x0C69, 0x0C6A, 0x0C6B, 0x0C6C, 0x0C6D, 0x0C6E, 0x0C6F,
              0x0C77, 0x0C78, 0x0C79, 0x0C7A, 0x0C7B, 0x0C7C, 0x0C7D, 0x0C7E, 0x0C7F]
    return generate_indic_combos(consonants, vowel_signs, virama, extras)


def fill_kannada() -> list[str]:
    consonants = list(range(0x0C95, 0x0CBA))
    vowel_signs = list(range(0x0CBE, 0x0CCD))
    virama = 0x0CCD
    extras = [0x0C81, 0x0C82, 0x0C83, 0x0CBC, 0x0CBD,
              0x0CDE,  # llla
              0x0CE0, 0x0CE1, 0x0CE2, 0x0CE3,
              0x0CE6, 0x0CE7, 0x0CE8, 0x0CE9, 0x0CEA, 0x0CEB, 0x0CEC, 0x0CED, 0x0CEE, 0x0CEF,
              0x0CF1, 0x0CF2]
    return generate_indic_combos(consonants, vowel_signs, virama, extras)


def fill_malayalam() -> list[str]:
    consonants = list(range(0x0D15, 0x0D3A))
    vowel_signs = list(range(0x0D3E, 0x0D4D))
    virama = 0x0D4D
    extras = [0x0D01, 0x0D02, 0x0D03, 0x0D3D,
              0x0D4E,  # dot reph
              0x0D54, 0x0D55, 0x0D56, 0x0D57,
              0x0D5F, 0x0D60, 0x0D61, 0x0D62, 0x0D63,
              0x0D66, 0x0D67, 0x0D68, 0x0D69, 0x0D6A, 0x0D6B, 0x0D6C, 0x0D6D, 0x0D6E, 0x0D6F,
              0x0D70, 0x0D71, 0x0D72, 0x0D73, 0x0D74, 0x0D75, 0x0D76, 0x0D77, 0x0D78, 0x0D79,
              0x0D7A, 0x0D7B, 0x0D7C, 0x0D7D, 0x0D7E, 0x0D7F]  # chillu letters
    return generate_indic_combos(consonants, vowel_signs, virama, extras)


def fill_gujarati() -> list[str]:
    consonants = list(range(0x0A95, 0x0AB0)) + [0x0AB2, 0x0AB3, 0x0AB5, 0x0AB6, 0x0AB7, 0x0AB8, 0x0AB9]
    vowel_signs = list(range(0x0ABE, 0x0ACD))
    virama = 0x0ACD
    extras = [0x0A81, 0x0A82, 0x0A83, 0x0ABC, 0x0ABD,
              0x0AD0,  # OM
              0x0AE0, 0x0AE1, 0x0AE2, 0x0AE3,
              0x0AE6, 0x0AE7, 0x0AE8, 0x0AE9, 0x0AEA, 0x0AEB, 0x0AEC, 0x0AED, 0x0AEE, 0x0AEF,
              0x0AF0, 0x0AF1, 0x0AF9, 0x0AFA, 0x0AFB, 0x0AFC, 0x0AFD, 0x0AFE, 0x0AFF]
    return generate_indic_combos(consonants, vowel_signs, virama, extras)


def fill_gurmukhi() -> list[str]:
    consonants = list(range(0x0A15, 0x0A30)) + [0x0A32, 0x0A33, 0x0A35, 0x0A36, 0x0A38, 0x0A39]
    vowel_signs = list(range(0x0A3E, 0x0A4D))
    virama = 0x0A4D
    extras = [0x0A01, 0x0A02, 0x0A03, 0x0A3C,
              0x0A59, 0x0A5A, 0x0A5B, 0x0A5C, 0x0A5E,  # nukta forms
              0x0A66, 0x0A67, 0x0A68, 0x0A69, 0x0A6A, 0x0A6B, 0x0A6C, 0x0A6D, 0x0A6E, 0x0A6F,
              0x0A70, 0x0A71, 0x0A72, 0x0A73, 0x0A74, 0x0A75, 0x0A76]
    return generate_indic_combos(consonants, vowel_signs, virama, extras)


# ---- Thai / Lao generators ----

def fill_thai() -> list[str]:
    consonants = list(range(0x0E01, 0x0E2F))  # ko kai to ho nokhuk
    vowels_above_below = [0x0E31, 0x0E34, 0x0E35, 0x0E36, 0x0E37, 0x0E38, 0x0E39, 0x0E47]
    tone_marks = [0x0E48, 0x0E49, 0x0E4A, 0x0E4B]
    extras = [0x0E2F, 0x0E30, 0x0E32, 0x0E33, 0x0E40, 0x0E41, 0x0E42, 0x0E43, 0x0E44, 0x0E45, 0x0E46,
              0x0E4C, 0x0E4D, 0x0E4E, 0x0E4F,
              0x0E50, 0x0E51, 0x0E52, 0x0E53, 0x0E54, 0x0E55, 0x0E56, 0x0E57, 0x0E58, 0x0E59,  # digits
              0x0E3A, 0x0E3F, 0x0E5A, 0x0E5B]  # baht sign, etc.
    words = []
    # Consonant + vowel above/below + optional tone
    for c in consonants:
        for v in vowels_above_below:
            words.append(chr(c) + chr(v))
            for t in tone_marks:
                words.append(chr(c) + chr(v) + chr(t))
    # Leading vowels + consonant
    for lv in [0x0E40, 0x0E41, 0x0E42, 0x0E43, 0x0E44]:
        for c in consonants[:10]:
            words.append(chr(lv) + chr(c))
    # Extras in context
    for cp in extras:
        words.append(chr(consonants[0]) + chr(cp))
    return words


def fill_lao() -> list[str]:
    consonants = list(range(0x0E81, 0x0EAE))
    # Filter to only assigned Lao consonants
    consonants = [cp for cp in consonants if chr(cp).isprintable()]
    vowels = [0x0EB1, 0x0EB4, 0x0EB5, 0x0EB6, 0x0EB7, 0x0EB8, 0x0EB9, 0x0EBB]
    tone_marks = [0x0EC8, 0x0EC9, 0x0ECA, 0x0ECB]
    extras = [0x0EAF, 0x0EB0, 0x0EB2, 0x0EB3,
              0x0EBC, 0x0EBD,
              0x0EC0, 0x0EC1, 0x0EC2, 0x0EC3, 0x0EC4, 0x0EC6,
              0x0ECC, 0x0ECD,
              0x0ED0, 0x0ED1, 0x0ED2, 0x0ED3, 0x0ED4, 0x0ED5, 0x0ED6, 0x0ED7, 0x0ED8, 0x0ED9,
              0x0EDC, 0x0EDD, 0x0EDE, 0x0EDF]
    words = []
    for c in consonants:
        for v in vowels:
            words.append(chr(c) + chr(v))
            for t in tone_marks:
                words.append(chr(c) + chr(v) + chr(t))
    for lv in [0x0EC0, 0x0EC1, 0x0EC2, 0x0EC3, 0x0EC4]:
        for c in consonants[:10]:
            words.append(chr(lv) + chr(c))
    for cp in extras:
        if chr(cp).isprintable():
            words.append(chr(consonants[0]) + chr(cp))
    return words


# ---- CJK / Korean ----

def fill_cjk() -> list[str]:
    """Add common CJK characters as 2-char pairs."""
    words = []
    # HSK-style frequency bands — add missing common characters
    important_ranges = [
        (0x4E00, 0x5FFF),  # most common CJK block
        (0x6000, 0x7FFF),
        (0x8000, 0x9FFF),
        (0x3040, 0x309F),  # hiragana (all)
        (0x30A0, 0x30FF),  # katakana (all)
    ]
    chars = []
    for start, end in important_ranges:
        for cp in range(start, end + 1):
            try:
                c = chr(cp)
                if c.isprintable() and not c.isspace():
                    chars.append(c)
            except:
                pass
    # Create pairs from consecutive characters
    random.seed(42)
    random.shuffle(chars)
    for i in range(0, len(chars) - 1, 2):
        words.append(chars[i] + chars[i + 1])
    return words


def fill_korean() -> list[str]:
    """Generate Korean syllable blocks to fill coverage gaps."""
    words = []
    # Korean syllables: (initial * 21 + medial) * 28 + final + 0xAC00
    # Initial: 0-18, Medial: 0-20, Final: 0-27
    # Generate a spread across the syllable space
    random.seed(42)
    for _ in range(3000):
        initial = random.randint(0, 18)
        medial = random.randint(0, 20)
        final = random.randint(0, 27)
        cp = 0xAC00 + (initial * 21 + medial) * 28 + final
        if cp <= 0xD7A3:
            # Create a 2-syllable word
            initial2 = random.randint(0, 18)
            medial2 = random.randint(0, 20)
            final2 = random.randint(0, 27)
            cp2 = 0xAC00 + (initial2 * 21 + medial2) * 28 + final2
            if cp2 <= 0xD7A3:
                words.append(chr(cp) + chr(cp2))
    return words


# ---- Arabic / Hebrew ----

def fill_arabic() -> list[str]:
    """Add Arabic characters in word context."""
    words = []
    # Basic Arabic block — all characters
    for cp in range(0x0600, 0x06FF):
        c = chr(cp)
        if c.isprintable() and not c.isspace():
            # Wrap in alef + char + ba for context
            words.append("\u0627" + c + "\u0628")
    # Arabic Supplement
    for cp in range(0x0750, 0x077F):
        c = chr(cp)
        if c.isprintable() and not c.isspace():
            words.append("\u0627" + c + "\u0628")
    # Presentation forms (important for positional rendering)
    for cp in range(0xFE70, 0xFEFF):
        c = chr(cp)
        if c.isprintable() and not c.isspace():
            words.append(c + "\u0627")
    return words


def fill_hebrew() -> list[str]:
    """Add Hebrew characters including vowel points."""
    words = []
    base_letters = list(range(0x05D0, 0x05EB))  # alef to tav
    # Vowel points (nikkud)
    vowel_points = list(range(0x05B0, 0x05BE)) + [0x05BF, 0x05C1, 0x05C2, 0x05C4, 0x05C5, 0x05C7]
    # Cantillation marks
    cantillation = list(range(0x0591, 0x05B0))
    # Letter + each vowel point
    for letter in base_letters:
        for vp in vowel_points:
            words.append(chr(letter) + chr(vp))
    # Letter + cantillation
    for letter in base_letters[:5]:
        for cm in cantillation:
            words.append(chr(letter) + chr(cm))
    # Presentation forms
    for cp in range(0xFB1D, 0xFB50):
        c = chr(cp)
        if c.isprintable() and not c.isspace():
            words.append(c + chr(base_letters[0]))
    return words


def fill_greek() -> list[str]:
    """Add Greek with polytonic marks."""
    words = []
    # Basic Greek
    for cp in range(0x0370, 0x03FF):
        c = chr(cp)
        if c.isprintable() and not c.isspace() and c.isalpha():
            words.append(c + "α")
            words.append("κ" + c)
    # Extended Greek (polytonic)
    for cp in range(0x1F00, 0x1FFF):
        c = chr(cp)
        if c.isprintable() and not c.isspace():
            words.append(c + "ς")
    return words


def fill_cyrillic() -> list[str]:
    """Add rare Cyrillic characters."""
    words = []
    for cp in range(0x0400, 0x052F):
        c = chr(cp)
        if c.isprintable() and not c.isspace() and c.isalpha():
            words.append(c + "а")
            words.append("к" + c)
    return words


# ---- Main ----

SCRIPT_FILLERS = {
    "devanagari":  (["devanagari.txt", "marathi.txt"], fill_devanagari),
    "bengali":     (["bengali.txt"], fill_bengali),
    "tamil":       (["tamil.txt"], fill_tamil),
    "telugu":      (["telugu.txt"], fill_telugu),
    "kannada":     (["kannada.txt"], fill_kannada),
    "malayalam":   (["malayalam.txt"], fill_malayalam),
    "gujarati":    (["gujarati.txt"], fill_gujarati),
    "gurmukhi":    (["gurmukhi.txt"], fill_gurmukhi),
    "thai":        (["thai.txt"], fill_thai),
    "lao":         (["lao.txt"], fill_lao),
    "han_kana":         (["han_kana.txt", "japanese.txt"], fill_cjk),
    "korean":      (["korean.txt"], fill_korean),
    "arabic":      (["arabic.txt"], fill_arabic),
    "hebrew":      (["hebrew.txt"], fill_hebrew),
    "greek":       (["greek.txt"], fill_greek),
    "cyrillic":    (["cyrillic.txt"], fill_cyrillic),
}


def main():
    parser = argparse.ArgumentParser(description="Fill character coverage gaps")
    parser.add_argument("--dry-run", action="store_true", help="Just report, don't write")
    args = parser.parse_args()

    for script, (files, generator) in SCRIPT_FILLERS.items():
        words = generator()
        # Deduplicate
        words = list(set(w for w in words if len(w) >= 2))

        if args.dry_run:
            print(f"{script}: would add up to {len(words)} synthetic entries")
            continue

        for filename in files:
            filepath = WORD_LIST_DIR / filename
            added = append_words(filepath, words)
            if added:
                # Count chars after
                text = filepath.read_text(encoding="utf-8", errors="ignore")
                chars = len(set(c for c in text if not c.isspace() and ord(c) > 127))
                print(f"  {filename}: +{added} entries ({chars} unique chars)")


if __name__ == "__main__":
    main()
