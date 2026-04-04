"""
Font discovery for OCR training data generation.

Uses an explicit font-to-script mapping — no pattern matching.
Each font is mapped to the scripts it can actually render.
Prevents tofu rendering from fonts that pass cmap checks but
can't display the correct glyphs.
"""

import os
from pathlib import Path

from src.data.renderer import font_can_render, font_has_codepoint


# Common font directories by platform
_FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    "/System/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/Library/Fonts",
    # Project-local fonts
    str(Path(__file__).parent.parent.parent / "training_data" / "fonts"),
]

# Explicit font filename → scripts mapping.
# Only fonts in this list are used. No pattern matching.
# Derived from setup_fonts.py — these are the fonts we download and control.
_FONT_TO_SCRIPTS: dict[str, list[str]] = {
    # --- Latin / Cyrillic / Greek (base Noto + serif) ---
    "NotoSans-Regular.ttf": ["latin", "cyrillic", "greek"],
    "NotoSans-Bold.ttf": ["latin", "cyrillic", "greek"],
    "NotoSans-Italic.ttf": ["latin", "cyrillic", "greek"],
    "NotoSans-Light.ttf": ["latin", "cyrillic", "greek"],
    "NotoSansMono-Regular.ttf": ["latin", "cyrillic", "greek"],
    "NotoSerif-Regular.ttf": ["latin", "cyrillic", "greek"],
    "NotoSerif-Bold.ttf": ["latin", "cyrillic", "greek"],
    "NotoSerif-Italic.ttf": ["latin", "cyrillic", "greek"],
    # Latin handwriting / display
    "Caveat[wght].ttf": ["latin"],
    "DancingScript[wght].ttf": ["latin"],
    "IndieFlower-Regular.ttf": ["latin"],
    "PatrickHand-Regular.ttf": ["latin"],
    "ShadowsIntoLight.ttf": ["latin"],
    "PermanentMarker-Regular.ttf": ["latin"],
    "AmaticSC-Regular.ttf": ["latin"],
    "Lobster-Regular.ttf": ["latin"],
    "Pacifico-Regular.ttf": ["latin"],
    "ComicNeue-Regular.ttf": ["latin"],
    "SpecialElite-Regular.ttf": ["latin"],
    "Poppins-Regular.ttf": ["latin", "devanagari"],
    # --- Arabic ---
    "NotoSansArabic-Regular.ttf": ["arabic"],
    "NotoSansArabic-Bold.ttf": ["arabic"],
    "NotoNaskhArabic-Regular.ttf": ["arabic"],
    "NotoNaskhArabic-Bold.ttf": ["arabic"],
    "NotoNastaliqUrdu-Regular.ttf": ["arabic"],
    "NotoKufiArabic-Regular.ttf": ["arabic"],
    "Amiri-Regular.ttf": ["arabic"],
    "Amiri-Bold.ttf": ["arabic"],
    "ScheherazadeNew-Regular.ttf": ["arabic"],
    "Lateef-Regular.ttf": ["arabic"],
    # --- Hebrew ---
    "NotoSansHebrew-Regular.ttf": ["hebrew"],
    "NotoSansHebrew-Bold.ttf": ["hebrew"],
    "NotoSerifHebrew-Regular.ttf": ["hebrew"],
    "FrankRuhlLibre[wght].ttf": ["hebrew"],
    "Rubik[wght].ttf": ["hebrew"],
    "SecularOne-Regular.ttf": ["hebrew"],
    "Heebo[wght].ttf": ["hebrew"],
    "Assistant[wght].ttf": ["hebrew"],
    "SuezOne-Regular.ttf": ["hebrew"],
    "DavidLibre-Regular.ttf": ["hebrew"],
    "Karantina-Regular.ttf": ["hebrew"],
    # --- CJK (han_kana) ---
    "NotoSansSC[wght].ttf": ["han_kana"],
    "NotoSansJP[wght].ttf": ["han_kana"],
    "NotoSansCJKsc-Regular.otf": ["han_kana"],
    "NotoSansCJKjp-Regular.otf": ["han_kana"],
    "NotoSerifCJKsc-Regular.otf": ["han_kana"],
    "HachiMaruPop-Regular.ttf": ["han_kana"],
    "KleeOne-Regular.ttf": ["han_kana"],
    "Yomogi-Regular.ttf": ["han_kana"],
    "MaShanZheng-Regular.ttf": ["han_kana"],
    "LiuJianMaoCao-Regular.ttf": ["han_kana"],
    "LongCang-Regular.ttf": ["han_kana"],
    "ZhiMangXing-Regular.ttf": ["han_kana"],
    "ShipporiMincho-Regular.ttf": ["han_kana"],
    "ZenMaruGothic-Regular.ttf": ["han_kana"],
    "ZenKurenaido-Regular.ttf": ["han_kana"],
    "ZCOOLQingKeHuangYou-Regular.ttf": ["han_kana"],
    "ZCOOLKuaiLe-Regular.ttf": ["han_kana"],
    # --- Korean ---
    "NotoSansKR[wght].ttf": ["korean"],
    "NotoSansCJKkr-Regular.otf": ["korean"],
    "NotoSerifCJKkr-Regular.otf": ["korean"],
    "NanumGothic-Regular.ttf": ["korean"],
    "NanumMyeongjo-Regular.ttf": ["korean"],
    "NanumPenScript-Regular.ttf": ["korean"],
    # --- Devanagari ---
    "NotoSansDevanagari-Regular.ttf": ["devanagari"],
    "NotoSansDevanagari-Bold.ttf": ["devanagari"],
    "NotoSerifDevanagari-Regular.ttf": ["devanagari"],
    "TiroDevanagariHindi-Regular.ttf": ["devanagari"],
    "Laila-Regular.ttf": ["devanagari"],
    "Kalam-Regular.ttf": ["devanagari", "latin"],
    # --- Bengali ---
    "NotoSansBengali-Regular.ttf": ["bengali"],
    "NotoSansBengali-Bold.ttf": ["bengali"],
    "NotoSerifBengali-Regular.ttf": ["bengali"],
    "TiroBangla-Regular.ttf": ["bengali"],
    "HindSiliguri-Regular.ttf": ["bengali"],
    "BalooDa2[wght].ttf": ["bengali"],
    "Atma-Regular.ttf": ["bengali"],
    "Galada-Regular.ttf": ["bengali"],
    "Mina-Regular.ttf": ["bengali"],
    # --- Gurmukhi ---
    "NotoSansGurmukhi-Regular.ttf": ["gurmukhi"],
    "NotoSansGurmukhi-Bold.ttf": ["gurmukhi"],
    "NotoSerifGurmukhi-Regular.ttf": ["gurmukhi"],
    "BalooPaaji2[wght].ttf": ["gurmukhi"],
    "MuktaMahee-Regular.ttf": ["gurmukhi"],
    "Langar-Regular.ttf": ["gurmukhi"],
    # --- Gujarati ---
    "NotoSansGujarati-Regular.ttf": ["gujarati"],
    "NotoSansGujarati-Bold.ttf": ["gujarati"],
    "NotoSerifGujarati-Regular.ttf": ["gujarati"],
    "HindVadodara-Regular.ttf": ["gujarati"],
    # --- Odia ---
    "NotoSansOriya-Regular.ttf": ["odia"],
    "NotoSansOriya-Bold.ttf": ["odia"],
    "BalooBhaina2[wght].ttf": ["odia"],
    # --- Tamil ---
    "NotoSansTamil-Regular.ttf": ["tamil"],
    "NotoSansTamil-Bold.ttf": ["tamil"],
    "NotoSerifTamil-Regular.ttf": ["tamil"],
    "TiroTamil-Regular.ttf": ["tamil"],
    "Kavivanar-Regular.ttf": ["tamil"],
    "Catamaran[wght].ttf": ["tamil"],
    "HindMadurai-Regular.ttf": ["tamil"],
    "MuktaMalar-Regular.ttf": ["tamil"],
    "BalooThambi2[wght].ttf": ["tamil"],
    # --- Telugu ---
    "NotoSansTelugu-Regular.ttf": ["telugu"],
    "NotoSansTelugu-Bold.ttf": ["telugu"],
    "NotoSerifTelugu-Regular.ttf": ["telugu"],
    "TiroTelugu-Regular.ttf": ["telugu"],
    "LakkiReddy-Regular.ttf": ["telugu"],
    "HindGuntur-Regular.ttf": ["telugu"],
    "Mandali-Regular.ttf": ["telugu"],
    "Ramabhadra-Regular.ttf": ["telugu"],
    "BalooTammudu2[wght].ttf": ["telugu"],
    "Peddana-Regular.ttf": ["telugu"],
    # --- Kannada ---
    "NotoSansKannada-Regular.ttf": ["kannada"],
    "NotoSansKannada-Bold.ttf": ["kannada"],
    "NotoSerifKannada-Regular.ttf": ["kannada"],
    "TiroKannada-Regular.ttf": ["kannada"],
    "AkayaKanadaka-Regular.ttf": ["kannada"],
    "Benne-Regular.ttf": ["kannada"],
    "HindMysuru-Regular.ttf": ["kannada"],
    "BalooTamma2[wght].ttf": ["kannada"],
    # --- Malayalam ---
    "NotoSansMalayalam-Regular.ttf": ["malayalam"],
    "NotoSansMalayalam-Bold.ttf": ["malayalam"],
    "NotoSerifMalayalam-Regular.ttf": ["malayalam"],
    "Chilanka-Regular.ttf": ["malayalam"],
    "Manjari-Regular.ttf": ["malayalam"],
    "Gayathri-Regular.ttf": ["malayalam"],
    "BalooChettan2[wght].ttf": ["malayalam"],
    # --- Sinhala ---
    "NotoSansSinhala-Regular.ttf": ["sinhala"],
    "NotoSansSinhala-Bold.ttf": ["sinhala"],
    "NotoSerifSinhala-Regular.ttf": ["sinhala"],
    "AbhayaLibre-Regular.ttf": ["sinhala"],
    "Yaldevi[wght].ttf": ["sinhala"],
    "GemunuLibre[wght].ttf": ["sinhala"],
    # --- Thai ---
    "NotoSansThai-Regular.ttf": ["thai"],
    "NotoSansThai-Bold.ttf": ["thai"],
    "NotoSerifThai-Regular.ttf": ["thai"],
    "Kanit-Regular.ttf": ["thai"],
    "Sarabun-Regular.ttf": ["thai"],
    "Prompt-Regular.ttf": ["thai"],
    # --- Lao ---
    "NotoSansLao-Regular.ttf": ["lao"],
    "NotoSansLao-Bold.ttf": ["lao"],
    "NotoSerifLao-Regular.ttf": ["lao"],
    "PhetsarathOT-Regular.ttf": ["lao"],
    # --- Burmese ---
    "NotoSansMyanmar-Regular.ttf": ["burmese"],
    "NotoSansMyanmar-Bold.ttf": ["burmese"],
    "NotoSerifMyanmar-Regular.ttf": ["burmese"],
    "Padauk-Regular.ttf": ["burmese"],
    "Padauk-Bold.ttf": ["burmese"],
    # --- Khmer ---
    "NotoSansKhmer-Regular.ttf": ["khmer"],
    "NotoSansKhmer-Bold.ttf": ["khmer"],
    "NotoSerifKhmer-Regular.ttf": ["khmer"],
    "Battambang-Regular.ttf": ["khmer"],
    "Hanuman[wght].ttf": ["khmer"],
    "Moul-Regular.ttf": ["khmer"],
    "Siemreap.ttf": ["khmer"],
    "Koulen-Regular.ttf": ["khmer"],
    "Fasthand-Regular.ttf": ["khmer"],
    "Freehand-Regular.ttf": ["khmer"],
    "Dangrek-Regular.ttf": ["khmer"],
    "Bayon-Regular.ttf": ["khmer"],
    "Content-Regular.ttf": ["khmer"],
    # --- Armenian ---
    "NotoSansArmenian-Regular.ttf": ["armenian"],
    "NotoSansArmenian-Bold.ttf": ["armenian"],
    "NotoSerifArmenian-Regular.ttf": ["armenian"],
    # --- Georgian ---
    "NotoSansGeorgian-Regular.ttf": ["georgian"],
    "NotoSansGeorgian-Bold.ttf": ["georgian"],
    "NotoSerifGeorgian-Regular.ttf": ["georgian"],
    # --- Ethiopic ---
    "NotoSansEthiopic-Regular.ttf": ["ethiopic"],
    "NotoSansEthiopic-Bold.ttf": ["ethiopic"],
    "NotoSerifEthiopic-Regular.ttf": ["ethiopic"],
    "AbyssinicaSIL-Regular.ttf": ["ethiopic"],
    # --- Tibetan ---
    "NotoSansTibetan-Regular.ttf": ["tibetan"],
    "NotoSansTibetan-Bold.ttf": ["tibetan"],
    "NotoSerifTibetan-Regular.ttf": ["tibetan"],
    "Jomolhari-Regular.ttf": ["tibetan"],
}

# Build reverse mapping: script → set of font filenames
_SCRIPT_TO_FONTS: dict[str, set[str]] = {}
for _font, _scripts in _FONT_TO_SCRIPTS.items():
    for _script in _scripts:
        _SCRIPT_TO_FONTS.setdefault(_script, set()).add(_font)

# Keywords for font style weighting
_HANDWRITING_KEYWORDS = [
    "caveat", "dancing", "indie", "patrick", "shadow", "kalam",
    "nanumpen", "chilanka", "handwrit", "cursive",
]
_DISPLAY_KEYWORDS = [
    "permanent", "amatic", "lobster", "pacifico", "special", "display",
]


def find_system_fonts() -> list[str]:
    """Find all .ttf/.otf font files on the system."""
    fonts = []
    seen = set()
    for d in _FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf")):
                    path = os.path.join(root, f)
                    if path not in seen:
                        seen.add(path)
                        fonts.append(path)
    return fonts


def find_fonts_for_script(script: str) -> list[str]:
    """Find fonts that can render a given script.

    Uses explicit font-to-script mapping. Only returns fonts that are
    known to correctly render this script — no pattern matching.
    """
    allowed_names = _SCRIPT_TO_FONTS.get(script, set())
    if not allowed_names:
        return []

    all_fonts = find_system_fonts()
    return [f for f in all_fonts if Path(f).name in allowed_names]


def build_weighted_font_list(
    fonts: list[str],
    sample_text: str,
) -> list[str]:
    """Build a weighted font list for training diversity.

    Weights: 70% clean/regular, 20% handwriting, 10% display.
    Validates each font can actually render the sample text.
    """
    valid = [f for f in fonts if font_can_render(f, sample_text)]
    if not valid:
        return []

    weighted = []
    for f in valid:
        name = Path(f).name.lower()
        if any(k in name for k in _HANDWRITING_KEYWORDS):
            weighted.extend([f] * 2)  # 20% weight
        elif any(k in name for k in _DISPLAY_KEYWORDS):
            weighted.extend([f] * 1)  # 10% weight
        else:
            weighted.extend([f] * 7)  # 70% weight

    return weighted
