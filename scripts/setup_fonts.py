#!/usr/bin/env python3
"""Download fonts for all 26 scripts. Fast, parallel, reliable."""

import os
import sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FONT_DIR = Path(__file__).parent.parent / "training_data" / "fonts"

NOTO = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts"
GFONTS = "https://github.com/google/fonts/raw/main/ofl"
CJK_RAW = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF"
CJK_SERIF_RAW = "https://github.com/notofonts/noto-cjk/raw/main/Serif/OTF"

FONTS = {
    # (url, filename)
    # Latin / Cyrillic / Greek
    (f"{NOTO}/NotoSans/full/ttf/NotoSans-Regular.ttf", "NotoSans-Regular.ttf"),
    (f"{NOTO}/NotoSans/full/ttf/NotoSans-Bold.ttf", "NotoSans-Bold.ttf"),
    (f"{NOTO}/NotoSans/full/ttf/NotoSans-Italic.ttf", "NotoSans-Italic.ttf"),
    (f"{NOTO}/NotoSans/full/ttf/NotoSans-Light.ttf", "NotoSans-Light.ttf"),
    (f"{NOTO}/NotoSansMono/full/ttf/NotoSansMono-Regular.ttf", "NotoSansMono-Regular.ttf"),
    # Serif (different repo path)
    ("https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSerif/NotoSerif-Regular.ttf", "NotoSerif-Regular.ttf"),
    ("https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSerif/NotoSerif-Bold.ttf", "NotoSerif-Bold.ttf"),
    ("https://raw.githubusercontent.com/googlefonts/noto-fonts/main/hinted/ttf/NotoSerif/NotoSerif-Italic.ttf", "NotoSerif-Italic.ttf"),
    # Arabic
    (f"{NOTO}/NotoSansArabic/full/ttf/NotoSansArabic-Regular.ttf", "NotoSansArabic-Regular.ttf"),
    (f"{NOTO}/NotoSansArabic/full/ttf/NotoSansArabic-Bold.ttf", "NotoSansArabic-Bold.ttf"),
    (f"{NOTO}/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Regular.ttf", "NotoNaskhArabic-Regular.ttf"),
    (f"{NOTO}/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Bold.ttf", "NotoNaskhArabic-Bold.ttf"),
    (f"{NOTO}/NotoNastaliqUrdu/full/ttf/NotoNastaliqUrdu-Regular.ttf", "NotoNastaliqUrdu-Regular.ttf"),
    (f"{NOTO}/NotoKufiArabic/full/ttf/NotoKufiArabic-Regular.ttf", "NotoKufiArabic-Regular.ttf"),
    (f"{GFONTS}/amiri/Amiri-Regular.ttf", "Amiri-Regular.ttf"),
    (f"{GFONTS}/amiri/Amiri-Bold.ttf", "Amiri-Bold.ttf"),
    (f"{GFONTS}/scheherazadenew/ScheherazadeNew-Regular.ttf", "ScheherazadeNew-Regular.ttf"),
    (f"{GFONTS}/lateef/Lateef-Regular.ttf", "Lateef-Regular.ttf"),
    # Hebrew
    (f"{NOTO}/NotoSansHebrew/full/ttf/NotoSansHebrew-Regular.ttf", "NotoSansHebrew-Regular.ttf"),
    (f"{NOTO}/NotoSansHebrew/full/ttf/NotoSansHebrew-Bold.ttf", "NotoSansHebrew-Bold.ttf"),
    (f"{NOTO}/NotoSerifHebrew/full/ttf/NotoSerifHebrew-Regular.ttf", "NotoSerifHebrew-Regular.ttf"),
    (f"{GFONTS}/frankruhllibre/FrankRuhlLibre%5Bwght%5D.ttf", "FrankRuhlLibre[wght].ttf"),
    (f"{GFONTS}/rubik/Rubik%5Bwght%5D.ttf", "Rubik[wght].ttf"),
    (f"{GFONTS}/secularone/SecularOne-Regular.ttf", "SecularOne-Regular.ttf"),
    # CJK (Google Fonts variable weight versions — smaller, reliable)
    (f"{GFONTS}/notosanssc/NotoSansSC%5Bwght%5D.ttf", "NotoSansSC[wght].ttf"),
    (f"{GFONTS}/notosansjp/NotoSansJP%5Bwght%5D.ttf", "NotoSansJP[wght].ttf"),
    # CJK (raw repo — OTF versions)
    (f"{CJK_RAW}/SimplifiedChinese/NotoSansCJKsc-Regular.otf", "NotoSansCJKsc-Regular.otf"),
    (f"{CJK_RAW}/Japanese/NotoSansCJKjp-Regular.otf", "NotoSansCJKjp-Regular.otf"),
    (f"{CJK_SERIF_RAW}/SimplifiedChinese/NotoSerifCJKsc-Regular.otf", "NotoSerifCJKsc-Regular.otf"),
    # Korean
    (f"{GFONTS}/notosanskr/NotoSansKR%5Bwght%5D.ttf", "NotoSansKR[wght].ttf"),
    (f"{CJK_RAW}/Korean/NotoSansCJKkr-Regular.otf", "NotoSansCJKkr-Regular.otf"),
    (f"{CJK_SERIF_RAW}/Korean/NotoSerifCJKkr-Regular.otf", "NotoSerifCJKkr-Regular.otf"),
    (f"{GFONTS}/nanumgothic/NanumGothic-Regular.ttf", "NanumGothic-Regular.ttf"),
    (f"{GFONTS}/nanummyeongjo/NanumMyeongjo-Regular.ttf", "NanumMyeongjo-Regular.ttf"),
    (f"{GFONTS}/nanumpenscript/NanumPenScript-Regular.ttf", "NanumPenScript-Regular.ttf"),
    # Devanagari
    (f"{NOTO}/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Regular.ttf", "NotoSansDevanagari-Regular.ttf"),
    (f"{NOTO}/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Bold.ttf", "NotoSansDevanagari-Bold.ttf"),
    (f"{NOTO}/NotoSerifDevanagari/full/ttf/NotoSerifDevanagari-Regular.ttf", "NotoSerifDevanagari-Regular.ttf"),
    (f"{GFONTS}/poppins/Poppins-Regular.ttf", "Poppins-Regular.ttf"),
    (f"{GFONTS}/tirodevanagarihindi/TiroDevanagariHindi-Regular.ttf", "TiroDevanagariHindi-Regular.ttf"),
    (f"{GFONTS}/laila/Laila-Regular.ttf", "Laila-Regular.ttf"),
    (f"{GFONTS}/kalam/Kalam-Regular.ttf", "Kalam-Regular.ttf"),
    # Bengali
    (f"{NOTO}/NotoSansBengali/full/ttf/NotoSansBengali-Regular.ttf", "NotoSansBengali-Regular.ttf"),
    (f"{NOTO}/NotoSansBengali/full/ttf/NotoSansBengali-Bold.ttf", "NotoSansBengali-Bold.ttf"),
    (f"{NOTO}/NotoSerifBengali/full/ttf/NotoSerifBengali-Regular.ttf", "NotoSerifBengali-Regular.ttf"),
    (f"{GFONTS}/tirobangla/TiroBangla-Regular.ttf", "TiroBangla-Regular.ttf"),
    (f"{GFONTS}/hindsiliguri/HindSiliguri-Regular.ttf", "HindSiliguri-Regular.ttf"),
    # Gurmukhi
    (f"{NOTO}/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Regular.ttf", "NotoSansGurmukhi-Regular.ttf"),
    (f"{NOTO}/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Bold.ttf", "NotoSansGurmukhi-Bold.ttf"),
    (f"{NOTO}/NotoSerifGurmukhi/full/ttf/NotoSerifGurmukhi-Regular.ttf", "NotoSerifGurmukhi-Regular.ttf"),
    # Gujarati
    (f"{NOTO}/NotoSansGujarati/full/ttf/NotoSansGujarati-Regular.ttf", "NotoSansGujarati-Regular.ttf"),
    (f"{NOTO}/NotoSansGujarati/full/ttf/NotoSansGujarati-Bold.ttf", "NotoSansGujarati-Bold.ttf"),
    (f"{NOTO}/NotoSerifGujarati/full/ttf/NotoSerifGujarati-Regular.ttf", "NotoSerifGujarati-Regular.ttf"),
    (f"{GFONTS}/hindvadodara/HindVadodara-Regular.ttf", "HindVadodara-Regular.ttf"),
    # Tamil
    (f"{NOTO}/NotoSansTamil/full/ttf/NotoSansTamil-Regular.ttf", "NotoSansTamil-Regular.ttf"),
    (f"{NOTO}/NotoSansTamil/full/ttf/NotoSansTamil-Bold.ttf", "NotoSansTamil-Bold.ttf"),
    (f"{NOTO}/NotoSerifTamil/full/ttf/NotoSerifTamil-Regular.ttf", "NotoSerifTamil-Regular.ttf"),
    (f"{GFONTS}/tirotamil/TiroTamil-Regular.ttf", "TiroTamil-Regular.ttf"),
    # Telugu
    (f"{NOTO}/NotoSansTelugu/full/ttf/NotoSansTelugu-Regular.ttf", "NotoSansTelugu-Regular.ttf"),
    (f"{NOTO}/NotoSansTelugu/full/ttf/NotoSansTelugu-Bold.ttf", "NotoSansTelugu-Bold.ttf"),
    (f"{NOTO}/NotoSerifTelugu/full/ttf/NotoSerifTelugu-Regular.ttf", "NotoSerifTelugu-Regular.ttf"),
    (f"{GFONTS}/tirotelugu/TiroTelugu-Regular.ttf", "TiroTelugu-Regular.ttf"),
    # Kannada
    (f"{NOTO}/NotoSansKannada/full/ttf/NotoSansKannada-Regular.ttf", "NotoSansKannada-Regular.ttf"),
    (f"{NOTO}/NotoSansKannada/full/ttf/NotoSansKannada-Bold.ttf", "NotoSansKannada-Bold.ttf"),
    (f"{NOTO}/NotoSerifKannada/full/ttf/NotoSerifKannada-Regular.ttf", "NotoSerifKannada-Regular.ttf"),
    (f"{GFONTS}/tirokannada/TiroKannada-Regular.ttf", "TiroKannada-Regular.ttf"),
    # Malayalam
    (f"{NOTO}/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Regular.ttf", "NotoSansMalayalam-Regular.ttf"),
    (f"{NOTO}/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Bold.ttf", "NotoSansMalayalam-Bold.ttf"),
    (f"{NOTO}/NotoSerifMalayalam/full/ttf/NotoSerifMalayalam-Regular.ttf", "NotoSerifMalayalam-Regular.ttf"),
    (f"{GFONTS}/chilanka/Chilanka-Regular.ttf", "Chilanka-Regular.ttf"),
    # Thai
    (f"{NOTO}/NotoSansThai/full/ttf/NotoSansThai-Regular.ttf", "NotoSansThai-Regular.ttf"),
    (f"{NOTO}/NotoSansThai/full/ttf/NotoSansThai-Bold.ttf", "NotoSansThai-Bold.ttf"),
    (f"{NOTO}/NotoSerifThai/full/ttf/NotoSerifThai-Regular.ttf", "NotoSerifThai-Regular.ttf"),
    (f"{GFONTS}/kanit/Kanit-Regular.ttf", "Kanit-Regular.ttf"),
    (f"{GFONTS}/sarabun/Sarabun-Regular.ttf", "Sarabun-Regular.ttf"),
    (f"{GFONTS}/prompt/Prompt-Regular.ttf", "Prompt-Regular.ttf"),
    # Lao
    (f"{NOTO}/NotoSansLao/full/ttf/NotoSansLao-Regular.ttf", "NotoSansLao-Regular.ttf"),
    (f"{NOTO}/NotoSansLao/full/ttf/NotoSansLao-Bold.ttf", "NotoSansLao-Bold.ttf"),
    (f"{NOTO}/NotoSerifLao/full/ttf/NotoSerifLao-Regular.ttf", "NotoSerifLao-Regular.ttf"),
    (f"{GFONTS}/phetsarathot/PhetsarathOT-Regular.ttf", "PhetsarathOT-Regular.ttf"),
    # Odia
    (f"{NOTO}/NotoSansOriya/full/ttf/NotoSansOriya-Regular.ttf", "NotoSansOriya-Regular.ttf"),
    (f"{NOTO}/NotoSansOriya/full/ttf/NotoSansOriya-Bold.ttf", "NotoSansOriya-Bold.ttf"),
    # Sinhala
    (f"{NOTO}/NotoSansSinhala/full/ttf/NotoSansSinhala-Regular.ttf", "NotoSansSinhala-Regular.ttf"),
    (f"{NOTO}/NotoSansSinhala/full/ttf/NotoSansSinhala-Bold.ttf", "NotoSansSinhala-Bold.ttf"),
    (f"{NOTO}/NotoSerifSinhala/full/ttf/NotoSerifSinhala-Regular.ttf", "NotoSerifSinhala-Regular.ttf"),
    # Burmese / Myanmar
    (f"{NOTO}/NotoSansMyanmar/full/ttf/NotoSansMyanmar-Regular.ttf", "NotoSansMyanmar-Regular.ttf"),
    (f"{NOTO}/NotoSansMyanmar/full/ttf/NotoSansMyanmar-Bold.ttf", "NotoSansMyanmar-Bold.ttf"),
    (f"{NOTO}/NotoSerifMyanmar/full/ttf/NotoSerifMyanmar-Regular.ttf", "NotoSerifMyanmar-Regular.ttf"),
    (f"{GFONTS}/padauk/Padauk-Regular.ttf", "Padauk-Regular.ttf"),
    (f"{GFONTS}/padauk/Padauk-Bold.ttf", "Padauk-Bold.ttf"),
    # Khmer
    (f"{NOTO}/NotoSansKhmer/full/ttf/NotoSansKhmer-Regular.ttf", "NotoSansKhmer-Regular.ttf"),
    (f"{NOTO}/NotoSansKhmer/full/ttf/NotoSansKhmer-Bold.ttf", "NotoSansKhmer-Bold.ttf"),
    (f"{NOTO}/NotoSerifKhmer/full/ttf/NotoSerifKhmer-Regular.ttf", "NotoSerifKhmer-Regular.ttf"),
    (f"{GFONTS}/battambang/Battambang-Regular.ttf", "Battambang-Regular.ttf"),
    # Armenian
    (f"{NOTO}/NotoSansArmenian/full/ttf/NotoSansArmenian-Regular.ttf", "NotoSansArmenian-Regular.ttf"),
    (f"{NOTO}/NotoSansArmenian/full/ttf/NotoSansArmenian-Bold.ttf", "NotoSansArmenian-Bold.ttf"),
    (f"{NOTO}/NotoSerifArmenian/full/ttf/NotoSerifArmenian-Regular.ttf", "NotoSerifArmenian-Regular.ttf"),
    # Georgian
    (f"{NOTO}/NotoSansGeorgian/full/ttf/NotoSansGeorgian-Regular.ttf", "NotoSansGeorgian-Regular.ttf"),
    (f"{NOTO}/NotoSansGeorgian/full/ttf/NotoSansGeorgian-Bold.ttf", "NotoSansGeorgian-Bold.ttf"),
    (f"{NOTO}/NotoSerifGeorgian/full/ttf/NotoSerifGeorgian-Regular.ttf", "NotoSerifGeorgian-Regular.ttf"),
    # Ethiopic / Amharic
    (f"{NOTO}/NotoSansEthiopic/full/ttf/NotoSansEthiopic-Regular.ttf", "NotoSansEthiopic-Regular.ttf"),
    (f"{NOTO}/NotoSansEthiopic/full/ttf/NotoSansEthiopic-Bold.ttf", "NotoSansEthiopic-Bold.ttf"),
    (f"{NOTO}/NotoSerifEthiopic/full/ttf/NotoSerifEthiopic-Regular.ttf", "NotoSerifEthiopic-Regular.ttf"),
    # Tibetan
    (f"{NOTO}/NotoSansTibetan/full/ttf/NotoSansTibetan-Regular.ttf", "NotoSansTibetan-Regular.ttf"),
    (f"{NOTO}/NotoSansTibetan/full/ttf/NotoSansTibetan-Bold.ttf", "NotoSansTibetan-Bold.ttf"),
    (f"{NOTO}/NotoSerifTibetan/full/ttf/NotoSerifTibetan-Regular.ttf", "NotoSerifTibetan-Regular.ttf"),
    # Handwriting / Display (Latin)
    (f"{GFONTS}/caveat/Caveat%5Bwght%5D.ttf", "Caveat[wght].ttf"),
    (f"{GFONTS}/dancingscript/DancingScript%5Bwght%5D.ttf", "DancingScript[wght].ttf"),
    (f"{GFONTS}/indieflower/IndieFlower-Regular.ttf", "IndieFlower-Regular.ttf"),
    (f"{GFONTS}/patrickhand/PatrickHand-Regular.ttf", "PatrickHand-Regular.ttf"),
    (f"{GFONTS}/shadowsintolight/ShadowsIntoLight.ttf", "ShadowsIntoLight.ttf"),
    (f"{GFONTS}/permanentmarker/PermanentMarker-Regular.ttf", "PermanentMarker-Regular.ttf"),
    (f"{GFONTS}/amaticsc/AmaticSC-Regular.ttf", "AmaticSC-Regular.ttf"),
    (f"{GFONTS}/lobster/Lobster-Regular.ttf", "Lobster-Regular.ttf"),
    (f"{GFONTS}/pacifico/Pacifico-Regular.ttf", "Pacifico-Regular.ttf"),
    (f"{GFONTS}/comicneue/ComicNeue-Regular.ttf", "ComicNeue-Regular.ttf"),
    (f"{GFONTS}/specialelite/SpecialElite-Regular.ttf", "SpecialElite-Regular.ttf"),
}


def download(url_name):
    url, name = url_name
    dest = FONT_DIR / name
    if dest.exists() and dest.stat().st_size > 1000:
        return f"  SKIP: {name}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LipiOCR/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 1000:
            return f"  FAIL: {name} (too small)"
        dest.write_bytes(data)
        return f"  OK:   {name} ({len(data)//1024}KB)"
    except Exception as e:
        return f"  FAIL: {name} ({e})"


def main():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(FONTS)} fonts to {FONT_DIR}/\n")

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(download, f): f for f in FONTS}
        ok = fail = skip = 0
        for future in as_completed(futures):
            result = future.result()
            print(result)
            if "OK:" in result:
                ok += 1
            elif "FAIL:" in result:
                fail += 1
            else:
                skip += 1

    total = len(list(FONT_DIR.glob("*.ttf"))) + len(list(FONT_DIR.glob("*.otf")))
    print(f"\nDone: {ok} downloaded, {skip} skipped, {fail} failed. {total} fonts total.")


if __name__ == "__main__":
    main()
