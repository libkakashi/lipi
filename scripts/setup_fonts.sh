#!/bin/bash
# Download diverse fonts for all 18 scripts.
# Includes: sans-serif, serif, handwriting/cursive, display, monospace
# Run once on any machine before training.

set -e

FONT_DIR="$(dirname "$0")/../training_data/fonts"
mkdir -p "$FONT_DIR"
cd "$FONT_DIR"

echo "Downloading fonts to $FONT_DIR ..."

NOTO="https://github.com/notofonts/notofonts.github.io/raw/main/fonts"

dl() {
    local url="$1"
    local name="$(basename "$url")"
    if [ ! -f "$name" ]; then
        echo "  $name"
        curl -sL -o "$name" "$url" || wget -q -O "$name" "$url" || echo "    FAILED: $name"
    fi
}

echo ""
echo "=== Latin / Cyrillic / Greek ==="
# Sans
dl "$NOTO/NotoSans/full/ttf/NotoSans-Regular.ttf"
dl "$NOTO/NotoSans/full/ttf/NotoSans-Bold.ttf"
dl "$NOTO/NotoSans/full/ttf/NotoSans-Italic.ttf"
dl "$NOTO/NotoSans/full/ttf/NotoSans-Light.ttf"
# Serif
dl "$NOTO/NotoSerif/full/ttf/NotoSerif-Regular.ttf"
dl "$NOTO/NotoSerif/full/ttf/NotoSerif-Bold.ttf"
dl "$NOTO/NotoSerif/full/ttf/NotoSerif-Italic.ttf"
# Mono
dl "$NOTO/NotoSansMono/full/ttf/NotoSansMono-Regular.ttf"
# Google Fonts - handwriting/cursive/display
dl "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/permanentmarker/PermanentMarker-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/amaticsc/AmaticSC-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/lobster/Lobster-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/specialelite/SpecialElite-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/courierprimetarget/CourierPrime-Regular.ttf"

echo ""
echo "=== Arabic ==="
dl "$NOTO/NotoSansArabic/full/ttf/NotoSansArabic-Regular.ttf"
dl "$NOTO/NotoSansArabic/full/ttf/NotoSansArabic-Bold.ttf"
dl "$NOTO/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Regular.ttf"
dl "$NOTO/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Bold.ttf"
dl "$NOTO/NotoNastaliqUrdu/full/ttf/NotoNastaliqUrdu-Regular.ttf"
dl "$NOTO/NotoKufiArabic/full/ttf/NotoKufiArabic-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Bold.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/scheherazadenew/ScheherazadeNew-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/lateef/Lateef-Regular.ttf"

echo ""
echo "=== Hebrew ==="
dl "$NOTO/NotoSansHebrew/full/ttf/NotoSansHebrew-Regular.ttf"
dl "$NOTO/NotoSansHebrew/full/ttf/NotoSansHebrew-Bold.ttf"
dl "$NOTO/NotoSerifHebrew/full/ttf/NotoSerifHebrew-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/frankruhllibre/FrankRuhlLibre%5Bwght%5D.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/rubik/Rubik%5Bwght%5D.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/secular_one/SecularOne-Regular.ttf"

echo ""
echo "=== CJK ==="
dl "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/01_NotoSansCJKsc-Regular.otf"
dl "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/01_NotoSansCJKsc-Bold.otf"
dl "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/03_NotoSansCJKjp-Regular.otf"
dl "https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/01_NotoSerifCJKsc-Regular.otf"
dl "https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/03_NotoSerifCJKjp-Regular.otf"

echo ""
echo "=== Korean ==="
dl "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/05_NotoSansCJKkr-Regular.otf"
dl "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/05_NotoSansCJKkr-Bold.otf"
dl "https://github.com/notofonts/noto-cjk/releases/download/Serif2.003/05_NotoSerifCJKkr-Regular.otf"
dl "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/nanumpenscript/NanumPenScript-Regular.ttf"

echo ""
echo "=== Devanagari ==="
dl "$NOTO/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Regular.ttf"
dl "$NOTO/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Bold.ttf"
dl "$NOTO/NotoSerifDevanagari/full/ttf/NotoSerifDevanagari-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/tirodevanagarihindi/TiroDevanagariHindi-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/laila/Laila-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf"

echo ""
echo "=== Bengali ==="
dl "$NOTO/NotoSansBengali/full/ttf/NotoSansBengali-Regular.ttf"
dl "$NOTO/NotoSansBengali/full/ttf/NotoSansBengali-Bold.ttf"
dl "$NOTO/NotoSerifBengali/full/ttf/NotoSerifBengali-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/tirobangla/TiroBangla-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/hindsiliguri/HindSiliguri-Regular.ttf"

echo ""
echo "=== Gurmukhi ==="
dl "$NOTO/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Regular.ttf"
dl "$NOTO/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Bold.ttf"
dl "$NOTO/NotoSerifGurmukhi/full/ttf/NotoSerifGurmukhi-Regular.ttf"

echo ""
echo "=== Gujarati ==="
dl "$NOTO/NotoSansGujarati/full/ttf/NotoSansGujarati-Regular.ttf"
dl "$NOTO/NotoSansGujarati/full/ttf/NotoSansGujarati-Bold.ttf"
dl "$NOTO/NotoSerifGujarati/full/ttf/NotoSerifGujarati-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/hindvadodara/HindVadodara-Regular.ttf"

echo ""
echo "=== Tamil ==="
dl "$NOTO/NotoSansTamil/full/ttf/NotoSansTamil-Regular.ttf"
dl "$NOTO/NotoSansTamil/full/ttf/NotoSansTamil-Bold.ttf"
dl "$NOTO/NotoSerifTamil/full/ttf/NotoSerifTamil-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/tirotamil/TiroTamil-Regular.ttf"

echo ""
echo "=== Telugu ==="
dl "$NOTO/NotoSansTelugu/full/ttf/NotoSansTelugu-Regular.ttf"
dl "$NOTO/NotoSansTelugu/full/ttf/NotoSansTelugu-Bold.ttf"
dl "$NOTO/NotoSerifTelugu/full/ttf/NotoSerifTelugu-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/tirotelugu/TiroTelugu-Regular.ttf"

echo ""
echo "=== Kannada ==="
dl "$NOTO/NotoSansKannada/full/ttf/NotoSansKannada-Regular.ttf"
dl "$NOTO/NotoSansKannada/full/ttf/NotoSansKannada-Bold.ttf"
dl "$NOTO/NotoSerifKannada/full/ttf/NotoSerifKannada-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/tirokannada/TiroKannada-Regular.ttf"

echo ""
echo "=== Malayalam ==="
dl "$NOTO/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Regular.ttf"
dl "$NOTO/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Bold.ttf"
dl "$NOTO/NotoSerifMalayalam/full/ttf/NotoSerifMalayalam-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/chilanka/Chilanka-Regular.ttf"

echo ""
echo "=== Thai ==="
dl "$NOTO/NotoSansThai/full/ttf/NotoSansThai-Regular.ttf"
dl "$NOTO/NotoSansThai/full/ttf/NotoSansThai-Bold.ttf"
dl "$NOTO/NotoSerifThai/full/ttf/NotoSerifThai-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/kanit/Kanit-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/prompt/Prompt-Regular.ttf"

echo ""
echo "=== Lao ==="
dl "$NOTO/NotoSansLao/full/ttf/NotoSansLao-Regular.ttf"
dl "$NOTO/NotoSansLao/full/ttf/NotoSansLao-Bold.ttf"
dl "$NOTO/NotoSerifLao/full/ttf/NotoSerifLao-Regular.ttf"
dl "https://github.com/google/fonts/raw/main/ofl/phetsarathot/PhetsarathOT-Regular.ttf"

echo ""
echo "Done."
ls -1 "$FONT_DIR"/*.ttf "$FONT_DIR"/*.otf 2>/dev/null | wc -l | xargs -I{} echo "{} font files downloaded"
echo "Fonts in: $FONT_DIR"
