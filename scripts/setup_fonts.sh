#!/bin/bash
# Download diverse fonts for all 18 scripts (parallel).
set -e

FONT_DIR="$(dirname "$0")/../training_data/fonts"
mkdir -p "$FONT_DIR"
cd "$FONT_DIR"

NOTO="https://github.com/notofonts/notofonts.github.io/raw/main/fonts"
GFONTS="https://github.com/google/fonts/raw/main/ofl"
CJK_SANS="https://github.com/notofonts/noto-cjk/releases/download/Sans2.004"
CJK_SERIF="https://github.com/notofonts/noto-cjk/releases/download/Serif2.003"

# Build URL list
URLS=(
# Latin/Cyrillic/Greek — sans, serif, mono, handwriting, display
"$NOTO/NotoSans/full/ttf/NotoSans-Regular.ttf"
"$NOTO/NotoSans/full/ttf/NotoSans-Bold.ttf"
"$NOTO/NotoSans/full/ttf/NotoSans-Italic.ttf"
"$NOTO/NotoSans/full/ttf/NotoSans-Light.ttf"
"$NOTO/NotoSerif/full/ttf/NotoSerif-Regular.ttf"
"$NOTO/NotoSerif/full/ttf/NotoSerif-Bold.ttf"
"$NOTO/NotoSerif/full/ttf/NotoSerif-Italic.ttf"
"$NOTO/NotoSansMono/full/ttf/NotoSansMono-Regular.ttf"
"$GFONTS/caveat/Caveat%5Bwght%5D.ttf"
"$GFONTS/dancingscript/DancingScript%5Bwght%5D.ttf"
"$GFONTS/indieflower/IndieFlower-Regular.ttf"
"$GFONTS/patrickhand/PatrickHand-Regular.ttf"
"$GFONTS/shadowsintolight/ShadowsIntoLight.ttf"
"$GFONTS/permanentmarker/PermanentMarker-Regular.ttf"
"$GFONTS/amaticsc/AmaticSC-Regular.ttf"
"$GFONTS/lobster/Lobster-Regular.ttf"
"$GFONTS/pacifico/Pacifico-Regular.ttf"
"$GFONTS/comicneue/ComicNeue-Regular.ttf"
"$GFONTS/specialelite/SpecialElite-Regular.ttf"
# Arabic
"$NOTO/NotoSansArabic/full/ttf/NotoSansArabic-Regular.ttf"
"$NOTO/NotoSansArabic/full/ttf/NotoSansArabic-Bold.ttf"
"$NOTO/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Regular.ttf"
"$NOTO/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Bold.ttf"
"$NOTO/NotoNastaliqUrdu/full/ttf/NotoNastaliqUrdu-Regular.ttf"
"$NOTO/NotoKufiArabic/full/ttf/NotoKufiArabic-Regular.ttf"
"$GFONTS/amiri/Amiri-Regular.ttf"
"$GFONTS/amiri/Amiri-Bold.ttf"
"$GFONTS/scheherazadenew/ScheherazadeNew-Regular.ttf"
"$GFONTS/lateef/Lateef-Regular.ttf"
# Hebrew
"$NOTO/NotoSansHebrew/full/ttf/NotoSansHebrew-Regular.ttf"
"$NOTO/NotoSansHebrew/full/ttf/NotoSansHebrew-Bold.ttf"
"$NOTO/NotoSerifHebrew/full/ttf/NotoSerifHebrew-Regular.ttf"
"$GFONTS/frankruhllibre/FrankRuhlLibre%5Bwght%5D.ttf"
"$GFONTS/rubik/Rubik%5Bwght%5D.ttf"
"$GFONTS/secularone/SecularOne-Regular.ttf"
# CJK
"$CJK_SANS/01_NotoSansCJKsc-Regular.otf"
"$CJK_SANS/01_NotoSansCJKsc-Bold.otf"
"$CJK_SANS/03_NotoSansCJKjp-Regular.otf"
"$CJK_SERIF/01_NotoSerifCJKsc-Regular.otf"
"$CJK_SERIF/03_NotoSerifCJKjp-Regular.otf"
# Korean
"$CJK_SANS/05_NotoSansCJKkr-Regular.otf"
"$CJK_SANS/05_NotoSansCJKkr-Bold.otf"
"$CJK_SERIF/05_NotoSerifCJKkr-Regular.otf"
"$GFONTS/nanumgothic/NanumGothic-Regular.ttf"
"$GFONTS/nanummyeongjo/NanumMyeongjo-Regular.ttf"
"$GFONTS/nanumpenscript/NanumPenScript-Regular.ttf"
# Devanagari
"$NOTO/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Regular.ttf"
"$NOTO/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Bold.ttf"
"$NOTO/NotoSerifDevanagari/full/ttf/NotoSerifDevanagari-Regular.ttf"
"$GFONTS/poppins/Poppins-Regular.ttf"
"$GFONTS/tirodevanagarihindi/TiroDevanagariHindi-Regular.ttf"
"$GFONTS/laila/Laila-Regular.ttf"
"$GFONTS/kalam/Kalam-Regular.ttf"
# Bengali
"$NOTO/NotoSansBengali/full/ttf/NotoSansBengali-Regular.ttf"
"$NOTO/NotoSansBengali/full/ttf/NotoSansBengali-Bold.ttf"
"$NOTO/NotoSerifBengali/full/ttf/NotoSerifBengali-Regular.ttf"
"$GFONTS/tirobangla/TiroBangla-Regular.ttf"
"$GFONTS/hindsiliguri/HindSiliguri-Regular.ttf"
# Gurmukhi
"$NOTO/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Regular.ttf"
"$NOTO/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Bold.ttf"
"$NOTO/NotoSerifGurmukhi/full/ttf/NotoSerifGurmukhi-Regular.ttf"
# Gujarati
"$NOTO/NotoSansGujarati/full/ttf/NotoSansGujarati-Regular.ttf"
"$NOTO/NotoSansGujarati/full/ttf/NotoSansGujarati-Bold.ttf"
"$NOTO/NotoSerifGujarati/full/ttf/NotoSerifGujarati-Regular.ttf"
"$GFONTS/hindvadodara/HindVadodara-Regular.ttf"
# Tamil
"$NOTO/NotoSansTamil/full/ttf/NotoSansTamil-Regular.ttf"
"$NOTO/NotoSansTamil/full/ttf/NotoSansTamil-Bold.ttf"
"$NOTO/NotoSerifTamil/full/ttf/NotoSerifTamil-Regular.ttf"
"$GFONTS/tirotamil/TiroTamil-Regular.ttf"
# Telugu
"$NOTO/NotoSansTelugu/full/ttf/NotoSansTelugu-Regular.ttf"
"$NOTO/NotoSansTelugu/full/ttf/NotoSansTelugu-Bold.ttf"
"$NOTO/NotoSerifTelugu/full/ttf/NotoSerifTelugu-Regular.ttf"
"$GFONTS/tirotelugu/TiroTelugu-Regular.ttf"
# Kannada
"$NOTO/NotoSansKannada/full/ttf/NotoSansKannada-Regular.ttf"
"$NOTO/NotoSansKannada/full/ttf/NotoSansKannada-Bold.ttf"
"$NOTO/NotoSerifKannada/full/ttf/NotoSerifKannada-Regular.ttf"
"$GFONTS/tirokannada/TiroKannada-Regular.ttf"
# Malayalam
"$NOTO/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Regular.ttf"
"$NOTO/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Bold.ttf"
"$NOTO/NotoSerifMalayalam/full/ttf/NotoSerifMalayalam-Regular.ttf"
"$GFONTS/chilanka/Chilanka-Regular.ttf"
# Thai
"$NOTO/NotoSansThai/full/ttf/NotoSansThai-Regular.ttf"
"$NOTO/NotoSansThai/full/ttf/NotoSansThai-Bold.ttf"
"$NOTO/NotoSerifThai/full/ttf/NotoSerifThai-Regular.ttf"
"$GFONTS/kanit/Kanit-Regular.ttf"
"$GFONTS/sarabun/Sarabun-Regular.ttf"
"$GFONTS/prompt/Prompt-Regular.ttf"
# Lao
"$NOTO/NotoSansLao/full/ttf/NotoSansLao-Regular.ttf"
"$NOTO/NotoSansLao/full/ttf/NotoSansLao-Bold.ttf"
"$NOTO/NotoSerifLao/full/ttf/NotoSerifLao-Regular.ttf"
"$GFONTS/phetsarathot/PhetsarathOT-Regular.ttf"
)

echo "Downloading ${#URLS[@]} fonts (parallel)..."

# Download all in parallel (up to 20 at a time), skip existing
printf '%s\n' "${URLS[@]}" | xargs -P 20 -I{} sh -c '
    name="$(basename "{}" | sed "s/%5B/[/g; s/%5D/]/g")"
    if [ ! -f "'"$FONT_DIR"'/$name" ]; then
        curl -sL -o "'"$FONT_DIR"'/$name" "{}" && echo "  OK: $name" || echo "  FAIL: $name"
    fi
'

echo ""
count=$(ls -1 "$FONT_DIR"/*.ttf "$FONT_DIR"/*.otf 2>/dev/null | wc -l)
echo "Done. $count font files in $FONT_DIR"
