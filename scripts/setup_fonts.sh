#!/bin/bash
# Download diverse fonts for all 18 scripts (parallel).
set -e

FONT_DIR="$(cd "$(dirname "$0")/.." && pwd)/training_data/fonts"
mkdir -p "$FONT_DIR"

NOTO="https://github.com/notofonts/notofonts.github.io/raw/main/fonts"
GFONTS="https://github.com/google/fonts/raw/main/ofl"
CJK_SANS="https://github.com/notofonts/noto-cjk/releases/download/Sans2.004"
CJK_SERIF="https://github.com/notofonts/noto-cjk/releases/download/Serif2.003"

# Write all URLs to a temp file
URLFILE=$(mktemp)
cat > "$URLFILE" << 'ENDURLS'
NOTO/NotoSans/full/ttf/NotoSans-Regular.ttf
NOTO/NotoSans/full/ttf/NotoSans-Bold.ttf
NOTO/NotoSans/full/ttf/NotoSans-Italic.ttf
NOTO/NotoSans/full/ttf/NotoSans-Light.ttf
NOTO/NotoSerif/full/ttf/NotoSerif-Regular.ttf
NOTO/NotoSerif/full/ttf/NotoSerif-Bold.ttf
NOTO/NotoSerif/full/ttf/NotoSerif-Italic.ttf
NOTO/NotoSansMono/full/ttf/NotoSansMono-Regular.ttf
NOTO/NotoSansArabic/full/ttf/NotoSansArabic-Regular.ttf
NOTO/NotoSansArabic/full/ttf/NotoSansArabic-Bold.ttf
NOTO/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Regular.ttf
NOTO/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Bold.ttf
NOTO/NotoNastaliqUrdu/full/ttf/NotoNastaliqUrdu-Regular.ttf
NOTO/NotoKufiArabic/full/ttf/NotoKufiArabic-Regular.ttf
NOTO/NotoSansHebrew/full/ttf/NotoSansHebrew-Regular.ttf
NOTO/NotoSansHebrew/full/ttf/NotoSansHebrew-Bold.ttf
NOTO/NotoSerifHebrew/full/ttf/NotoSerifHebrew-Regular.ttf
NOTO/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Regular.ttf
NOTO/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Bold.ttf
NOTO/NotoSerifDevanagari/full/ttf/NotoSerifDevanagari-Regular.ttf
NOTO/NotoSansBengali/full/ttf/NotoSansBengali-Regular.ttf
NOTO/NotoSansBengali/full/ttf/NotoSansBengali-Bold.ttf
NOTO/NotoSerifBengali/full/ttf/NotoSerifBengali-Regular.ttf
NOTO/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Regular.ttf
NOTO/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Bold.ttf
NOTO/NotoSerifGurmukhi/full/ttf/NotoSerifGurmukhi-Regular.ttf
NOTO/NotoSansGujarati/full/ttf/NotoSansGujarati-Regular.ttf
NOTO/NotoSansGujarati/full/ttf/NotoSansGujarati-Bold.ttf
NOTO/NotoSerifGujarati/full/ttf/NotoSerifGujarati-Regular.ttf
NOTO/NotoSansTamil/full/ttf/NotoSansTamil-Regular.ttf
NOTO/NotoSansTamil/full/ttf/NotoSansTamil-Bold.ttf
NOTO/NotoSerifTamil/full/ttf/NotoSerifTamil-Regular.ttf
NOTO/NotoSansTelugu/full/ttf/NotoSansTelugu-Regular.ttf
NOTO/NotoSansTelugu/full/ttf/NotoSansTelugu-Bold.ttf
NOTO/NotoSerifTelugu/full/ttf/NotoSerifTelugu-Regular.ttf
NOTO/NotoSansKannada/full/ttf/NotoSansKannada-Regular.ttf
NOTO/NotoSansKannada/full/ttf/NotoSansKannada-Bold.ttf
NOTO/NotoSerifKannada/full/ttf/NotoSerifKannada-Regular.ttf
NOTO/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Regular.ttf
NOTO/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Bold.ttf
NOTO/NotoSerifMalayalam/full/ttf/NotoSerifMalayalam-Regular.ttf
NOTO/NotoSansThai/full/ttf/NotoSansThai-Regular.ttf
NOTO/NotoSansThai/full/ttf/NotoSansThai-Bold.ttf
NOTO/NotoSerifThai/full/ttf/NotoSerifThai-Regular.ttf
NOTO/NotoSansLao/full/ttf/NotoSansLao-Regular.ttf
NOTO/NotoSansLao/full/ttf/NotoSansLao-Bold.ttf
NOTO/NotoSerifLao/full/ttf/NotoSerifLao-Regular.ttf
ENDURLS

# Google Fonts (separate because different base URL)
GFFILE=$(mktemp)
cat > "$GFFILE" << 'ENDGF'
caveat/Caveat[wght].ttf
dancingscript/DancingScript[wght].ttf
indieflower/IndieFlower-Regular.ttf
patrickhand/PatrickHand-Regular.ttf
shadowsintolight/ShadowsIntoLight.ttf
permanentmarker/PermanentMarker-Regular.ttf
amaticsc/AmaticSC-Regular.ttf
lobster/Lobster-Regular.ttf
pacifico/Pacifico-Regular.ttf
comicneue/ComicNeue-Regular.ttf
specialelite/SpecialElite-Regular.ttf
amiri/Amiri-Regular.ttf
amiri/Amiri-Bold.ttf
scheherazadenew/ScheherazadeNew-Regular.ttf
lateef/Lateef-Regular.ttf
frankruhllibre/FrankRuhlLibre[wght].ttf
rubik/Rubik[wght].ttf
secularone/SecularOne-Regular.ttf
nanumgothic/NanumGothic-Regular.ttf
nanummyeongjo/NanumMyeongjo-Regular.ttf
nanumpenscript/NanumPenScript-Regular.ttf
poppins/Poppins-Regular.ttf
tirodevanagarihindi/TiroDevanagariHindi-Regular.ttf
laila/Laila-Regular.ttf
kalam/Kalam-Regular.ttf
tirobangla/TiroBangla-Regular.ttf
hindsiliguri/HindSiliguri-Regular.ttf
hindvadodara/HindVadodara-Regular.ttf
tirotamil/TiroTamil-Regular.ttf
tirotelugu/TiroTelugu-Regular.ttf
tirokannada/TiroKannada-Regular.ttf
chilanka/Chilanka-Regular.ttf
kanit/Kanit-Regular.ttf
sarabun/Sarabun-Regular.ttf
prompt/Prompt-Regular.ttf
phetsarathot/PhetsarathOT-Regular.ttf
ENDGF

# CJK (separate — large files, different URLs)
CJK_URLS=(
"$CJK_SANS/01_NotoSansCJKsc-Regular.otf"
"$CJK_SANS/01_NotoSansCJKsc-Bold.otf"
"$CJK_SANS/03_NotoSansCJKjp-Regular.otf"
"$CJK_SANS/05_NotoSansCJKkr-Regular.otf"
"$CJK_SANS/05_NotoSansCJKkr-Bold.otf"
"$CJK_SERIF/01_NotoSerifCJKsc-Regular.otf"
"$CJK_SERIF/03_NotoSerifCJKjp-Regular.otf"
"$CJK_SERIF/05_NotoSerifCJKkr-Regular.otf"
)

download_one() {
    local url="$1"
    local dest="$2"
    local name="$(basename "$dest")"
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        return
    fi
    if curl -sL -f -o "$dest" "$url"; then
        echo "  OK: $name"
    else
        rm -f "$dest"
        echo "  FAIL: $name"
    fi
}
export -f download_one

total=0

# Noto fonts (parallel)
echo "=== Noto fonts ==="
while IFS= read -r line; do
    [ -z "$line" ] && continue
    url="${NOTO/NOTO/}"; url="https://github.com/notofonts/notofonts.github.io/raw/main/fonts/${line#NOTO/}"
    name="$(basename "$line")"
    dest="$FONT_DIR/$name"
    total=$((total+1))
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        continue
    fi
    curl -sL -f -o "$dest" "$url" && echo "  OK: $name" || { rm -f "$dest"; echo "  FAIL: $name"; } &
    # Limit parallel jobs
    [ $((total % 20)) -eq 0 ] && wait
done < "$URLFILE"
wait

# Google Fonts (parallel)
echo "=== Google Fonts ==="
while IFS= read -r line; do
    [ -z "$line" ] && continue
    name="$(basename "$line")"
    dest="$FONT_DIR/$name"
    total=$((total+1))
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        continue
    fi
    curl -sL -f -o "$dest" "$GFONTS/$line" && echo "  OK: $name" || { rm -f "$dest"; echo "  FAIL: $name"; } &
    [ $((total % 20)) -eq 0 ] && wait
done < "$GFFILE"
wait

# CJK fonts (parallel — these are big)
echo "=== CJK fonts ==="
for url in "${CJK_URLS[@]}"; do
    name="$(basename "$url")"
    dest="$FONT_DIR/$name"
    total=$((total+1))
    if [ -f "$dest" ] && [ -s "$dest" ]; then
        continue
    fi
    curl -sL -f -o "$dest" "$url" && echo "  OK: $name" || { rm -f "$dest"; echo "  FAIL: $name"; } &
done
wait

rm -f "$URLFILE" "$GFFILE"

echo ""
count=$(ls -1 "$FONT_DIR"/*.ttf "$FONT_DIR"/*.otf 2>/dev/null | wc -l | tr -d ' ')
echo "Done. $count fonts in $FONT_DIR"
