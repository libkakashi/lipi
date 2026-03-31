#!/bin/bash
# Download Google Noto fonts for all 18 scripts.
# Run once on any machine before training.

set -e

FONT_DIR="$(dirname "$0")/../training_data/fonts"
mkdir -p "$FONT_DIR"
cd "$FONT_DIR"

echo "Downloading Noto fonts to $FONT_DIR ..."

# Base URL for Noto releases
NOTO="https://github.com/notofonts/notofonts.github.io/raw/main/fonts"

# Function to download if not already present
dl() {
    local url="$1"
    local name="$(basename "$url")"
    if [ ! -f "$name" ]; then
        echo "  Downloading $name ..."
        curl -sL -o "$name" "$url" || wget -q -O "$name" "$url"
    else
        echo "  Already have $name"
    fi
}

# Latin, Cyrillic, Greek (covered by NotoSans)
dl "$NOTO/NotoSans/full/ttf/NotoSans-Regular.ttf"
dl "$NOTO/NotoSans/full/ttf/NotoSans-Bold.ttf"

# Arabic
dl "$NOTO/NotoSansArabic/full/ttf/NotoSansArabic-Regular.ttf"
dl "$NOTO/NotoNaskhArabic/full/ttf/NotoNaskhArabic-Regular.ttf"

# Hebrew
dl "$NOTO/NotoSansHebrew/full/ttf/NotoSansHebrew-Regular.ttf"

# CJK (Simplified Chinese, Japanese, Korean all in one)
dl "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/01_NotoSansCJKsc-Regular.otf"
dl "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/03_NotoSansCJKjp-Regular.otf"
dl "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/05_NotoSansCJKkr-Regular.otf"

# Devanagari
dl "$NOTO/NotoSansDevanagari/full/ttf/NotoSansDevanagari-Regular.ttf"

# Bengali
dl "$NOTO/NotoSansBengali/full/ttf/NotoSansBengali-Regular.ttf"

# Gurmukhi
dl "$NOTO/NotoSansGurmukhi/full/ttf/NotoSansGurmukhi-Regular.ttf"

# Gujarati
dl "$NOTO/NotoSansGujarati/full/ttf/NotoSansGujarati-Regular.ttf"

# Tamil
dl "$NOTO/NotoSansTamil/full/ttf/NotoSansTamil-Regular.ttf"

# Telugu
dl "$NOTO/NotoSansTelugu/full/ttf/NotoSansTelugu-Regular.ttf"

# Kannada
dl "$NOTO/NotoSansKannada/full/ttf/NotoSansKannada-Regular.ttf"

# Malayalam
dl "$NOTO/NotoSansMalayalam/full/ttf/NotoSansMalayalam-Regular.ttf"

# Thai
dl "$NOTO/NotoSansThai/full/ttf/NotoSansThai-Regular.ttf"

# Lao
dl "$NOTO/NotoSansLao/full/ttf/NotoSansLao-Regular.ttf"

echo ""
echo "Done. Fonts in: $FONT_DIR"
ls -1 "$FONT_DIR"/*.ttf "$FONT_DIR"/*.otf 2>/dev/null | wc -l | xargs -I{} echo "{} font files downloaded"
