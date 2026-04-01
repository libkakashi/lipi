#!/usr/bin/env python3
"""
Phase 0: Train and validate hierarchical LID.

Tests two levels of script identification:
  LID-1 (stem only): Coarse grouping into 8 groups
  LID-2 (stem + Stage 1 SWA): Fine-grained 18-script classification

Generates multi-script synthetic word crops on-the-fly using system fonts.
Runs on CPU/MPS in minutes.

Usage:
    python scripts/train_lid.py
    python scripts/train_lid.py --level both --epochs 20
    python scripts/train_lid.py --level coarse  # test stem-only grouping
    python scripts/train_lid.py --level fine     # test with Stage 1
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.stem import ConvNeXtStem, ResNetStem
from src.model.lid import (
    LIDCoarse, LIDFine,
    SCRIPTS, SCRIPT_TO_ID, NUM_SCRIPTS,
    GROUPS, GROUP_TO_ID, NUM_GROUPS,
    SCRIPT_TO_GROUP, GROUP_SCRIPTS, script_to_group_id,
)
from src.data.color import rgb_to_input, INPUT_CHANNELS, ColorProjection
from src.data.augmentation import RandAugmentOCR


WORD_LIST_DIR = Path(__file__).parent.parent / "training_data" / "word_lists"


# Extra language word lists that belong to each script
_SCRIPT_EXTRA_FILES = {
    "latin": ["english_common.txt", "french.txt", "german.txt", "spanish.txt",
              "turkish.txt", "vietnamese.txt", "italian.txt", "portuguese.txt",
              "polish.txt", "dutch.txt", "romanian.txt", "czech.txt",
              "hungarian.txt", "swedish.txt", "norwegian.txt", "danish.txt",
              "finnish.txt", "croatian.txt", "indonesian.txt", "malay.txt",
              "swahili.txt", "afrikaans.txt", "albanian.txt", "basque.txt",
              "catalan.txt", "estonian.txt", "galician.txt", "icelandic.txt",
              "latvian.txt", "lithuanian.txt", "maltese.txt", "slovak.txt",
              "slovenian.txt", "welsh.txt", "irish.txt", "tagalog.txt"],
    "cyrillic": ["ukrainian.txt"],
    "devanagari": ["marathi.txt"],
    "arabic": ["persian.txt", "urdu.txt"],
    "han_kana": ["japanese.txt"],
}


def load_word_list(script: str, fallback: list[str]) -> list[str]:
    """Load and merge all word list files for a script."""
    files = [WORD_LIST_DIR / f"{script}.txt"]
    for extra in _SCRIPT_EXTRA_FILES.get(script, []):
        files.append(WORD_LIST_DIR / extra)

    words = []
    for path in files:
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if 2 <= len(w) <= 15 and w and not w[0].isdigit():
                    words.append(w)

    if words:
        words = list(set(words))
        random.shuffle(words)
        return words
    return fallback


# Fallback word samples per script (used when no word list file exists)
_SCRIPT_SAMPLES_FALLBACK = {
    "latin": [
        "hello", "world", "justice", "court", "legal", "document", "appeal",
        "judge", "order", "case", "file", "law", "right", "party", "trial",
        "evidence", "motion", "ruling", "verdict", "penalty", "defense",
        "plaintiff", "attorney", "statute", "contract", "agreement", "clause",
    ],
    "cyrillic": [
        "привет", "мир", "суд", "закон", "право", "дело", "решение",
        "истец", "ответчик", "документ", "судья", "апелляция", "протокол",
        "москва", "россия", "газета", "книга", "школа", "работа", "время",
    ],
    "greek": [
        "δικαιοσύνη", "νόμος", "δικαστήριο", "αλήθεια", "κόσμος",
        "αριθμός", "σύμβολο", "λόγος", "πόλη", "χρόνος",
        "άνθρωπος", "βιβλίο", "σχολείο", "εργασία", "ζωή",
    ],
    "devanagari": [
        "नमस्ते", "न्याय", "अदालत", "कानून", "अधिकार", "फैसला",
        "मुकदमा", "वकील", "सरकार", "भारत", "दिल्ली", "हिंदी",
        "काम", "समय", "लोग", "देश", "पानी", "घर", "बात", "दिन",
    ],
    "bengali": [
        "নমস্কার", "বিচার", "আদালত", "আইন", "অধিকার", "সিদ্ধান্ত",
        "মামলা", "উকিল", "সরকার", "বাংলা", "কলকাতা", "ঢাকা",
        "কাজ", "সময়", "মানুষ", "দেশ", "জল", "বাড়ি", "কথা", "দিন",
    ],
    "tamil": [
        "வணக்கம்", "நீதி", "நீதிமன்றம்", "சட்டம்", "உரிமை",
        "தீர்ப்பு", "வழக்கு", "வக்கீல்", "அரசு", "தமிழ்",
        "சென்னை", "காலம்", "மனிதன்", "நாடு", "வீடு", "நாள்",
    ],
    "telugu": [
        "నమస్కారం", "న్యాయం", "కోర్టు", "చట్టం", "హక్కు",
        "తీర్పు", "కేసు", "న్యాయవాది", "ప్రభుత్వం", "తెలుగు",
        "హైదరాబాద్", "సమయం", "మనిషి", "దేశం", "ఇల్లు", "రోజు",
    ],
    "kannada": [
        "ನಮಸ್ಕಾರ", "ನ್ಯಾಯ", "ನ್ಯಾಯಾಲಯ", "ಕಾನೂನು", "ಹಕ್ಕು",
        "ತೀರ್ಪು", "ಮೊಕದ್ದಮೆ", "ವಕೀಲ", "ಸರ್ಕಾರ", "ಕನ್ನಡ",
        "ಬೆಂಗಳೂರು", "ಸಮಯ", "ಮನುಷ್ಯ", "ದೇಶ", "ಮನೆ", "ದಿನ",
    ],
    "malayalam": [
        "നമസ്കാരം", "നീതി", "കോടതി", "നിയമം", "അവകാശം",
        "വിധി", "കേസ്", "അഭിഭാഷകൻ", "സർക്കാർ", "മലയാളം",
        "കൊച്ചി", "സമയം", "മനുഷ്യൻ", "രാജ്യം", "വീട്", "ദിവസം",
    ],
    "gujarati": [
        "નમસ્તે", "ન્યાય", "અદાલત", "કાયદો", "અધિકાર",
        "ચુકાદો", "કેસ", "વકીલ", "સરકાર", "ગુજરાતી",
        "અમદાવાદ", "સમય", "માણસ", "દેશ", "ઘર", "દિવસ",
    ],
    "gurmukhi": [
        "ਨਿਆਂ", "ਅਦਾਲਤ", "ਕਾਨੂੰਨ", "ਅਧਿਕਾਰ",
        "ਫੈਸਲਾ", "ਕੇਸ", "ਵਕੀਲ", "ਸਰਕਾਰ", "ਪੰਜਾਬੀ",
        "ਚੰਡੀਗੜ੍ਹ", "ਸਮਾਂ", "ਬੰਦਾ", "ਦੇਸ਼", "ਘਰ", "ਦਿਨ",
    ],
    "arabic": [
        "مرحبا", "عدالة", "محكمة", "قانون", "حق", "حكم",
        "قضية", "محامي", "حكومة", "عربي", "القاهرة", "وقت",
        "إنسان", "بلد", "بيت", "يوم", "كتاب", "مدرسة",
    ],
    "han_kana": [
        "你好", "正义", "法院", "法律", "权利", "判决",
        "案件", "律师", "政府", "中文", "北京", "时间",
        "人民", "国家", "房子", "今天", "学校", "工作",
    ],
    "korean": [
        "안녕하세요", "정의", "법원", "법률", "권리", "판결",
        "사건", "변호사", "정부", "한국어", "서울", "시간",
        "사람", "나라", "집", "오늘", "학교", "일",
    ],
    "thai": [
        "สวัสดี", "ความยุติธรรม", "ศาล", "กฎหมาย", "สิทธิ",
        "คำพิพากษา", "คดี", "ทนายความ", "รัฐบาล", "ภาษาไทย",
        "กรุงเทพ", "เวลา", "คน", "ประเทศ", "บ้าน", "วัน",
    ],
    "hebrew": [
        "שלום", "משפט", "חוק", "זכות", "עורך", "דין", "שופט",
        "ממשלה", "ישראל", "ירושלים", "עברית", "ספר", "בית",
        "זמן", "אדם", "ארץ", "מים", "יום", "לילה", "שנה",
        "עיר", "דרך", "מלך", "אמת", "צדק", "תורה", "כנסת",
    ],
    "lao": [
        "ສະບາຍດີ", "ກົດໝາຍ", "ສານ", "ປະເທດ", "ເມືອງ",
        "ຄົນ", "ເຮືອນ", "ນ້ຳ", "ເວລາ", "ວຽກ", "ໂຮງຮຽນ",
        "ຕະຫຼາດ", "ທາງ", "ພູ", "ແມ່ນ້ຳ", "ກິນ", "ດື່ມ",
    ],
}

# Loaded at runtime — file-based word lists merged with fallbacks
SCRIPT_SAMPLES: dict[str, list[str]] = {}


def load_all_script_samples():
    """Load word lists from files, falling back to hardcoded samples."""
    for script, fallback in _SCRIPT_SAMPLES_FALLBACK.items():
        SCRIPT_SAMPLES[script] = load_word_list(script, fallback)
    # Emoji doesn't have words — handled separately in rendering
    SCRIPT_SAMPLES["emoji"] = ["emoji"]


FONT_DIR = Path(__file__).parent.parent / "training_data" / "fonts"

# Local fonts -> scripts they cover (downloaded by setup_fonts.sh)
# Includes sans, serif, handwriting, display for visual diversity
_LOCAL_FONT_MAP = {
    "latin": [
        "NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf", "NotoSans-Light.ttf",
        "NotoSerif-Regular.ttf", "NotoSerif-Bold.ttf", "NotoSerif-Italic.ttf",
        "NotoSansMono-Regular.ttf",
        "Caveat[wght].ttf", "DancingScript[wght].ttf", "IndieFlower-Regular.ttf",
        "PatrickHand-Regular.ttf", "ShadowsIntoLight.ttf", "PermanentMarker-Regular.ttf",
        "AmaticSC-Regular.ttf", "Lobster-Regular.ttf", "Pacifico-Regular.ttf",
        "ComicNeue-Regular.ttf", "SpecialElite-Regular.ttf",
    ],
    "cyrillic": [
        "NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf",
        "NotoSerif-Regular.ttf", "NotoSerif-Bold.ttf",
        "Caveat[wght].ttf", "ComicNeue-Regular.ttf", "Pacifico-Regular.ttf",
    ],
    "greek": [
        "NotoSans-Regular.ttf", "NotoSans-Bold.ttf",
        "NotoSerif-Regular.ttf",
    ],
    "arabic": [
        "NotoSansArabic-Regular.ttf", "NotoSansArabic-Bold.ttf",
        "NotoNaskhArabic-Regular.ttf", "NotoNaskhArabic-Bold.ttf",
        "NotoNastaliqUrdu-Regular.ttf", "NotoKufiArabic-Regular.ttf",
        "Amiri-Regular.ttf", "Amiri-Bold.ttf",
        "ScheherazadeNew-Regular.ttf", "Lateef-Regular.ttf",
    ],
    "hebrew": [
        "NotoSansHebrew-Regular.ttf", "NotoSansHebrew-Bold.ttf",
        "NotoSerifHebrew-Regular.ttf",
        "FrankRuhlLibre[wght].ttf", "Rubik[wght].ttf", "SecularOne-Regular.ttf",
    ],
    "han_kana": [
        "NotoSansSC[wght].ttf", "NotoSansJP[wght].ttf",
        "NotoSansCJKsc-Regular.otf", "NotoSansCJKjp-Regular.otf",
        "NotoSerifCJKsc-Regular.otf",
    ],
    "korean": [
        "NotoSansKR[wght].ttf", "NotoSansCJKkr-Regular.otf",
        "NotoSerifCJKkr-Regular.otf",
        "NanumGothic-Regular.ttf", "NanumMyeongjo-Regular.ttf", "NanumPenScript-Regular.ttf",
    ],
    "devanagari": [
        "NotoSansDevanagari-Regular.ttf", "NotoSansDevanagari-Bold.ttf",
        "NotoSerifDevanagari-Regular.ttf",
        "Poppins-Regular.ttf", "TiroDevanagariHindi-Regular.ttf",
        "Laila-Regular.ttf", "Kalam-Regular.ttf",
    ],
    "bengali": [
        "NotoSansBengali-Regular.ttf", "NotoSansBengali-Bold.ttf",
        "NotoSerifBengali-Regular.ttf",
        "TiroBangla-Regular.ttf", "HindSiliguri-Regular.ttf",
    ],
    "gurmukhi": [
        "NotoSansGurmukhi-Regular.ttf", "NotoSansGurmukhi-Bold.ttf",
        "NotoSerifGurmukhi-Regular.ttf",
    ],
    "gujarati": [
        "NotoSansGujarati-Regular.ttf", "NotoSansGujarati-Bold.ttf",
        "NotoSerifGujarati-Regular.ttf", "HindVadodara-Regular.ttf",
    ],
    "tamil": [
        "NotoSansTamil-Regular.ttf", "NotoSansTamil-Bold.ttf",
        "NotoSerifTamil-Regular.ttf", "TiroTamil-Regular.ttf",
    ],
    "telugu": [
        "NotoSansTelugu-Regular.ttf", "NotoSansTelugu-Bold.ttf",
        "NotoSerifTelugu-Regular.ttf", "TiroTelugu-Regular.ttf",
    ],
    "kannada": [
        "NotoSansKannada-Regular.ttf", "NotoSansKannada-Bold.ttf",
        "NotoSerifKannada-Regular.ttf", "TiroKannada-Regular.ttf",
    ],
    "malayalam": [
        "NotoSansMalayalam-Regular.ttf", "NotoSansMalayalam-Bold.ttf",
        "NotoSerifMalayalam-Regular.ttf", "Chilanka-Regular.ttf",
    ],
    "thai": [
        "NotoSansThai-Regular.ttf", "NotoSansThai-Bold.ttf",
        "NotoSerifThai-Regular.ttf",
        "Kanit-Regular.ttf", "Sarabun-Regular.ttf", "Prompt-Regular.ttf",
    ],
    "lao": [
        "NotoSansLao-Regular.ttf", "NotoSansLao-Bold.ttf",
        "NotoSerifLao-Regular.ttf", "PhetsarathOT-Regular.ttf",
    ],
}

# fc-list language codes for system font fallback
_LANG_MAP = {
    "latin": "en", "cyrillic": "ru", "greek": "el",
    "devanagari": "hi", "bengali": "bn", "tamil": "ta",
    "telugu": "te", "kannada": "kn", "malayalam": "ml",
    "gujarati": "gu", "gurmukhi": "pa",
    "arabic": "ar", "hebrew": "he", "han_kana": "zh", "korean": "ko",
    "thai": "th", "lao": "lo",
}


def find_fonts_for_script(script: str) -> list[str]:
    """Find fonts for a script. Checks local Noto fonts first, then system."""
    if script == "emoji":
        return ["__emoji__"]

    fonts = []

    # 1. Check local downloaded fonts
    for name in _LOCAL_FONT_MAP.get(script, []):
        path = FONT_DIR / name
        if path.exists():
            fonts.append(str(path))

    # 2. Fallback to system fonts via fc-list
    if not fonts:
        lang = _LANG_MAP.get(script)
        if lang:
            try:
                result = subprocess.run(
                    ["fc-list", f":lang={lang}", "file"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.strip().split("\n"):
                    path = line.strip().rstrip(":")
                    if path and Path(path).exists():
                        fonts.append(path)
            except Exception:
                pass

    return fonts


from src.data.renderer import render_text, font_can_render


def random_ink_color():
    """Random text color — documents, signs, banners, screens, neon."""
    style = random.choice([
        # Documents (50%)
        "black", "black", "black",
        "dark_gray",
        "blue_ink",
        # Signs & banners (25%)
        "white_text",          # white text on dark bg
        "yellow_text",         # yellow on dark (road signs, banners)
        "bright_red",          # warning signs, sale banners
        "bright_blue",         # info signs, digital
        "neon_green",          # digital displays, neon signs
        # Wild (25%)
        "orange",
        "purple",
        "teal",
        "magenta",
        "random_bright",       # fully random saturated color
    ])
    colors = {
        "black": (random.randint(0, 40),) * 3,
        "dark_gray": (random.randint(50, 90),) * 3,
        "blue_ink": (random.randint(0, 40), random.randint(0, 50), random.randint(130, 210)),
        "white_text": (random.randint(220, 255),) * 3,
        "yellow_text": (random.randint(220, 255), random.randint(200, 240), random.randint(0, 60)),
        "bright_red": (random.randint(200, 255), random.randint(0, 60), random.randint(0, 60)),
        "bright_blue": (random.randint(0, 60), random.randint(80, 160), random.randint(200, 255)),
        "neon_green": (random.randint(0, 80), random.randint(200, 255), random.randint(0, 80)),
        "orange": (random.randint(220, 255), random.randint(120, 180), random.randint(0, 50)),
        "purple": (random.randint(120, 180), random.randint(0, 60), random.randint(180, 240)),
        "teal": (random.randint(0, 60), random.randint(180, 230), random.randint(180, 230)),
        "magenta": (random.randint(220, 255), random.randint(0, 80), random.randint(180, 240)),
        "random_bright": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    }
    return colors[style]


def random_bg_color():
    """Random background — paper, signs, screens, posters, walls."""
    style = random.choice([
        # Paper/document (40%)
        "white", "white",
        "offwhite",
        "cream",
        # Colored surfaces (30%)
        "dark_bg",             # dark sign/banner background
        "dark_blue",           # blue banner, slide
        "dark_red",            # red banner
        "dark_green",          # green chalkboard
        "brown",               # wood, cardboard
        # Misc (30%)
        "light_blue",
        "light_pink",
        "light_green",
        "gray",
        "yellow",              # post-it note, caution sign
        "random_pastel",       # random light color
    ])
    colors = {
        "white": (random.randint(240, 255),) * 3,
        "offwhite": (random.randint(230, 250), random.randint(228, 248), random.randint(220, 240)),
        "cream": (random.randint(240, 255), random.randint(235, 250), random.randint(200, 225)),
        "dark_bg": (random.randint(10, 50),) * 3,
        "dark_blue": (random.randint(10, 40), random.randint(20, 60), random.randint(80, 140)),
        "dark_red": (random.randint(100, 160), random.randint(10, 40), random.randint(10, 40)),
        "dark_green": (random.randint(10, 40), random.randint(60, 110), random.randint(10, 40)),
        "brown": (random.randint(120, 170), random.randint(80, 120), random.randint(40, 70)),
        "light_blue": (random.randint(220, 240), random.randint(230, 248), random.randint(245, 255)),
        "light_pink": (random.randint(245, 255), random.randint(220, 235), random.randint(225, 240)),
        "light_green": (random.randint(225, 242), random.randint(245, 255), random.randint(225, 242)),
        "gray": (random.randint(180, 220),) * 3,
        "yellow": (random.randint(245, 255), random.randint(240, 255), random.randint(140, 190)),
        "random_pastel": (random.randint(180, 240), random.randint(180, 240), random.randint(180, 240)),
    }
    return colors[style]


def render_word(text: str, font_path: str, height: int = 32) -> Image.Image | None:
    """Render a word with random ink and paper colors using FreeType."""
    # Pick colors with enough contrast
    for _ in range(5):
        bg = random_bg_color()
        ink = random_ink_color()
        bg_lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        ink_lum = 0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2]
        if abs(bg_lum - ink_lum) > 60:
            break

    font_size = random.randint(18, 26)
    return render_text(text, font_path, font_size, ink=ink, bg=bg, height=height,
                       pad_x=random.randint(2, 8), pad_y=random.randint(2, 6))


def render_emoji(height: int = 32, max_width: int = 192) -> Image.Image:
    """Render a synthetic emoji-like colorful block image.

    Emojis are visually distinct: bright colors, round shapes, high contrast.
    We simulate this with colored circles/rectangles on white backgrounds.
    """
    w = random.randint(height, min(height * 3, max_width))
    img = Image.new("RGB", (w, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw 1-3 colorful shapes
    for _ in range(random.randint(1, 3)):
        color = (random.randint(150, 255), random.randint(50, 255), random.randint(0, 200))
        cx = random.randint(4, w - 4)
        cy = random.randint(4, height - 4)
        r = random.randint(4, min(12, height // 3))
        shape = random.choice(["circle", "rect"])
        if shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        else:
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)

    return img


def _generate_batch(args_tuple):
    """Worker function for parallel image generation. Must be at module level for pickling."""
    script, count, fonts, words, h, mw, do_augment = args_tuple
    aug = RandAugmentOCR(n_ops=2, p=0.5) if do_augment else None
    results = []
    attempts = 0
    while len(results) < count and attempts < count * 5:
        attempts += 1
        if script == "emoji":
            img = render_emoji(h, mw)
        else:
            img = render_word(random.choice(words), random.choice(fonts), h)
        if img is None:
            continue
        if img.width > mw:
            img = img.resize((mw, h), Image.BILINEAR)
        elif img.width < mw:
            padded = Image.new("RGB", (mw, h), (240, 240, 240))
            padded.paste(img, (0, 0))
            img = padded
        if aug is not None:
            img = aug(img)
        results.append(rgb_to_input(img))
    return results


class MultiScriptDataset(Dataset):
    """Pre-generated multi-script word images with both coarse and fine labels."""

    def __init__(self, samples_per_script: int = 1000, height: int = 32, max_width: int = 192, balance_groups: bool = False, augment: bool = False):
        self.height = height
        self.max_width = max_width
        self.augmentor = RandAugmentOCR(n_ops=2, p=0.5) if augment else None

        print("Discovering fonts per script...")
        self.script_fonts = {}
        self.active_scripts = []

        for script in SCRIPTS:
            if script == "emoji":
                # Emoji rendered as colored blocks, no font needed
                self.script_fonts[script] = ["__emoji__"]
                self.active_scripts.append(script)
                print(f"  {'emoji':<15}   - (synthetic colored blocks)")
                continue
            if script not in SCRIPT_SAMPLES:
                continue
            fonts = find_fonts_for_script(script)
            sample_text = SCRIPT_SAMPLES[script][0]
            valid_fonts = [f for f in fonts if font_can_render(f, sample_text)]
            if valid_fonts:
                # Build weighted font list: 70% clean, 20% handwriting, 10% display
                weighted = []
                for f in valid_fonts:
                    name = Path(f).name.lower()
                    if any(k in name for k in ["caveat", "dancing", "indie", "patrick",
                            "shadow", "kalam", "nanum_pen", "nanumpen", "chilanka",
                            "handwrit", "cursive", "script"]):
                        weighted.append((f, 2))   # handwriting: weight 2
                    elif any(k in name for k in ["permanent", "amatic", "lobster",
                            "pacifico", "special", "display"]):
                        weighted.append((f, 1))   # display: weight 1
                    else:
                        weighted.append((f, 7))   # clean: weight 7
                # Expand into sampling list
                self.script_fonts[script] = []
                for font, weight in weighted:
                    self.script_fonts[script].extend([font] * weight)
                self.active_scripts.append(script)
                print(f"  {script:<15} {len(valid_fonts):>3} fonts")
            else:
                print(f"  {script:<15}   0 fonts — SKIPPED")

        if len(self.active_scripts) < 2:
            raise RuntimeError("Need at least 2 scripts with valid fonts")

        self.num_scripts = len(self.active_scripts)
        self.script_to_idx = {s: i for i, s in enumerate(self.active_scripts)}

        # Build group mapping for active scripts
        self.active_groups = []
        seen_groups = set()
        for s in self.active_scripts:
            g = SCRIPT_TO_GROUP[s]
            if g not in seen_groups:
                self.active_groups.append(g)
                seen_groups.add(g)
        self.group_to_idx = {g: i for i, g in enumerate(self.active_groups)}
        self.num_groups = len(self.active_groups)

        # Pre-generate images (parallelized across CPU cores)
        if balance_groups:
            samples_per_group = samples_per_script
            total_est = self.num_groups * samples_per_group
        else:
            total_est = self.num_scripts * samples_per_script

        print(f"\nGenerating {total_est} images ({'balanced per group' if balance_groups else 'per script'})...")

        # Build task list: (script, target_count)
        tasks = []
        for script in self.active_scripts:
            group = SCRIPT_TO_GROUP[script]
            if balance_groups:
                scripts_in_group = [s for s in self.active_scripts if SCRIPT_TO_GROUP[s] == group]
                target = samples_per_group // len(scripts_in_group)
            else:
                target = samples_per_script
            tasks.append((script, target))

        # Parallel generation — one process per script, return all at once
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        n_workers = min(len(tasks), os.cpu_count() or 4)
        self.images = []
        self.script_labels = []
        self.group_labels = []

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for script, target in tasks:
                fonts = self.script_fonts[script]
                words = SCRIPT_SAMPLES.get(script, ["placeholder"])
                f = pool.submit(_generate_batch,
                                (script, target, fonts, words, self.height, self.max_width, augment))
                futures[f] = script

            for future in as_completed(futures):
                script = futures[future]
                group = SCRIPT_TO_GROUP[script]
                batch_tensors = future.result()
                for t in batch_tensors:
                    self.images.append(t)
                    self.script_labels.append(self.script_to_idx[script])
                    self.group_labels.append(self.group_to_idx[group])
                print(f"  {script:<15} {len(batch_tensors):>5} images  (group: {group})")

        self.total = len(self.images)
        print(f"Total: {self.total} images, {self.num_scripts} scripts, {self.num_groups} groups\n")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        return self.images[idx], self.script_labels[idx], self.group_labels[idx]


def train_coarse(stem, lid_coarse, dataset, train_set, val_set, args, device, color_proj=None,
                  resume_state=None, shared_swa=None, swa_proj=None):
    """Train LID-1: stem [+ shared SWA] + coarse group classifier."""
    swa_str = f" + {len(shared_swa)} shared SWA blocks" if shared_swa else ""
    print(f"\n{'='*55}")
    print(f"LID-1: COARSE GROUP CLASSIFICATION ({dataset.num_groups} groups)")
    print(f"Pipeline: color_proj → stem{swa_str} → LID-1")
    print(f"{'='*55}\n")
    print(f"Groups: {dataset.active_groups}\n")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    params = list(stem.parameters()) + list(lid_coarse.parameters())
    if color_proj is not None:
        params += list(color_proj.parameters())
    if shared_swa is not None:
        params += list(swa_proj.parameters()) + list(shared_swa.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def forward_to_lid(images):
        """color_proj → stem [→ shared SWA] → LID logits."""
        x = color_proj(images) if color_proj is not None else images
        x = stem(x)  # (B, C, H, W)
        if shared_swa is not None:
            B, C, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
            x = swa_proj(x)
            for block in shared_swa:
                x = block(x, h=h, w=w)
            return lid_coarse.forward_seq(x)  # attention pooling + classify
        return lid_coarse(x)

    start_epoch = 1
    best_val_acc = 0.0
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
        last_val_acc = resume_state.get("val_acc", 0.0)
        best_val_acc = resume_state.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}, val acc: {last_val_acc:.1f}%, best: {best_val_acc:.1f}%\n")

    # AMP setup
    device_type = device if isinstance(device, str) else device.type
    use_amp = device_type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    if use_amp:
        torch.backends.cudnn.benchmark = True
        print(f"AMP: {amp_dtype}, cudnn.benchmark: True")

    for epoch in range(start_epoch, args.epochs + 1):
        stem.train()
        lid_coarse.train()
        total_loss = correct = total = 0

        for images, _, group_labels in train_loader:
            images = images.to(device, non_blocking=True)
            group_labels = group_labels.to(device, non_blocking=True) if isinstance(group_labels, torch.Tensor) else torch.tensor(group_labels, dtype=torch.long, device=device)

            with torch.amp.autocast(device_type, enabled=use_amp, dtype=amp_dtype):
                logits = forward_to_lid(images)
                loss = F.cross_entropy(logits, group_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(-1) == group_labels).sum().item()
            total += images.size(0)

        scheduler.step()

        # Validate
        stem.eval()
        lid_coarse.eval()
        if shared_swa is not None:
            swa_proj.eval()
            shared_swa.eval()
        val_correct = val_total = 0
        per_group_correct = {}
        per_group_total = {}

        confusion = [[0] * dataset.num_groups for _ in range(dataset.num_groups)]
        with torch.no_grad():
            for images, _, group_labels in val_loader:
                images = images.to(device)
                group_labels = group_labels.to(device) if isinstance(group_labels, torch.Tensor) else torch.tensor(group_labels, dtype=torch.long, device=device)
                preds = forward_to_lid(images).argmax(-1)
                val_correct += (preds == group_labels).sum().item()
                val_total += images.size(0)

                for p, l in zip(preds.cpu().tolist(), group_labels.cpu().tolist()):
                    confusion[l][p] += 1
                    g = dataset.active_groups[l]
                    per_group_total[g] = per_group_total.get(g, 0) + 1
                    if p == l:
                        per_group_correct[g] = per_group_correct.get(g, 0) + 1

        val_acc = 100 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch:>2}/{args.epochs}  "
              f"loss={total_loss/total:.4f}  train={100*correct/total:.1f}%  "
              f"val={val_acc:.1f}%")

        # Per-epoch checkpoint
        ckpt_dir = Path(args.save).parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        epoch_ckpt = {
            "stem": stem.state_dict(),
            "lid_coarse": lid_coarse.state_dict(),
            "coarse_optimizer": optimizer.state_dict(),
            "coarse_scheduler": scheduler.state_dict(),
            "coarse_epoch": epoch,
            "coarse_val_acc": val_acc,
            "coarse_best_val_acc": best_val_acc,
            "stem_type": args.stem,
            "stem_depth": args.stem_depth,
            "args": vars(args),
        }
        if color_proj is not None:
            epoch_ckpt["color_proj"] = color_proj.state_dict()
        if shared_swa is not None:
            epoch_ckpt["swa_proj"] = swa_proj.state_dict()
            epoch_ckpt["shared_swa"] = shared_swa.state_dict()
        # Save latest (overwritten each epoch) + best
        torch.save(epoch_ckpt, ckpt_dir / "lid_latest.pt")
        if val_acc >= best_val_acc:
            torch.save(epoch_ckpt, ckpt_dir / "lid_best.pt")

        # Per-epoch confusion matrix
        short_names = [g[:8] for g in dataset.active_groups]
        for i, g in enumerate(dataset.active_groups):
            row = confusion[i]
            row_total = sum(row)
            errs = []
            for j, count in enumerate(row):
                if count > 0 and i != j:
                    pct = 100 * count / row_total if row_total else 0
                    if pct >= 2:
                        errs.append(f"{short_names[j]}:{pct:.0f}%")
            if errs:
                acc = 100 * row[i] / row_total if row_total else 0
                print(f"    {short_names[i]:>8} {acc:.0f}% ok | confused with {', '.join(errs)}")

        # Color projection diagnostic
        if color_proj is not None:
            color_proj.eval()
            test_inputs = {
                "L=0,a=.5": [0.0, 0.5], "L=1,a=.5": [1.0, 0.5],
                "L=.5,a=.5": [0.5, 0.5], "L=.5,a=.7": [0.5, 0.7], "L=.5,a=.3": [0.5, 0.3],
            }
            parts = []
            with torch.no_grad():
                for name, la in test_inputs.items():
                    t = torch.tensor(la, dtype=torch.float32).reshape(1,2,1,1).to(device)
                    out = color_proj(t).squeeze()
                    parts.append(f"{name}→{out.item():.2f}")
            print(f"    color_proj: {' | '.join(parts)}")

    print(f"\nBest coarse val accuracy: {best_val_acc:.1f}%")
    print(f"\nPer-group accuracy (final):")
    for g in dataset.active_groups:
        sc = per_group_correct.get(g, 0)
        st = per_group_total.get(g, 0)
        acc = 100 * sc / st if st > 0 else 0
        scripts_in_group = [s for s in dataset.active_scripts if SCRIPT_TO_GROUP[s] == g]
        print(f"  {g:<20} {acc:>6.1f}%  ({sc}/{st})  scripts: {scripts_in_group}")

    # Confusion matrix
    short_names = [g[:8] for g in dataset.active_groups]
    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    header = "          " + "".join(f"{s:>9}" for s in short_names)
    print(header)
    for i, g in enumerate(dataset.active_groups):
        row = confusion[i]
        row_total = sum(row)
        cells = []
        for j, count in enumerate(row):
            if count == 0:
                cells.append(f"{'·':>9}")
            elif i == j:
                cells.append(f"{count:>9}")
            else:
                pct = 100 * count / row_total if row_total else 0
                cells.append(f"{count:>5}{pct:3.0f}%")
        print(f"{short_names[i]:>9} " + "".join(cells))

    # Print what the projection does to known L+a values
    if color_proj is not None:
        color_proj.eval()
        test_inputs = {
            "black (L=0,a=.5)": [0.0, 0.5],
            "white (L=1,a=.5)": [1.0, 0.5],
            "gray  (L=.5,a=.5)": [0.5, 0.5],
            "red   (L=.5,a=.7)": [0.5, 0.7],
            "green (L=.5,a=.3)": [0.5, 0.3],
        }
        print(f"\nColor projection outputs (L+a → 1ch):")
        with torch.no_grad():
            for name, la in test_inputs.items():
                t = torch.tensor(la, dtype=torch.float32).reshape(1,2,1,1).to(device)
                out = color_proj(t).squeeze()
                print(f"  {name:<20} → {out.item():.3f}")

    return best_val_acc, optimizer.state_dict(), scheduler.state_dict()


def train_fine(stem, stage1_blocks, proj1, lid_fine, dataset, train_set, val_set, args, device, color_proj=None, use_moe=False):
    """Train LID-2: stem + Stage 1 SWA + fine script classifier."""
    print(f"\n{'='*55}")
    print(f"LID-2: FINE SCRIPT CLASSIFICATION ({dataset.num_scripts} scripts)")
    print(f"{'='*55}\n")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_params = (
        list(stem.parameters()) +
        list(proj1.parameters()) +
        [p for block in stage1_blocks for p in block.parameters()] +
        list(lid_fine.parameters())
    )
    if color_proj is not None:
        all_params += list(color_proj.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        stem.train()
        proj1.train()
        for block in stage1_blocks:
            block.train()
        lid_fine.train()

        total_loss = correct = total = 0

        for images, script_labels, group_labels in train_loader:
            images = images.to(device)
            script_labels = script_labels.to(device) if isinstance(script_labels, torch.Tensor) else torch.tensor(script_labels, dtype=torch.long, device=device)
            group_labels = group_labels.to(device) if isinstance(group_labels, torch.Tensor) else torch.tensor(group_labels, dtype=torch.long, device=device)

            # Color projection (learned mode)
            inp = color_proj(images) if color_proj is not None else images

            # Stem
            x = stem(inp)  # (B, 64, 8, W/4)
            B, C, h, w = x.shape

            # Reshape to sequence
            x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)

            # Channel projection
            x = proj1(x)

            # Stage 1 blocks (MoE or shared)
            for block in stage1_blocks:
                if use_moe:
                    x = block(x, h=h, w=w, group_ids=group_labels)
                else:
                    x = block(x, h=h, w=w)

            # LID-2 classifier
            logits = lid_fine(x)
            loss = F.cross_entropy(logits, script_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(-1) == script_labels).sum().item()
            total += images.size(0)

        scheduler.step()

        # Validate
        stem.eval()
        proj1.eval()
        for block in stage1_blocks:
            block.eval()
        lid_fine.eval()

        val_correct = val_total = 0
        per_script_correct = {}
        per_script_total = {}

        confusion = [[0] * dataset.num_scripts for _ in range(dataset.num_scripts)]
        with torch.no_grad():
            for images, script_labels, group_labels in val_loader:
                images = images.to(device)
                script_labels = script_labels.to(device) if isinstance(script_labels, torch.Tensor) else torch.tensor(script_labels, dtype=torch.long, device=device)
                group_labels = group_labels.to(device) if isinstance(group_labels, torch.Tensor) else torch.tensor(group_labels, dtype=torch.long, device=device)

                inp = color_proj(images) if color_proj is not None else images
                x = stem(inp)
                B, C, h, w = x.shape
                x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)
                x = proj1(x)
                for block in stage1_blocks:
                    if use_moe:
                        x = block(x, h=h, w=w, group_ids=group_labels)
                    else:
                        x = block(x, h=h, w=w)

                preds = lid_fine(x).argmax(-1)
                val_correct += (preds == script_labels).sum().item()
                val_total += images.size(0)

                for p, l in zip(preds.cpu().tolist(), script_labels.cpu().tolist()):
                    confusion[l][p] += 1
                    s = dataset.active_scripts[l]
                    per_script_total[s] = per_script_total.get(s, 0) + 1
                    if p == l:
                        per_script_correct[s] = per_script_correct.get(s, 0) + 1

        val_acc = 100 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch:>2}/{args.epochs}  "
              f"loss={total_loss/total:.4f}  train={100*correct/total:.1f}%  "
              f"val={val_acc:.1f}%")

    print(f"\nBest fine val accuracy: {best_val_acc:.1f}%")
    print(f"\nPer-script accuracy (final):")
    for s in dataset.active_scripts:
        sc = per_script_correct.get(s, 0)
        st = per_script_total.get(s, 0)
        acc = 100 * sc / st if st > 0 else 0
        print(f"  {s:<15} {acc:>6.1f}%  ({sc}/{st})  group: {SCRIPT_TO_GROUP[s]}")

    # Confusion matrix
    short_names = [s[:6] for s in dataset.active_scripts]
    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    header = "          " + "".join(f"{s:>7}" for s in short_names)
    print(header)
    for i, s in enumerate(dataset.active_scripts):
        row = confusion[i]
        row_total = sum(row)
        cells = []
        for j, count in enumerate(row):
            if count == 0:
                cells.append(f"{'·':>7}")
            elif i == j:
                cells.append(f"{count:>7}")
            else:
                pct = 100 * count / row_total if row_total else 0
                cells.append(f"{count:>3}{pct:3.0f}%")
        print(f"{short_names[i]:>9} " + "".join(cells))

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train LID-1 group classifier")
    parser.add_argument("--stem", type=str, default="resnet", choices=["convnext", "resnet"])
    parser.add_argument("--stem-depth", type=int, default=3, choices=[2, 3],
                        help="ResNet stem depth (default: 3)")
    parser.add_argument("--shared-swa-blocks", type=int, default=0,
                        help="Shared SWA blocks before LID (0=stem only, 2-3 for GPU)")
    parser.add_argument("--swa-dim", type=int, default=288,
                        help="Shared SWA channel dim (default: 288)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-script", type=int, default=2000)
    parser.add_argument("--balance-groups", action="store_true",
                        help="Balance samples per GROUP instead of per script")
    parser.add_argument("--augment", action="store_true",
                        help="Apply RandAugmentOCR to training images")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster training")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-width", type=int, default=192)
    parser.add_argument("--data", type=str, default=None,
                        help="Pre-generated data file from generate_lid_data.py (skips generation)")
    parser.add_argument("--save", type=str, default="checkpoints/lid_hierarchical.pt")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}\n")

    # Load dataset — either from pre-generated file or generate on the fly
    if args.data:
        data_path = Path(args.data)
        if data_path.is_dir():
            # Shard directory from generate_lid_data.py
            print(f"Loading shards from {data_path}/...")
            meta = torch.load(data_path / "metadata.pt", weights_only=False)
            shard_files = sorted(data_path.glob("shard_*.pt"))
            print(f"  {len(shard_files)} shards found")

            from concurrent.futures import ThreadPoolExecutor
            def _load_shard(p):
                return torch.load(p, weights_only=False)

            all_imgs, all_slabels, all_glabels = [], [], []
            with ThreadPoolExecutor(max_workers=16) as pool:
                for shard in pool.map(_load_shard, shard_files):
                    all_imgs.append(shard["images"])
                    all_slabels.append(shard.get("script_labels", shard.get("script_ids")))
                    all_glabels.append(shard.get("group_labels", shard.get("group_ids")))

            print(f"  Concatenating...")
            images = torch.cat(all_imgs)
            script_labels = torch.cat(all_slabels)
            group_labels = torch.cat(all_glabels)
            del all_imgs, all_slabels, all_glabels
        else:
            # Single .pt file
            print(f"Loading data from {data_path}...")
            saved = torch.load(data_path, weights_only=False)
            meta = saved
            images = saved["images"]
            script_labels = saved["script_labels"]
            group_labels = saved["group_labels"]

        # Remap global group IDs to contiguous 0..N-1
        active_groups = meta["active_groups"]
        global_to_local = {}
        for local_id, gname in enumerate(active_groups):
            from src.model.lid import GROUP_TO_ID as _G2ID
            global_to_local[_G2ID[gname]] = local_id
        remapped = group_labels.clone()
        for gid, lid in global_to_local.items():
            remapped[group_labels == gid] = lid
        group_labels = remapped

        dataset = torch.utils.data.TensorDataset(images, script_labels, group_labels)
        dataset.active_scripts = meta["active_scripts"]
        dataset.active_groups = meta["active_groups"]
        dataset.script_to_idx = meta["script_to_idx"]
        dataset.group_to_idx = meta["group_to_idx"]
        dataset.num_scripts = len(meta["active_scripts"])
        dataset.num_groups = len(meta["active_groups"])
        print(f"  {len(dataset)} images, {dataset.num_scripts} scripts, {dataset.num_groups} groups")
    else:
        load_all_script_samples()
        dataset = MultiScriptDataset(
            samples_per_script=args.samples_per_script,
            height=32,
            max_width=args.max_width,
            balance_groups=args.balance_groups,
            augment=args.augment,
        )

    # 80/20 split
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"Train: {n_train}, Val: {n_val}\n")

    color_proj = ColorProjection().to(device)
    proj_params = sum(p.numel() for p in color_proj.parameters())
    print(f"Color projection: L+a + learned correction, {proj_params} params")

    stem_channels = 64
    if args.stem == "resnet":
        stem = ResNetStem(in_channels=INPUT_CHANNELS, out_channels=stem_channels,
                          depth=args.stem_depth).to(device)
    else:
        stem = ConvNeXtStem(in_channels=INPUT_CHANNELS, out_channels=stem_channels).to(device)

    stem_params = sum(p.numel() for p in stem.parameters())
    print(f"{args.stem} stem (depth={args.stem_depth}): {stem_params:,} params")

    # Optional shared SWA blocks before LID
    shared_swa = None
    swa_proj = None
    lid_in_channels = stem_channels
    if args.shared_swa_blocks > 0:
        from src.model.attention import SWABlock

        swa_proj = nn.Linear(stem_channels, args.swa_dim).to(device)
        shared_swa = nn.ModuleList([
            SWABlock(
                dim=args.swa_dim,
                num_heads=args.swa_dim // 32,
                window_h=4, window_w=4,
                shift=(i % 2 == 1), mlp_ratio=4,
            )
            for i in range(args.shared_swa_blocks)
        ]).to(device)
        lid_in_channels = args.swa_dim

        swa_params = sum(p.numel() for p in swa_proj.parameters()) + sum(p.numel() for p in shared_swa.parameters())
        print(f"Shared SWA: {swa_params:,} params ({args.shared_swa_blocks} blocks, dim={args.swa_dim})")

    lid_coarse = LIDCoarse(in_channels=lid_in_channels, num_groups=dataset.num_groups).to(device)
    print(f"LID-1 (coarse): {sum(p.numel() for p in lid_coarse.parameters()):,} params")

    # torch.compile
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        stem = torch.compile(stem)
        lid_coarse = torch.compile(lid_coarse)
        color_proj = torch.compile(color_proj)
        if shared_swa is not None:
            swa_proj = torch.compile(swa_proj)
            shared_swa = torch.compile(shared_swa)
        print("  Done.")

    # Resume
    coarse_resume = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        stem.load_state_dict(ckpt["stem"])
        if "color_proj" in ckpt:
            color_proj.load_state_dict(ckpt["color_proj"])
        # Only load LID/SWA if architecture matches
        try:
            lid_coarse.load_state_dict(ckpt["lid_coarse"])
        except (RuntimeError, KeyError):
            print("  LID weights incompatible (architecture changed), starting fresh")
        if shared_swa is not None and "shared_swa" in ckpt:
            swa_proj.load_state_dict(ckpt["swa_proj"])
            shared_swa.load_state_dict(ckpt["shared_swa"])
        if "coarse_optimizer" in ckpt:
            coarse_resume = {
                "optimizer": ckpt["coarse_optimizer"],
                "scheduler": ckpt["coarse_scheduler"],
                "epoch": ckpt["coarse_epoch"],
                "val_acc": ckpt.get("coarse_val_acc", 0.0),
                "best_val_acc": ckpt.get("coarse_best_val_acc", 0.0),
            }
        print(f"Loaded checkpoint from {args.resume}")

    start = time.time()
    coarse_acc, coarse_optim_state, coarse_sched_state = train_coarse(
        stem, lid_coarse, dataset, train_set, val_set, args, device,
        color_proj=color_proj, resume_state=coarse_resume,
        shared_swa=shared_swa, swa_proj=swa_proj,
    )

    elapsed = time.time() - start

    # Save
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "stem": stem.state_dict(),
        "stem_type": args.stem,
        "stem_depth": args.stem_depth,
        "active_scripts": dataset.active_scripts,
        "active_groups": dataset.active_groups,
        "script_to_idx": dataset.script_to_idx,
        "group_to_idx": dataset.group_to_idx,
        "args": vars(args),
        "lid_coarse": lid_coarse.state_dict(),
        "coarse_acc": coarse_acc,
        "coarse_optimizer": coarse_optim_state,
        "coarse_scheduler": coarse_sched_state,
        "coarse_epoch": args.epochs,
        "coarse_best_val_acc": coarse_acc,
    }
    if color_proj is not None:
        save_dict["color_proj"] = color_proj.state_dict()
    if shared_swa is not None:
        save_dict["swa_proj"] = swa_proj.state_dict()
        save_dict["shared_swa"] = shared_swa.state_dict()

    torch.save(save_dict, save_path)

    print(f"\n{'='*55}")
    print(f"Total time: {elapsed:.0f}s")
    swa_str = f" + {args.shared_swa_blocks} SWA blocks" if args.shared_swa_blocks > 0 else ""
    print(f"LID-1 ({dataset.num_groups} groups, stem{swa_str}): {coarse_acc:.1f}%")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
