#!/usr/bin/env python3
"""
MoE Encoder Training Script.

Trains the full LipiMoEEncoder with three losses:
  - CTC loss (weight 1.0): character recognition via per-script BiLSTM heads
  - LID-1 CE loss (weight 0.1): coarse group classification
  - LID-2 CE loss (weight 0.1): fine script classification

Supports two data modes:
  --synth: on-the-fly synthetic rendering (laptop testing, no LMDB needed)
  --train_dir: LMDB datasets with PARSeq format (full GPU training)

Usage:
    # Quick 2-expert test on laptop with synthetic data
    python scripts/train_moe.py --synth --scripts latin,devanagari --epochs 5

    # Full training on GPU with LMDB data
    python scripts/train_moe.py --train_dir data/train --scripts latin,devanagari --epochs 20

    # All scripts, synthetic data
    python scripts/train_moe.py --synth --scripts all --epochs 10
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.moe_encoder import LipiMoEEncoder
from src.model.lid import (
    SCRIPTS, SCRIPT_TO_ID, NUM_SCRIPTS,
    GROUPS, GROUP_TO_ID, NUM_GROUPS,
    SCRIPT_TO_GROUP,
)
from src.data.color import rgb_to_input
from src.data.script_detect import detect_script_id, detect_group_id
from src.data.bigrams import LipiTokenizer, SCRIPT_CHARSETS, BASE_CHARS, BLANK_TOKEN

# Import synthetic rendering utilities from train_lid
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Script name -> tokenizer script_id mapping
# The LipiTokenizer uses language codes ("en", "hi", etc.) while the MoE
# encoder uses script names ("latin", "devanagari", etc.).
# ---------------------------------------------------------------------------
SCRIPT_TO_LANG = {
    "latin": "en",
    "cyrillic": "en",     # Cyrillic chars not in SCRIPT_CHARSETS, use base
    "greek": "en",        # Same
    "arabic": "ur",
    "hebrew": "en",       # Hebrew chars not in SCRIPT_CHARSETS
    "cjk": "en",
    "korean": "en",
    "devanagari": "hi",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "bengali": "bn_as",
    "kannada": "kn",
    "telugu": "te",
    "malayalam": "ml",
    "tamil": "ta",
    "thai": "en",
    "lao": "en",
    "emoji": "en",
}


# ---------------------------------------------------------------------------
# Synthetic data: reuse rendering from train_lid.py
# ---------------------------------------------------------------------------
WORD_LIST_DIR = Path(__file__).parent.parent / "training_data" / "word_lists"


# Extra language word lists that belong to each script
_SCRIPT_EXTRA_FILES = {
    "latin": ["english_common.txt", "french.txt", "german.txt", "spanish.txt",
              "turkish.txt", "vietnamese.txt", "italian.txt", "portuguese.txt",
              "polish.txt", "dutch.txt", "romanian.txt", "czech.txt",
              "hungarian.txt", "swedish.txt", "norwegian.txt", "danish.txt",
              "finnish.txt", "croatian.txt", "indonesian.txt", "malay.txt",
              "swahili.txt"],
    "cyrillic": ["ukrainian.txt"],
    "devanagari": ["marathi.txt"],
    "arabic": ["persian.txt", "urdu.txt"],
    "cjk": ["japanese.txt"],
}


def load_word_list(script: str, fallback: list[str]) -> list[str]:
    """Load and merge all word list files for a script."""
    files = [WORD_LIST_DIR / f"{script}.txt"]
    for extra in _SCRIPT_EXTRA_FILES.get(script, []):
        files.append(WORD_LIST_DIR / extra)

    words = []
    loaded_files = []
    for path in files:
        if path.exists():
            count_before = len(words)
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if 2 <= len(w) <= 15 and w and not w[0].isdigit():
                    words.append(w)
            if len(words) > count_before:
                loaded_files.append(f"{path.name}({len(words) - count_before})")

    if words:
        words = list(set(words))
        random.shuffle(words)
        if loaded_files:
            print(f"    {script}: {len(words)} unique words from {', '.join(loaded_files)}")
        return words
    return fallback


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
    "cjk": [
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
    ],
    "lao": [
        "ສະບາຍດີ", "ກົດໝາຍ", "ສານ", "ປະເທດ", "ເມືອງ",
        "ຄົນ", "ເຮືອນ", "ນ້ຳ", "ເວລາ", "ວຽກ", "ໂຮງຮຽນ",
        "ຕະຫຼາດ", "ທາງ", "ພູ", "ແມ່ນ້ຳ", "ກິນ", "ດື່ມ",
    ],
}

# Populated at runtime by load_script_samples()
SCRIPT_SAMPLES: dict[str, list[str]] = {}


def load_script_samples():
    """Load word lists for all scripts. Uses files when available, else fallback."""
    for script, fallback in _SCRIPT_SAMPLES_FALLBACK.items():
        SCRIPT_SAMPLES[script] = load_word_list(script, fallback)


FONT_DIR = Path(__file__).parent.parent / "training_data" / "fonts"

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
    "greek": ["NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSerif-Regular.ttf"],
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
    "cjk": [
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

_LANG_MAP = {
    "latin": "en", "cyrillic": "ru", "greek": "el",
    "devanagari": "hi", "bengali": "bn", "tamil": "ta",
    "telugu": "te", "kannada": "kn", "malayalam": "ml",
    "gujarati": "gu", "gurmukhi": "pa",
    "arabic": "ar", "hebrew": "he", "cjk": "zh", "korean": "ko",
    "thai": "th", "lao": "lo",
}


def find_fonts_for_script(script: str) -> list[str]:
    """Find fonts for a script. Checks local Noto fonts first, then system."""
    import subprocess
    fonts = []

    for name in _LOCAL_FONT_MAP.get(script, []):
        path = FONT_DIR / name
        if path.exists():
            fonts.append(str(path))

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


def font_can_render(font_path: str, text: str, size: int = 24) -> bool:
    """Check if a font produces ink for the given text."""
    try:
        font = ImageFont.truetype(font_path, size=size)
        img = Image.new("L", (300, 50), 255)
        ImageDraw.Draw(img).text((5, 5), text, fill=0, font=font)
        return (np.array(img) < 200).sum() > len(text) * 3
    except Exception:
        return False


def random_ink_color():
    """Random text color -- realistic document ink colors."""
    style = random.choice([
        "black", "black", "black", "black",
        "dark_gray", "dark_gray",
        "blue_ink",
        "red", "red",
        "dark_green",
    ])
    colors = {
        "black": (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40)),
        "dark_gray": (random.randint(50, 90),) * 3,
        "blue_ink": (random.randint(0, 40), random.randint(0, 50), random.randint(130, 210)),
        "red": (random.randint(160, 230), random.randint(0, 50), random.randint(0, 50)),
        "dark_green": (random.randint(0, 50), random.randint(90, 150), random.randint(0, 50)),
    }
    return colors[style]


def random_bg_color():
    """Random background -- paper colors."""
    style = random.choice([
        "white", "white", "white",
        "offwhite", "offwhite",
        "cream", "light_blue", "light_pink", "gray", "light_green",
    ])
    colors = {
        "white": (random.randint(240, 255),) * 3,
        "offwhite": (random.randint(230, 250), random.randint(228, 248), random.randint(220, 240)),
        "cream": (random.randint(240, 255), random.randint(235, 250), random.randint(200, 225)),
        "light_blue": (random.randint(220, 240), random.randint(230, 248), random.randint(245, 255)),
        "light_pink": (random.randint(245, 255), random.randint(220, 235), random.randint(225, 240)),
        "gray": (random.randint(200, 225),) * 3,
        "light_green": (random.randint(225, 242), random.randint(245, 255), random.randint(225, 242)),
    }
    return colors[style]


def render_word(text: str, font_path: str, height: int = 32) -> Image.Image | None:
    """Render a word with random ink and paper colors."""
    try:
        font_size = random.randint(18, 26)
        font = ImageFont.truetype(font_path, size=font_size)

        dummy = Image.new("RGB", (1, 1))
        bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= 0 or text_h <= 0:
            return None

        pad_x = random.randint(2, 8)
        pad_y = random.randint(2, 6)
        img_w = text_w + 2 * pad_x
        img_h = text_h + 2 * pad_y

        bg = random_bg_color()
        ink = random_ink_color()
        img = Image.new("RGB", (img_w, img_h), bg)
        ImageDraw.Draw(img).text(
            (pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink, font=font,
        )

        scale = height / img_h
        new_w = max(4, int(img_w * scale))
        return img.resize((new_w, height), Image.BILINEAR)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build a unified tokenizer covering all selected scripts
# ---------------------------------------------------------------------------
def build_multi_script_tokenizer(scripts: list[str]) -> LipiTokenizer:
    """Build a character-level tokenizer covering all selected scripts.

    Merges BASE_CHARS with all script-specific character sets needed.
    No bigrams -- character-level only for now.
    """
    all_chars = list(BASE_CHARS)
    seen = set(BASE_CHARS)

    for script in scripts:
        lang = SCRIPT_TO_LANG.get(script, "en")
        script_chars = SCRIPT_CHARSETS.get(lang, [])
        for ch in script_chars:
            if ch not in seen:
                seen.add(ch)
                all_chars.append(ch)

    vocab = [BLANK_TOKEN] + all_chars
    return LipiTokenizer(vocab=vocab, bigrams=set())


# ---------------------------------------------------------------------------
# Synthetic dataset: renders word images with text labels for CTC training
# ---------------------------------------------------------------------------
class SyntheticMoEDataset(Dataset):
    """On-the-fly synthetic word image dataset with text labels.

    Each sample: (rgb_tensor, text_label, script_id, group_id)
    Images are rendered at 32px height with random fonts/colors.
    """

    def __init__(
        self,
        scripts: list[str],
        samples_per_script: int = 500,
        height: int = 32,
        max_width: int = 192,
        augment: bool = True,
    ):
        self.height = height
        self.max_width = max_width
        from src.data.augmentation import RandAugmentOCR
        self.augmentor = RandAugmentOCR(n_ops=2, p=0.5) if augment else None

        print("Discovering fonts per script...")
        self.script_fonts: dict[str, list[str]] = {}
        self.active_scripts: list[str] = []

        for script in scripts:
            if script not in SCRIPT_SAMPLES:
                print(f"  {script:<15}   no word samples -- SKIPPED")
                continue
            fonts = find_fonts_for_script(script)
            sample_text = SCRIPT_SAMPLES[script][0]
            valid_fonts = [f for f in fonts if font_can_render(f, sample_text)]
            if valid_fonts:
                self.script_fonts[script] = valid_fonts
                self.active_scripts.append(script)
                print(f"  {script:<15} {len(valid_fonts):>3} fonts")
            else:
                print(f"  {script:<15}   0 fonts -- SKIPPED")

        if len(self.active_scripts) < 1:
            raise RuntimeError("Need at least 1 script with valid fonts")

        # Pre-generate samples
        self.images: list[torch.Tensor] = []
        self.labels: list[str] = []
        self.script_ids: list[int] = []
        self.group_ids: list[int] = []

        print(f"\nGenerating {len(self.active_scripts) * samples_per_script} synthetic samples...")

        for script in self.active_scripts:
            fonts = self.script_fonts[script]
            words = SCRIPT_SAMPLES[script]
            sid = SCRIPT_TO_ID[script]
            gid = GROUP_TO_ID[SCRIPT_TO_GROUP[script]]

            generated = 0
            attempts = 0
            while generated < samples_per_script and attempts < samples_per_script * 5:
                attempts += 1
                word = random.choice(words)
                font_path = random.choice(fonts)
                img = render_word(word, font_path, self.height)
                if img is None:
                    continue

                # Clamp width
                if img.width > self.max_width:
                    img = img.resize((self.max_width, self.height), Image.BILINEAR)

                # Augment (on RGB, before color conversion)
                if self.augmentor is not None:
                    img = self.augmentor(img)

                # Convert to tensor via rgb_to_input
                tensor = rgb_to_input(img)
                self.images.append(tensor)
                self.labels.append(word)
                self.script_ids.append(sid)
                self.group_ids.append(gid)
                generated += 1

            print(f"  {script:<15} {generated:>5} samples  (script_id={sid}, group={SCRIPT_TO_GROUP[script]})")

        self.total = len(self.images)
        print(f"Total: {self.total} samples, {len(self.active_scripts)} scripts\n")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.script_ids[idx], self.group_ids[idx]


# ---------------------------------------------------------------------------
# LMDB dataset wrapper: auto-detects script from labels
# ---------------------------------------------------------------------------
class LMDBMoEDataset(Dataset):
    """Wraps PARSeqLMDB with script/group auto-detection from text labels."""

    def __init__(self, lmdb_path: str, max_width: int = 192, augment: bool = False):
        from src.data.parseq_lmdb import PARSeqLMDB
        self.ds = PARSeqLMDB(lmdb_path, max_width=max_width, augment=augment)
        self._max_width = max_width

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_arr, label = self.ds[idx]
        # PARSeqLMDB returns (C, H, W) numpy, normalized to [-1,1].
        # Undo normalization to get [0,1] RGB, convert to PIL, then rgb_to_input.
        arr = (img_arr.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)  # (H, W, 3) in [0,1]
        pil_img = Image.fromarray((arr * 255).astype(np.uint8))
        tensor = rgb_to_input(pil_img)

        script_id = detect_script_id(label)
        group_id = detect_group_id(label)
        return tensor, label, script_id, group_id

    def close(self):
        if hasattr(self.ds, 'close'):
            self.ds.close()


# ---------------------------------------------------------------------------
# Collate function: pads images to max width in batch
# ---------------------------------------------------------------------------
def collate_moe(
    batch: list[tuple[torch.Tensor, str, int, int]],
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-width images into a padded batch.

    Args:
        batch: List of (image_tensor, label, script_id, group_id) tuples.

    Returns:
        images: (B, C, H, max_W) tensor.
        labels: list of B label strings.
        script_ids: (B,) long tensor.
        group_ids: (B,) long tensor.
        widths: (B,) long tensor of original widths.
    """
    images, labels, script_ids, group_ids = zip(*batch)

    C = images[0].shape[0]
    H = images[0].shape[1]
    max_w = max(img.shape[2] for img in images)
    # Ensure width is divisible by 4 (stem stride)
    max_w = ((max_w + 3) // 4) * 4
    B = len(images)

    padded = torch.zeros(B, C, H, max_w, dtype=torch.float32)
    widths = torch.zeros(B, dtype=torch.long)

    for i, img in enumerate(images):
        w = img.shape[2]
        padded[i, :, :, :w] = img
        widths[i] = w

    script_ids_t = torch.tensor(script_ids, dtype=torch.long)
    group_ids_t = torch.tensor(group_ids, dtype=torch.long)

    return padded, list(labels), script_ids_t, group_ids_t, widths


# ---------------------------------------------------------------------------
# CTC greedy decode
# ---------------------------------------------------------------------------
def ctc_greedy_decode(logits: torch.Tensor, tokenizer: LipiTokenizer) -> list[str]:
    """Greedy CTC decode: argmax -> collapse repeats -> remove blank."""
    preds = logits.argmax(dim=-1)
    results = []
    for i in range(preds.shape[0]):
        p = preds[i].tolist()
        collapsed = [p[0]] + [p[j] for j in range(1, len(p)) if p[j] != p[j - 1]]
        collapsed = [x for x in collapsed if x != 0]
        results.append(tokenizer.decode(collapsed))
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: LipiMoEEncoder,
    tokenizer: LipiTokenizer,
    dataloader: DataLoader,
    device: torch.device,
    active_scripts: list[str],
) -> dict:
    """Evaluate CTC accuracy, LID-1 accuracy, LID-2 accuracy, per-script CTC accuracy."""
    model.eval()

    total_correct = 0
    total_samples = 0
    lid1_correct = 0
    lid2_correct = 0
    per_script_correct: dict[str, int] = {}
    per_script_total: dict[str, int] = {}

    for batch_imgs, batch_labels, batch_sids, batch_gids, widths in dataloader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        batch_sids = batch_sids.to(device, non_blocking=True)
        batch_gids = batch_gids.to(device, non_blocking=True)

        out = model(batch_imgs, script_ids=None, group_ids=None)
        logits = out["logits"]
        decoded = ctc_greedy_decode(logits.float().cpu(), tokenizer)

        # LID accuracy (compare predicted vs ground truth)
        pred_gids = out["group_logits"].argmax(dim=-1)
        pred_sids = out["script_logits"].argmax(dim=-1)
        lid1_correct += (pred_gids == batch_gids).sum().item()
        lid2_correct += (pred_sids == batch_sids).sum().item()

        for i, (dec, label) in enumerate(zip(decoded, batch_labels)):
            sid = batch_sids[i].item()
            script_name = SCRIPTS[sid] if 0 <= sid < len(SCRIPTS) else "unknown"

            if script_name not in per_script_total:
                per_script_total[script_name] = 0
                per_script_correct[script_name] = 0
            per_script_total[script_name] += 1

            if dec.lower() == label.lower():
                total_correct += 1
                per_script_correct[script_name] += 1
            total_samples += 1

    # Print results
    ctc_acc = total_correct / max(total_samples, 1) * 100
    lid1_acc = lid1_correct / max(total_samples, 1) * 100
    lid2_acc = lid2_correct / max(total_samples, 1) * 100

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-' * 40}")
    print(f"  {'CTC Accuracy':<25} {ctc_acc:>9.1f}%")
    print(f"  {'LID-1 (Group) Accuracy':<25} {lid1_acc:>9.1f}%")
    print(f"  {'LID-2 (Script) Accuracy':<25} {lid2_acc:>9.1f}%")

    print(f"\n  {'Script':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-' * 50}")
    for script in active_scripts:
        if script in per_script_total:
            sc = per_script_correct.get(script, 0)
            st = per_script_total[script]
            acc = sc / max(st, 1) * 100
            print(f"  {script:<20} {sc:>8} {st:>8} {acc:>9.1f}%")

    model.train()

    return {
        "ctc_accuracy": ctc_acc,
        "lid1_accuracy": lid1_acc,
        "lid2_accuracy": lid2_acc,
        "per_script": {s: per_script_correct.get(s, 0) / max(per_script_total.get(s, 1), 1) * 100
                       for s in active_scripts if s in per_script_total},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Lipi MoE Encoder")
    parser.add_argument("--scripts", type=str, default="latin,devanagari",
                        help="Comma-separated script names, or 'all' (default: latin,devanagari)")
    parser.add_argument("--synth", action="store_true",
                        help="Generate synthetic data on-the-fly (no LMDB needed)")
    parser.add_argument("--train_dir", type=str, nargs="*", default=None,
                        help="LMDB training data directories")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--max-width", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader workers (-1 = auto based on CPU cores)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() for faster training (requires PyTorch 2.0+)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints/moe")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--synth-samples", type=int, default=500,
                        help="Samples per script in synthetic mode (default: 500)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--log-interval", type=int, default=20,
                        help="Log every N batches")

    # Model size overrides
    parser.add_argument("--stem-depth", type=int, default=3)
    parser.add_argument("--shared-blocks", type=int, default=3,
                        help="Shared SWA blocks before LID")
    parser.add_argument("--shared-dim", type=int, default=288)
    parser.add_argument("--stage1-dim", type=int, default=288)
    parser.add_argument("--stage1-blocks", type=int, default=5)
    parser.add_argument("--stage2-dim", type=int, default=576)
    parser.add_argument("--stage2-blocks", type=int, default=9)
    parser.add_argument("--head-hidden", type=int, default=246)

    # Loss weights
    parser.add_argument("--w-ctc", type=float, default=1.0, help="CTC loss weight")
    parser.add_argument("--w-lid1", type=float, default=0.1, help="LID-1 CE loss weight")
    parser.add_argument("--w-lid2", type=float, default=0.1, help="LID-2 CE loss weight")

    args = parser.parse_args()

    # ---- Device ----
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Parse scripts ----
    if args.scripts.lower() == "all":
        selected_scripts = [s for s in SCRIPTS if s != "emoji"]
    else:
        selected_scripts = [s.strip() for s in args.scripts.split(",")]
    for s in selected_scripts:
        if s not in SCRIPTS:
            print(f"ERROR: Unknown script '{s}'. Valid: {SCRIPTS}")
            sys.exit(1)
    print(f"Scripts: {selected_scripts}")

    # ---- Load word lists ----
    load_script_samples()

    # ---- Validate data mode ----
    if not args.synth and not args.train_dir:
        print("ERROR: Must specify --synth or --train_dir")
        sys.exit(1)

    # ---- Build tokenizer ----
    tokenizer = build_multi_script_tokenizer(selected_scripts)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ---- Build dataset ----
    if args.synth:
        full_dataset = SyntheticMoEDataset(
            scripts=selected_scripts,
            samples_per_script=args.synth_samples,
            height=32,
            max_width=args.max_width,
        )
        active_scripts = full_dataset.active_scripts
    else:
        from torch.utils.data import ConcatDataset
        datasets = []
        for path in args.train_dir:
            print(f"  Loading LMDB: {path}")
            ds = LMDBMoEDataset(path, max_width=args.max_width)
            print(f"    {len(ds)} samples")
            datasets.append(ds)
        if len(datasets) == 1:
            full_dataset = datasets[0]
        else:
            full_dataset = ConcatDataset(datasets)
        print(f"  Total: {len(full_dataset)} samples")
        active_scripts = selected_scripts

    # ---- Train/val split ----
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train}, Val: {n_val}")

    if args.num_workers < 0:
        import os
        n_workers = min(os.cpu_count() or 4, 16) if device.type == "cuda" else 0
    else:
        n_workers = args.num_workers
    is_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_moe, num_workers=n_workers,
        pin_memory=is_cuda,
        persistent_workers=n_workers > 0,
        prefetch_factor=4 if n_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_moe, num_workers=n_workers,
        pin_memory=is_cuda,
        persistent_workers=n_workers > 0,
        prefetch_factor=4 if n_workers > 0 else None,
    )

    # ---- Build model ----
    model = LipiMoEEncoder(
        stem_depth=args.stem_depth,
        shared_dim=args.shared_dim,
        shared_blocks=args.shared_blocks,
        stage1_dim=args.stage1_dim,
        stage1_blocks=args.stage1_blocks,
        stage2_dim=args.stage2_dim,
        stage2_blocks=args.stage2_blocks,
        vocab_size=tokenizer.vocab_size,
        head_hidden=args.head_hidden,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params / 1e6:.2f}M")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ---- AMP setup ----
    use_amp = device.type in ("cuda", "mps")
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # GradScaler only needed for fp16, not bf16
        use_scaler = amp_dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    elif device.type == "mps":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler(enabled=False)
    else:
        amp_dtype = torch.float32
        scaler = torch.amp.GradScaler(enabled=False)

    # ---- torch.compile ----
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("  Done.")

    # ---- Scheduler ----
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(steps_per_epoch, total_steps // 10)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )

    # ---- Resume ----
    start_epoch = 1
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed at epoch {start_epoch}, loss was {ckpt.get('loss', '?')}")

    # ---- Loss functions ----
    ce_loss_fn = nn.CrossEntropyLoss()

    # ---- Training loop ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    eff_batch = args.batch_size * args.grad_accum
    print(f"\n{'=' * 60}")
    print(f"TRAINING: epochs {start_epoch}-{args.epochs}, lr={args.lr}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} accum = {eff_batch} effective")
    print(f"  Workers: {n_workers}, AMP: {amp_dtype}, Scaler: {scaler.is_enabled()}")
    print(f"  Compile: {args.compile and hasattr(torch, 'compile')}")
    print(f"  Losses: CTC x{args.w_ctc} + LID1 x{args.w_lid1} + LID2 x{args.w_lid2}")
    print(f"{'=' * 60}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_ctc_loss = 0.0
        epoch_lid1_loss = 0.0
        epoch_lid2_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for batch_idx, (batch_imgs, batch_labels, batch_sids, batch_gids, widths) in enumerate(train_loader):
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_sids = batch_sids.to(device, non_blocking=True)
            batch_gids = batch_gids.to(device, non_blocking=True)

            # Encode targets for CTC
            target_ids = [tokenizer.encode(label) for label in batch_labels]
            target_lengths = torch.tensor(
                [len(ids) for ids in target_ids], dtype=torch.long, device=device,
            )

            # Skip batches with empty labels
            if (target_lengths == 0).any():
                continue

            max_tgt = target_lengths.max().item()
            targets = torch.zeros(len(batch_labels), max_tgt, dtype=torch.long, device=device)
            for i, ids in enumerate(target_ids):
                if ids:
                    targets[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            # Forward pass with AMP
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(batch_imgs, script_ids=batch_sids, group_ids=batch_gids)
                logits = out["logits"]
                enc_lengths = out["lengths"]

                # LID losses (inside autocast)
                lid1_loss = ce_loss_fn(out["group_logits"], batch_gids)
                lid2_loss = ce_loss_fn(out["script_logits"], batch_sids)

            # CTC loss (outside autocast for float32 stability)
            log_probs = logits.float().log_softmax(dim=-1).permute(1, 0, 2)  # (T, B, V)
            loss_ctc = F.ctc_loss(
                log_probs, targets,
                enc_lengths, target_lengths,
                blank=tokenizer.blank_id,
                reduction="mean", zero_infinity=True,
            )

            # Skip bad batches
            if torch.isinf(loss_ctc) or torch.isnan(loss_ctc):
                continue

            # Combined loss
            loss = (args.w_ctc * loss_ctc
                    + args.w_lid1 * lid1_loss.float()
                    + args.w_lid2 * lid2_loss.float())

            # Scale loss for gradient accumulation
            if args.grad_accum > 1:
                loss = loss / args.grad_accum

            # Backward
            scaler.scale(loss).backward()

            # Optimizer step every grad_accum batches
            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()

            # Track losses
            epoch_ctc_loss += loss_ctc.item()
            epoch_lid1_loss += lid1_loss.item()
            epoch_lid2_loss += lid2_loss.item()
            epoch_total_loss += loss.item()
            n_batches += 1

            if n_batches % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  [{epoch}/{args.epochs}] batch {n_batches}/{steps_per_epoch}  "
                      f"loss={loss.item():.4f} (ctc={loss_ctc.item():.4f} "
                      f"lid1={lid1_loss.item():.4f} lid2={lid2_loss.item():.4f})  "
                      f"lr={lr:.2e}")

        # Epoch summary
        if n_batches == 0:
            print(f"Epoch {epoch}: no valid batches")
            continue

        avg_total = epoch_total_loss / n_batches
        avg_ctc = epoch_ctc_loss / n_batches
        avg_lid1 = epoch_lid1_loss / n_batches
        avg_lid2 = epoch_lid2_loss / n_batches
        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch}/{args.epochs}: "
              f"total={avg_total:.4f} ctc={avg_ctc:.4f} "
              f"lid1={avg_lid1:.4f} lid2={avg_lid2:.4f}  "
              f"time={elapsed:.0f}s")

        # Save checkpoint
        ckpt_path = save_dir / f"moe_epoch{epoch}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "loss": avg_total,
            "args": vars(args),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # Evaluate
        print(f"\n  Evaluation after epoch {epoch}:")
        eval_results = evaluate(model, tokenizer, val_loader, device, active_scripts)

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 3600:.1f}h)")
    print(f"  Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
