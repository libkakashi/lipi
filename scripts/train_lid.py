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
from PIL import Image, ImageDraw, ImageFont

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


def find_fonts_for_script(script: str) -> list[str]:
    """Find system fonts that can render a given script."""
    if script == "emoji":
        return ["__emoji__"]  # special marker, rendered differently

    lang_map = {
        "latin": "en", "cyrillic": "ru", "greek": "el",
        "devanagari": "hi", "bengali": "bn", "tamil": "ta",
        "telugu": "te", "kannada": "kn", "malayalam": "ml",
        "gujarati": "gu", "gurmukhi": "pa",
        "arabic": "ar", "hebrew": "he", "cjk": "zh", "korean": "ko",
        "thai": "th", "lao": "lo",
    }
    lang = lang_map.get(script)
    if not lang:
        return []
    try:
        result = subprocess.run(
            ["fc-list", f":lang={lang}", "file"],
            capture_output=True, text=True, timeout=5,
        )
        fonts = []
        for line in result.stdout.strip().split("\n"):
            path = line.strip().rstrip(":")
            if path and Path(path).exists():
                fonts.append(path)
        return fonts
    except Exception:
        return []


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

        # Pick colors with enough contrast
        for _ in range(5):
            bg = random_bg_color()
            ink = random_ink_color()
            # Luminance contrast check
            bg_lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            ink_lum = 0.299 * ink[0] + 0.587 * ink[1] + 0.114 * ink[2]
            if abs(bg_lum - ink_lum) > 60:
                break
        img = Image.new("RGB", (img_w, img_h), bg)
        ImageDraw.Draw(img).text(
            (pad_x - bbox[0], pad_y - bbox[1]), text, fill=ink, font=font,
        )

        scale = height / img_h
        new_w = max(4, int(img_w * scale))
        return img.resize((new_w, height), Image.BILINEAR)
    except Exception:
        return None


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
                self.script_fonts[script] = valid_fonts
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

        # Pre-generate images
        # If balance_groups: each GROUP gets samples_per_script images total,
        # split evenly across its scripts. This prevents 9-script indic from
        # overwhelming 1-script thai.
        self.images = []
        self.script_labels = []
        self.group_labels = []

        if balance_groups:
            samples_per_group = samples_per_script
            print(f"\nGenerating {self.num_groups * samples_per_group} images (balanced per group)...")
        else:
            print(f"\nGenerating {self.num_scripts * samples_per_script} images...")

        for script in self.active_scripts:
            fonts = self.script_fonts[script]
            group = SCRIPT_TO_GROUP[script]

            if balance_groups:
                scripts_in_group = [s for s in self.active_scripts if SCRIPT_TO_GROUP[s] == group]
                target = samples_per_group // len(scripts_in_group)
            else:
                target = samples_per_script

            generated = 0
            attempts = 0
            while generated < target and attempts < target * 5:
                attempts += 1

                if script == "emoji":
                    img = render_emoji(self.height, self.max_width)
                else:
                    words = SCRIPT_SAMPLES[script]
                    img = render_word(random.choice(words), random.choice(fonts), self.height)
                if img is None:
                    continue

                if img.width > self.max_width:
                    img = img.resize((self.max_width, self.height), Image.BILINEAR)
                elif img.width < self.max_width:
                    padded = Image.new("RGB", (self.max_width, self.height), (240, 240, 240))
                    padded.paste(img, (0, 0))
                    img = padded

                # Augment (on RGB, before color conversion)
                if self.augmentor is not None:
                    img = self.augmentor(img)

                # Convert RGB → model input channels
                self.images.append(rgb_to_input(img))
                self.script_labels.append(self.script_to_idx[script])
                self.group_labels.append(self.group_to_idx[group])
                generated += 1

            print(f"  {script:<15} {generated:>5} images  (group: {group}, target: {target})")

        self.total = len(self.images)
        print(f"Total: {self.total} images, {self.num_scripts} scripts, {self.num_groups} groups\n")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        return self.images[idx], self.script_labels[idx], self.group_labels[idx]


def train_coarse(stem, lid_coarse, dataset, train_set, val_set, args, device, color_proj=None,
                  resume_state=None):
    """Train LID-1: stem + coarse group classifier."""
    print(f"\n{'='*55}")
    print(f"LID-1: COARSE GROUP CLASSIFICATION ({dataset.num_groups} groups)")
    print(f"{'='*55}\n")
    print(f"Groups: {dataset.active_groups}\n")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    params = list(stem.parameters()) + list(lid_coarse.parameters())
    if color_proj is not None:
        params += list(color_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_acc = 0.0
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
        last_val_acc = resume_state.get("val_acc", 0.0)
        best_val_acc = resume_state.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}, val acc: {last_val_acc:.1f}%, best: {best_val_acc:.1f}%\n")

    for epoch in range(start_epoch, args.epochs + 1):
        stem.train()
        lid_coarse.train()
        total_loss = correct = total = 0

        for images, _, group_labels in train_loader:
            images = images.to(device)
            group_labels = group_labels.to(device) if isinstance(group_labels, torch.Tensor) else torch.tensor(group_labels, dtype=torch.long, device=device)

            logits = lid_coarse(stem(color_proj(images) if color_proj is not None else images))
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
        val_correct = val_total = 0
        per_group_correct = {}
        per_group_total = {}

        confusion = [[0] * dataset.num_groups for _ in range(dataset.num_groups)]
        with torch.no_grad():
            for images, _, group_labels in val_loader:
                images = images.to(device)
                group_labels = group_labels.to(device) if isinstance(group_labels, torch.Tensor) else torch.tensor(group_labels, dtype=torch.long, device=device)
                preds = lid_coarse(stem(color_proj(images) if color_proj is not None else images)).argmax(-1)
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
    parser = argparse.ArgumentParser(description="Phase 0: Train hierarchical LID")
    parser.add_argument("--level", type=str, default="both", choices=["coarse", "fine", "both"])
    parser.add_argument("--stem", type=str, default="resnet", choices=["convnext", "resnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-script", type=int, default=2000)
    parser.add_argument("--balance-groups", action="store_true",
                        help="Balance samples per GROUP instead of per script")
    parser.add_argument("--augment", action="store_true",
                        help="Apply RandAugmentOCR to training images")
    parser.add_argument("--moe", action="store_true",
                        help="Use group-specific expert MLPs in Stage 1")
    parser.add_argument("--stem-depth", type=int, default=2, choices=[2, 3],
                        help="ResNet stem depth: 2 (default) or 3 (extra ResBlock)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-width", type=int, default=192)
    parser.add_argument("--stage1-dim", type=int, default=192,
                        help="Stage 1 channel dim (192 for 12M, 288 for 56M)")
    parser.add_argument("--stage1-blocks", type=int, default=3,
                        help="Number of Stage 1 SWA blocks (3 for 12M, 8 for 56M)")
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

    # Load word lists from files
    load_all_script_samples()

    # Generate dataset
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
    print(f"{args.stem} stem: {stem_params:,} params")

    start = time.time()

    # LID-1: Coarse
    if args.level in ("coarse", "both"):
        lid_coarse = LIDCoarse(in_channels=stem_channels, num_groups=dataset.num_groups).to(device)
        print(f"LID-1 (coarse): {sum(p.numel() for p in lid_coarse.parameters()):,} params")

        # Resume coarse training if checkpoint exists
        coarse_resume = None
        if args.resume:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            stem.load_state_dict(ckpt["stem"])
            lid_coarse.load_state_dict(ckpt["lid_coarse"])
            if color_proj is not None and "color_proj" in ckpt:
                color_proj.load_state_dict(ckpt["color_proj"])
            if "coarse_optimizer" in ckpt:
                coarse_resume = {
                    "optimizer": ckpt["coarse_optimizer"],
                    "scheduler": ckpt["coarse_scheduler"],
                    "epoch": ckpt["coarse_epoch"],
                    "val_acc": ckpt.get("coarse_val_acc", 0.0),
                    "best_val_acc": ckpt.get("coarse_best_val_acc", 0.0),
                }
            print(f"Loaded checkpoint from {args.resume}")

        coarse_acc, coarse_optim_state, coarse_sched_state = train_coarse(
            stem, lid_coarse, dataset, train_set, val_set, args, device,
            color_proj=color_proj, resume_state=coarse_resume,
        )

    # LID-2: Fine (with Stage 1)
    if args.level in ("fine", "both"):
        from src.model.attention import SWABlock, SWABlockMoE

        proj1 = nn.Linear(stem_channels, args.stage1_dim).to(device)

        if args.moe:
            stage1_blocks = nn.ModuleList([
                SWABlockMoE(
                    dim=args.stage1_dim,
                    num_heads=args.stage1_dim // 32,
                    num_groups=dataset.num_groups,
                    window_h=4, window_w=4,
                    shift=(i % 2 == 1), mlp_ratio=3,
                )
                for i in range(args.stage1_blocks)
            ]).to(device)
        else:
            stage1_blocks = nn.ModuleList([
                SWABlock(
                    dim=args.stage1_dim,
                    num_heads=args.stage1_dim // 32,
                    window_h=4, window_w=4,
                    shift=(i % 2 == 1), mlp_ratio=3,
                )
                for i in range(args.stage1_blocks)
            ]).to(device)

        lid_fine = LIDFine(in_dim=args.stage1_dim, num_scripts=dataset.num_scripts).to(device)

        s1_params = sum(p.numel() for p in proj1.parameters()) + sum(p.numel() for p in stage1_blocks.parameters())
        moe_str = f"MoE ({dataset.num_groups} experts)" if args.moe else "shared"
        print(f"\nStage 1: {s1_params:,} params ({args.stage1_blocks} blocks, dim={args.stage1_dim}, {moe_str})")
        print(f"LID-2 (fine): {sum(p.numel() for p in lid_fine.parameters()):,} params")

        fine_acc = train_fine(stem, stage1_blocks, proj1, lid_fine, dataset, train_set, val_set, args, device, color_proj=color_proj, use_moe=args.moe)

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
    }
    if color_proj is not None:
        save_dict["color_proj"] = color_proj.state_dict()
    if args.level in ("coarse", "both"):
        save_dict["lid_coarse"] = lid_coarse.state_dict()
        save_dict["coarse_acc"] = coarse_acc
        save_dict["coarse_optimizer"] = coarse_optim_state
        save_dict["coarse_scheduler"] = coarse_sched_state
        save_dict["coarse_epoch"] = args.epochs
        save_dict["coarse_best_val_acc"] = coarse_acc
    if args.level in ("fine", "both"):
        save_dict["proj1"] = proj1.state_dict()
        save_dict["stage1"] = stage1_blocks.state_dict()
        save_dict["lid_fine"] = lid_fine.state_dict()
        save_dict["fine_acc"] = fine_acc

    torch.save(save_dict, save_path)

    print(f"\n{'='*55}")
    print(f"Total time: {elapsed:.0f}s")
    if args.level in ("coarse", "both"):
        print(f"LID-1 (coarse, {dataset.num_groups} groups): {coarse_acc:.1f}%")
    if args.level in ("fine", "both"):
        print(f"LID-2 (fine, {dataset.num_scripts} scripts): {fine_acc:.1f}%")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
