"""
Font discovery and validation for OCR training data generation.

Finds fonts that can render each script using name-based matching.
Fonts must contain script-relevant keywords (e.g., "Ethiopic", "Tamil")
to be used for that script. This prevents system fonts (DejaVu, FreeSerif)
from rendering tofu/garbage for scripts they technically have in their
cmap but can't actually display.
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

# Keywords for font style weighting
_HANDWRITING_KEYWORDS = [
    "caveat", "dancing", "indie", "patrick", "shadow", "kalam",
    "nanumpen", "chilanka", "handwrit", "cursive", "script",
]
_DISPLAY_KEYWORDS = [
    "permanent", "amatic", "lobster", "pacifico", "special", "display",
]

# Font name patterns that indicate support for each script.
# A font must contain at least one of these substrings (case-insensitive)
# to be considered for that script. This prevents DejaVu/FreeSerif/etc
# from being used for scripts they have in cmap but render as tofu.
#
# "universal" fonts (explicitly designed for broad Unicode) are also listed.
_SCRIPT_FONT_PATTERNS = {
    "latin": ["notosans-", "notoserif-", "notosansmono",
              "dejavu", "free", "liberation", "arial", "helvetica",
              "times-", "times ", "verdana", "tahoma", "comic", "courier",
              "roboto", "opensans", "lato", "montserrat", "poppins",
              "inter-", "inter.", "raleway", "ubuntu", "source", "jetbrains",
              "caveat", "dancing", "indie", "patrick", "kalam",
              "lobster", "pacifico", "amatic", "permanent",
              "baloo", "hind", "tiro", "mukta", "gemunu", "heebo",
              "content"],
    "cyrillic": ["notosans-", "notoserif-", "notosansmono",
                 "dejavu", "free", "liberation", "arial", "helvetica",
                 "times-", "times ", "verdana", "roboto", "opensans",
                 "ubuntu", "source", "jetbrains", "inter-", "inter.",
                 "caveat", "dancing"],
    "greek": ["notosans-", "notoserif-", "notosansmono",
              "dejavu", "free", "liberation", "arial", "helvetica",
              "times-", "times ", "verdana", "roboto", "opensans",
              "source", "jetbrains", "inter-", "inter."],
    "arabic": ["arabic", "nastaliq", "naskh", "kufi", "urdu", "persian",
               "lateef", "scheherazade", "amiri", "harmattan", "alkalami",
               "reem", "mirza", "markazi", "tajawal", "cairo", "almarai"],
    "hebrew": ["hebrew", "david", "frank", "miriam"],
    "han_kana": ["cjk", "japanese", "chinese", "gothic", "mincho", "meiryo",
                 "hiragino", "kaiti", "songti", "heiti", "fangsong",
                 "source han", "hachi", "kosugi", "sawarabi",
                 "zen", "klee", "reggae", "rampart", "rocknroll",
                 "shippori", "dela", "potta", "yomogi", "yuji", "murecho"],
    "korean": ["korean", "hangul", "nanum", "gothic", "batang", "gulim",
               "malgun", "source han", "gamja", "jua",
               "black han", "do hyeon", "gaegu", "gugi", "hi melody",
               "poor story", "stylish", "sunflower", "single day"],
    "devanagari": ["devanagari", "hindi", "marathi", "sanskrit", "mangal",
                   "kokila", "gargi", "lohit", "poppins", "rajdhani",
                   "yantramanav", "khand", "biryani", "halant", "laila"],
    "gurmukhi": ["gurmukhi", "punjabi", "raavi"],
    "gujarati": ["gujarati", "shruti"],
    "bengali": ["bengali", "bangla", "vrinda", "shonar",
                "galada", "atma", "mina"],
    "odia": ["odia", "oriya", "kalinga"],
    "kannada": ["kannada", "tunga"],
    "telugu": ["telugu", "gautami",
               "mandali", "ramabhadra", "tenali", "gurajada", "lakki"],
    "malayalam": ["malayalam", "kartika", "rachana",
                  "chilanka", "gayathri", "manjari"],
    "tamil": ["tamil", "latha",
              "arima", "kavivanar", "meera", "catamaran"],
    "sinhala": ["sinhala", "sinhalese", "iskoola", "abhaya"],
    "thai": ["thai", "angsana", "browallia", "cordia",
             "sarabun", "kanit", "prompt", "mitr", "itim", "charm",
             "chonburi", "krub", "pridi", "taviraj", "trirong"],
    "lao": ["lao", "phetsarath", "saysettha"],
    "burmese": ["myanmar", "burmese", "padauk"],
    "khmer": ["khmer", "cambodian", "battambang", "bayon",
              "bokor", "chenla", "dangrek", "fasthand", "freehand",
              "hanuman", "metal", "moul", "siemreap", "suwannaphum",
              "taprom", "content"],
    "emoji": None,
    "armenian": ["armenian"],
    "georgian": ["georgian"],
    "ethiopic": ["ethiopic", "abyssinica"],
    "tibetan": ["tibetan", "jomolhari"],
}


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

    Uses name-based matching to prevent fonts with broad cmap tables
    (DejaVu, FreeSerif) from being used for scripts they render as tofu.
    """
    import unicodedata
    from src.data.script_detect import _SCRIPT_RANGES

    if script not in _SCRIPT_RANGES:
        return []

    # Find a valid sample codepoint for cmap verification
    ranges = _SCRIPT_RANGES[script]
    sample_cp = None
    for start, end in ranges:
        for cp in range(start, end + 1):
            ch = chr(cp)
            cat = unicodedata.category(ch)
            if cat != "Cn" and cat not in ("Mn", "Mc"):
                sample_cp = ch
                break
        if sample_cp:
            break
    if sample_cp is None:
        for start, end in ranges:
            for cp in range(start, end + 1):
                if unicodedata.category(chr(cp)) != "Cn":
                    sample_cp = chr(cp)
                    break
            if sample_cp:
                break
    if sample_cp is None:
        return []

    all_fonts = find_system_fonts()
    patterns = _SCRIPT_FONT_PATTERNS.get(script)

    valid = []
    for f in all_fonts:
        # Must have the codepoint in cmap
        if not font_has_codepoint(f, sample_cp):
            continue
        # If script has font name patterns, font must match at least one
        if patterns is not None:
            name_lower = Path(f).name.lower()
            if not any(p in name_lower for p in patterns):
                continue
        valid.append(f)

    return valid


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
