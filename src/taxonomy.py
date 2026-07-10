"""
Script and group taxonomy — shared by data, encoding, training, and the model.

Two-level identity system for LID routing:

Scripts (27): routed script labels — latin, cyrillic, greek, arabic,
    hebrew, sparse/dense Han, kana, korean, and the Brahmic / South Asian /
    SE Asian scripts.

  Groups (14): visual/linguistic families of scripts that share expert
    capacity — e.g. `cyrillic_greek` groups Cyrillic and Greek, `ne_indic`
    groups the 5 North Indian Brahmic scripts. Groups with only one script
    skip LID-2 at routing time.

This module is intentionally data-only: no torch, no nn.Module — so it's
safe to import from any layer without pulling the model in.

Population and rough coverage (in the ARCHITECTURE.md ordering):
  Latin (~800 chars, 36 langs, ~3B speakers), Cyrillic+Greek (~760, ~300M),
  Arabic (~490 incl. Persian/Urdu, ~500M), Hebrew (~190, ~9M),
  Han (two complexity-routed heads, Hanzi/Kanji/Hanja, ~1.4B),
  Kana (~260, Japanese syllabaries, ~125M),
  Korean (~1.5K jamo-decomposed tokens, ~80M),
  N+E Indic (~850, Devanagari/Gurmukhi/Gujarati/Bengali/Odia, ~1B+),
  Dravidian North (~500, Kannada/Telugu/Sinhala, ~200M),
  Dravidian South (~350, Malayalam/Tamil, ~100M),
  SE Asian (~770, Thai/Lao/Burmese/Khmer, ~150M),
  Caucasus (~330, Armenian/Georgian, ~10M),
  Ethiopic (~520, Amharic, ~57M), Tibetan (~270, ~6M).
"""


# --- Scripts (individual writing systems) ---

# Script IDs follow group/local flatten order — SCRIPT_TO_ID[s] equals the
# position of s when GROUP_SCRIPTS is flattened in GROUPS order. The encoder
# and shards rely on this invariant (tests pin it); keep the two in sync.
SCRIPTS = [
    # Group 1: Latin
    "latin",       # 0
    # Group 2: Cyrillic + Greek
    "cyrillic",    # 1
    "greek",       # 2
    # Group 3: Arabic (incl. Urdu, Persian)
    "arabic",      # 3
    # Group 4: Hebrew
    "hebrew",      # 4
    # Group 5: Han (Hanzi / Kanji / Hanja), split by glyph complexity
    "han_sparse",  # 5
    "han_dense",   # 6
    # Group 6: Kana (hiragana + katakana, Japanese syllabaries)
    "kana",        # 7
    # Group 7: Korean
    "korean",      # 8
    # Group 8: N+E Indian Brahmic
    "devanagari",  # 9
    "gurmukhi",    # 10
    "gujarati",    # 11
    "bengali",     # 12
    "odia",        # 13
    # Group 9: Dravidian North (curvy, similar stroke patterns)
    "kannada",     # 14
    "telugu",      # 15
    "sinhala",     # 16
    # Group 10: Dravidian South (round, loopy)
    "malayalam",   # 17
    "tamil",       # 18
    # Group 11: SE Asian
    "thai",        # 19
    "lao",         # 20
    "burmese",     # 21
    "khmer",       # 22
    # Group 12: Caucasus
    "armenian",    # 23
    "georgian",    # 24
    # Group 13: Ethiopic
    "ethiopic",    # 25
    # Group 14: Tibetan
    "tibetan",     # 26
]

SCRIPT_TO_ID = {name: i for i, name in enumerate(SCRIPTS)}
NUM_SCRIPTS = len(SCRIPTS)
# v3: han_dense moved from appended slot 26 to 6 (natural flatten order).
# Bumped whenever numeric script/group IDs shift; shards are stamped with it
# and load_shard_metadata refuses mismatches.
TAXONOMY_VERSION = 3

# Names found in shards/checkpoints created before the Han complexity split.
SCRIPT_ALIASES = {"han": "han_sparse"}


def canonical_script_name(name: str) -> str:
    """Return the current script name for a legacy or current label."""
    return SCRIPT_ALIASES.get(name, name)


# --- Groups (14 families) ---

GROUPS = [
    "latin",             # 0
    "cyrillic_greek",    # 1
    "arabic",            # 2
    "hebrew",            # 3
    "han",               # 4
    "kana",              # 5
    "korean",            # 6
    "ne_indic",          # 7
    "dravidian_north",   # 8  (kannada, telugu, sinhala)
    "dravidian_south",   # 9  (malayalam, tamil)
    "se_asian",          # 10
    "caucasus",          # 11
    "ethiopic",          # 12
    "tibetan",           # 13
]

GROUP_TO_ID = {name: i for i, name in enumerate(GROUPS)}
NUM_GROUPS = len(GROUPS)


# Map each script to its group
SCRIPT_TO_GROUP = {
    "latin": "latin",
    "cyrillic": "cyrillic_greek",
    "greek": "cyrillic_greek",
    "arabic": "arabic",
    "hebrew": "hebrew",
    "han_sparse": "han",
    "han_dense": "han",
    "kana": "kana",
    "korean": "korean",
    "devanagari": "ne_indic",
    "gurmukhi": "ne_indic",
    "gujarati": "ne_indic",
    "bengali": "ne_indic",
    "odia": "ne_indic",
    "kannada": "dravidian_north",
    "telugu": "dravidian_north",
    "sinhala": "dravidian_north",
    "malayalam": "dravidian_south",
    "tamil": "dravidian_south",
    "thai": "se_asian",
    "lao": "se_asian",
    "burmese": "se_asian",
    "khmer": "se_asian",
    "armenian": "caucasus",
    "georgian": "caucasus",
    "ethiopic": "ethiopic",
    "tibetan": "tibetan",
}

GROUP_SCRIPTS = {
    "latin": ["latin"],
    "cyrillic_greek": ["cyrillic", "greek"],
    "arabic": ["arabic"],
    "hebrew": ["hebrew"],
    "han": ["han_sparse", "han_dense"],
    "kana": ["kana"],
    "korean": ["korean"],
    "ne_indic": ["devanagari", "gurmukhi", "gujarati", "bengali", "odia"],
    "dravidian_north": ["kannada", "telugu", "sinhala"],
    "dravidian_south": ["malayalam", "tamil"],
    "se_asian": ["thai", "lao", "burmese", "khmer"],
    "caucasus": ["armenian", "georgian"],
    "ethiopic": ["ethiopic"],
    "tibetan": ["tibetan"],
}
