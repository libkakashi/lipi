"""
Script and group taxonomy — shared by data, encoding, training, and the model.

Two-level identity system for LID routing:

Scripts (26): routed script labels — latin, cyrillic, greek, arabic,
    hebrew, han, kana, korean, and the Brahmic / South Asian /
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
  Han (Hanzi/Kanji/Hanja, ~1.4B),
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
    # Group 5: Han (Hanzi / Kanji / Hanja)
    "han",         # 5
    # Group 6: Kana (hiragana + katakana, Japanese syllabaries)
    "kana",        # 6
    # Group 7: Korean
    "korean",      # 7
    # Group 8: N+E Indian Brahmic
    "devanagari",  # 8
    "gurmukhi",    # 9
    "gujarati",    # 10
    "bengali",     # 11
    "odia",        # 12
    # Group 9: Dravidian North (curvy, similar stroke patterns)
    "kannada",     # 13
    "telugu",      # 14
    "sinhala",     # 15
    # Group 10: Dravidian South (round, loopy)
    "malayalam",   # 16
    "tamil",       # 17
    # Group 11: SE Asian
    "thai",        # 18
    "lao",         # 19
    "burmese",     # 20
    "khmer",       # 21
    # Group 12: Caucasus
    "armenian",    # 22
    "georgian",    # 23
    # Group 13: Ethiopic
    "ethiopic",    # 24
    # Group 14: Tibetan
    "tibetan",     # 25
]

SCRIPT_TO_ID = {name: i for i, name in enumerate(SCRIPTS)}
NUM_SCRIPTS = len(SCRIPTS)
# v4: Han complexity split removed — single han head again (measured on the
# full 27.6K inventory 2026-07: the complexity axis is smooth and gapless,
# so any split needs band/dual-decode machinery that outweighs its benefit).
# Bumped whenever numeric script/group IDs shift; shards are stamped with it
# and load_shard_metadata refuses mismatches.
TAXONOMY_VERSION = 4

# Names found in shards/checkpoints created while the Han complexity split
# existed (no trained checkpoints ever shipped with it).
SCRIPT_ALIASES = {"han_sparse": "han", "han_dense": "han"}


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
    "han": "han",
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
    "han": ["han"],
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
