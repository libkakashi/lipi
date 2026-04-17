"""
Migrate a 13-group checkpoint to 15-group (han/kana split + dravidian split).

Usage:
    python scripts/migrate_checkpoint.py --input checkpoints/moe/moe_epoch20.pt \
                                         --output checkpoints/moe/moe_epoch20_migrated.pt

Changes from old 13 groups → new 15 groups:
  - Group 4 (sino_japanese) → Group 4 (han). Kana expert at group 5 = NEW.
  - Group 7 (south_indic: kannada,telugu,malayalam,tamil,sinhala) →
    Group 8 (dravidian_north: kannada,telugu,sinhala) +
    Group 9 (dravidian_south: malayalam,tamil) = SPLIT.
  - Groups after the splits shift up accordingly.
  - LID-1 head dropped (shape changes 14→16).
  - LID-2 for old south_indic dropped (script count changes).
  - CTC heads for south_indic split across two new groups.
  - CTC head for sino_japanese dropped (vocab changed).
"""

import argparse
import re
from pathlib import Path

import torch


# =========================================================================
# Old 13 groups → New 15 groups
# =========================================================================
#
# Old:                          New:
# 0  latin                      0  latin
# 1  cyrillic_greek             1  cyrillic_greek
# 2  arabic                     2  arabic
# 3  hebrew                     3  hebrew
# 4  sino_japanese              4  han
#                                5  kana (NEW)
# 5  korean                     6  korean
# 6  ne_indic                   7  ne_indic
# 7  south_indic                8  dravidian_north (kannada,telugu,sinhala)
#                                9  dravidian_south (malayalam,tamil) (NEW)
# 8  se_asian                   10 se_asian
# 9  emoji                      11 emoji
# 10 caucasus                   12 caucasus
# 11 ethiopic                   13 ethiopic
# 12 tibetan                    14 tibetan

OLD_TO_NEW_GROUP = {
    0: 0,   # latin
    1: 1,   # cyrillic_greek
    2: 2,   # arabic
    3: 3,   # hebrew
    4: 4,   # sino_japanese → han
    # 5 = kana (NEW)
    5: 6,   # korean
    6: 7,   # ne_indic
    7: 8,   # south_indic → dravidian_north
    # 9 = dravidian_south (NEW)
    8: 10,  # se_asian
    9: 11,  # emoji
    10: 12, # caucasus
    11: 13, # ethiopic
    12: 14, # tibetan
}

# Old 26 flat scripts → new 27 flat scripts
# Old south_indic had: kannada(12), telugu(13), malayalam(14), tamil(15), sinhala(16)
# New dravidian_north: kannada(13), telugu(14), sinhala(15)
# New dravidian_south: malayalam(16), tamil(17)
OLD_TO_NEW_SCRIPT = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,  # latin..han_kana→han
    # 6 = kana (NEW)
    6: 7,                                    # korean
    7: 8, 8: 9, 9: 10, 10: 11, 11: 12,     # ne_indic (shifted +1)
    12: 13,  # kannada → dravidian_north[0]
    13: 14,  # telugu → dravidian_north[1]
    14: 16,  # malayalam → dravidian_south[0]
    15: 17,  # tamil → dravidian_south[1]
    16: 15,  # sinhala → dravidian_north[2]
    17: 18, 18: 19, 19: 20, 20: 21,         # se_asian
    21: 22,                                   # emoji
    22: 23, 23: 24,                           # caucasus
    24: 25,                                   # ethiopic
    25: 26,                                   # tibetan
}

# Old LID-2 group keys → new LID-2 group keys
# Old group 7 (south_indic) is DROPPED — split changes script count
OLD_TO_NEW_LID2 = {
    1: 1,    # cyrillic_greek (2 scripts → 2 scripts)
    6: 7,    # ne_indic (5 → 5)
    # 7: SKIP (south_indic split)
    8: 10,   # se_asian (4 → 4)
    10: 12,  # caucasus (2 → 2)
}

# Old CTC group → new CTC group (simple groups that don't change structure)
OLD_TO_NEW_CTC_SIMPLE = {
    0: 0, 1: 1, 2: 2, 3: 3,
    # 4: SKIP (sino_japanese vocab changed)
    5: 6, 6: 7,
    # 7: SPECIAL (south_indic splits — handled separately)
    8: 10, 9: 11, 10: 12, 11: 13, 12: 14,
}

# South_indic CTC head split: old heads → new groups+heads
# Old group 7 heads: 0=kannada, 1=telugu, 2=malayalam, 3=tamil, 4=sinhala
# New group 8 (dravidian_north): 0=kannada, 1=telugu, 2=sinhala
# New group 9 (dravidian_south): 0=malayalam, 1=tamil
SOUTH_INDIC_CTC_REMAP = {
    # (old_head_idx): (new_group, new_head_idx)
    0: (8, 0),   # kannada
    1: (8, 1),   # telugu
    2: (9, 0),   # malayalam
    3: (9, 1),   # tamil
    4: (8, 2),   # sinhala
}


def _remap_indexed_key(key, prefix, old_to_new):
    pattern = rf'^({re.escape(prefix)}\.)(\d+)(\..*)?$'
    m = re.match(pattern, key)
    if not m:
        return key, True
    old_id = int(m.group(2))
    if old_id not in old_to_new:
        return None, False
    new_id = old_to_new[old_id]
    suffix = m.group(3) or ''
    return f'{m.group(1)}{new_id}{suffix}', True


def _remap_expert_key(key, block_prefix, old_to_new):
    if block_prefix not in key:
        return key, True
    for expert_type in ('expert_attns', 'expert_mlps'):
        pattern = rf'({re.escape(block_prefix)}\.\d+\.{expert_type}\.)(\d+)(\..*)'
        m = re.match(pattern, key)
        if m:
            old_id = int(m.group(2))
            if old_id not in old_to_new:
                return None, False
            new_id = old_to_new[old_id]
            return f'{m.group(1)}{new_id}{m.group(3)}', True
    return key, True


def migrate(old_state):
    new_state = {}
    skipped = []
    remapped = []

    for key, value in old_state.items():
        new_key = key
        keep = True

        # --- LID-1: drop (shape changes 14 → 16) ---
        if key.startswith('group_head.'):
            skipped.append(key)
            continue

        # --- Group expert blocks ---
        if key.startswith('group_local_blocks.') or key.startswith('group_wide_blocks.'):
            prefix = 'group_local_blocks' if 'group_local' in key else 'group_wide_blocks'
            new_key, keep = _remap_expert_key(key, prefix, OLD_TO_NEW_GROUP)

        # --- Group aggregates ---
        elif key.startswith('group_aggregates.'):
            new_key, keep = _remap_indexed_key(key, 'group_aggregates', OLD_TO_NEW_GROUP)

        # --- LID-2 heads ---
        elif key.startswith('lid2_heads.'):
            new_key, keep = _remap_indexed_key(key, 'lid2_heads', OLD_TO_NEW_LID2)

        # --- Script expert blocks ---
        elif key.startswith('script_local_blocks.') or key.startswith('script_wide_blocks.'):
            prefix = 'script_local_blocks' if 'script_local' in key else 'script_wide_blocks'
            new_key, keep = _remap_expert_key(key, prefix, OLD_TO_NEW_SCRIPT)

        # --- Script aggregates ---
        elif key.startswith('script_aggregates.'):
            new_key, keep = _remap_indexed_key(key, 'script_aggregates', OLD_TO_NEW_SCRIPT)

        # --- CTC modules (complex: south_indic splits) ---
        elif key.startswith('ctc_modules.'):
            # Parse: ctc_modules.{group}.{rest}
            m = re.match(r'^(ctc_modules\.)(\d+)(\..*)?$', key)
            if m:
                old_g = int(m.group(2))
                suffix = m.group(3) or ''

                if old_g in OLD_TO_NEW_CTC_SIMPLE:
                    new_g = OLD_TO_NEW_CTC_SIMPLE[old_g]
                    new_key = f'ctc_modules.{new_g}{suffix}'
                elif old_g == 7:
                    # South_indic split: remap per-head
                    head_match = re.match(r'\.heads\.(\d+)(\..*)', suffix)
                    if head_match:
                        old_head = int(head_match.group(1))
                        head_suffix = head_match.group(2)
                        if old_head in SOUTH_INDIC_CTC_REMAP:
                            new_g, new_head = SOUTH_INDIC_CTC_REMAP[old_head]
                            new_key = f'ctc_modules.{new_g}.heads.{new_head}{head_suffix}'
                        else:
                            keep = False
                    else:
                        # Non-head params (e.g. max_vocab buffer) — skip, will reinit
                        keep = False
                elif old_g == 4:
                    keep = False  # sino_japanese vocab changed
                else:
                    keep = False
            else:
                pass  # non-matching key, pass through

        if not keep:
            skipped.append(key)
            continue

        if new_key != key:
            remapped.append(f'{key} → {new_key}')
        new_state[new_key] = value

    return new_state, skipped, remapped


def main():
    parser = argparse.ArgumentParser(description="Migrate 13-group checkpoint to 15-group")
    parser.add_argument("--input", required=True, help="Path to old checkpoint")
    parser.add_argument("--output", required=True, help="Path to save migrated checkpoint")
    args = parser.parse_args()

    assert Path(args.input).exists(), f"Input not found: {args.input}"

    print(f"Loading {args.input}...")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    old_state = ckpt["model"]
    print(f"  {len(old_state)} parameters in old checkpoint")

    new_state, skipped, remapped = migrate(old_state)

    print(f"\n  Remapped: {len(remapped)} keys")
    for r in remapped[:15]:
        print(f"    {r}")
    if len(remapped) > 15:
        print(f"    ... and {len(remapped) - 15} more")

    print(f"\n  Skipped (will reinitialize): {len(skipped)} keys")
    prefixes = {}
    for s in skipped:
        p = '.'.join(s.split('.')[:2])
        prefixes[p] = prefixes.get(p, 0) + 1
    for p, n in sorted(prefixes.items()):
        print(f"    {p}.* ({n} params)")

    print(f"\n  Result: {len(new_state)} parameters in migrated checkpoint")

    out = {
        "model": new_state,
        "epoch": ckpt.get("epoch", 0),
        "args": ckpt.get("args", {}),
    }
    if "model_config" in ckpt:
        out["model_config"] = ckpt["model_config"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"\nSaved migrated checkpoint to {args.output}")
    print("\nWhat will reinitialize on load:")
    print("  - group_head (LID-1): 14→16 outputs")
    print("  - kana group expert (group 5): identity-init")
    print("  - kana script expert (flat 6): identity-init")
    print("  - dravidian_south group expert (group 9): identity-init")
    print("  - han CTC head (group 4): vocab changed")
    print("  - kana CTC head (group 5): new")
    print("  - LID-2 for dravidian_north (group 8): new (3 scripts vs old 5)")
    print("  - LID-2 for dravidian_south (group 9): new (2 scripts)")
    print("  - Optimizer/scheduler: dropped")
    print("\nPreserved from old south_indic:")
    print("  - Group expert → dravidian_north (group 8)")
    print("  - CTC heads: kannada,telugu → drav_north; malayalam,tamil → drav_south")
    print("  - sinhala CTC head → dravidian_north heads.2")
    print("\nRecommended training:")
    print("  1. python scripts/train.py --resume <migrated.pt> --freeze-except lid \\")
    print("       --ctc-weight 0 --epochs 3 --lr 1e-3")
    print("  2. python scripts/train.py --resume <lid_done.pt> --epochs 20 --lr 5e-5")


if __name__ == "__main__":
    main()
