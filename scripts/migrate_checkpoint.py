"""
Migrate checkpoints to 15-group layout (han/kana split + dravidian split),
and seed LID-0 super-group SWA-B stacks from the pre-LID-0 shared_b stack.

Auto-detects source layout from checkpoint's model_config:
  - 13 groups (original): applies han/kana split + dravidian split
  - 14 groups (post han/kana): applies dravidian split only

Then (if the checkpoint still has a single shared_b stack), replicates its
weights across all NUM_SUPER_GROUPS super_b stacks so each super-group
starts identical to the pre-LID-0 behavior and specializes during training.
The new lid0_head is left to random init.

Usage:
    python scripts/migrate_checkpoint.py --input checkpoints/moe/moe_epoch20.pt \
                                         --output checkpoints/moe/moe_migrated.pt
"""

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.lid import NUM_SUPER_GROUPS  # noqa: E402


# =========================================================================
# 13 → 15 migration (han/kana split + dravidian split)
# =========================================================================

_13_TO_15_GROUP = {
    0: 0, 1: 1, 2: 2, 3: 3,
    4: 4,   # sino_japanese → han
    5: 6,   # korean
    6: 7,   # ne_indic
    7: 8,   # south_indic → dravidian_north
    8: 10,  # se_asian
    9: 11,  # emoji
    10: 12, # caucasus
    11: 13, # ethiopic
    12: 14, # tibetan
}

_13_TO_15_SCRIPT = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,  # latin..han_kana→han
    6: 7,                                    # korean
    7: 8, 8: 9, 9: 10, 10: 11, 11: 12,     # ne_indic
    12: 13, 13: 14,                          # kannada, telugu → drav_north
    14: 16, 15: 17,                          # malayalam, tamil → drav_south
    16: 15,                                  # sinhala → drav_north[2]
    17: 18, 18: 19, 19: 20, 20: 21,         # se_asian
    21: 22, 22: 23, 23: 24, 24: 25, 25: 26, # emoji..tibetan
}

_13_TO_15_LID2 = {
    1: 1,    # cyrillic_greek
    6: 7,    # ne_indic
    # 7: SKIP (south_indic split)
    8: 10,   # se_asian
    10: 12,  # caucasus
}

_13_TO_15_CTC_SIMPLE = {
    0: 0, 1: 1, 2: 2, 3: 3,
    # 4: SKIP (han_kana vocab changed)
    5: 6, 6: 7,
    # 7: SPECIAL (south_indic splits)
    8: 10, 9: 11, 10: 12, 11: 13, 12: 14,
}

# =========================================================================
# 14 → 15 migration (dravidian split only)
# =========================================================================

_14_TO_15_GROUP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 8,   # south_indic → dravidian_north
    9: 10,  # se_asian
    10: 11, # emoji
    11: 12, # caucasus
    12: 13, # ethiopic
    13: 14, # tibetan
}

_14_TO_15_SCRIPT = {
    **{i: i for i in range(13)},  # 0-12 unchanged (latin..odia)
    13: 13, 14: 14,               # kannada, telugu → same flat IDs
    15: 16, 16: 17,               # malayalam, tamil → shifted
    17: 15,                        # sinhala → drav_north[2]
    **{i: i for i in range(18, 27)},  # 18-26 unchanged (se_asian..tibetan)
}

_14_TO_15_LID2 = {
    1: 1,    # cyrillic_greek
    7: 7,    # ne_indic
    # 8: SKIP (south_indic split)
    9: 10,   # se_asian
    11: 12,  # caucasus
}

_14_TO_15_CTC_SIMPLE = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    # 8: SPECIAL (south_indic splits)
    9: 10, 10: 11, 11: 12, 12: 13, 13: 14,
}

# =========================================================================
# South_indic CTC head split (same for both 13→15 and 14→15)
# Old heads: 0=kannada, 1=telugu, 2=malayalam, 3=tamil, 4=sinhala
# New drav_north (group 8): 0=kannada, 1=telugu, 2=sinhala
# New drav_south (group 9): 0=malayalam, 1=tamil
# =========================================================================

SOUTH_INDIC_CTC_REMAP = {
    0: (8, 0),   # kannada
    1: (8, 1),   # telugu
    2: (9, 0),   # malayalam
    3: (9, 1),   # tamil
    4: (8, 2),   # sinhala
}


# =========================================================================
# Remapping helpers
# =========================================================================

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


def _seed_super_b_from_shared_b(state_dict):
    """Replicate pre-LID-0 shared_b.* weights across all NUM_SUPER_GROUPS
    super_b.<sg>.* stacks. Idempotent — skips if super_b.* already present.

    Returns a new dict; does not mutate the input.
    """
    has_shared_b = any(k.startswith("shared_b.") for k in state_dict)
    has_super_b = any(k.startswith("super_b.") for k in state_dict)
    if not has_shared_b:
        # Either already migrated (super_b present) or checkpoint has neither
        return state_dict, 0
    if has_super_b:
        # Both present — trust super_b, drop shared_b
        return ({k: v for k, v in state_dict.items()
                 if not k.startswith("shared_b.")},
                0)

    new_state = {k: v for k, v in state_dict.items()
                 if not k.startswith("shared_b.")}
    replicated = 0
    for k, v in state_dict.items():
        if not k.startswith("shared_b."):
            continue
        # shared_b.<i>.<rest>  →  super_b.<sg>.<i>.<rest>  for sg in [0..)
        suffix = k[len("shared_b."):]
        for sg in range(NUM_SUPER_GROUPS):
            new_state[f"super_b.{sg}.{suffix}"] = v.clone()
            replicated += 1
    return new_state, replicated


def migrate(old_state, source_groups):
    """Migrate state dict to 15-group layout."""
    if source_groups == 13:
        GROUP_MAP = _13_TO_15_GROUP
        SCRIPT_MAP = _13_TO_15_SCRIPT
        LID2_MAP = _13_TO_15_LID2
        CTC_SIMPLE = _13_TO_15_CTC_SIMPLE
        south_indic_group = 7
    elif source_groups == 14:
        GROUP_MAP = _14_TO_15_GROUP
        SCRIPT_MAP = _14_TO_15_SCRIPT
        LID2_MAP = _14_TO_15_LID2
        CTC_SIMPLE = _14_TO_15_CTC_SIMPLE
        south_indic_group = 8
    else:
        raise ValueError(f"Unsupported source_groups={source_groups}")

    new_state = {}
    skipped = []
    remapped = []

    for key, value in old_state.items():
        new_key = key
        keep = True

        # --- LID-1: drop (output size changes) ---
        if key.startswith('group_head.'):
            skipped.append(key)
            continue

        # --- Group expert blocks ---
        if key.startswith('group_local_blocks.') or key.startswith('group_wide_blocks.'):
            prefix = 'group_local_blocks' if 'group_local' in key else 'group_wide_blocks'
            new_key, keep = _remap_expert_key(key, prefix, GROUP_MAP)

        # --- Group aggregates ---
        elif key.startswith('group_aggregates.'):
            new_key, keep = _remap_indexed_key(key, 'group_aggregates', GROUP_MAP)

        # --- LID-2 heads ---
        elif key.startswith('lid2_heads.'):
            new_key, keep = _remap_indexed_key(key, 'lid2_heads', LID2_MAP)

        # --- Script expert blocks ---
        elif key.startswith('script_local_blocks.') or key.startswith('script_wide_blocks.'):
            prefix = 'script_local_blocks' if 'script_local' in key else 'script_wide_blocks'
            new_key, keep = _remap_expert_key(key, prefix, SCRIPT_MAP)

        # --- Script aggregates ---
        elif key.startswith('script_aggregates.'):
            new_key, keep = _remap_indexed_key(key, 'script_aggregates', SCRIPT_MAP)

        # --- CTC modules ---
        elif key.startswith('ctc_modules.'):
            m = re.match(r'^(ctc_modules\.)(\d+)(\..*)?$', key)
            if m:
                old_g = int(m.group(2))
                suffix = m.group(3) or ''

                if old_g in CTC_SIMPLE:
                    new_g = CTC_SIMPLE[old_g]
                    new_key = f'ctc_modules.{new_g}{suffix}'
                elif old_g == south_indic_group:
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
                        keep = False
                else:
                    keep = False

        if not keep:
            skipped.append(key)
            continue

        if new_key != key:
            remapped.append(f'{key} → {new_key}')
        new_state[new_key] = value

    # Seed per-super-group SWA-B stacks from the old single shared_b stack
    new_state, seeded = _seed_super_b_from_shared_b(new_state)
    if seeded:
        remapped.append(f'shared_b.* → super_b.{{0..{NUM_SUPER_GROUPS-1}}}.* '
                        f'(replicated {seeded} tensors)')

    return new_state, skipped, remapped


def main():
    parser = argparse.ArgumentParser(description="Migrate checkpoint to 15-group layout")
    parser.add_argument("--input", required=True, help="Path to old checkpoint")
    parser.add_argument("--output", required=True, help="Path to save migrated checkpoint")
    args = parser.parse_args()

    assert Path(args.input).exists(), f"Input not found: {args.input}"

    print(f"Loading {args.input}...")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    source_groups = ckpt.get("model_config", {}).get("num_groups")
    if source_groups is None:
        # Fallback: count group experts
        keys = list(ckpt["model"].keys())
        group_ids = set()
        for k in keys:
            m = re.match(r'group_local_blocks\.\d+\.expert_attns\.(\d+)\.', k)
            if m:
                group_ids.add(int(m.group(1)))
        source_groups = max(group_ids) + 1 if group_ids else 13
    print(f"  Source: {source_groups} groups")

    if source_groups not in (13, 14, 15):
        print(f"  ERROR: Unsupported source layout ({source_groups} groups)")
        return

    old_state = ckpt["model"]
    print(f"  {len(old_state)} parameters")

    if source_groups == 15:
        # Skip group remap; only run the LID-0 seeding step.
        print("  15 groups — skipping group remap, running LID-0 seeding only.")
        new_state, seeded = _seed_super_b_from_shared_b(old_state)
        remapped = []
        if seeded:
            remapped.append(f'shared_b.* → super_b.{{0..{NUM_SUPER_GROUPS-1}}}.* '
                            f'(replicated {seeded} tensors)')
        skipped = []
        if not seeded and not any(k.startswith("shared_b.") for k in old_state):
            print("  No shared_b.* keys found — nothing to migrate.")
            return
    else:
        new_state, skipped, remapped = migrate(old_state, source_groups)

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
    print(f"\nSaved to {args.output}")

    if source_groups == 13:
        print("\nApplied: han/kana split + dravidian split (13 → 15)")
        print("  Reinit: LID-1, kana expert/CTC, han CTC, dravidian LID-2s")
    elif source_groups == 14:
        print("\nApplied: dravidian split only (14 → 15)")
        print("  Reinit: LID-1, dravidian_south expert, dravidian LID-2s")
    else:
        print("\nApplied: LID-0 super-group seeding only (15 → 15)")
    print("  Preserved: south_indic CTC heads split across dravidian_north/south")
    print(f"  LID-0 head (lid0_head.*) will randomly init on model load.")


if __name__ == "__main__":
    main()
