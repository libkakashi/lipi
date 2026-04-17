"""
Migrate a 13-group (han_kana) checkpoint to 14-group (han + kana) layout.

Usage:
    python scripts/migrate_checkpoint.py --input checkpoints/moe/moe_epoch19.pt \
                                         --output checkpoints/moe/moe_epoch19_migrated.pt

What it does:
    - Remaps group expert weights: old sino_japanese (4) → new han (4),
      inserts identity-init kana expert at new index 5, shifts 5-12 → 6-13.
    - Remaps script expert weights: old flat[5] (han_kana) → new flat[5] (han),
      inserts identity-init kana at flat[6], shifts 6-25 → 7-26.
    - Remaps group_aggregates, script_aggregates (same index shifts).
    - Remaps LID-2 heads (group index shift for multi-script groups).
    - Drops group_head (LID-1) — must reinitialize (14+1 outputs vs 13+1).
    - Drops CTC head for old group 4 (vocab size changed) and new group 5 (kana).
    - Copies all shared components (stem, shared_a/b, merge_a/b, norm) unchanged.
"""

import argparse
import re
from pathlib import Path

import torch


# Old 13 groups → new 14 groups mapping
OLD_TO_NEW_GROUP = {
    0: 0,   # latin
    1: 1,   # cyrillic_greek
    2: 2,   # arabic
    3: 3,   # hebrew
    4: 4,   # sino_japanese → han
    5: 6,   # korean
    6: 7,   # ne_indic
    7: 8,   # south_indic
    8: 9,   # se_asian
    9: 10,  # emoji
    10: 11, # caucasus
    11: 12, # ethiopic
    12: 13, # tibetan
}

# Old 26 flat scripts → new 27 flat scripts mapping
# old 0-5 → new 0-5 (latin, cyrillic, greek, arabic, hebrew, han_kana→han)
# new 6 = kana (fresh)
# old 6-25 → new 7-26
OLD_TO_NEW_SCRIPT = {}
for i in range(6):
    OLD_TO_NEW_SCRIPT[i] = i
for i in range(6, 26):
    OLD_TO_NEW_SCRIPT[i] = i + 1

# Old LID-2 group keys → new LID-2 group keys
OLD_TO_NEW_LID2 = {
    1: 1,    # cyrillic_greek
    6: 7,    # ne_indic
    7: 8,    # south_indic
    8: 9,    # se_asian
    10: 11,  # caucasus
}

# Old CTC group indices → new CTC group indices
# Group 4 (sino_japanese) has vocab mismatch → skip
OLD_TO_NEW_CTC = {
    0: 0, 1: 1, 2: 2, 3: 3,
    # 4: skip (vocab changed)
    5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13,
}


def _remap_indexed_key(key, prefix, old_to_new):
    """Remap a key like 'prefix.{old_id}.suffix' → 'prefix.{new_id}.suffix'.

    Returns (new_key, True) if remapped, (None, False) if old_id has no mapping.
    """
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
    """Remap expert_attns.{id} and expert_mlps.{id} inside block keys."""
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


def migrate(old_state, dim=256):
    """Remap old 13-group state_dict to new 14-group layout."""
    new_state = {}
    skipped = []
    remapped = []

    for key, value in old_state.items():
        new_key = key
        keep = True

        # --- LID-1: drop (shape changes 14 → 15) ---
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

        # --- CTC modules ---
        elif key.startswith('ctc_modules.'):
            new_key, keep = _remap_indexed_key(key, 'ctc_modules', OLD_TO_NEW_CTC)

        if not keep:
            skipped.append(key)
            continue

        if new_key != key:
            remapped.append(f'{key} → {new_key}')
        new_state[new_key] = value

    return new_state, skipped, remapped


def main():
    parser = argparse.ArgumentParser(description="Migrate 13-group checkpoint to 14-group")
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
    for r in remapped[:10]:
        print(f"    {r}")
    if len(remapped) > 10:
        print(f"    ... and {len(remapped) - 10} more")

    print(f"\n  Skipped (will reinitialize): {len(skipped)} keys")
    prefixes = {}
    for s in skipped:
        p = s.split('.')[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    for p, n in sorted(prefixes.items()):
        print(f"    {p}.* ({n} params)")

    print(f"\n  Result: {len(new_state)} parameters in migrated checkpoint")

    # Save with updated model state, drop optimizer (incompatible)
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
    print("  - group_head (LID-1): 14→15 outputs, needs retraining")
    print("  - kana group expert (new group 5): identity-init")
    print("  - kana script expert (new flat script 6): identity-init")
    print("  - kana group/script aggregates: identity-init")
    print("  - han CTC head (group 4): vocab changed, reinit")
    print("  - kana CTC head (group 5): new, reinit")
    print("  - Optimizer/scheduler state: dropped (fresh start)")
    print("\nRecommended training:")
    print("  1. Freeze everything except LID, train LID to convergence:")
    print("     python scripts/train.py --resume <migrated.pt> --freeze-except lid \\")
    print("       --ctc-weight 0 --epochs 3 --lr 1e-3")
    print("  2. Unfreeze all, continue training:")
    print("     python scripts/train.py --resume <lid_retrained.pt> --epochs 20 --lr 5e-5")


if __name__ == "__main__":
    main()
