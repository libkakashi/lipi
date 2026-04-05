"""
Merge two checkpoints: take most weights from a base checkpoint,
then overwrite specific groups' expert/CTC weights from a donor.

Usage:
    python -m scripts.merge_checkpoints \
        --base checkpoints/moe/moe_epoch50.pt \
        --donor checkpoints/moe/moe_epoch70.pt \
        --donor-groups 0 \
        --out checkpoints/moe/moe_merged.pt

This takes everything from --base, then overwrites stage1, stage2,
and ctc_modules weights for the specified group indices from --donor.
Shape mismatches (e.g., CTC head after vocab change) are skipped.
"""

import argparse
import re
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Merge expert weights from a donor checkpoint into a base checkpoint")
    parser.add_argument("--base", required=True,
                        help="Base checkpoint (provides shared encoder, LID, and all expert weights)")
    parser.add_argument("--donor", required=True,
                        help="Donor checkpoint (provides expert weights for specified groups)")
    parser.add_argument("--donor-groups", required=True, type=str,
                        help="Comma-separated group indices to take from donor (e.g., '0' or '0,1')")
    parser.add_argument("--out", required=True,
                        help="Output checkpoint path")
    args = parser.parse_args()

    donor_groups = {int(g) for g in args.donor_groups.split(",")}

    print(f"Base:   {args.base}")
    print(f"Donor:  {args.donor}")
    print(f"Groups from donor: {sorted(donor_groups)}")
    print(f"Output: {args.out}")

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    donor = torch.load(args.donor, map_location="cpu", weights_only=False)

    base_state = base["model"]
    donor_state = donor["model"]

    # Expert layer patterns: stage1, stage2, ctc_modules for specific group indices
    # Examples:
    #   stage1.0.expert_attns.{group}.qkv.weight
    #   stage2.0.expert_ffns.{group}.fc1.weight
    #   ctc_modules.{group}.heads.0.proj.weight
    expert_pattern = re.compile(
        r"^(stage[12]\.\d+\.expert_(?:attns|ffns)\.(\d+)\..+|"
        r"ctc_modules\.(\d+)\..+)$"
    )

    replaced = 0
    skipped_shape = 0
    skipped_missing = 0

    for key, donor_tensor in donor_state.items():
        m = expert_pattern.match(key)
        if not m:
            continue

        # Extract group index from the key
        group_idx = int(m.group(2)) if m.group(2) is not None else int(m.group(3))

        if group_idx not in donor_groups:
            continue

        if key not in base_state:
            skipped_missing += 1
            continue

        if base_state[key].shape != donor_tensor.shape:
            print(f"  Shape mismatch, skipped: {key} "
                  f"(base: {base_state[key].shape}, donor: {donor_tensor.shape})")
            skipped_shape += 1
            continue

        base_state[key] = donor_tensor
        replaced += 1

    print(f"\nReplaced {replaced} layers from donor")
    if skipped_shape:
        print(f"Skipped {skipped_shape} layers (shape mismatch — e.g., vocab change)")
    if skipped_missing:
        print(f"Skipped {skipped_missing} layers (not in base checkpoint)")

    # Keep base's optimizer/scheduler/scaler state — donor's won't match
    base["model"] = base_state
    torch.save(base, args.out)
    print(f"\nSaved merged checkpoint to {args.out}")


if __name__ == "__main__":
    main()
