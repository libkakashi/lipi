"""Measure inference FLOPs for the Lipi MoE encoder.

Uses torch.utils.flop_counter.FlopCounterMode which hooks into ATen ops
directly, so dynamic routing / data-dependent control flow are handled
correctly. Single-sample inference, single-script routing — measures the
*active* compute path.
"""

import torch
from torch.utils.flop_counter import FlopCounterMode

from src.model.encoder import LipiMoEEncoder

# Realistic group/script vocab sizes from ARCHITECTURE.md
GROUP_SCRIPT_VOCAB_SIZES = [
    [797],                        # 0  latin
    [364, 426],                   # 1  cyrillic_greek
    [500],                        # 2  arabic
    [192],                        # 3  hebrew
    [3811],                       # 4  han
    [263],                        # 5  kana
    [1500],                       # 6  korean
    [1000, 600, 900, 900, 850],   # 7  ne_indic
    [550, 950, 500],              # 8  dravidian_north
    [900, 350],                   # 9  dravidian_south
    [450, 500, 650, 950],         # 10 se_asian
    [107],                        # 11 emoji
    [150, 186],                   # 12 caucasus
    [521],                        # 13 ethiopic
    [550],                        # 14 tibetan
]


def measure_flops(model, image_w: int, group_id: int, script_id: int):
    """Run one forward pass, return total FLOPs."""
    images = torch.randn(1, 3, 32, image_w)
    group_ids = torch.tensor([group_id], dtype=torch.long)
    script_ids = torch.tensor([script_id], dtype=torch.long)

    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(images, group_ids=group_ids, script_ids=script_ids)
    return fc.get_total_flops()


def build_model(mlp_ratio: int, shared_mlp_ratio: int):
    return LipiMoEEncoder(
        dim=384,
        mlp_ratio=mlp_ratio,
        shared_mlp_ratio=shared_mlp_ratio,
        group_script_vocab_sizes=GROUP_SCRIPT_VOCAB_SIZES,
    ).eval()


def main():
    configs_to_compare = [
        ("baseline (all r=2)", 2, 2),
        ("all shared r=4, experts r=2", 2, 4),
        ("all r=4", 4, 4),
    ]

    print(f"{'Config':<32} {'Params':>10} {'GFLOPs@W=128':>15} {'vs base':>10}")
    print("-" * 70)

    base_flops = None
    for label, mlp_r, shared_r in configs_to_compare:
        model = build_model(mlp_ratio=mlp_r, shared_mlp_ratio=shared_r)

        n_params = sum(p.numel() for p in model.parameters())
        flops = measure_flops(model, image_w=128, group_id=0, script_id=0)
        gf = flops / 1e9
        if base_flops is None:
            base_flops = flops
            ratio_str = "1.00x"
        else:
            ratio_str = f"{flops/base_flops:.2f}x"
        print(f"{label:<32} {n_params/1e6:>8.1f}M {gf:>13.3f} {ratio_str:>10}")

    print()
    # Original detailed breakdown for the baseline:
    model = build_model(mlp_ratio=2, shared_mlp_ratio=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"--- Baseline (r=2 everywhere) detail ---")
    print(f"Total params: {n_params/1e6:.1f}M")
    print()

    # Per-component param breakdown
    breakdown = {
        "stem": model.stem,
        "convA (3 blocks)": model.convA,
        "convB (3 blocks)": model.convB,
        "blur_ab / blur_bc": [model.blur_ab, model.blur_bc],
        "swac_in_proj + swa_c (3 blocks)":
            [model.swac_in_proj, model.swa_c],
        "swa_d (3 blocks)": model.swa_d,
        "merge_cd / merge_d1": [model.merge_cd, model.merge_d1],
        "lid1_attn + group_head": [model.lid1_attn, model.group_head],
        "group experts (15 × local+wide)":
            [model.group_local_blocks, model.group_wide_blocks],
        "group aggregates (15)": model.group_aggregates,
        "lid2 heads": model.lid2_heads,
        "script experts (27 × local+wide)":
            [model.script_local_blocks, model.script_wide_blocks],
        "script aggregates (27)": model.script_aggregates,
        "norm": model.norm,
        "ctc heads (27 scripts)": model.ctc_modules,
    }
    print("Param breakdown:")
    for name, mod in breakdown.items():
        if isinstance(mod, list):
            n = sum(sum(p.numel() for p in m.parameters()) for m in mod)
        else:
            n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:40s} {n/1e6:6.2f}M")
    print()

    # Measure FLOPs for several widths and scripts
    print("Active-path inference FLOPs (single sample):")
    print(f"{'Width':>6} {'L=W/4':>6} {'Script':>10} {'Group':>3} "
          f"{'GFLOPs':>8} {'M·L':>8}")
    print("-" * 56)

    configs = [
        # (width, group_id, script_id, label)
        (64,  0, 0,  "latin"),
        (128, 0, 0,  "latin"),
        (256, 0, 0,  "latin"),
        (128, 4, 0,  "han"),       # large vocab
        (128, 6, 0,  "korean"),    # multi-script via decomposition
        (128, 7, 0,  "devanagari"),# multi-script group → fires LID-2
        (128, 11, 0, "emoji"),     # tiny vocab
    ]
    for w, gid, sid, label in configs:
        flops = measure_flops(model, w, gid, sid)
        L = w // 4
        gflops = flops / 1e9
        per_l = flops / L / 1e6
        print(f"{w:>6} {L:>6} {label:>10} {gid:>3} "
              f"{gflops:>8.3f} {per_l:>8.2f}")


if __name__ == "__main__":
    main()
