#!/usr/bin/env python3
"""
Export all model components to ONNX.

Usage:
    python scripts/export_onnx.py \
        --backbone checkpoints/phase1/phase1_best.pt \
        --adapter_dir checkpoints/phase2 \
        --lid checkpoints/lid.pt \
        --output_dir models/
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.onnx_export import export_backbone, export_rnnt_head, export_lid
from src.model.lid import SCRIPT_NAMES


def main():
    parser = argparse.ArgumentParser(description="Export all components to ONNX")
    parser.add_argument("--backbone", type=str, required=True, help="Phase 1 checkpoint")
    parser.add_argument("--adapter_dir", type=str, help="Directory with Phase 2 checkpoints")
    parser.add_argument("--lid", type=str, help="LID checkpoint")
    parser.add_argument("--output_dir", type=str, default="models/")
    parser.add_argument("--languages", nargs="+", default=None, help="Languages to export")
    args = parser.parse_args()

    output = Path(args.output_dir)

    # Export backbone
    print("=" * 60)
    print("Exporting backbone...")
    export_backbone(args.backbone, output / "backbone.onnx")

    # Export LID
    if args.lid and Path(args.lid).exists():
        print("\n" + "=" * 60)
        print("Exporting LID...")
        export_lid(args.lid, output / "lid.onnx")

    # Export RNN-T heads per language
    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        languages = args.languages or SCRIPT_NAMES

        for lang in languages:
            checkpoint = adapter_dir / lang / f"adapter_best.pt"
            if not checkpoint.exists():
                # Try numbered checkpoints
                checkpoints = sorted(
                    (adapter_dir / lang).glob("adapter_epoch*.pt"),
                    key=lambda p: p.stat().st_mtime,
                ) if (adapter_dir / lang).exists() else []
                if checkpoints:
                    checkpoint = checkpoints[-1]
                else:
                    print(f"\nSkipping {lang}: no checkpoint found")
                    continue

            print(f"\n{'='*60}")
            print(f"Exporting {lang} RNN-T head from: {checkpoint}")

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            export_rnnt_head(
                pred_net_state=state.get("pred_net", {}),
                joint_net_state=state.get("joint_net", {}),
                output_dir=output / "heads",
                language=lang,
            )

    print(f"\n{'='*60}")
    print(f"Export complete. Output: {output}")


if __name__ == "__main__":
    main()
