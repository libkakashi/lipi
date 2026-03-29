#!/usr/bin/env python3
"""
Convert LoRA adapters to ONNX Runtime .onnx_adapter format.

Usage:
    python scripts/export_adapters.py \
        --adapter_dir checkpoints/phase2 \
        --output_dir models/adapters
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.export.lora_export import export_adapter_native, extract_lora_state_dict
from src.model.lid import SCRIPT_NAMES


def main():
    parser = argparse.ArgumentParser(description="Export LoRA adapters")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/adapters")
    parser.add_argument("--languages", nargs="+", default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    adapter_dir = Path(args.adapter_dir)
    languages = args.languages or SCRIPT_NAMES

    for lang in languages:
        # Find latest checkpoint
        lang_dir = adapter_dir / lang
        if not lang_dir.exists():
            print(f"Skipping {lang}: directory not found")
            continue

        checkpoints = sorted(lang_dir.glob("adapter_epoch*.pt"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            print(f"Skipping {lang}: no checkpoints found")
            continue

        checkpoint = checkpoints[-1]
        print(f"\nExporting {lang} from: {checkpoint}")

        lora_state = extract_lora_state_dict(checkpoint)
        if not lora_state:
            print(f"  No LoRA parameters found, skipping")
            continue

        output_path = output / f"{lang}.onnx_adapter"
        try:
            export_adapter_native(lora_state, output_path)
        except RuntimeError as e:
            print(f"  Native export failed: {e}")
            print(f"  Try: pip install onnxruntime>=1.24.0")

    print(f"\nAdapter export complete. Output: {output}")


if __name__ == "__main__":
    main()
