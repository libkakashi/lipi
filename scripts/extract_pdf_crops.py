#!/usr/bin/env python3
"""
Extract word crops + ground truth from clean legal PDFs.

Usage:
    python scripts/extract_pdf_crops.py \
        --pdf_dir training_data/pdfs/english \
        --output training_data/datasets/en_pdf_crops \
        --dpi 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pdf_extractor import extract_pdf_directory


def main():
    parser = argparse.ArgumentParser(description="Extract word crops from PDFs")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory with PDF files")
    parser.add_argument("--output", type=str, required=True, help="Output LMDB path")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()

    extract_pdf_directory(
        pdf_dir=args.pdf_dir,
        output_lmdb=args.output,
        dpi=args.dpi,
        target_height=args.height,
    )


if __name__ == "__main__":
    main()
