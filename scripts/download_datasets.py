#!/usr/bin/env python3
"""
Download real-world OCR datasets for fine-tuning.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset iiit-indic
    python scripts/download_datasets.py --list
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "training_data" / "real_datasets"


def run(cmd, cwd=None):
    """Run a shell command."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: command exited with {result.returncode}")
    return result.returncode == 0


def check_hf():
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub  # noqa
        return True
    except ImportError:
        print("Installing huggingface_hub...")
        run("pip install huggingface_hub[cli]")
        return True


def check_kaggle():
    """Check if kaggle CLI is configured."""
    if not Path(os.path.expanduser("~/.kaggle/kaggle.json")).exists():
        print("  Kaggle API not configured. Set up at https://www.kaggle.com/docs/api")
        print("  Place kaggle.json in ~/.kaggle/")
        return False
    return True


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

DATASETS = {}


def register(name, scripts, description):
    def decorator(func):
        DATASETS[name] = {
            "fn": func,
            "scripts": scripts,
            "description": description,
        }
        return func
    return decorator


@register("iiit-indic",
          ["devanagari", "bengali", "gurmukhi", "gujarati", "odia",
           "kannada", "telugu", "malayalam", "tamil"],
          "IIIT-INDIC-HW-WORDS: ~1M handwritten word images, 10 Indic scripts")
def download_iiit_indic(out_dir):
    """Download from HuggingFace mirrors."""
    check_hf()
    dest = out_dir / "iiit-indic-hw"
    dest.mkdir(parents=True, exist_ok=True)

    # Hindi/Devanagari subset
    run(f"huggingface-cli download c3rl/IIIT-INDIC-HW-WORDS-Hindi "
        f"--repo-type dataset --local-dir {dest}/hindi")

    # Full dataset from Kaggle
    if check_kaggle():
        run(f"kaggle datasets download -d himanshukumarrajak/indic-data "
            f"-p {dest} --unzip")

    print(f"  Saved to {dest}/")


@register("indic-scene",
          ["devanagari", "bengali", "gurmukhi", "gujarati", "odia",
           "kannada", "telugu", "malayalam", "tamil", "latin"],
          "IndicSTR12 + Bharat Scene Text: 127K+ real Indic scene text words")
def download_indic_scene(out_dir):
    dest = out_dir / "indic-scene"
    dest.mkdir(parents=True, exist_ok=True)

    # Bharat Scene Text Dataset (GitHub, direct download)
    run(f"git clone https://github.com/Bhashini-IITJ/BharatSceneTextDataset.git "
        f"{dest}/bstd")

    # IndicSTR12 real images (direct download)
    indicstr_dest = dest / "indicstr12"
    indicstr_dest.mkdir(parents=True, exist_ok=True)
    run(f"wget -P {indicstr_dest} https://cvit.iiit.ac.in/images/datasets/IndicSTR12/real.zip")
    run(f"unzip -o {indicstr_dest}/real.zip -d {indicstr_dest}")

    # Mozhi (printed Indic text) — requires form submission
    print("  Mozhi (1.2M+ printed word images, 13 languages):")
    print("  Submit form at: https://cvit.iiit.ac.in/usodi/tdocrmil.php")
    print(f"  Extract to: {dest}/mozhi/")

    print(f"  Saved to {dest}/")


@register("iam",
          ["latin"],
          "IAM Handwriting Database: 115K English word images, 657 writers")
def download_iam(out_dir):
    """Download IAM from HuggingFace."""
    check_hf()
    dest = out_dir / "iam"
    dest.mkdir(parents=True, exist_ok=True)

    # Word-level dataset
    run(f"huggingface-cli download Teklia/IAM-line "
        f"--repo-type dataset --local-dir {dest}/lines")

    # Also try Kaggle word-level
    if check_kaggle():
        run(f"kaggle datasets download -d nibinv23/iam-handwriting-word-database "
            f"-p {dest} --unzip")

    print(f"  Saved to {dest}/")


@register("casia-hwdb",
          ["han_kana"],
          "CASIA-HWDB2: Chinese handwritten text lines (HuggingFace)")
def download_casia(out_dir):
    check_hf()
    dest = out_dir / "casia-hwdb"
    dest.mkdir(parents=True, exist_ok=True)

    run(f"huggingface-cli download Teklia/CASIA-HWDB2-line "
        f"--repo-type dataset --local-dir {dest}")

    print(f"  Saved to {dest}/")


@register("hkr",
          ["cyrillic"],
          "HKR: 63K Russian/Kazakh handwritten sentences")
def download_hkr(out_dir):
    dest = out_dir / "hkr"
    dest.mkdir(parents=True, exist_ok=True)
    run(f"git clone https://github.com/abdoelsayed2016/HKR_Dataset.git {dest}/repo")
    print(f"  Saved to {dest}/")


@register("hebrew-htr",
          ["hebrew"],
          "HebHTR: 100K Hebrew handwritten word images")
def download_hebrew(out_dir):
    check_hf()
    dest = out_dir / "hebrew"
    dest.mkdir(parents=True, exist_ok=True)

    run(f"huggingface-cli download sivan22/hebrew-handwritten-dataset "
        f"--repo-type dataset --local-dir {dest}/hhd")

    run(f"git clone https://github.com/Lotemn102/HebHTR.git {dest}/hebhtr")

    print(f"  Saved to {dest}/")


@register("arabic",
          ["arabic"],
          "Arabic OCR: 2.16M words + KHATT handwriting + Muharaf + EvArEST scene text")
def download_arabic(out_dir):
    check_hf()
    dest = out_dir / "arabic"
    dest.mkdir(parents=True, exist_ok=True)

    # Arabic OCR Dataset — 2.16M word images (biggest Arabic dataset)
    run(f"huggingface-cli download mssqpi/Arabic-OCR-Dataset "
        f"--repo-type dataset --local-dir {dest}/arabic-ocr-2m")

    # KHATT handwritten Arabic
    run(f"huggingface-cli download johnlockejrr/KHATT_v1.0_dataset "
        f"--repo-type dataset --local-dir {dest}/khatt")

    # EvArEST Arabic scene text
    run(f"huggingface-cli download Melaraby/EvArEST-dataset-for-Arabic-scene-text-recognition "
        f"--repo-type dataset --local-dir {dest}/evarest")

    # Muharaf (historical Arabic manuscripts)
    run(f"git clone https://github.com/MehreenMehreen/muharaf.git {dest}/muharaf")

    # OpenITI (Arabic + Persian + Urdu printed)
    run(f"huggingface-cli download --repo-type dataset "
        f"OpenITI/arabic-script-ocr-training-data --local-dir {dest}/openiti 2>/dev/null || "
        f"wget -P {dest}/openiti https://zenodo.org/records/7050296/files/training_data.zip")

    # Kaggle Arabic docs
    if check_kaggle():
        run(f"kaggle datasets download -d humansintheloop/arabic-documents-ocr-dataset "
            f"-p {dest} --unzip")

    print(f"  IFN/ENIT requires registration: http://www.ifnenit.com/download.htm")
    print(f"  KHATT: https://gts.ai/dataset-download/khatt-arabic-dataset/")
    print(f"  Saved to {dest}/")


@register("ethiopic",
          ["ethiopic"],
          "HHD-Ethiopic: 80K historical Ethiopic text-line images")
def download_ethiopic(out_dir):
    check_hf()
    dest = out_dir / "ethiopic"
    dest.mkdir(parents=True, exist_ok=True)

    run(f"huggingface-cli download OCR-Ethiopic/HHD-Ethiopic "
        f"--repo-type dataset --local-dir {dest}")

    print(f"  Saved to {dest}/")


@register("thai",
          ["thai"],
          "iApp Thai: 4.9K handwritten sentences from 2K writers")
def download_thai(out_dir):
    check_hf()
    dest = out_dir / "thai"
    dest.mkdir(parents=True, exist_ok=True)

    run(f"huggingface-cli download iapp/thai_handwriting_dataset "
        f"--repo-type dataset --local-dir {dest}")

    print(f"  Saved to {dest}/")


@register("khmer",
          ["khmer"],
          "KhmerST + benchmark: Khmer scene text")
def download_khmer(out_dir):
    dest = out_dir / "khmer"
    dest.mkdir(parents=True, exist_ok=True)

    run(f"git clone https://gitlab.com/vannkinhnom123/khmerst.git {dest}/khmerst")
    run(f"git clone https://github.com/EKYCSolutions/khmer-ocr-benchmark-dataset.git "
        f"{dest}/benchmark")

    print(f"  Saved to {dest}/")


@register("bengali",
          ["bengali"],
          "BN-HTRd: 108K Bengali handwritten word instances")
def download_bengali(out_dir):
    dest = out_dir / "bengali"
    dest.mkdir(parents=True, exist_ok=True)

    print("  BN-HTRd available at:")
    print("  https://data.mendeley.com/datasets/743k6dm543/1")
    print(f"  Download and extract to {dest}/")


@register("chinese-text",
          ["han_kana"],
          "Chinese text recognition: 500K line images")
def download_chinese_text(out_dir):
    check_hf()
    dest = out_dir / "chinese-text"
    dest.mkdir(parents=True, exist_ok=True)
    run(f"huggingface-cli download priyank-m/chinese_text_recognition "
        f"--repo-type dataset --local-dir {dest}")
    print(f"  Saved to {dest}/")


@register("russian-hw",
          ["cyrillic"],
          "HWR200: 30K Russian handwritten sentences, 3 photo conditions")
def download_russian_hw(out_dir):
    check_hf()
    dest = out_dir / "russian-hw"
    dest.mkdir(parents=True, exist_ok=True)
    run(f"huggingface-cli download AntiplagiatCompany/HWR200 "
        f"--repo-type dataset --local-dir {dest}")
    print(f"  Saved to {dest}/")


@register("burmese-real",
          ["burmese"],
          "Burmese OCR: 9K clean line images (real)")
def download_burmese_real(out_dir):
    check_hf()
    dest = out_dir / "burmese-real"
    dest.mkdir(parents=True, exist_ok=True)
    run(f"huggingface-cli download alexbeatson/burmese_ocr_data "
        f"--repo-type dataset --local-dir {dest}")
    print(f"  Saved to {dest}/")


@register("sanskrit-ocr",
          ["devanagari"],
          "Sanskrit typed OCR: 3.5K Devanagari word images")
def download_sanskrit(out_dir):
    check_hf()
    dest = out_dir / "sanskrit-ocr"
    dest.mkdir(parents=True, exist_ok=True)
    run(f"huggingface-cli download Process-Venue/Sanskrit-OCR-Typed-Dataset "
        f"--repo-type dataset --local-dir {dest}")
    print(f"  Saved to {dest}/")


@register("burmese",
          ["burmese"],
          "myOCR: 25K Myanmar text-line images (synthetic)")
def download_burmese(out_dir):
    check_hf()
    dest = out_dir / "burmese"
    dest.mkdir(parents=True, exist_ok=True)

    run(f"huggingface-cli download LULab/myOCR "
        f"--repo-type dataset --local-dir {dest}")

    print(f"  Saved to {dest}/")


@register("tibetan",
          ["tibetan"],
          "BDRC Tibetan: 30K woodblock print line images")
def download_tibetan(out_dir):
    dest = out_dir / "tibetan"
    dest.mkdir(parents=True, exist_ok=True)

    print("  BDRC Tibetan OCR data available at:")
    print("  https://huggingface.co/buda-base")
    print("  https://github.com/buda-base/tibetan-ocr-app/releases")
    print(f"  Download and extract to {dest}/")


@register("scene-text",
          ["latin", "cyrillic", "arabic", "devanagari", "bengali",
           "han_kana", "korean"],
          "ICDAR 2019 MLT + TextOCR: multilingual scene text")
def download_scene_text(out_dir):
    dest = out_dir / "scene-text"
    dest.mkdir(parents=True, exist_ok=True)

    # TextOCR from HuggingFace
    check_hf()
    run(f"huggingface-cli download yunusserhat/TextOCR-Dataset "
        f"--repo-type dataset --local-dir {dest}/textocr")

    # Google HierText
    run(f"git clone https://github.com/google-research-datasets/hiertext.git "
        f"{dest}/hiertext")

    # ICDAR MLT from Kaggle
    if check_kaggle():
        run(f"kaggle datasets download -d zubairalibhutto/mlt-19-ocr-dataset "
            f"-p {dest} --unzip")

    print(f"  Saved to {dest}/")


@register("rimes",
          ["latin"],
          "RIMES: 12K French handwritten pages")
def download_rimes(out_dir):
    check_hf()
    dest = out_dir / "rimes"
    dest.mkdir(parents=True, exist_ok=True)

    run(f"huggingface-cli download Teklia/RIMES-2011-line "
        f"--repo-type dataset --local-dir {dest}")

    print(f"  Saved to {dest}/")


@register("sinhala",
          ["sinhala"],
          "AksharaOCR + SinOCR: Sinhala printed + handwritten")
def download_sinhala(out_dir):
    dest = out_dir / "sinhala"
    dest.mkdir(parents=True, exist_ok=True)

    run(f"git clone https://github.com/SriDoc/datasets.git {dest}/sinocr")

    print("  AksharaOCR available at:")
    print("  https://ieee-dataport.org/documents/aksharaocr-real-world-image-based-sinhala-and-sinhala-english-mixed-ocr-datasets")
    print(f"  Saved to {dest}/")


@register("greek",
          ["greek"],
          "Greek handwritten characters from GCDB")
def download_greek(out_dir):
    dest = out_dir / "greek"
    dest.mkdir(parents=True, exist_ok=True)

    if check_kaggle():
        run(f"kaggle datasets download -d vrushalipatel/handwritten-greek-characters-from-gcdb "
            f"-p {dest} --unzip")

    print(f"  Saved to {dest}/")


@register("cyrillic-extra",
          ["cyrillic"],
          "Cyrillic handwriting dataset (Kaggle)")
def download_cyrillic(out_dir):
    dest = out_dir / "cyrillic"
    dest.mkdir(parents=True, exist_ok=True)

    if check_kaggle():
        run(f"kaggle datasets download -d constantinwerner/cyrillic-handwriting-dataset "
            f"-p {dest} --unzip")

    print(f"  Saved to {dest}/")


@register("chinese-scene",
          ["han_kana"],
          "RCTW-17 + CTW: Chinese scene text in the wild")
def download_chinese_scene(out_dir):
    dest = out_dir / "chinese-scene"
    dest.mkdir(parents=True, exist_ok=True)

    print("  RCTW-17: https://rrc.cvc.uab.es/?ch=12")
    print("  CTW: https://ctwdataset.github.io/")
    print("  LSVT: https://rrc.cvc.uab.es/ (search LSVT)")
    print(f"  Download and extract to {dest}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download real-world OCR datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Download specific dataset (use --list to see options)")
    parser.add_argument("--all", action="store_true",
                        help="Download all available datasets")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    parser.add_argument("--out", type=str, default=str(DATA_DIR),
                        help=f"Output directory (default: {DATA_DIR})")
    args = parser.parse_args()

    if args.list or (not args.dataset and not args.all):
        print("\nAvailable datasets:\n")
        for name, info in sorted(DATASETS.items()):
            scripts = ", ".join(info["scripts"])
            print(f"  {name:<20s} [{scripts}]")
            print(f"  {'':20s} {info['description']}\n")
        print(f"Usage: python {sys.argv[0]} --dataset <name>")
        print(f"       python {sys.argv[0]} --all")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = list(DATASETS.keys())
    else:
        targets = [d.strip() for d in args.dataset.split(",")]
        for t in targets:
            if t not in DATASETS:
                print(f"Unknown dataset: {t}")
                print(f"Available: {', '.join(sorted(DATASETS.keys()))}")
                return

    for name in targets:
        info = DATASETS[name]
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"  {info['description']}")
        print(f"  Scripts: {', '.join(info['scripts'])}")
        print(f"{'='*60}\n")
        try:
            info["fn"](out_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Done. Datasets saved to {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
