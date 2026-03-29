"""
PDF Word Crop Extractor.

Extracts individual word crops with perfect ground truth labels
from clean (digitally-produced) legal PDFs using PyMuPDF.

This is the primary training data source — labels are perfect
because they come directly from the PDF's text stream, not OCR.

Pipeline:
  1. Open PDF with PyMuPDF
  2. For each page, extract word-level spans with bounding boxes
  3. Render each word as a crop image
  4. Pair (crop_image, text_label)
  5. Write to LMDB

Handles multi-script PDFs (English + Hindi + Tamil in same document).
"""

from pathlib import Path
from dataclasses import dataclass

from PIL import Image


@dataclass
class WordCrop:
    """A single extracted word crop with metadata."""
    image: Image.Image
    text: str
    page_num: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_name: str = ""
    font_size: float = 0.0


def extract_words_from_pdf(
    pdf_path: str | Path,
    dpi: int = 300,
    min_word_length: int = 1,
    max_word_length: int = 30,
    target_height: int = 32,
) -> list[WordCrop]:
    """Extract word crops from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Rendering DPI for crop extraction.
        min_word_length: Minimum characters per word.
        max_word_length: Maximum characters per word.
        target_height: Target crop height in pixels.

    Returns:
        List of WordCrop objects.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    crops = []
    zoom = dpi / 72  # PDF uses 72 DPI

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Get word-level text blocks
        words = page.get_text("words")
        # Each word: (x0, y0, x1, y1, text, block_no, line_no, word_no)

        # Render page at target DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        for word_data in words:
            x0, y0, x1, y1, text = word_data[0], word_data[1], word_data[2], word_data[3], word_data[4]

            # Filter
            text = text.strip()
            if len(text) < min_word_length or len(text) > max_word_length:
                continue

            # Scale coordinates to pixel space
            px0 = int(x0 * zoom)
            py0 = int(y0 * zoom)
            px1 = int(x1 * zoom)
            py1 = int(y1 * zoom)

            # Add padding
            pad = 2
            px0 = max(0, px0 - pad)
            py0 = max(0, py0 - pad)
            px1 = min(pix.width, px1 + pad)
            py1 = min(pix.height, py1 + pad)

            if px1 <= px0 or py1 <= py0:
                continue

            # Crop
            crop_img = page_img.crop((px0, py0, px1, py1))

            # Resize to target height
            w, h = crop_img.size
            if h > 0:
                new_w = int(w * target_height / h)
                new_w = max(new_w, 1)
                crop_img = crop_img.resize((new_w, target_height), Image.BILINEAR)

            crops.append(WordCrop(
                image=crop_img,
                text=text,
                page_num=page_num,
                bbox=(x0, y0, x1, y1),
            ))

    doc.close()
    return crops


def extract_pdf_directory(
    pdf_dir: str | Path,
    output_lmdb: str | Path,
    dpi: int = 300,
    target_height: int = 32,
):
    """Extract word crops from all PDFs in a directory.

    Args:
        pdf_dir: Directory containing PDF files.
        output_lmdb: Output LMDB path.
        dpi: Rendering DPI.
        target_height: Target crop height.
    """
    from src.data.dataset import create_lmdb

    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    all_images = []
    all_labels = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            crops = extract_words_from_pdf(pdf_path, dpi=dpi, target_height=target_height)
            for crop in crops:
                all_images.append(crop.image)
                all_labels.append(crop.text)
            print(f"  Extracted {len(crops)} words")
        except Exception as e:
            print(f"  Error: {e}")

    if all_images:
        print(f"\nWriting {len(all_images)} crops to LMDB: {output_lmdb}")
        create_lmdb(output_lmdb, all_images, all_labels)
    else:
        print("No crops extracted")
