"""
LMDB Dataset Loader.

Loads word crop images and their text labels from LMDB databases.
Supports variable-width images with collation for batching.

LMDB format:
  key: f"image-{index:09d}" -> value: PNG/JPEG bytes
  key: f"label-{index:09d}" -> value: UTF-8 text
  key: "num-samples" -> value: count as string
"""

import io
from pathlib import Path

import lmdb
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from src.data.augmentation import RandAugmentOCR


def preprocess_crop(
    img: Image.Image,
    target_height: int = 32,
    max_width: int = 320,
) -> np.ndarray:
    """Normalize a word crop to fixed height with preserved aspect ratio.

    1. Resize height to target_height (32px), scale width proportionally
    2. If width > max_width, resize width to max_width (squash)
    3. Normalize pixel values to [-1, 1]
    4. Convert HWC -> CHW

    Args:
        img: PIL Image (RGB).
        target_height: Target height in pixels.
        max_width: Maximum width in pixels.

    Returns:
        (3, target_height, W') float32 array.
    """
    w, h = img.size
    new_w = int(w * target_height / h)
    new_w = min(new_w, max_width)
    new_w = max(new_w, 1)

    img = img.resize((new_w, target_height), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # Ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Normalize to [-1, 1]
    arr = arr / 127.5 - 1.0

    # HWC -> CHW
    return arr.transpose(2, 0, 1)


class LMDBDataset(Dataset):
    """Dataset that reads word crops and labels from an LMDB database."""

    def __init__(
        self,
        lmdb_path: str | Path,
        target_height: int = 32,
        max_width: int = 320,
        augment: bool = False,
        n_aug_ops: int = 2,
    ):
        self.lmdb_path = str(lmdb_path)
        self.target_height = target_height
        self.max_width = max_width
        self.augmentor = RandAugmentOCR(n_ops=n_aug_ops) if augment else None

        # Open LMDB in read-only mode
        self.env = lmdb.open(
            self.lmdb_path,
            max_readers=8,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get(b"num-samples").decode())

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        """
        Returns:
            image: (3, H, W) float32 array, normalized to [-1, 1].
            label: text string.
        """
        with self.env.begin(write=False) as txn:
            img_key = f"image-{idx:09d}".encode()
            lbl_key = f"label-{idx:09d}".encode()

            img_bytes = txn.get(img_key)
            label = txn.get(lbl_key).decode("utf-8")

        # Decode image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Apply augmentation
        if self.augmentor is not None:
            img = self.augmentor(img)

        # Preprocess
        image = preprocess_crop(img, self.target_height, self.max_width)

        return image, label

    def close(self):
        self.env.close()


def collate_ocr(
    batch: list[tuple[np.ndarray, str]],
) -> tuple[Tensor, list[str], Tensor]:
    """Collate variable-width images into a padded batch.

    Pads images on the right with zeros (black) to the max width
    in the batch. Returns widths for masking.

    Args:
        batch: List of (image_array, label_string) tuples.

    Returns:
        images: (B, 3, H, max_W) float32 tensor.
        labels: List of B label strings.
        widths: (B,) int tensor of original widths.
    """
    images, labels = zip(*batch)

    # Find max width in batch
    max_w = max(img.shape[2] for img in images)
    B = len(images)
    H = images[0].shape[1]

    # Create padded tensor
    padded = torch.zeros(B, 3, H, max_w, dtype=torch.float32)
    widths = torch.zeros(B, dtype=torch.long)

    for i, img in enumerate(images):
        w = img.shape[2]
        padded[i, :, :, :w] = torch.from_numpy(img)
        widths[i] = w

    return padded, list(labels), widths


def create_lmdb(
    output_path: str | Path,
    images: list[Image.Image],
    labels: list[str],
    map_size: int = 1 << 30,  # 1GB default
):
    """Write images and labels to an LMDB database.

    Args:
        output_path: Path for the LMDB directory.
        images: List of PIL Images.
        labels: List of text labels.
        map_size: Maximum size of the database in bytes.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(str(output_path), map_size=map_size)

    with env.begin(write=True) as txn:
        for i, (img, label) in enumerate(zip(images, labels)):
            # Encode image as PNG
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            txn.put(f"image-{i:09d}".encode(), img_bytes)
            txn.put(f"label-{i:09d}".encode(), label.encode("utf-8"))

        txn.put(b"num-samples", str(len(images)).encode())

    env.close()
