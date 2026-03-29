"""
PARSeq LMDB Dataset Adapter.

The PARSeq dataset uses a different LMDB key format than our internal format.
This adapter reads PARSeq LMDBs and presents them through our standard interface.

PARSeq LMDB format:
  key: b"image-%09d" -> value: image bytes
  key: b"label-%09d" -> value: UTF-8 text
  key: b"num-samples" -> value: count as string

This happens to match our format exactly, so we can use LMDBDataset directly.
But the images may be different sizes and formats (not pre-normalized to 32px).
"""

import io
import lmdb
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from src.data.dataset import preprocess_crop


def _decode_image(args: tuple) -> tuple[np.ndarray, str]:
    """Decode a single image from raw bytes. Module-level for pickling."""
    img_bytes, label, target_h, max_w = args
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    crop = preprocess_crop(img, target_h, max_w)
    return (crop, label)


class PARSeqLMDB(Dataset):
    """Read a PARSeq-format LMDB dataset.

    Images are loaded as-is and preprocessed to 32px height on the fly.
    Labels are filtered to remove non-alphanumeric characters if needed.
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        target_height: int = 32,
        max_width: int = 320,
        max_label_len: int = 25,
        case_sensitive: bool = True,
    ):
        self.lmdb_path = str(lmdb_path)
        self.target_height = target_height
        self.max_width = max_width
        self.max_label_len = max_label_len
        self.case_sensitive = case_sensitive

        self.env = lmdb.open(
            self.lmdb_path,
            max_readers=128,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self.num_samples = int(txn.get(b"num-samples").decode())

        self._valid_indices = list(range(self.num_samples))
        self._raw_cache: list[tuple[bytes, str]] | None = None

    def preload_raw(self, max_samples: int | None = None):
        """Load raw image bytes + labels into RAM (no decoding).

        LMDB sequential read is fast (~1 min for 1.5M samples).
        Images are decoded lazily in __getitem__ when the dataloader
        requests them — but from RAM instead of disk.
        """
        import time
        n = min(max_samples or len(self), len(self))
        print(f"Loading {n} raw samples into RAM...")
        t0 = time.time()

        self._raw_cache: list[tuple[bytes, str]] = []
        with self.env.begin(write=False) as txn:
            for i in range(n):
                real_idx = self._valid_indices[i]

                img_key = f"image-{real_idx + 1:09d}".encode()
                lbl_key = f"label-{real_idx + 1:09d}".encode()

                img_bytes = txn.get(img_key)
                if img_bytes is None:
                    img_key = f"image-{real_idx:09d}".encode()
                    lbl_key = f"label-{real_idx:09d}".encode()
                    img_bytes = txn.get(img_key)

                if img_bytes is None:
                    continue

                label_bytes = txn.get(lbl_key)
                label = label_bytes.decode("utf-8") if label_bytes else ""
                self._raw_cache.append((bytes(img_bytes), label))

                if (i + 1) % 500000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {i+1}/{n} read ({elapsed:.0f}s)")

        self._valid_indices = list(range(len(self._raw_cache)))
        elapsed = time.time() - t0
        size_gb = sum(len(b) for b, _ in self._raw_cache) / 1e9
        print(f"  Loaded {len(self._raw_cache)} samples ({size_gb:.1f} GB) in {elapsed:.0f}s")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        # Fast path: raw bytes in RAM, decode with PIL per-worker
        if self._raw_cache is not None:
            img_bytes, label = self._raw_cache[self._valid_indices[idx]]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            crop = preprocess_crop(img, self.target_height, self.max_width)
            return crop, label

        # Slow path: read from LMDB on the fly
        real_idx = self._valid_indices[idx]

        with self.env.begin(write=False) as txn:
            img_key = f"image-{real_idx + 1:09d}".encode()
            lbl_key = f"label-{real_idx + 1:09d}".encode()

            img_bytes = txn.get(img_key)
            if img_bytes is None:
                img_key = f"image-{real_idx:09d}".encode()
                lbl_key = f"label-{real_idx:09d}".encode()
                img_bytes = txn.get(img_key)

            if img_bytes is None:
                blank = np.zeros((3, self.target_height, 32), dtype=np.float32)
                return blank, ""

            label_bytes = txn.get(lbl_key)
            label = label_bytes.decode("utf-8") if label_bytes else ""

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        crop = preprocess_crop(img, self.target_height, self.max_width)
        return crop, label

    def close(self):
        self.env.close()


def discover_parseq_structure(data_dir: str | Path) -> dict:
    """Discover available PARSeq datasets in a directory tree.

    Returns dict mapping dataset name -> path.
    """
    data_dir = Path(data_dir)
    datasets = {}

    # Look for LMDB directories (contain data.mdb)
    for mdb_file in data_dir.rglob("data.mdb"):
        lmdb_dir = mdb_file.parent
        # Use the parent directory names as the dataset name
        rel = lmdb_dir.relative_to(data_dir)
        name = str(rel).replace("/", "_").replace("\\", "_")
        datasets[name] = str(lmdb_dir)

    return datasets
