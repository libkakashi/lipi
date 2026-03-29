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
        self._cache: list[tuple[np.ndarray, str]] | None = None

    def preload(self, max_samples: int | None = None, num_workers: int = 16):
        """Preload all images into RAM using parallel decoding.

        Reads raw bytes from LMDB (fast, sequential), then decodes
        images in parallel across multiple processes.
        """
        import multiprocessing as mp
        from functools import partial

        n = min(max_samples or len(self), len(self))
        print(f"Preloading {n} images into RAM ({num_workers} workers)...")

        # Step 1: Read raw bytes from LMDB (sequential, fast)
        print("  Reading raw bytes from LMDB...")
        raw_data: list[tuple[bytes, str]] = []
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
                raw_data.append((bytes(img_bytes), label))

                if (i + 1) % 500000 == 0:
                    print(f"    {i+1}/{n} read...")

        print(f"  Read {len(raw_data)} samples. Decoding images...")

        # Step 2: Decode images with thread pool (PIL releases GIL)
        from concurrent.futures import ThreadPoolExecutor

        target_h = self.target_height
        max_w = self.max_width

        def _decode(item):
            img_bytes, label = item
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return (preprocess_crop(img, target_h, max_w), label)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            self._cache = list(pool.map(_decode, raw_data, chunksize=1000))

        self._valid_indices = list(range(len(self._cache)))
        print(f"  Preloaded {len(self._cache)} images into RAM")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        # Fast path: serve from RAM cache
        if self._cache is not None:
            return self._cache[self._valid_indices[idx]]

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
