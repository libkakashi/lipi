"""
Width-bucketed batch sampler for OCR training.

Groups images by similar width so each batch has minimal padding waste.
Narrow batches (short words) process 3-5x faster than wide batches.

Usage:
    sampler = WidthBucketSampler(dataset, batch_size=600)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_ocr)
"""

import struct
import random
from torch.utils.data import Sampler


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read JPEG width/height from header without decoding.

    Returns (width, height) or None if not JPEG.
    Only reads first ~20KB of the file — very fast.
    """
    if len(data) < 4 or data[0:2] != b'\xff\xd8':
        return None

    pos = 2
    while pos < len(data) - 8:
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]

        # SOF markers (Start of Frame) contain dimensions
        if marker in (0xC0, 0xC1, 0xC2):
            height = struct.unpack('>H', data[pos+5:pos+7])[0]
            width = struct.unpack('>H', data[pos+7:pos+9])[0]
            return width, height

        # Skip to next marker
        if marker == 0xD8 or marker == 0xD9:
            pos += 2
        elif marker == 0x00:
            pos += 1
        else:
            length = struct.unpack('>H', data[pos+2:pos+4])[0]
            pos += 2 + length

    return None


def get_image_widths(dataset, target_height: int = 32, max_width: int = 320) -> list[int]:
    """Get normalized widths for all images in dataset.

    Reads JPEG headers from raw cache if available (fast, no decode).
    Falls back to default width if header read fails.
    """
    widths = []
    n = len(dataset)

    # Handle ConcatDataset — iterate over sub-datasets
    from torch.utils.data import ConcatDataset
    sub_datasets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]

    for ds in sub_datasets:
        if hasattr(ds, '_raw_cache') and ds._raw_cache is not None:
            for img_bytes, _ in ds._raw_cache:
                dims = jpeg_dimensions(img_bytes)
                if dims:
                    w, h = dims
                    new_w = int(w * target_height / max(h, 1))
                    new_w = min(max(new_w, 1), max_width)
                    widths.append(new_w)
                else:
                    widths.append(max_width // 2)
        else:
            widths.extend([max_width // 2] * len(ds))

    return widths


class WidthBucketSampler(Sampler):
    """Batch sampler that groups images by similar width.

    Sorts by width, then creates batches from consecutive sorted indices.
    Adds slight randomness within buckets so training isn't fully ordered.

    Result: each batch has images of similar width → minimal padding → fast.
    """

    def __init__(self, widths: list[int], batch_size: int, drop_last: bool = True,
                 bucket_multiplier: int = 10):
        """
        Args:
            widths: pre-computed width for each sample index.
            batch_size: target batch size.
            drop_last: drop the last incomplete batch.
            bucket_multiplier: how many batches to group for shuffling.
                Higher = more sorted (faster) but less random (worse training).
                10 is a good balance.
        """
        self.widths = widths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket_size = batch_size * bucket_multiplier

    def __iter__(self):
        # Sort indices by width
        indices = sorted(range(len(self.widths)), key=lambda i: self.widths[i])

        # Shuffle within buckets (groups of bucket_size consecutive sorted indices)
        # This keeps similar widths together but adds randomness
        bucketed = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            random.shuffle(bucket)
            bucketed.extend(bucket)

        # Shuffle the bucket order (not within buckets)
        n_buckets = len(bucketed) // self.bucket_size
        bucket_order = list(range(n_buckets))
        random.shuffle(bucket_order)
        shuffled = []
        for b in bucket_order:
            start = b * self.bucket_size
            shuffled.extend(bucketed[start:start + self.bucket_size])
        # Add remaining
        shuffled.extend(bucketed[n_buckets * self.bucket_size:])

        # Create batches
        batches = []
        for start in range(0, len(shuffled), self.batch_size):
            batch = shuffled[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)

        # Shuffle batch order
        random.shuffle(batches)

        yield from batches

    def __len__(self):
        n = len(self.widths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
