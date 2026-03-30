"""
NVIDIA DALI pipeline for GPU-accelerated OCR data loading.

Decodes JPEG/PNG on GPU using nvJPEG (hardware decoder),
resizes, normalizes, and augments — all on GPU in a single
fused C++ pipeline. Zero Python/PIL in the training loop.

Requires: pip install nvidia-dali-cuda120
"""

import lmdb
import time
import numpy as np
import torch

from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


class LMDBExternalSource:
    """Feeds raw image bytes + label indices to DALI from RAM.

    DALI calls __next__ to get a batch of raw encoded image bytes.
    Labels are tracked separately (DALI doesn't handle strings well).
    """

    def __init__(self, image_bytes: list[bytes], labels: list[str],
                 batch_size: int, shuffle: bool = True):
        from collections import deque
        self.image_bytes = image_bytes
        self.labels = labels
        self.batch_size = batch_size
        self.n = len(image_bytes)
        self.indices = list(range(self.n))
        self.shuffle = shuffle
        self.pos = 0
        # FIFO queue for labels — DALI prefetches N batches ahead,
        # so _current_batch_labels would be wrong. Queue stays in sync.
        self._label_queue = deque()
        if shuffle:
            import random
            random.shuffle(self.indices)

    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        self.pos = 0
        self._label_queue.clear()
        return self

    def __next__(self):
        if self.pos >= self.n:
            self.__iter__()
            raise StopIteration

        end = min(self.pos + self.batch_size, self.n)
        batch_indices = self.indices[self.pos:end]
        self.pos = end

        # Push labels to FIFO queue — popped in order by DALIOCRLoader.__next__
        self._label_queue.append([self.labels[i] for i in batch_indices])

        # Return raw bytes as numpy arrays (DALI ExternalSource format)
        return [np.frombuffer(self.image_bytes[i], dtype=np.uint8) for i in batch_indices]


@pipeline_def
def ocr_train_pipeline(source, target_height=32, max_width=320, augment=False):
    """DALI pipeline: nvJPEG GPU decode → resize → augment → normalize."""
    encoded = fn.external_source(source=source, dtype=types.UINT8)

    # GPU decode via nvJPEG hardware decoder
    images = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)

    # Resize height to target, preserve aspect ratio
    # DALI resize_y sets height, width scales proportionally
    images = fn.resize(
        images,
        resize_y=target_height,
    )

    if augment:
        # Brightness + contrast jitter
        images = fn.brightness_contrast(
            images,
            brightness=fn.random.uniform(range=(0.8, 1.2)),
            contrast=fn.random.uniform(range=(0.8, 1.2)),
        )
        # Slight rotation
        images = fn.rotate(
            images,
            angle=fn.random.uniform(range=(-3.0, 3.0)),
            fill_value=255,
        )
        # Gaussian blur (random)
        should_blur = fn.random.coin_flip(probability=0.2)
        blurred = fn.gaussian_blur(images, sigma=fn.random.uniform(range=(0.5, 1.5)))
        images = should_blur * blurred + (1 - should_blur) * images

    # Normalize to [-1, 1], convert to CHW float32
    images = fn.crop_mirror_normalize(
        images,
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        output_layout="CHW",
        dtype=types.FLOAT,
    )

    return images


class DALIOCRLoader:
    """DALI-powered OCR data loader. Drop-in replacement for DataLoader.

    Yields (padded_images, labels, widths) like collate_ocr.
    All image processing happens on GPU via nvJPEG + DALI kernels.
    """

    def __init__(
        self,
        lmdb_path: str,
        batch_size: int,
        max_samples: int | None = None,
        target_height: int = 32,
        max_width: int = 320,
        augment: bool = False,
        device_id: int = 0,
    ):
        self.batch_size = batch_size
        self.target_height = target_height
        self.max_width = max_width
        self.augment = augment
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")

        # Load raw bytes into RAM (fast sequential read, ~10s for 3M)
        self.image_bytes, self.labels = self._load_raw(lmdb_path, max_samples)
        self.n = len(self.image_bytes)

    def _load_raw(self, lmdb_path, max_samples):
        env = lmdb.open(lmdb_path, max_readers=4, readonly=True,
                        lock=False, readahead=True, meminit=False)

        with env.begin(write=False) as txn:
            num_samples = int(txn.get(b"num-samples").decode())

        n = min(max_samples or num_samples, num_samples)
        print(f"Loading {n} raw samples into RAM...")
        t0 = time.time()

        image_bytes = []
        labels = []

        with env.begin(write=False) as txn:
            for i in range(n):
                img_key = f"image-{i + 1:09d}".encode()
                lbl_key = f"label-{i + 1:09d}".encode()

                img_data = txn.get(img_key)
                if img_data is None:
                    img_key = f"image-{i:09d}".encode()
                    lbl_key = f"label-{i:09d}".encode()
                    img_data = txn.get(img_key)

                if img_data is None:
                    continue

                label = txn.get(lbl_key)
                label = label.decode("utf-8") if label else ""

                image_bytes.append(bytes(img_data))
                labels.append(label)

                if (i + 1) % 500000 == 0:
                    print(f"  {i+1}/{n} read ({time.time()-t0:.0f}s)")

        env.close()
        elapsed = time.time() - t0
        size_gb = sum(len(b) for b in image_bytes) / 1e9
        print(f"  Loaded {len(image_bytes)} samples ({size_gb:.1f} GB) in {elapsed:.0f}s")

        return image_bytes, labels

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Create fresh source + pipeline each epoch (reshuffles)
        source = LMDBExternalSource(
            self.image_bytes, self.labels, self.batch_size, shuffle=True
        )

        pipe = ocr_train_pipeline(
            source=source,
            target_height=self.target_height,
            max_width=self.max_width,
            augment=self.augment,
            batch_size=self.batch_size,
            num_threads=4,
            device_id=self.device_id,
            prefetch_queue_depth=3,
        )
        pipe.build()

        self._source = source
        self._pipe = pipe
        self._steps = 0
        self._total = len(self)
        return self

    def __next__(self):
        if self._steps >= self._total:
            raise StopIteration
        self._steps += 1

        try:
            (images_tl,) = self._pipe.run()
        except StopIteration:
            raise StopIteration

        # Convert DALI TensorListGPU to padded PyTorch tensor
        # DALI tensors on GPU → PyTorch tensors on GPU via DLPack (zero-copy)
        from nvidia.dali.plugin.pytorch import feed_ndarray
        import torch.utils.dlpack as dlpack

        B = len(images_tl)
        widths = []
        tensors = []

        for i in range(B):
            # Zero-copy: DALI GPU tensor → DLPack → PyTorch GPU tensor
            dali_tensor = images_tl[i]  # TensorGPU
            pt_tensor = torch.empty(
                dali_tensor.shape(), dtype=torch.float32, device=self.device
            )
            feed_ndarray(dali_tensor, pt_tensor)
            tensors.append(pt_tensor)
            widths.append(pt_tensor.shape[2])

        max_w = max(widths)
        padded = torch.zeros(B, 3, self.target_height, max_w, device=self.device)
        width_tensor = torch.tensor(widths, dtype=torch.long, device=self.device)

        for i, t in enumerate(tensors):
            padded[i, :, :, :widths[i]] = t

        del tensors

        labels = self._source._label_queue.popleft()[:B]

        return padded, labels, width_tensor
