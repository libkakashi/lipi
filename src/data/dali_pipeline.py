"""
GPU-accelerated data loader with background prefetch.

Decodes JPEG on GPU via torchvision.io.decode_jpeg.
Uses a background thread + separate CUDA stream to overlap
decode of next batch with model training on current batch.

Usage:
    loader = GPULoader(lmdb_path, batch_size=600, augment=True)
    for images, labels, widths in loader:
        # images already on GPU, model trains immediately
"""

import io
import random
import time
import threading
import queue

import lmdb
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image

from src.data.dataset import preprocess_crop


class GPULoader:
    """GPU-decoded data loader with background prefetch.

    Architecture:
      Main thread:     trains model on batch N
      BG thread:       decodes batch N+1 and N+2 on a separate CUDA stream

    The bg thread reads raw bytes from RAM, decodes JPEG on GPU,
    resizes, normalizes, pads, and puts the ready tensor into a queue.
    Main thread just pops from the queue — zero wait if prefetch keeps up.
    """

    def __init__(
        self,
        lmdb_path: str,
        batch_size: int,
        max_samples: int | None = None,
        target_height: int = 32,
        max_width: int = 320,
        augment: bool = False,
        device: str = "cuda:0",
        prefetch: int = 3,
    ):
        self.batch_size = batch_size
        self.target_height = target_height
        self.max_width = max_width
        self.augment = augment
        self.device = torch.device(device)
        self.prefetch = prefetch

        # Load raw bytes into RAM
        self.image_bytes, self.labels = self._load_raw(lmdb_path, max_samples)
        self.n = len(self.image_bytes)

        # Check GPU JPEG decode support
        self._gpu_decode_available = self._check_gpu_decode()

    def _check_gpu_decode(self):
        try:
            buf = torch.frombuffer(bytearray(self.image_bytes[0]), dtype=torch.uint8)
            torchvision.io.decode_jpeg(buf, device=self.device)
            print(f"  GPU JPEG decode: available")
            return True
        except Exception:
            print(f"  GPU JPEG decode: not available, using CPU")
            return False

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

    def _decode_one(self, raw_bytes: bytes) -> torch.Tensor:
        """Decode one image to (3, H, W) float32 on GPU. Returns None on failure."""
        try:
            buf = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
            if self._gpu_decode_available:
                img = torchvision.io.decode_jpeg(buf, device=self.device)
            else:
                img = torchvision.io.decode_image(buf)
                img = img.to(self.device)
        except Exception:
            # Fallback for PNG or corrupt images
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                arr = np.array(pil_img)
                img = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
            except Exception:
                return None

        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        elif img.shape[0] == 4:
            img = img[:3]

        h, w = img.shape[1], img.shape[2]
        new_w = int(w * self.target_height / h)
        new_w = min(max(new_w, 1), self.max_width)

        img = TF.resize(img, [self.target_height, new_w], antialias=False)
        img = img.float().div_(127.5).sub_(1.0)

        if self.augment and random.random() > 0.5:
            if random.random() < 0.3:
                img = img * random.uniform(0.7, 1.3)
            if random.random() < 0.3:
                mean = img.mean()
                img = (img - mean) * random.uniform(0.7, 1.3) + mean
            if random.random() < 0.2:
                img = img + torch.randn_like(img) * 0.05
            img = img.clamp(-1, 1)

        return img

    def _build_batch(self, batch_indices: list[int], stream: torch.cuda.Stream) -> tuple:
        """Decode a batch of images on a given CUDA stream."""
        with torch.cuda.stream(stream):
            crops = []
            batch_labels = []

            for idx in batch_indices:
                crop = self._decode_one(self.image_bytes[idx])
                if crop is not None:
                    crops.append(crop)
                    batch_labels.append(self.labels[idx])

            if not crops:
                return None, None, None

            max_w = max(c.shape[2] for c in crops)
            B = len(crops)
            padded = torch.zeros(B, 3, self.target_height, max_w, device=self.device)
            widths = torch.zeros(B, dtype=torch.long, device=self.device)

            for i, c in enumerate(crops):
                padded[i, :, :, :c.shape[2]] = c
                widths[i] = c.shape[2]

            # Don't hold references to individual crops
            del crops

            return padded, batch_labels, widths

    def _prefetch_worker(self, batch_list: list[list[int]], out_queue: queue.Queue,
                         stream: torch.cuda.Stream):
        """Background thread: decode batches and push to queue."""
        for batch_indices in batch_list:
            result = self._build_batch(batch_indices, stream)
            out_queue.put(result)
        out_queue.put(None)  # Sentinel

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(self.n))
        random.shuffle(indices)

        # Split into batches
        batch_list = []
        for i in range(0, self.n, self.batch_size):
            batch_list.append(indices[i:i + self.batch_size])

        # Create a separate CUDA stream for decoding
        decode_stream = torch.cuda.Stream(device=self.device)

        # Start background prefetch thread
        q = queue.Queue(maxsize=self.prefetch)
        thread = threading.Thread(
            target=self._prefetch_worker,
            args=(batch_list, q, decode_stream),
            daemon=True,
        )
        thread.start()

        # Yield batches from queue
        while True:
            result = q.get()
            if result is None:
                break
            padded, labels, widths = result
            if padded is None:
                continue
            # Sync the decode stream before yielding to training
            torch.cuda.current_stream(self.device).wait_stream(decode_stream)
            yield padded, labels, widths

        thread.join()
