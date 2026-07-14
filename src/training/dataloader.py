"""
Data loading and tokenization for MoE training.

Uses MosaicML Streaming for efficient shard-based data loading with
automatic caching, multi-worker support, and memory-mapped I/O.
"""

import bisect
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from streaming import Stream, StreamingDataset

from src.taxonomy import (
    SCRIPT_TO_GROUP, SCRIPT_TO_ID, GROUP_TO_ID, NUM_GROUPS, TAXONOMY_VERSION,
)
from src.data.augmentation import (
    AUGMENT_OPS, SCENARIO_CHAINS, X_TRANSFORM_OPS, RandAugmentOCR,
)
from src.encoding.decompose import encode_text, script_vocab_size


# ---------------------------------------------------------------------------
# Shard metadata
# ---------------------------------------------------------------------------

def load_shard_metadata(data_path: Path) -> dict | None:
    """Load shard metadata and reject shards from an older taxonomy.

    Shards store numeric script/group IDs, so any taxonomy renumbering
    silently mislabels every sample generated before it. Metadata lives next
    to the shard dir or one level up (MDS layouts use data_path/train,val).
    Returns the metadata dict, or None when no metadata.pt exists.
    """
    meta_path = data_path / "metadata.pt"
    if not meta_path.exists():
        meta_path = data_path.parent / "metadata.pt"
    if not meta_path.exists():
        return None
    meta = torch.load(meta_path, weights_only=False)
    if "emoji" in meta.get("active_scripts", []):
        raise RuntimeError(
            "These shards contain the removed Emoji script/group. "
            "Regenerate them with the current 14-group taxonomy.")
    if meta.get("taxonomy_version", 1) < TAXONOMY_VERSION:
        raise RuntimeError(
            f"These shards use taxonomy v{meta.get('taxonomy_version', 1)} "
            f"numeric script/group IDs; current is v{TAXONOMY_VERSION}. "
            "IDs shifted when the taxonomy changed, so old shards would "
            "silently mislabel scripts. Regenerate the shards.")
    return meta


# ---------------------------------------------------------------------------
# Tokenizer / vocab building
# ---------------------------------------------------------------------------

def build_script_tokenizers(
    active_scripts: list[str],
    active_groups: list[str],
) -> tuple[list[list[None]], list[list[int]], list[list[str]]]:
    """Build per-script vocab sizes organized by group.

    Returns:
        group_tokenizers[g][s]: always None (kept for API compat)
        group_script_vocab_sizes[g][s]: vocab size
        group_script_names[g][s]: script name
    """
    assert active_scripts, "active_scripts must be non-empty"
    assert active_groups, "active_groups must be non-empty"

    print("  Building vocabs...")
    group_tokenizers: list[list[None]] = []
    group_vocab_sizes: list[list[int]] = []
    group_script_names: list[list[str]] = []

    for g, group_name in enumerate(active_groups):
        scripts = [s for s in active_scripts
                   if SCRIPT_TO_GROUP.get(s) == group_name]
        sizes: list[int] = []
        for script in scripts:
            vs = script_vocab_size(script)
            sizes.append(vs)
            print(f"    Group {g} ({group_name}) / {script}: {vs} tokens")

        group_tokenizers.append([None] * len(scripts))
        group_vocab_sizes.append(sizes)
        group_script_names.append(scripts)

    return group_tokenizers, group_vocab_sizes, group_script_names


# ---------------------------------------------------------------------------
# Pixel-space label remapping under augmentation x-transforms
# ---------------------------------------------------------------------------

def shift_labels_x(labels: np.ndarray, a: float, b: float,
                   fill: int) -> np.ndarray:
    """Resample a per-pixel label row under x_new = a * x_old + b.

    Output pixel x takes the label of source pixel (x - b) / a; pixels
    that map outside the source get `fill` (whitespace/blank).
    """
    W = labels.shape[0]
    src = np.round((np.arange(W) - b) / a).astype(np.int64)
    valid = (src >= 0) & (src < W)
    out = np.full_like(labels, fill)
    out[valid] = labels[src[valid]]
    return out


def shift_segments_x(segments: list[dict], a: float, b: float,
                     width: int) -> list[dict]:
    """Map segment pixel offsets/widths under x_new = a * x_old + b.

    Segments pushed entirely outside the image (partial_crop can do this
    to edge segments) are dropped — their pixels are gone, so keeping the
    text would train the model to hallucinate. Partially visible segments
    keep their full text, matching partial_crop's intent that the model
    learns to read clipped characters.
    """
    out = []
    for seg in segments:
        x0 = seg["offset"] * a + b
        x1 = (seg["offset"] + seg["width"]) * a + b
        x0c = max(0, int(round(x0)))
        x1c = min(width, int(round(x1)))
        if x1c - x0c < 1:
            continue
        seg["offset"] = x0c
        seg["width"] = x1c - x0c
        out.append(seg)
    return out


# ---------------------------------------------------------------------------
# Streaming dataset
# ---------------------------------------------------------------------------

def _stream_has_samples(chunk_dir: Path) -> bool:
    """True if an MDS chunk's index.json advertises at least one sample."""
    idx = chunk_dir / "index.json"
    if not idx.exists():
        return False
    try:
        return sum(s.get("samples", 0)
                   for s in json.loads(idx.read_text()).get("shards", [])) > 0
    except (json.JSONDecodeError, OSError):
        return False


class LipiStreamingDataset(Dataset):
    """MDS-backed map-style dataset with global→local ID remapping.

    Uses MosaicML StreamingDataset internally for efficient shard I/O,
    but exposes a standard map-style Dataset interface so it works with
    batch_sampler and num_workers in DataLoader.
    """

    def __init__(
        self,
        local: str,
        active_scripts: list[str],
        active_groups: list[str],
        augment: bool = False,
        augment_p: float = 0.75,
        two_views: bool = False,
    ):
        # Train-time augmentation: shards store clean renders and each
        # epoch sees a fresh degradation. (Previously augmentation was
        # baked into the shards at generation time — one fixed appearance
        # per sample forever, and ~65% of samples fully clean.)
        # All ops are size-preserving, so stored widths stay valid.
        # Ops that shift content horizontally (partial_crop /
        # pad_with_border) report an x-affine that __getitem__ applies to
        # segment offsets and per-pixel group labels.
        self._aug = RandAugmentOCR(n_ops=2, p=augment_p) if augment else None

        # Two-view mode (consistency regularization): each sample also
        # returns a second, independently augmented view of the same
        # clean render. View 2 uses only x-preserving ops so its frames
        # align with view 1 whenever view 1's transform is identity —
        # the returned `aligned` flag gates the consistency loss.
        self._aug2 = None
        if augment and two_views:
            safe_ops = [op for op in AUGMENT_OPS
                        if op not in X_TRANSFORM_OPS]
            safe_chains = [
                (name, [op for op in ops if op not in X_TRANSFORM_OPS], w)
                for name, ops, w in SCENARIO_CHAINS
            ]
            self._aug2 = RandAugmentOCR(n_ops=2, p=augment_p,
                                        ops=safe_ops, chains=safe_chains)

        local_path = Path(local)
        # Support both flat MDS dirs and dirs with chunk_* sub-directories.
        # Empty chunks (index.json with samples=0) come from generate.py's
        # train/val split when a chunk happens to land entirely on one side;
        # StreamingDataset refuses to open them, so drop them here.
        chunk_dirs = sorted(local_path.glob("chunk_*"))
        if chunk_dirs:
            nonempty = [d for d in chunk_dirs if _stream_has_samples(d)]
            streams = [Stream(local=str(d)) for d in nonempty]
            self._ds = StreamingDataset(streams=streams, shuffle=False)
        else:
            self._ds = StreamingDataset(local=local, shuffle=False)

        # Build ID remap tables
        self._global_to_local_group: dict[int, int] = {}
        for local_id, gname in enumerate(active_groups):
            self._global_to_local_group[GROUP_TO_ID[gname]] = local_id

        self._global_sid_to_local: dict[int, int] = {}
        for g, group_name in enumerate(active_groups):
            members = [s for s in active_scripts
                       if SCRIPT_TO_GROUP.get(s) == group_name]
            for local_s, script in enumerate(members):
                self._global_sid_to_local[SCRIPT_TO_ID[script]] = local_s

        # LUT for the per-pixel group-label remap: one np.take instead of
        # a masked-assign pass per group over every pixel row.
        self._group_lut = np.full(NUM_GROUPS + 1, NUM_GROUPS, dtype=np.int64)
        for gid_global, gid_local in self._global_to_local_group.items():
            self._group_lut[gid_global] = gid_local

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        sample = self._ds[idx]

        raw_np = sample["image"]                            # (3, 32, W) uint8
        img_np = raw_np
        xa, xb = 1.0, 0.0
        if self._aug is not None:
            pil = Image.fromarray(raw_np.transpose(1, 2, 0))
            # (xa, xb) is the composed x-geometry map of any content-shifting
            # ops (partial_crop / pad_with_border): x_new = xa * x_old + xb.
            # Pixel-space labels below are remapped to follow the content.
            pil, (xa, xb) = self._aug.apply_with_transform(pil)
            img_np = np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1)
        img = torch.from_numpy(img_np.copy())              # (3, 32, W) uint8

        # Second view for consistency training (x-preserving ops only)
        img2 = None
        if self._aug2 is not None:
            pil2 = Image.fromarray(raw_np.transpose(1, 2, 0))
            pil2 = self._aug2(pil2)
            img2 = torch.from_numpy(
                np.asarray(pil2, dtype=np.uint8).transpose(2, 0, 1).copy())
        label = sample["label"]                             # str
        global_sid = sample["script_id"]                    # int
        global_gid = sample["group_id"]                     # int

        # Remap to local IDs. Blank-primary samples (digit/punct-only
        # lines get group_id == NUM_GROUPS from the generator) map to the
        # blank group — same convention as the per-segment remap below;
        # their CTC supervision rides on the segments, the per-image id
        # is routing/eval metadata.
        local_gid = self._global_to_local_group.get(global_gid, NUM_GROUPS)
        local_sid = self._global_sid_to_local.get(global_sid, 0)

        tids = torch.from_numpy(sample["target_ids"].copy())  # (L,) int64
        tlen = torch.tensor(sample["target_len"], dtype=torch.long)

        # Per-pixel group labels → remap to local group IDs
        # Blank pixels (NUM_GROUPS) must stay as NUM_GROUPS, not become 0
        gl = sample["group_labels"].copy()
        if (xa, xb) != (1.0, 0.0):
            gl = shift_labels_x(gl, xa, xb, fill=NUM_GROUPS)
        remapped = self._group_lut[np.clip(gl, 0, NUM_GROUPS)]
        group_labels = torch.from_numpy(remapped).to(torch.long)

        # Segments: per-word metadata for mixed-script CTC loss
        segments = json.loads(sample["segments"])
        if (xa, xb) != (1.0, 0.0):
            segments = shift_segments_x(segments, xa, xb, img_np.shape[2])
        for seg in segments:
            seg["group_id"] = self._global_to_local_group.get(seg["group_id"], NUM_GROUPS)
            seg["script_id"] = self._global_sid_to_local.get(seg["script_id"], 0)

        base = (img, tids, tlen,
                torch.tensor(local_gid, dtype=torch.long),
                torch.tensor(local_sid, dtype=torch.long),
                label, group_labels, segments)
        if self._aug2 is None:
            return base
        # Two-view mode: view 2 is x-preserving, so the pair is aligned
        # frame-for-frame iff view 1's x-transform was identity.
        return base + (img2, (xa, xb) == (1.0, 0.0))


# ---------------------------------------------------------------------------
# Static-shape bucket batching
# ---------------------------------------------------------------------------

def compute_bucket_edges(widths: np.ndarray | list[int],
                         max_buckets: int = 16,
                         min_gain: float = 0.005) -> list[int]:
    """Padding-optimal bucket edges derived from the width distribution.

    Exact DP over unique 4px-rounded widths: for each bucket count K the
    edges minimizing total padded pixels, then the smallest K where
    adding one more bucket would save less than `min_gain` of the
    dataset's pixels. Each shape costs a cudnn autotune + torch.compile
    specialization, so a bucket must buy real padding savings to earn
    its keep — the distribution decides, not a flag. The last edge
    always covers max(widths).
    """
    w = np.asarray(widths, dtype=np.int64)
    w4 = (w + 3) // 4 * 4
    uniq, counts = np.unique(w4, return_counts=True)
    U = len(uniq)
    if U == 1:
        return [int(uniq[0])]

    # cost[i, j]: padded-pixel waste of one bucket covering uniq[i..j],
    # every sample padded to uniq[j].
    n_cum = np.concatenate([[0], np.cumsum(counts)])
    px_cum = np.concatenate([[0], np.cumsum(counts * uniq)])
    i_idx = np.arange(U)[:, None]
    j_idx = np.arange(U)[None, :]
    n_ij = n_cum[j_idx + 1] - n_cum[i_idx]
    px_ij = px_cum[j_idx + 1] - px_cum[i_idx]
    cost = np.where(i_idx <= j_idx, uniq[j_idx] * n_ij - px_ij,
                    np.inf).astype(np.float64)

    total_px = float(px_cum[-1])
    k_max = min(max_buckets, U)

    # waste_k[j] = min waste covering uniq[0..j] with k buckets;
    # parent_k[j] = start index of the last bucket in that optimum.
    waste = cost[0].copy()
    tables = [waste.copy()]
    parents = [np.zeros(U, dtype=np.int64)]
    for _ in range(2, k_max + 1):
        m = np.full((U, U), np.inf)
        m[1:, :] = waste[:-1, None] + cost[1:, :]
        parent = m.argmin(axis=0)
        waste = m[parent, np.arange(U)]
        tables.append(waste.copy())
        parents.append(parent)

    K = k_max
    for k in range(1, k_max):
        if (tables[k - 1][-1] - tables[k][-1]) / total_px < min_gain:
            K = k
            break

    edges = []
    j = U - 1
    for k in range(K, 0, -1):
        edges.append(int(uniq[j]))
        j = int(parents[k - 1][j]) - 1
    return edges[::-1]


class BucketBatchSampler(Sampler):
    """Static-shape batches: quantile width buckets × measured batch sizes.

    Each sample belongs to the smallest bucket edge >= its width; each
    batch holds exactly `bucket_capacities[edge]` samples from one
    bucket, so collate (padding to the edge) produces one of
    len(bucket_edges) fixed (B_k, C, H, W_k) shapes. Static shapes are
    what makes the measured VRAM capacities valid forever — and they let
    cudnn.benchmark / torch.compile specialize each shape exactly once.

    Batch composition reshuffles every epoch (the old width-sorted
    sampler froze composition across epochs). Each bucket's leftover
    partial batch rides up into the next wider bucket; the single
    remainder at the widest edge is topped up with random extra samples
    (any width fits there) so even it keeps a static shape.

    With `sample_weights`, each epoch draws len(widths) indices with
    replacement (p ∝ weights) instead of visiting every index once —
    used for script-balanced sampling. Oversampled indices repeat within
    the epoch (each occurrence gets a fresh train-time augmentation);
    undersampled ones rotate across epochs via the per-epoch redraw.
    """

    def __init__(self, widths: list[int] | np.ndarray,
                 bucket_edges: list[int],
                 bucket_capacities: dict[int, int],
                 sample_weights: np.ndarray | None = None):
        self.widths = np.asarray(widths, dtype=np.int64)
        self.edges = np.asarray(sorted(bucket_edges), dtype=np.int64)
        assert self.widths.max() <= self.edges[-1], \
            f"max width {self.widths.max()} exceeds last bucket edge {self.edges[-1]}"
        self.capacities = [int(bucket_capacities[int(e)]) for e in self.edges]
        # Smallest edge >= width (edges are right-inclusive)
        self.bucket_of = np.searchsorted(self.edges, self.widths)

        self.sample_weights = None
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float64)
            assert len(w) == len(self.widths), \
                f"sample_weights length {len(w)} != widths length {len(self.widths)}"
            self.sample_weights = w / w.sum()

        # Pre-build batches so __len__ is accurate. __iter__ reuses this
        # first build (the weighted redraw over millions of samples costs
        # seconds), then redraws for every later epoch.
        self._batches = self._build_batches(self._draw_indices())
        self._fresh = True

    def _draw_indices(self):
        n = len(self.widths)
        if self.sample_weights is None:
            return np.arange(n)
        return np.random.choice(n, size=n, replace=True, p=self.sample_weights)

    def _build_batches(self, indices):
        batches = []
        buckets = self.bucket_of[indices]
        carry = np.empty(0, dtype=np.int64)
        for k in range(len(self.edges)):
            idx = np.concatenate([carry, indices[buckets == k]])
            np.random.shuffle(idx)
            B = self.capacities[k]
            n_full = (idx.size // B) * B
            batches.extend(idx[i:i + B].tolist()
                           for i in range(0, n_full, B))
            carry = idx[n_full:]
        if carry.size:
            B = self.capacities[-1]
            extra = np.random.choice(indices, size=B - carry.size,
                                     replace=indices.size < B - carry.size)
            batches.append(np.concatenate([carry, extra]).tolist())
        random.shuffle(batches)
        return batches

    def __iter__(self):
        # Fresh shuffle (and weighted redraw) every epoch; the first epoch
        # uses the batches already built in __init__.
        if self._fresh:
            self._fresh = False
        else:
            self._batches = self._build_batches(self._draw_indices())
        yield from self._batches

    def __len__(self):
        return len(self._batches)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _pad_images_to(imgs, max_w):
    """Right-pad each (C, H, W) image with zeros to width max_w."""
    padded = []
    for img in imgs:
        w = img.shape[2]
        if w < max_w:
            pad = torch.zeros(img.shape[0], img.shape[1], max_w - w,
                              dtype=img.dtype)
            img = torch.cat([img, pad], dim=2)
        padded.append(img)
    return padded


def collate_moe(batch, pad_to_widths: list[int] | None = None):
    """Stack batch, padding images, targets, and group_labels to max size.

    pad_to_widths: optional ascending list of static bucket widths. When
    given, images and group labels pad to the smallest entry >= the
    batch's max width (instead of the max width itself) so every batch
    from a bucket collates to the same static shape. None keeps dynamic
    padding (max width rounded up to a multiple of 4) — used by val/eval
    loaders where shape variety is harmless under no_grad.

    Two-view batches (10-tuples from two_views datasets) additionally
    return (images2, aligned) appended to the standard 8-tuple.
    """
    two_views = len(batch[0]) == 10
    if two_views:
        (imgs, targets, tgt_lens, gids, sids, labels, group_labels,
         segments, imgs2, aligneds) = zip(*batch)
    else:
        imgs, targets, tgt_lens, gids, sids, labels, group_labels, segments = \
            zip(*batch)

    max_w = max(img.shape[2] for img in imgs)
    if pad_to_widths is not None and max_w <= pad_to_widths[-1]:
        max_w = pad_to_widths[bisect.bisect_left(pad_to_widths, max_w)]
    else:
        max_w = (max_w + 3) // 4 * 4
    padded = _pad_images_to(imgs, max_w)

    # Pad targets to max length in batch
    max_tgt = max(t.shape[0] for t in targets)
    padded_tgt = []
    for t in targets:
        if t.shape[0] < max_tgt:
            padded_tgt.append(torch.cat([t, torch.zeros(max_tgt - t.shape[0], dtype=t.dtype)]))
        else:
            padded_tgt.append(t)

    # Pad group_labels to max_w
    # Whitespace between words = NUM_GROUPS (learnable class, from MDS data)
    # Padding to batch width = -100 (ignored by CrossEntropyLoss)
    PAD_IGNORE = -100
    padded_gl = []
    for gl in group_labels:
        if gl.shape[0] < max_w:
            pad = torch.full((max_w - gl.shape[0],), PAD_IGNORE, dtype=gl.dtype)
            padded_gl.append(torch.cat([gl, pad]))
        else:
            padded_gl.append(gl[:max_w])

    base = (torch.stack(padded), torch.stack(padded_tgt), torch.stack(tgt_lens),
            torch.stack(gids), torch.stack(sids), list(labels),
            torch.stack(padded_gl), list(segments))
    if not two_views:
        return base
    padded2 = _pad_images_to(imgs2, max_w)
    return base + (torch.stack(padded2),
                   torch.tensor(aligneds, dtype=torch.bool))
