"""
Data loading and tokenization for MoE training.

Uses MosaicML Streaming for efficient shard-based data loading with
automatic caching, multi-worker support, and memory-mapped I/O.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from streaming import Stream, StreamingDataset

from src.model.lid import SCRIPT_TO_GROUP, SCRIPT_TO_ID, GROUP_TO_ID
from src.encoding.decompose import encode_text, script_vocab_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scripts_in_group(active_scripts: list[str], group_name: str) -> list[str]:
    """Return the subset of *active_scripts* that belong to *group_name*."""
    return [s for s in active_scripts if SCRIPT_TO_GROUP.get(s) == group_name]


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
# Streaming dataset
# ---------------------------------------------------------------------------

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
    ):
        local_path = Path(local)
        # Support both flat MDS dirs and dirs with chunk_* sub-directories
        chunk_dirs = sorted(local_path.glob("chunk_*"))
        if chunk_dirs:
            streams = [Stream(local=str(d)) for d in chunk_dirs]
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

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        sample = self._ds[idx]

        img = torch.from_numpy(sample["image"].copy())     # (3, 32, W) uint8
        label = sample["label"]                             # str
        global_sid = sample["script_id"]                    # int
        global_gid = sample["group_id"]                     # int

        # Remap to local IDs
        local_gid = self._global_to_local_group.get(global_gid, 0)
        local_sid = self._global_sid_to_local.get(global_sid, 0)

        tids = torch.from_numpy(sample["target_ids"].copy())  # (L,) int64
        tlen = torch.tensor(sample["target_len"], dtype=torch.long)

        # Per-pixel group labels → remap to local group IDs
        # Blank pixels (NUM_GROUPS) must stay as NUM_GROUPS, not become 0
        from src.model.lid import NUM_GROUPS
        gl = sample["group_labels"].copy()
        remapped = np.full_like(gl, NUM_GROUPS)  # default to blank
        for gid_global, gid_local in self._global_to_local_group.items():
            remapped[gl == gid_global] = gid_local
        group_labels = torch.from_numpy(remapped).to(torch.long)

        # Segments: per-word metadata for mixed-script CTC loss
        segments = json.loads(sample["segments"])
        for seg in segments:
            seg["group_id"] = self._global_to_local_group.get(seg["group_id"], 0)
            seg["script_id"] = self._global_sid_to_local.get(seg["script_id"], 0)

        return (img, tids, tlen,
                torch.tensor(local_gid, dtype=torch.long),
                torch.tensor(local_sid, dtype=torch.long),
                label, group_labels, segments)


# ---------------------------------------------------------------------------
# Dynamic batch sampler
# ---------------------------------------------------------------------------

class WidthSortedBatchSampler(Sampler):
    """Sort by width, budget-aware batching, shuffle batch order.

    Images sorted by width so similar widths are batched together.
    Each batch gets up to `max_batch_size` images, but is also capped
    by a pixel budget (max_batch_size * max_width) so wide images
    get smaller batches and narrow images get larger batches.
    """

    def __init__(self, widths: list[int] | np.ndarray, max_batch_size: int,
                 max_width: int = 0, pixel_budget: int = 0):
        self.max_batch_size = max_batch_size
        if isinstance(widths, np.ndarray):
            widths = widths.tolist()
        self.sorted_indices = sorted(range(len(widths)), key=lambda i: widths[i])
        self.widths = widths

        # Pixel budget: either provided directly or derived from max batch size
        if pixel_budget > 0:
            self.pixel_budget = pixel_budget
        else:
            if max_width <= 0:
                max_width = max(widths)
            self.pixel_budget = max_batch_size * max_width

        # Pre-build batches so __len__ is accurate
        self._batches = self._build_batches()

    def _build_batches(self):
        batches = []
        i = 0
        while i < len(self.sorted_indices):
            # Width of the widest image in this batch (last one, since sorted)
            # Peek ahead to find how many fit under the budget
            batch = [self.sorted_indices[i]]
            batch_width = self.widths[self.sorted_indices[i]]
            i += 1
            while i < len(self.sorted_indices) and len(batch) < self.max_batch_size:
                w = self.widths[self.sorted_indices[i]]
                # All images padded to max width in batch, so cost = (len+1) * w
                if (len(batch) + 1) * w > self.pixel_budget:
                    break
                batch.append(self.sorted_indices[i])
                i += 1
            batches.append(batch)
        return batches

    def __iter__(self):
        batches = list(self._batches)
        random.shuffle(batches)
        yield from batches

    def __len__(self):
        return len(self._batches)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_moe(batch):
    """Stack batch, padding images, targets, and group_labels to max size."""
    imgs, targets, tgt_lens, gids, sids, labels, group_labels, segments = zip(*batch)

    # Pad images to max width in batch, rounded up to multiple of 4
    max_w = max(img.shape[2] for img in imgs)
    max_w = (max_w + 3) // 4 * 4
    padded = []
    for img in imgs:
        w = img.shape[2]
        if w < max_w:
            pad = torch.zeros(img.shape[0], img.shape[1], max_w - w,
                              dtype=img.dtype)
            img = torch.cat([img, pad], dim=2)
        padded.append(img)

    # Pad targets to max length in batch
    max_tgt = max(t.shape[0] for t in targets)
    padded_tgt = []
    for t in targets:
        if t.shape[0] < max_tgt:
            padded_tgt.append(torch.cat([t, torch.zeros(max_tgt - t.shape[0], dtype=t.dtype)]))
        else:
            padded_tgt.append(t)

    # Pad group_labels to max_w — padded region labeled as blank (NUM_GROUPS)
    from src.model.lid import NUM_GROUPS
    padded_gl = []
    for gl in group_labels:
        if gl.shape[0] < max_w:
            pad = torch.full((max_w - gl.shape[0],), NUM_GROUPS, dtype=gl.dtype)
            padded_gl.append(torch.cat([gl, pad]))
        else:
            padded_gl.append(gl[:max_w])

    return (torch.stack(padded), torch.stack(padded_tgt), torch.stack(tgt_lens),
            torch.stack(gids), torch.stack(sids), list(labels),
            torch.stack(padded_gl), list(segments))
