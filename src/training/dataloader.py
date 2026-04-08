"""
Data loading and tokenization for MoE training.

Uses MosaicML Streaming for efficient shard-based data loading with
automatic caching, multi-worker support, and memory-mapped I/O.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from streaming import StreamingDataset

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

        img = torch.from_numpy(sample["image"].copy())     # (2, 32, W) uint8
        label = sample["label"]                             # str
        global_sid = sample["script_id"]                    # int
        global_gid = sample["group_id"]                     # int

        # Remap to local IDs
        local_gid = self._global_to_local_group.get(global_gid, 0)
        local_sid = self._global_sid_to_local.get(global_sid, 0)

        tids = torch.from_numpy(sample["target_ids"].copy())  # (L,) int64
        tlen = torch.tensor(sample["target_len"], dtype=torch.long)

        return (img, tids, tlen,
                torch.tensor(local_gid, dtype=torch.long),
                torch.tensor(local_sid, dtype=torch.long),
                label)


# ---------------------------------------------------------------------------
# Dynamic batch sampler
# ---------------------------------------------------------------------------

class WidthBudgetBatchSampler(Sampler):
    """Batch sampler that packs batches by pixel budget, not fixed count.

    Sorts by width, then greedily fills each batch until adding another
    sample would exceed max_pixels. Wide-image batches get fewer samples,
    narrow ones get more. Shuffles batch order each epoch.

    Usage: pass as batch_sampler to DataLoader (NOT sampler).
    """

    def __init__(self, widths: list[int] | np.ndarray, max_pixels: int):
        self.max_pixels = max_pixels
        self.widths = widths if isinstance(widths, list) else widths.tolist()
        self.sorted_indices = sorted(
            range(len(self.widths)), key=lambda i: self.widths[i])
        self._batches = self._build_batches()

    def _build_batches(self) -> list[list[int]]:
        batches = []
        current_batch = []
        current_max_w = 0
        for idx in self.sorted_indices:
            w = self.widths[idx]
            new_max_w = max(current_max_w, w)
            if current_batch and (len(current_batch) + 1) * new_max_w > self.max_pixels:
                batches.append(current_batch)
                current_batch = [idx]
                current_max_w = w
            else:
                current_batch.append(idx)
                current_max_w = new_max_w
        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        random.shuffle(batches)
        yield from batches

    def __len__(self):
        return len(self._batches)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_moe(batch) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, list[str]
]:
    """Stack batch, padding images and targets to max size in batch."""
    imgs, targets, tgt_lens, gids, sids, labels = zip(*batch)

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

    return (torch.stack(padded), torch.stack(padded_tgt), torch.stack(tgt_lens),
            torch.stack(gids), torch.stack(sids), list(labels))
