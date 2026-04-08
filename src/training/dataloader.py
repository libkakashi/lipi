"""
Data loading and tokenization for MoE training.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset

from src.model.lid import SCRIPT_TO_GROUP, SCRIPT_TO_ID, GROUP_TO_ID
from src.encoding.decompose import encode_text, script_vocab_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scripts_in_group(active_scripts: list[str], group_name: str) -> list[str]:
    """Return the subset of *active_scripts* that belong to *group_name*."""
    return [s for s in active_scripts if SCRIPT_TO_GROUP.get(s) == group_name]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_shards(shard_dir: Path) -> tuple[
    torch.Tensor, list[str], torch.Tensor, torch.Tensor, dict,
    torch.Tensor | None, torch.Tensor | None
]:
    """Load all shards (word + char) in parallel.

    Returns:
        images, labels, script_ids, group_ids, meta, target_ids, target_lens
    """
    if not shard_dir.exists():
        raise FileNotFoundError(f"Shard directory does not exist: {shard_dir}")

    meta_path = shard_dir / "metadata.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.pt not found in {shard_dir}")

    meta = torch.load(meta_path, weights_only=False)
    shard_files = (sorted(shard_dir.glob("shard_*.pt"))
                   + sorted(shard_dir.glob("char_shard_*.pt")))

    if not shard_files:
        raise RuntimeError(f"No shard files found in {shard_dir}")

    print(f"  {len(shard_files)} shards")

    expected_keys = {"images", "labels", "script_ids", "group_ids"}

    all_imgs, all_labels, all_sids, all_gids = [], [], [], []
    all_tids, all_tlens = [], []
    has_targets = None
    with ThreadPoolExecutor(max_workers=16) as pool:
        for shard in pool.map(lambda p: torch.load(p, weights_only=False), shard_files):
            missing = expected_keys - shard.keys()
            if missing:
                raise KeyError(f"Shard missing expected keys: {missing}")
            all_imgs.append(shard["images"])
            all_labels.extend(shard["labels"])
            all_sids.append(shard["script_ids"])
            all_gids.append(shard["group_ids"])

            shard_has = "target_ids" in shard and "target_lens" in shard
            if has_targets is None:
                has_targets = shard_has
            elif has_targets != shard_has:
                has_targets = False
            if shard_has:
                all_tids.append(shard["target_ids"])
                all_tlens.append(shard["target_lens"])

    # Pad images to uniform width across shards (real-world data may vary)
    max_w = max(t.shape[3] for t in all_imgs)
    padded_imgs = []
    for t in all_imgs:
        if t.shape[3] < max_w:
            pad = torch.zeros(t.shape[0], t.shape[1], t.shape[2],
                              max_w - t.shape[3], dtype=t.dtype)
            padded_imgs.append(torch.cat([t, pad], dim=3))
        else:
            padded_imgs.append(t)
    images = torch.cat(padded_imgs)
    script_ids = torch.cat(all_sids)
    group_ids = torch.cat(all_gids)
    del all_imgs, padded_imgs, all_sids, all_gids

    if has_targets and all_tids:
        max_len = max(t.shape[1] for t in all_tids)
        padded = []
        for t in all_tids:
            if t.shape[1] < max_len:
                pad = torch.zeros(t.shape[0], max_len - t.shape[1], dtype=torch.long)
                padded.append(torch.cat([t, pad], dim=1))
            else:
                padded.append(t)
        target_ids = torch.cat(padded)
        target_lens = torch.cat(all_tlens)
        del padded, all_tids, all_tlens
        print(f"  {len(all_labels)} images loaded (pre-encoded targets found)")
    else:
        target_ids = None
        target_lens = None
        del all_tids, all_tlens
        print(f"  {len(all_labels)} images loaded (no pre-encoded targets)")

    return images, all_labels, script_ids, group_ids, meta, target_ids, target_lens


def build_script_tokenizers(
    active_scripts: list[str],
    active_groups: list[str],
) -> tuple[list[list[None]], list[list[int]], list[list[str]]]:
    """Build per-script vocab sizes organized by group.

    All encoding now goes through the unified encode_text()/decode_ids()
    interface — no LipiTokenizer needed.

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
        scripts_in_group = [s for s in active_scripts
                            if SCRIPT_TO_GROUP.get(s) == group_name]
        sizes: list[int] = []
        for script in scripts_in_group:
            vs = script_vocab_size(script)
            sizes.append(vs)
            print(f"    Group {g} ({group_name}) / {script}: {vs} tokens")

        group_tokenizers.append([None] * len(scripts_in_group))
        group_vocab_sizes.append(sizes)
        group_script_names.append(scripts_in_group)

    return group_tokenizers, group_vocab_sizes, group_script_names


def _encode_one(args: tuple) -> list[int]:
    """Encode a single label using the unified encode_text interface."""
    label, script_name = args
    return encode_text(label, script_name)


def encode_labels(
    labels: list[str],
    group_ids: torch.Tensor,
    local_script_ids: torch.Tensor,
    active_groups: list[str],
    group_tokenizers: list[list[None]],
    group_script_names: list[list[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-encode all labels via unified encode_text() interface.

    Returns:
        target_tensor: (N, max_len) padded token IDs
        target_len_tensor: (N,) actual lengths
    """
    n_groups = len(active_groups)
    gid_list = group_ids.tolist()
    lsid_list = local_script_ids.tolist()

    tasks = []
    for label, gid, lsid in zip(labels, gid_list, lsid_list):
        if gid < 0 or gid >= n_groups:
            raise IndexError(f"group_id {gid} out of range [0, {n_groups})")
        if lsid < 0 or lsid >= len(group_script_names[gid]):
            raise IndexError(
                f"local_script_id {lsid} out of range for group {gid} "
                f"(has {len(group_script_names[gid])} scripts)")

        script_name = group_script_names[gid][lsid]
        tasks.append((label, script_name))

    max_len = 0
    encoded = []

    if len(tasks) > 10000:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for ids in pool.map(_encode_one, tasks, chunksize=1000):
                encoded.append(ids)
                max_len = max(max_len, len(ids))
    else:
        for task in tasks:
            ids = _encode_one(task)
            encoded.append(ids)
            max_len = max(max_len, len(ids))

    if max_len == 0:
        max_len = 1
    target_tensor = torch.zeros(len(encoded), max_len, dtype=torch.long)
    target_len_tensor = torch.zeros(len(encoded), dtype=torch.long)
    for i, ids in enumerate(encoded):
        target_len_tensor[i] = len(ids)
        if ids:
            target_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    return target_tensor, target_len_tensor


def remap_ids(
    active_scripts: list[str],
    active_groups: list[str],
    script_ids_global: torch.Tensor,
    group_ids_global: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    """Remap global script/group IDs to local contiguous 0..N-1."""
    global_to_local_group: dict[int, int] = {}
    for local_id, gname in enumerate(active_groups):
        global_to_local_group[GROUP_TO_ID[gname]] = local_id

    group_ids = group_ids_global.clone()
    for gid, lid in global_to_local_group.items():
        group_ids[group_ids_global == gid] = lid

    local_script_ids = torch.zeros_like(group_ids)
    for g, group_name in enumerate(active_groups):
        members = scripts_in_group(active_scripts, group_name)
        for local_s, script in enumerate(members):
            global_sid = SCRIPT_TO_ID[script]
            mask = (script_ids_global == global_sid)
            local_script_ids[mask] = local_s

    assert group_ids.min() >= 0 and group_ids.max() < len(active_groups), (
        f"Remapped group_ids out of range [0, {len(active_groups)})")

    return group_ids, local_script_ids, global_to_local_group


class MoEDataset(Dataset):
    """Pre-encoded dataset with group IDs, local script IDs, and string labels."""

    def __init__(self, images, targets, target_lens, group_ids, local_script_ids, labels):
        self.images = images
        self.targets = targets
        self.target_lens = target_lens
        self.group_ids = group_ids
        self.local_script_ids = local_script_ids
        self.labels = labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx) -> tuple:
        return (self.images[idx], self.targets[idx], self.target_lens[idx],
                self.group_ids[idx], self.local_script_ids[idx], self.labels[idx])


def collate_moe(batch) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, list[str]
]:
    """Stack pre-encoded batch, padding images to max width in batch."""
    imgs, targets, tgt_lens, gids, sids, labels = zip(*batch)
    # Pad/crop images to max width in this batch, capped at 384px
    max_w = min(max(img.shape[2] for img in imgs), 384)
    padded = []
    for img in imgs:
        w = img.shape[2]
        if w > max_w:
            img = img[:, :, :max_w]  # truncate wide images
        elif w < max_w:
            pad = torch.zeros(img.shape[0], img.shape[1], max_w - w,
                              dtype=img.dtype)
            img = torch.cat([img, pad], dim=2)
        padded.append(img)
    # Pad targets to max length in this batch
    max_tgt = max(t.shape[0] for t in targets)
    padded_tgt = []
    for t in targets:
        if t.shape[0] < max_tgt:
            padded_tgt.append(torch.cat([t, torch.zeros(max_tgt - t.shape[0], dtype=t.dtype)]))
        else:
            padded_tgt.append(t)
    return (torch.stack(padded), torch.stack(padded_tgt), torch.stack(tgt_lens),
            torch.stack(gids), torch.stack(sids), list(labels))


# ---------------------------------------------------------------------------
# Streaming shard dataset (constant memory)
# ---------------------------------------------------------------------------

class ShardStreamDataset(Dataset):
    """Loads one shard at a time instead of all data into memory.

    Keeps an index of (shard_path, offset) per sample. Caches the
    most recently loaded shard to avoid re-reading for sequential access.
    Remaps global script/group IDs to local IDs on the fly.
    """

    def __init__(
        self,
        shard_files: list[Path],
        active_scripts: list[str],
        active_groups: list[str],
    ):
        self._shard_files = shard_files
        self._index: list[tuple[int, int]] = []  # (shard_idx, sample_idx)

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

        # Build index from cached shard sizes or scan
        index_path = shard_files[0].parent / ".shard_index.pt"
        if index_path.exists():
            shard_sizes = torch.load(index_path, weights_only=True)
            if len(shard_sizes) == len(shard_files):
                for si, n in enumerate(shard_sizes.tolist()):
                    for i in range(n):
                        self._index.append((si, i))
            else:
                index_path = None  # mismatch, rescan

        if not self._index:
            # Scan shard sizes in parallel
            def _get_size(path):
                s = torch.load(path, weights_only=False)
                n = s["script_ids"].shape[0]
                del s
                return n

            with ThreadPoolExecutor(max_workers=16) as pool:
                sizes = list(pool.map(_get_size, shard_files))

            for si, n in enumerate(sizes):
                for i in range(n):
                    self._index.append((si, i))
            # Cache for next time
            torch.save(torch.tensor(sizes), index_path)

        # LRU cache — keep recent shards in memory
        self._cache: dict[int, dict] = {}
        self._cache_order: list[int] = []
        self._cache_max = 32  # ~3GB

    def __len__(self):
        return len(self._index)

    def _load_shard(self, shard_idx: int):
        if shard_idx not in self._cache:
            if len(self._cache) >= self._cache_max:
                evict = self._cache_order.pop(0)
                del self._cache[evict]
            self._cache[shard_idx] = torch.load(
                self._shard_files[shard_idx], weights_only=False)
            self._cache_order.append(shard_idx)
        else:
            # Move to end (most recently used)
            self._cache_order.remove(shard_idx)
            self._cache_order.append(shard_idx)

    def __getitem__(self, idx):
        shard_idx, sample_idx = self._index[idx]
        self._load_shard(shard_idx)
        s = self._cache[shard_idx]
        img = s["images"][sample_idx]
        label = s["labels"][sample_idx]
        global_sid = s["script_ids"][sample_idx].item()
        global_gid = s["group_ids"][sample_idx].item()

        # Remap to local IDs
        local_gid = self._global_to_local_group.get(global_gid, 0)
        local_sid = self._global_sid_to_local.get(global_sid, 0)

        tid = s["target_ids"][sample_idx] if "target_ids" in s else torch.zeros(1, dtype=torch.long)
        tlen = s["target_lens"][sample_idx] if "target_lens" in s else torch.tensor(0, dtype=torch.long)
        return img, tid, tlen, torch.tensor(local_gid, dtype=torch.long), \
               torch.tensor(local_sid, dtype=torch.long), label


def load_shard_metadata(shard_dir: Path) -> tuple[list[Path], dict]:
    """Scan shard directory, return shard file list and metadata."""
    if not shard_dir.exists():
        raise FileNotFoundError(f"Shard directory does not exist: {shard_dir}")

    meta_path = shard_dir / "metadata.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.pt not found in {shard_dir}")

    meta = torch.load(meta_path, weights_only=False)
    shard_files = (sorted(shard_dir.glob("shard_*.pt"))
                   + sorted(shard_dir.glob("char_shard_*.pt"))
                   + sorted(shard_dir.glob("real_*.pt"))
                   + sorted(shard_dir.glob("mlt50m_*.pt")))

    if not shard_files:
        raise RuntimeError(f"No shard files found in {shard_dir}")

    return shard_files, meta
