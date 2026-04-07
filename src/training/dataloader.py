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

    images = torch.cat(all_imgs)
    script_ids = torch.cat(all_sids)
    group_ids = torch.cat(all_gids)
    del all_imgs, all_sids, all_gids

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
    """Stack pre-encoded batch."""
    imgs, targets, tgt_lens, gids, sids, labels = zip(*batch)
    return (torch.stack(imgs), torch.stack(targets), torch.stack(tgt_lens),
            torch.stack(gids), torch.stack(sids), list(labels))
