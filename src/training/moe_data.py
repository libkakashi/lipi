"""
Data loading and tokenization for MoE training.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset

from src.model.lid import SCRIPT_TO_GROUP, SCRIPT_TO_ID, GROUP_TO_ID
from src.data.bigrams import LipiTokenizer
from src.data.decompose import decompose_text, DECOMPOSE_GROUPS
from src.data.vocab import get_all_script_vocabs


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

    target_ids and target_lens are None if shards don't contain pre-encoded
    targets (backward compat with old shards).
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
    has_targets = None  # tri-state: None=unknown, True/False after first shard
    with ThreadPoolExecutor(max_workers=16) as pool:
        for shard in pool.map(lambda p: torch.load(p, weights_only=False), shard_files):
            missing = expected_keys - shard.keys()
            if missing:
                raise KeyError(f"Shard missing expected keys: {missing}")
            all_imgs.append(shard["images"])
            all_labels.extend(shard["labels"])
            all_sids.append(shard["script_ids"])
            all_gids.append(shard["group_ids"])

            # Pre-encoded targets (optional, for backward compat)
            shard_has = "target_ids" in shard and "target_lens" in shard
            if has_targets is None:
                has_targets = shard_has
            elif has_targets != shard_has:
                # Mixed shards: some have targets, some don't — fall back
                has_targets = False
            if shard_has:
                all_tids.append(shard["target_ids"])
                all_tlens.append(shard["target_lens"])

    images = torch.cat(all_imgs)
    script_ids = torch.cat(all_sids)
    group_ids = torch.cat(all_gids)
    del all_imgs, all_sids, all_gids

    if has_targets and all_tids:
        # Pad target_ids to uniform max_len across all shards
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
) -> tuple[list[list[LipiTokenizer]], list[list[int]], list[list[str]]]:
    """Build per-script tokenizers with fixed vocabs (not data-dependent).

    Vocabs are defined by Unicode ranges and decomposition rules.
    This prevents dirty word lists from inflating vocab sizes.

    Returns:
        group_tokenizers[g][s]: tokenizer for script s in group g
        group_script_vocab_sizes[g][s]: vocab size
        group_script_names[g][s]: script name
    """
    assert active_scripts, "active_scripts must be non-empty"
    assert active_groups, "active_groups must be non-empty"

    print("  Building fixed vocabs from Unicode ranges + decomposition rules...")
    group_vocabs, group_vocab_sizes = get_all_script_vocabs(active_scripts, active_groups)

    group_tokenizers: list[list[LipiTokenizer]] = []
    group_script_names: list[list[str]] = []

    for g, group_name in enumerate(active_groups):
        members = scripts_in_group(active_scripts, group_name)
        tokenizers = []
        for s, script in enumerate(members):
            vocab = group_vocabs[g][s]
            tok = LipiTokenizer(vocab=vocab, bigrams=set())
            tokenizers.append(tok)

        group_tokenizers.append(tokenizers)
        group_script_names.append(members)

    return group_tokenizers, group_vocab_sizes, group_script_names


def encode_labels(
    labels: list[str],
    group_ids: torch.Tensor,
    local_script_ids: torch.Tensor,
    active_groups: list[str],
    group_tokenizers: list[list[LipiTokenizer]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-encode all labels using each sample's script tokenizer.

    Returns:
        target_tensor: (N, max_len) padded token IDs
        target_len_tensor: (N,) actual lengths
    """
    n_groups = len(active_groups)
    max_len = 0
    encoded = []
    oov_chars = 0
    oov_samples = 0
    for label, gid, lsid in zip(labels, group_ids.tolist(), local_script_ids.tolist()):
        if gid < 0 or gid >= n_groups:
            raise IndexError(f"group_id {gid} out of range [0, {n_groups})")
        if lsid < 0 or lsid >= len(group_tokenizers[gid]):
            raise IndexError(
                f"local_script_id {lsid} out of range for group {gid} "
                f"(has {len(group_tokenizers[gid])} scripts)"
            )

        group_name = active_groups[gid]
        if group_name in DECOMPOSE_GROUPS:
            label_tokens = decompose_text(label, group_name)
        else:
            label_tokens = label
        tok = group_tokenizers[gid][lsid]
        ids = tok.encode(label_tokens)
        n_dropped = len(label_tokens) - len(ids)
        if n_dropped > 0:
            oov_chars += n_dropped
            oov_samples += 1
        encoded.append(ids)
        max_len = max(max_len, len(ids))
    if oov_chars > 0:
        print(f"  WARNING: {oov_chars} OOV chars dropped across {oov_samples} samples "
              f"({100*oov_samples/len(labels):.1f}% of data has corrupted labels)")

    # Prevent zero-width tensor when every label encodes to an empty sequence
    # (e.g. all-OOV batch). At least 1 column is needed for valid downstream ops.
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
    """Remap global script/group IDs to local contiguous 0..N-1.

    Returns:
        group_ids: (N,) local group IDs
        local_script_ids: (N,) local script IDs within each group
        global_to_local_group: mapping dict
    """
    # Group ID remapping
    global_to_local_group: dict[int, int] = {}
    for local_id, gname in enumerate(active_groups):
        global_to_local_group[GROUP_TO_ID[gname]] = local_id

    group_ids = group_ids_global.clone()
    for gid, lid in global_to_local_group.items():
        group_ids[group_ids_global == gid] = lid

    # Local script IDs within each group
    local_script_ids = torch.zeros_like(group_ids)
    for g, group_name in enumerate(active_groups):
        members = scripts_in_group(active_scripts, group_name)
        for local_s, script in enumerate(members):
            global_sid = SCRIPT_TO_ID[script]
            mask = (script_ids_global == global_sid)
            local_script_ids[mask] = local_s

    # Sanity check: all remapped group IDs must be in valid range
    assert group_ids.min() >= 0 and group_ids.max() < len(active_groups), (
        f"Remapped group_ids out of range [0, {len(active_groups)}): "
        f"min={group_ids.min().item()}, max={group_ids.max().item()}"
    )

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
