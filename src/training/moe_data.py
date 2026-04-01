"""
Data loading and tokenization for MoE training.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset

from src.model.lid import SCRIPTS, SCRIPT_TO_GROUP, SCRIPT_TO_ID, GROUP_TO_ID
from src.data.bigrams import LipiTokenizer, BASE_CHARS, BLANK_TOKEN
from src.data.decompose import decompose_text, get_vocab_tokens, DECOMPOSE_GROUPS


def load_shards(shard_dir: Path):
    """Load all shards (word + char) in parallel."""
    meta = torch.load(shard_dir / "metadata.pt", weights_only=False)
    shard_files = (sorted(shard_dir.glob("shard_*.pt"))
                   + sorted(shard_dir.glob("char_shard_*.pt")))
    print(f"  {len(shard_files)} shards")

    all_imgs, all_labels, all_sids, all_gids = [], [], [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for shard in pool.map(lambda p: torch.load(p, weights_only=False), shard_files):
            all_imgs.append(shard["images"])
            all_labels.extend(shard["labels"])
            all_sids.append(shard["script_ids"])
            all_gids.append(shard["group_ids"])

    images = torch.cat(all_imgs)
    script_ids = torch.cat(all_sids)
    group_ids = torch.cat(all_gids)
    del all_imgs, all_sids, all_gids
    print(f"  {len(all_labels)} images loaded")
    return images, all_labels, script_ids, group_ids, meta


def build_tokenizer(words: list[str]) -> LipiTokenizer:
    """Build vocab from training words."""
    chars = set(BASE_CHARS)
    for word in words:
        for ch in word:
            chars.add(ch)
    vocab = [BLANK_TOKEN] + sorted(chars)
    return LipiTokenizer(vocab=vocab, bigrams=set())


def build_script_tokenizers(
    labels: list[str], script_ids: torch.Tensor,
    active_scripts: list[str], active_groups: list[str],
    global_to_local_group: dict[int, int],
) -> tuple[list[list[LipiTokenizer]], list[list[int]], list[list[str]]]:
    """Build per-script tokenizers organized by group.

    Returns:
        group_tokenizers[g][s]: tokenizer for script s in group g
        group_script_vocab_sizes[g][s]: vocab size
        group_script_names[g][s]: script name
    """
    # Collect words per script
    script_words: dict[str, list[str]] = {s: [] for s in active_scripts}
    active_set = set(active_scripts)
    for label, sid in zip(labels, script_ids.tolist()):
        script = SCRIPTS[sid] if sid < len(SCRIPTS) else None
        if script and script in active_set:
            script_words[script].append(label)

    group_tokenizers = []
    group_script_vocab_sizes = []
    group_script_names = []

    for g, group_name in enumerate(active_groups):
        scripts_in_group = [s for s in active_scripts
                            if SCRIPT_TO_GROUP.get(s) == group_name]
        tokenizers = []
        vocab_sizes = []

        for script in scripts_in_group:
            words = script_words.get(script, [])
            if group_name in DECOMPOSE_GROUPS:
                vocab_tokens = get_vocab_tokens(group_name)
                all_tokens = set(BASE_CHARS) | set(vocab_tokens)
                for word in words:
                    for ch in decompose_text(word, group_name):
                        all_tokens.add(ch)
                vocab = [BLANK_TOKEN] + sorted(all_tokens)
                tok = LipiTokenizer(vocab=vocab, bigrams=set())
                tag = "decomposed"
            else:
                tok = build_tokenizer(words)
                tag = "chars"
            tokenizers.append(tok)
            vocab_sizes.append(tok.vocab_size)
            print(f"    Group {g} ({group_name}) / {script}: "
                  f"{tok.vocab_size} {tag}, {len(words)} words")

        group_tokenizers.append(tokenizers)
        group_script_vocab_sizes.append(vocab_sizes)
        group_script_names.append(scripts_in_group)

    return group_tokenizers, group_script_vocab_sizes, group_script_names


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
    max_len = 0
    encoded = []
    for label, gid, lsid in zip(labels, group_ids.tolist(), local_script_ids.tolist()):
        group_name = active_groups[gid] if gid < len(active_groups) else ""
        if group_name in DECOMPOSE_GROUPS:
            label_tokens = decompose_text(label, group_name)
        else:
            label_tokens = label
        tok = group_tokenizers[gid][lsid]
        ids = tok.encode(label_tokens)
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
    """Remap global script/group IDs to local contiguous 0..N-1.

    Returns:
        group_ids: (N,) local group IDs
        local_script_ids: (N,) local script IDs within each group
        global_to_local_group: mapping dict
    """
    # Group ID remapping
    global_to_local_group = {}
    for local_id, gname in enumerate(active_groups):
        global_to_local_group[GROUP_TO_ID[gname]] = local_id

    group_ids = group_ids_global.clone()
    for gid, lid in global_to_local_group.items():
        group_ids[group_ids_global == gid] = lid

    # Local script IDs within each group
    local_script_ids = torch.zeros_like(group_ids)
    for g, group_name in enumerate(active_groups):
        scripts_in_group = [s for s in active_scripts
                            if SCRIPT_TO_GROUP.get(s) == group_name]
        for local_s, script in enumerate(scripts_in_group):
            global_sid = SCRIPT_TO_ID[script]
            mask = (script_ids_global == global_sid)
            local_script_ids[mask] = local_s

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

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return (self.images[idx], self.targets[idx], self.target_lens[idx],
                self.group_ids[idx], self.local_script_ids[idx], self.labels[idx])


def collate_moe(batch):
    """Stack pre-encoded batch."""
    imgs, targets, tgt_lens, gids, sids, labels = zip(*batch)
    return (torch.stack(imgs), torch.stack(targets), torch.stack(tgt_lens),
            torch.stack(gids), torch.stack(sids), list(labels))
