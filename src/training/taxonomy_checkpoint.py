"""Warm-start migration across script/group taxonomy changes."""

from __future__ import annotations

import re

import torch

from src.encoding.config import get_han_codec, get_legacy_han_codec
from src.encoding.han_split import HAN_DENSE, HAN_SPARSE
from src.taxonomy import SCRIPT_TO_GROUP, SCRIPT_TO_ID, canonical_script_name


_GROUP_EXPERT_RE = re.compile(
    r"^(group_layers\.\d+\.routed_mlps\.)(\d+)(\..+)$")
_SCRIPT_EXPERT_RE = re.compile(
    r"^(script_layers\.\d+\.routed_mlps\.)(\d+)(\..+)$")
_CTC_RE = re.compile(r"^ctc_modules\.(\d+)\.heads\.(\d+)(\..+)$")
_LID2_RE = re.compile(r"^lid2_heads\.(\d+)(\..+)$")


def _group_name(scripts: list[str]) -> str:
    first = canonical_script_name(scripts[0])
    if first == "emoji":
        return "emoji"
    return SCRIPT_TO_GROUP[first]


def _layout(group_script_names: list[list[str]]) -> tuple[
        list[str], dict[str, tuple[int, int]], dict[str, int]]:
    groups = [_group_name(scripts) for scripts in group_script_names]
    locations = {}
    for group_id, scripts in enumerate(group_script_names):
        for local_id, script in enumerate(scripts):
            locations[script] = (group_id, local_id)

    # Expert flat IDs are group/local flatten order in every era: pre-split
    # checkpoints matched their SCRIPT_TO_ID by construction, and since
    # taxonomy v3 the invariant is explicit (SCRIPT_TO_ID == flatten order).
    # The only exception — han-split-with-emoji checkpoints, which appended
    # han_dense after emoji — never existed: no training ran in that window.
    flat_ids = {}
    flat = 0
    for scripts in group_script_names:
        for script in scripts:
            flat_ids[script] = flat
            flat += 1
    return groups, locations, flat_ids


def needs_taxonomy_migration(old_config: dict | None,
                             new_config: dict) -> bool:
    old_names = (old_config or {}).get("group_script_names")
    new_names = new_config.get("group_script_names")
    return bool(old_names and new_names and old_names != new_names)


def _source_script(new_script: str, old_locations: dict) -> str | None:
    if new_script in old_locations:
        return new_script
    if new_script in {HAN_SPARSE, HAN_DENSE} and "han" in old_locations:
        return "han"
    return None


def _ctc_row_map(new_codec, old_codec) -> dict[int, int]:
    old_token_ids = {token: i + 1 for i, token in enumerate(old_codec.tokens)}
    rows = {0: 0}
    for new_id, token in enumerate(new_codec.tokens, start=1):
        old_id = old_token_ids.get(token)
        if old_id is not None:
            rows[new_id] = old_id
    for slot in range(len(new_codec._alt_id_set)):
        rows[new_codec._alt_base_id + slot] = old_codec._alt_base_id + slot
    return rows


def _remap_projection(source: torch.Tensor, target: torch.Tensor,
                      row_map: dict[int, int]) -> torch.Tensor:
    result = target.clone()
    for new_row, old_row in row_map.items():
        if new_row < result.shape[0] and old_row < source.shape[0]:
            result[new_row].copy_(source[old_row])
    return result


def migrate_taxonomy_state(
    state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    old_model_config: dict | None,
    new_model_config: dict,
) -> int:
    """Remap group/script modules by name and seed split Han projections."""
    if not needs_taxonomy_migration(old_model_config, new_model_config):
        return 0

    old_names = (old_model_config or {})["group_script_names"]
    new_names = new_model_config["group_script_names"]
    old_groups, old_locations, old_flat_ids = _layout(old_names)
    new_groups, new_locations, _ = _layout(new_names)
    old_group_ids = {name: i for i, name in enumerate(old_groups)}
    source_state = dict(state)
    changed = 0

    # Remove every layout-dependent key first so obsolete Emoji/numeric keys
    # cannot accidentally load into a different current module.
    for key in list(state):
        if (_GROUP_EXPERT_RE.match(key) or _SCRIPT_EXPERT_RE.match(key)
                or _CTC_RE.match(key) or _LID2_RE.match(key)
                or key in {"group_head.2.weight", "group_head.2.bias"}):
            del state[key]

    for target_key, target_value in current_state.items():
        match = _GROUP_EXPERT_RE.match(target_key)
        if match:
            new_group_id = int(match.group(2))
            old_group_id = old_group_ids.get(new_groups[new_group_id])
            if old_group_id is not None:
                source_key = f"{match.group(1)}{old_group_id}{match.group(3)}"
                source = source_state.get(source_key)
                if source is not None and source.shape == target_value.shape:
                    state[target_key] = source.clone()
                    changed += 1
            continue

        match = _SCRIPT_EXPERT_RE.match(target_key)
        if match:
            new_flat_id = int(match.group(2))
            new_script = next((name for name, sid in SCRIPT_TO_ID.items()
                               if sid == new_flat_id), None)
            source_script = (_source_script(new_script, old_locations)
                             if new_script else None)
            old_flat_id = old_flat_ids.get(source_script) if source_script else None
            if old_flat_id is not None:
                source_key = f"{match.group(1)}{old_flat_id}{match.group(3)}"
                source = source_state.get(source_key)
                if source is not None and source.shape == target_value.shape:
                    state[target_key] = source.clone()
                    changed += 1
            continue

        match = _CTC_RE.match(target_key)
        if match:
            new_group_id, new_local_id = int(match.group(1)), int(match.group(2))
            new_script = new_names[new_group_id][new_local_id]
            source_script = _source_script(new_script, old_locations)
            if source_script is None:
                continue
            old_group_id, old_local_id = old_locations[source_script]
            source_key = (f"ctc_modules.{old_group_id}.heads.{old_local_id}"
                          f"{match.group(3)}")
            source = source_state.get(source_key)
            if source is None:
                continue
            if source_script == "han" and new_script in {HAN_SPARSE, HAN_DENSE}:
                rows = _ctc_row_map(get_han_codec(new_script),
                                    get_legacy_han_codec())
                state[target_key] = _remap_projection(source, target_value, rows)
            elif source.shape == target_value.shape:
                state[target_key] = source.clone()
            else:
                continue
            changed += 1
            continue

        match = _LID2_RE.match(target_key)
        if match:
            new_group_id = int(match.group(1))
            old_group_id = old_group_ids.get(new_groups[new_group_id])
            if old_group_id is not None:
                source_key = f"lid2_heads.{old_group_id}{match.group(2)}"
                source = source_state.get(source_key)
                if source is not None and source.shape == target_value.shape:
                    state[target_key] = source.clone()
                    changed += 1

    # LID-1 classifier rows follow group names; the final row is blank.
    for suffix in ("weight", "bias"):
        key = f"group_head.2.{suffix}"
        source = source_state.get(key)
        target = current_state.get(key)
        if source is None or target is None:
            continue
        result = target.clone()
        for new_group_id, group_name in enumerate(new_groups):
            old_group_id = old_group_ids.get(group_name)
            if old_group_id is not None:
                result[new_group_id].copy_(source[old_group_id])
        result[len(new_groups)].copy_(source[len(old_groups)])
        state[key] = result
        changed += 1

    return changed
