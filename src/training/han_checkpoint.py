"""Warm-start migration from the legacy single-Han checkpoint layout."""

from __future__ import annotations

import torch

from src.encoding.config import get_han_codec, get_legacy_han_codec
from src.encoding.han_split import HAN_DENSE, HAN_SPARSE
from src.taxonomy import SCRIPT_TO_ID


def has_legacy_han_layout(model_config: dict | None) -> bool:
    """Whether a checkpoint predates the sparse/dense Han split."""
    groups = (model_config or {}).get("group_script_names", [])
    names = {name for scripts in groups for name in scripts}
    return "han" in names and HAN_SPARSE not in names


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


def migrate_legacy_han_state(
    state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    old_model_config: dict | None,
    new_model_config: dict,
) -> int:
    """Seed both new heads and the dense expert from a legacy Han model.

    Returns the number of state entries created or replaced. The input state
    is mutated so the caller can continue with ordinary shape validation.
    """
    if not has_legacy_han_layout(old_model_config):
        return 0

    old_groups = (old_model_config or {}).get("group_script_names", [])
    new_groups = new_model_config.get("group_script_names", [])
    try:
        old_group = next(i for i, scripts in enumerate(old_groups)
                         if "han" in scripts)
        new_group = next(i for i, scripts in enumerate(new_groups)
                         if HAN_SPARSE in scripts)
    except StopIteration:
        return 0

    old_codec = get_legacy_han_codec()
    source_prefix = f"ctc_modules.{old_group}.heads.0.proj"
    source_projection = {
        suffix: state.get(f"{source_prefix}.{suffix}")
        for suffix in ("weight", "bias")
    }
    changed = 0
    for script in (HAN_SPARSE, HAN_DENSE):
        local_id = new_groups[new_group].index(script)
        target_prefix = f"ctc_modules.{new_group}.heads.{local_id}.proj"
        rows = _ctc_row_map(get_han_codec(script), old_codec)
        for suffix in ("weight", "bias"):
            target_key = f"{target_prefix}.{suffix}"
            source = source_projection[suffix]
            if source is not None and target_key in current_state:
                state[target_key] = _remap_projection(
                    source, current_state[target_key], rows)
                changed += 1

    # The old expert at global ID 5 saw every Han character. Keep it for
    # han_sparse and clone it into the newly appended dense expert at ID 27.
    sparse_id = SCRIPT_TO_ID[HAN_SPARSE]
    dense_id = SCRIPT_TO_ID[HAN_DENSE]
    sparse_marker = f".routed_mlps.{sparse_id}."
    dense_marker = f".routed_mlps.{dense_id}."
    for target_key, target_value in current_state.items():
        if dense_marker not in target_key:
            continue
        source_key = target_key.replace(dense_marker, sparse_marker)
        source = state.get(source_key)
        if source is not None and source.shape == target_value.shape:
            state[target_key] = source.clone()
            changed += 1

    return changed
