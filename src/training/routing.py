"""
Routing mask computation for MoE training.

Builds the masks that control which samples contribute to which losses:
  - lid1_ok: LID-1 predicted correctly → used for LID-2 loss
  - ctc_ok: LID-1 AND LID-2 correct → used for CTC loss
"""

import numpy as np
import torch
from torch import Tensor

from src.model.lid import (
    GROUP_ID_TO_SUPER_GROUP_ID, NUM_SUPER_GROUPS, NUM_GROUPS,
)


_GROUP_TO_SUPER_GROUP_TENSOR: Tensor | None = None


def _group_to_super_group_tensor(device: torch.device) -> Tensor:
    """Lookup table: index num_groups (blank) maps to num_super_groups (blank).
    Cached per-device so we don't re-allocate every batch.
    """
    global _GROUP_TO_SUPER_GROUP_TENSOR
    if (_GROUP_TO_SUPER_GROUP_TENSOR is None
            or _GROUP_TO_SUPER_GROUP_TENSOR.device != device):
        _GROUP_TO_SUPER_GROUP_TENSOR = torch.tensor(
            GROUP_ID_TO_SUPER_GROUP_ID + [NUM_SUPER_GROUPS],
            dtype=torch.long, device=device)
    return _GROUP_TO_SUPER_GROUP_TENSOR


def derive_super_group_labels(
    group_labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Map per-frame group labels to super-group labels.

    Preserves -100 padding for CE ignore. Blank group (NUM_GROUPS) maps to
    blank super-group (NUM_SUPER_GROUPS). All others map via the fixed
    GROUP_ID_TO_SUPER_GROUP_ID table.

    Args:
        group_labels: (B, T) long tensor. Values in {-100} ∪ [0, NUM_GROUPS].
        ignore_index: CE ignore value (default -100). Passed through unchanged.

    Returns:
        (B, T) long tensor. Values in {ignore_index} ∪ [0, NUM_SUPER_GROUPS].
    """
    lut = _group_to_super_group_tensor(group_labels.device)
    # Clamp to [0, NUM_GROUPS] for safe indexing; restore ignore_index afterward.
    ignore_mask = (group_labels == ignore_index)
    clamped = group_labels.clamp(0, NUM_GROUPS)
    out = lut[clamped]
    return torch.where(ignore_mask, torch.full_like(out, ignore_index), out)


def build_frame_labels_from_segments(
    segments_batch: list[list[dict]],
    T: int,
    blank_group_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build per-frame (group_id, script_id) labels from segment metadata.

    Uses the same offset→frame arithmetic as compute_ctc_loss_segments
    (frame = pixel // 2) so model routing and CTC segment ranges agree
    frame-for-frame. Frames outside any segment are labeled blank_group_id
    (model skips expert processing) and script_id 0.

    Returns:
        gl_for_model: (B, T) — group_id per frame, blank_group_id for
            inter-segment / padding frames.
        sl_frames: (B, T) — local script_id within the frame's group, 0
            for blank frames.
    """
    # Build on CPU as NumPy to avoid one GPU kernel launch per segment
    # (was ~10 segments × batch 192 → ~2000 tiny fill_ ops per batch).
    # One host→device transfer for each tensor, non-blocking.
    B = len(segments_batch)
    gl_np = np.full((B, T), blank_group_id, dtype=np.int64)
    sl_np = np.zeros((B, T), dtype=np.int64)
    for b in range(B):
        for seg in segments_batch[b]:
            fs = seg["offset"] // 2
            fe = min((seg["offset"] + seg["width"] + 1) // 2, T)
            if fe <= fs:
                continue
            gl_np[b, fs:fe] = seg["group_id"]
            sl_np[b, fs:fe] = seg.get("script_id", 0)
    gl_for_model = torch.from_numpy(gl_np).to(device, non_blocking=True)
    sl_frames = torch.from_numpy(sl_np).to(device, non_blocking=True)
    return gl_for_model, sl_frames


def get_predicted_script_ids(
    script_logits_per_group: list[tuple],
    true_script_ids: Tensor,
) -> Tensor:
    """Extract per-sample predicted script IDs from LID-2.

    For single-script groups (no LID-2), defaults to true IDs.
    For multi-script groups, uses LID-2 argmax prediction.
    """
    pred_sids = true_script_ids.clone()
    for _g, script_logits, group_mask in script_logits_per_group:
        if script_logits is not None:
            pred_sids[group_mask] = script_logits.argmax(-1)
    return pred_sids


def build_routing_masks(
    pred_group_ids: Tensor,
    true_group_ids: Tensor,
    pred_script_ids: Tensor,
    true_script_ids: Tensor,
    tgt_lens: Tensor,
    enc_lengths: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build routing masks for loss computation.

    Returns:
        lid1_ok: (B,) — LID-1 routed correctly. Used for LID-2 loss.
        ctc_ok: (B,) — LID-1 AND LID-2 correct, valid lengths. Used for CTC.
    """
    lid1_ok = (pred_group_ids == true_group_ids)
    ctc_ok = (lid1_ok
              & (pred_script_ids == true_script_ids)
              & (tgt_lens <= enc_lengths)
              & (tgt_lens > 0))
    return lid1_ok, ctc_ok
