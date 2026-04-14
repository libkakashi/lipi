"""
Routing mask computation for MoE training.

Builds the masks that control which samples contribute to which losses:
  - lid1_ok: LID-1 predicted correctly → used for LID-2 loss
  - ctc_ok: LID-1 AND LID-2 correct → used for CTC loss
"""

import torch
from torch import Tensor


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
    B = len(segments_batch)
    gl_for_model = torch.full((B, T), blank_group_id,
                              dtype=torch.long, device=device)
    sl_frames = torch.zeros(B, T, dtype=torch.long, device=device)
    for b in range(B):
        for seg in segments_batch[b]:
            fs = seg["offset"] // 2
            fe = min((seg["offset"] + seg["width"] + 1) // 2, T)
            gl_for_model[b, fs:fe] = seg["group_id"]
            sl_frames[b, fs:fe] = seg.get("script_id", 0)
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
