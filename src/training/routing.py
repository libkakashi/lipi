"""
Routing mask computation for MoE training.

Builds the masks that control which samples contribute to which losses:
  - lid1_ok: LID-1 predicted correctly → used for LID-2 loss
  - ctc_ok: LID-1 AND LID-2 correct → used for CTC loss
"""

import torch
from torch import Tensor


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
