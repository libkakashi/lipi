"""
Frame-level routing labels for MoE training.

Builds the per-frame (group_id, script_id) labels used to route each frame
to its group/script expert and to keep CTC segment ranges frame-aligned.
"""

import numpy as np
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
    (frame = pixel // 4, matching the encoder's 4× width downsample) so
    model routing and CTC segment ranges agree frame-for-frame. Frames
    outside any segment are labeled blank_group_id (model skips expert
    processing) and script_id 0.

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
            fs = seg["offset"] // 4
            fe = min((seg["offset"] + seg["width"] + 3) // 4, T)
            if fe <= fs:
                continue
            gl_np[b, fs:fe] = seg["group_id"]
            sl_np[b, fs:fe] = seg.get("script_id", 0)
    gl_for_model = torch.from_numpy(gl_np).to(device, non_blocking=True)
    sl_frames = torch.from_numpy(sl_np).to(device, non_blocking=True)
    return gl_for_model, sl_frames
