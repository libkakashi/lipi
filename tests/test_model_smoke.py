"""Smoke test: model instantiation, forward pass, loss computation."""

import torch
import pytest


def test_model_forward_pass():
    """Smoke test: model instantiation, forward pass, basic output checks."""
    from src.model.encoder import LipiMoEEncoder

    model = LipiMoEEncoder(
        shared_dim=64,
        stage1_dim=64,
        stage2_dim=64,
        shared_blocks_4x4=1,
        shared_blocks_4x16=1,
        stage1_blocks=1,
        stage2_blocks=1,
        num_groups=2,
        group_script_vocab_sizes=[[100], [100]],
        group_script_names=[["test1"], ["test2"]],
    )

    # Forward pass — model expects (B, 2, H, W) input (L+a channels)
    B, H, W = 2, 32, 64
    images = torch.randn(B, 2, H, W)
    group_ids = torch.tensor([0, 1])

    model.eval()
    with torch.no_grad():
        out = model(images, group_ids=group_ids)

    assert "logits" in out
    assert "lengths" in out
    assert "group_logits" in out
    assert out["logits"].shape[0] == B
    assert out["lengths"].shape[0] == B
