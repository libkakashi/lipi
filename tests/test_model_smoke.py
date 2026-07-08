"""Smoke test: model instantiation, forward pass, output checks."""

import torch


def test_model_forward_pass():
    """Smoke test: v4 encoder with two-level expert routing."""
    from src.model.encoder import LipiMoEEncoder

    model = LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [80, 120]],  # group 1 is multi-script
        group_script_names=[["test1"], ["test2a", "test2b"]],
    )

    B, H, W = 2, 32, 64
    images = torch.randn(B, 3, H, W)
    T = W // 4

    # With ground truth group + script routing
    group_ids = torch.tensor([[0]*T, [1]*T])  # per-frame
    script_ids = torch.tensor([[0]*T, [1]*T])  # per-frame

    model.eval()
    with torch.no_grad():
        out = model(images, group_ids=group_ids, script_ids=script_ids)

    assert "logits" in out
    assert "lengths" in out
    assert "group_logits" in out
    assert "group_ids" in out
    assert "lid2_logits_per_group" in out
    assert "frame_scripts" in out
    assert out["logits"].shape[0] == B
    assert out["lengths"].shape[0] == B
    assert out["lengths"][0].item() == T
    # group_logits: (B, T, num_groups+1) with blank
    assert out["group_logits"].shape == (B, T, 3)
    # LID-2 only for multi-script group (group 1)
    assert 1 in out["lid2_logits_per_group"]
    assert 0 not in out["lid2_logits_per_group"]
    assert out["lid2_logits_per_group"][1].shape == (B, T, 2)


def test_model_inference_mode():
    """Test inference (no ground truth routing)."""
    from src.model.encoder import LipiMoEEncoder

    model = LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [80, 120]],
        group_script_names=[["test1"], ["test2a", "test2b"]],
    )

    B, H, W = 2, 32, 64
    images = torch.randn(B, 3, H, W)

    model.eval()
    with torch.no_grad():
        out = model(images)  # no group_ids, no script_ids

    assert out["logits"].shape[0] == B
    assert out["group_logits"].shape[2] == 3
