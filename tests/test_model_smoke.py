"""Smoke test: model instantiation, forward pass, output checks."""

import torch


def test_model_forward_pass():
    """Smoke test: v3 encoder with HGNetV2 backbone."""
    from src.model.encoder import LipiMoEEncoder

    model = LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [100]],
        group_script_names=[["test1"], ["test2"]],
    )

    B, H, W = 2, 32, 64
    images = torch.randn(B, 3, H, W)
    group_ids = torch.tensor([0, 1])

    model.eval()
    with torch.no_grad():
        out = model(images, group_ids=group_ids)

    assert "logits" in out
    assert "lengths" in out
    assert "group_logits" in out
    assert "group_ids" in out
    assert "script_logits_per_group" in out
    assert out["logits"].shape[0] == B
    assert out["lengths"].shape[0] == B
    # T = W/2
    assert out["lengths"][0].item() == W // 2
    assert out["group_logits"].shape == (B, 2)
