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


def test_inter_ctc_logits():
    """inter_ctc=True returns pre-script-stack logits with the same shape
    and routing as the final logits, and no inter_logits otherwise."""
    from src.model.encoder import LipiMoEEncoder

    model = LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [80, 120]],
        group_script_names=[["test1"], ["test2a", "test2b"]],
    )

    B, H, W = 2, 32, 64
    images = torch.randn(B, 3, H, W)
    T = W // 4
    group_ids = torch.tensor([[0] * T, [1] * T])
    script_ids = torch.tensor([[0] * T, [1] * T])

    model.eval()
    with torch.no_grad():
        out = model(images, group_ids=group_ids, script_ids=script_ids,
                    inter_ctc=True)
        out_plain = model(images, group_ids=group_ids, script_ids=script_ids)

    assert "inter_logits" not in out_plain
    assert out["inter_logits"].shape == out["logits"].shape
    # Same head dispatch: unrouted vocab tail stays zero in both
    assert torch.equal(out["inter_logits"][0, :, 100:],
                       torch.zeros_like(out["inter_logits"][0, :, 100:]))
    # Final logits pass through the script stack, intermediate don't —
    # they must differ (script MoE layers aren't identity even at init
    # due to shared MLP + attention branches).
    assert not torch.allclose(out["inter_logits"], out["logits"])


def test_scheduled_routing_sampling():
    """route_sample_p=1.0 routes every frame by predicted LID; p=0 keeps GT."""
    from src.model.encoder import LipiMoEEncoder

    torch.manual_seed(0)
    model = LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [80, 120]],
        group_script_names=[["test1"], ["test2a", "test2b"]],
        drop_path_rate=0.0,  # deterministic in train mode
    )

    B, H, W = 2, 32, 64
    images = torch.randn(B, 3, H, W)
    T = W // 4
    group_ids = torch.tensor([[0] * T, [1] * T])
    script_ids = torch.tensor([[0] * T, [1] * T])

    model.train()
    with torch.no_grad():
        out_full = model(images, group_ids=group_ids, script_ids=script_ids,
                         route_sample_p=1.0)
        out_gt = model(images, group_ids=group_ids, script_ids=script_ids,
                       route_sample_p=0.0)

    # p=1: every frame routed by LID-1 argmax
    assert torch.equal(out_full["group_ids"],
                       out_full["group_logits"].argmax(dim=-1))
    # p=0: GT routing untouched
    assert torch.equal(out_gt["group_ids"], group_ids)
    assert torch.equal(out_gt["frame_scripts"], script_ids)

    # eval mode ignores route_sample_p entirely
    model.eval()
    with torch.no_grad():
        out_eval = model(images, group_ids=group_ids, script_ids=script_ids,
                         route_sample_p=1.0)
    assert torch.equal(out_eval["group_ids"], group_ids)
