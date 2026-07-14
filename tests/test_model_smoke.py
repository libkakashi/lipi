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
    # 2 emission slots per frame: logits + lengths run at 2T
    assert out["logits"].shape[1] == T * model.emit_per_frame
    assert out["lengths"].shape[0] == B
    assert out["lengths"][0].item() == T * model.emit_per_frame
    # group_logits stay at frame granularity: (B, T, num_groups+1)
    assert out["group_logits"].shape == (B, T, 3)
    # LID-2 only for multi-script group (group 1)
    assert 1 in out["lid2_logits_per_group"]
    assert 0 not in out["lid2_logits_per_group"]
    assert out["lid2_logits_per_group"][1].shape == (B, T, 2)


def test_in_height_64_output_geometry():
    """The 64px trunk (h_c=8 SWA-C grid) produces the same output
    geometry as the 32px trunk at equal width: T frames, 2T emissions,
    LID at T. Wrong input height fails loudly."""
    from src.model.encoder import LipiMoEEncoder

    kwargs = dict(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [80, 120]],
        group_script_names=[["test1"], ["test2a", "test2b"]],
        drop_path_rate=0.0,
    )
    m64 = LipiMoEEncoder(in_height=64, **kwargs)
    assert m64.h_c == 8
    assert m64.swa_c[0].attn.window_h == 8

    B, W = 2, 64
    T = W // 4
    group_ids = torch.tensor([[0] * T, [1] * T])
    script_ids = torch.tensor([[0] * T, [0] * T])
    m64.eval()
    with torch.no_grad():
        out = m64(torch.randn(B, 3, 64, W),
                  group_ids=group_ids, script_ids=script_ids)
    assert out["lengths"][0].item() == T * m64.emit_per_frame
    assert out["logits"].shape[1] == T * m64.emit_per_frame
    assert out["group_logits"].shape == (B, T, 3)

    # Feeding the wrong height must fail loudly, not silently mis-shape.
    try:
        m64(torch.randn(B, 3, 32, W), group_ids=group_ids,
            script_ids=script_ids)
        raise AssertionError("expected height-mismatch ValueError")
    except ValueError as e:
        assert "in_height" in str(e)


def test_relook_is_noop_at_init():
    """The ReLook's output projection is zero-init: outputs are
    bit-identical whatever its q/k/v projections and row embeddings
    hold, so the retrieval path starts silent and grows only if useful."""
    from src.model.encoder import LipiMoEEncoder

    torch.manual_seed(0)
    model = LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [80, 120]],
        group_script_names=[["test1"], ["test2a", "test2b"]],
        drop_path_rate=0.0,
    )
    model.eval()
    B, W = 2, 64
    T = W // 4
    imgs = torch.randn(B, 3, 32, W)
    gids = torch.zeros(B, T, dtype=torch.long)
    sids = torch.zeros(B, T, dtype=torch.long)

    with torch.no_grad():
        out1 = model(imgs, group_ids=gids, script_ids=sids)
        model.relook.q_proj.weight.normal_()   # scramble everything
        model.relook.k_proj.weight.normal_()   # except out_proj
        model.relook.v_proj.weight.normal_()
        model.relook.row_emb.normal_()
        out2 = model(imgs, group_ids=gids, script_ids=sids)
    assert torch.equal(out1["logits"], out2["logits"])
    assert torch.equal(model.relook.out_proj.weight,
                       torch.zeros_like(model.relook.out_proj.weight))


def test_readouts_start_as_row_mean():
    """Both attention collapses degrade to plain row mean-pooling when
    their queries/row embeddings are zero (uniform attention + identity
    v/out projections) — the trunk's pooling-at-step-0 convention. With
    the real trunc-normal init they sit within noise of that mean."""
    from src.model.blocks import AttentionReadout, RoutedReadout

    torch.manual_seed(0)
    B, h, w, C = 2, 8, 6, 32

    ro = AttentionReadout(C, 48, n_rows=h, n_queries=2, num_heads=2).eval()
    with torch.no_grad():
        ro.queries.zero_()
        ro.row_emb.zero_()
        x = torch.randn(B, h * w, C)
        out = ro(x, h, w)  # (B, 2*w, 48)
        mean = ro.norm(x).reshape(B, h, w, C).mean(dim=1)  # (B, w, C)
    out_rows = out.reshape(B, 2, w, 48)
    for q in range(2):
        assert torch.allclose(out_rows[:, q, :, :C], mean, atol=1e-5)
        assert torch.allclose(out_rows[:, q, :, C:],
                              torch.zeros(B, w, 48 - C), atol=1e-6)

    rc = RoutedReadout(C, num_ids=5, num_heads=2).eval()
    with torch.no_grad():
        rc.queries.zero_()
        rc.row_emb.zero_()
        x = torch.randn(B, 2 * w, C)
        ids_a = torch.zeros(B, w, dtype=torch.long)
        ids_b = torch.full((B, w), -1, dtype=torch.long)  # blank/default
        out_a = rc(x, ids_a, 2, w)
        out_b = rc(x, ids_b, 2, w)
        mean = rc.norm(x).reshape(B, 2, w, C).mean(dim=1)
    assert torch.allclose(out_a, mean, atol=1e-5)
    assert torch.allclose(out_b, mean, atol=1e-5)  # id-independent at init


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


def test_self_conditioned_ctc():
    """Self-conditioning: intermediate logits returned in training mode
    only, feedback gate is zero-init, and the tied-weight feedback path
    is actually wired into the script stack."""
    from src.model.encoder import LipiMoEEncoder

    torch.manual_seed(0)
    model = LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [80, 120]],
        group_script_names=[["test1"], ["test2a", "test2b"]],
        drop_path_rate=0.0,
    )

    B, H, W = 2, 32, 64
    images = torch.randn(B, 3, H, W)
    T = W // 4
    group_ids = torch.tensor([[0] * T, [1] * T])
    script_ids = torch.tensor([[0] * T, [1] * T])

    # Zero-init gate: feedback starts as a no-op
    assert torch.equal(model.self_cond_ls.gamma,
                       torch.zeros_like(model.self_cond_ls.gamma))

    model.train()
    with torch.no_grad():
        out_train = model(images, group_ids=group_ids, script_ids=script_ids)
    assert "inter_logits" in out_train
    assert out_train["inter_logits"].shape == out_train["logits"].shape
    # Same head dispatch: unrouted vocab tail stays zero
    assert torch.equal(out_train["inter_logits"][0, :, 100:],
                       torch.zeros_like(out_train["inter_logits"][0, :, 100:]))
    # Intermediate = pre-script-stack pass; must differ from final
    assert not torch.allclose(out_train["inter_logits"], out_train["logits"])

    # Eval mode skips returning the (max_vocab-wide) intermediate logits
    model.eval()
    with torch.no_grad():
        out_gated = model(images, group_ids=group_ids, script_ids=script_ids)
    assert "inter_logits" not in out_gated

    # Open the gate: final logits must change — proves the feedback
    # actually feeds the script stack.
    with torch.no_grad():
        model.self_cond_ls.gamma.fill_(1.0)
        out_open = model(images, group_ids=group_ids, script_ids=script_ids)
    assert not torch.allclose(out_gated["logits"], out_open["logits"])


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


def test_routed_mlp_scatter_dtype_under_autocast():
    """Routed-MLP scatter must not crash when its buffer dtype differs from
    the expert output dtype.

    Under CUDA autocast, LayerNorm is forced to fp32, so norm2's output (fed
    to _routed_mlp) is fp32 while the expert Linear runs in bf16 — the
    masked-scatter's dst (fp32) and src (bf16) then mismatch and index_put
    raises. This killed the capacity probe's first forward on H100. We
    reproduce the exact fp32-in / bf16-expert split by calling _routed_mlp
    with an fp32 input inside CPU autocast (CPU autocast happens to preserve
    LayerNorm's input dtype, so the full-model forward can't surface it —
    the direct call mirrors the CUDA path faithfully)."""
    from src.model.blocks import MoELayer

    torch.manual_seed(0)
    layer = MoELayer(dim=64, num_heads=4, num_experts=3, window_w=16)

    B, T = 2, 16
    x = torch.randn(B, T, 64)  # fp32, like CUDA-autocast norm2 output
    expert_ids = torch.randint(0, 3, (B, T))

    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = layer._routed_mlp(x, expert_ids, None)
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


def test_layer_scale_init_near_identity():
    """MoE layers start near-identity (LayerScale 1e-4) like the backbone;
    lid1_attn keeps LayerScale 1.0 (its identity comes from zero-init
    projections instead)."""
    from src.model.encoder import LipiMoEEncoder

    model = LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[100], [80, 120]],
        group_script_names=[["test1"], ["test2a", "test2b"]],
    )

    for layer in list(model.group_layers) + list(model.script_layers):
        assert torch.allclose(layer.ls1.gamma,
                              torch.full_like(layer.ls1.gamma, 1e-4))
        assert torch.allclose(layer.ls2.gamma,
                              torch.full_like(layer.ls2.gamma, 1e-4))
    assert torch.allclose(model.lid1_attn.ls1.gamma,
                          torch.ones_like(model.lid1_attn.ls1.gamma))
    assert model.config["layer_scale_init"] == 1e-4
