"""Equivalence of the grouped MoE dispatch vs the boolean-mask reference.

The grouped gather/scatter dispatch in MoELayer._routed_mlp must produce
byte-for-byte the same output and gradients as the old per-expert
boolean-mask loop — the rewrite is purely a performance change (the loop's
full-size masked-scatter backward dominated the training step)."""

import torch

from src.model.blocks import MoELayer, _per_sample_key_lens


def _ref_routed_mlp(layer, x, expert_ids):
    """Old implementation: per-expert boolean mask select + scatter."""
    out = torch.zeros_like(x)
    for e in range(layer.num_experts):
        mask = (expert_ids == e)
        if not mask.any():
            continue
        out[mask] = layer.routed_mlps[e](x[mask]).to(out.dtype)
    return out


def _make_layer():
    torch.manual_seed(0)
    return MoELayer(dim=64, num_heads=4, num_experts=5, window_w=16)


def test_routed_mlp_matches_boolean_reference():
    layer = _make_layer()
    B, T, D = 3, 20, 64
    torch.manual_seed(1)
    x = torch.randn(B, T, D)
    # Mix of routed experts and unrouted frames (id == num_experts → skipped).
    expert_ids = torch.randint(0, layer.num_experts + 1, (B, T))

    ref = _ref_routed_mlp(layer, x, expert_ids)

    lens = _per_sample_key_lens(expert_ids, layer.num_experts).tolist()
    got_precomp = layer._routed_mlp(x, expert_ids, lens)
    got_bincount = layer._routed_mlp(x, expert_ids, None)

    assert torch.allclose(got_precomp, ref, atol=1e-6), \
        (got_precomp - ref).abs().max().item()
    assert torch.allclose(got_bincount, ref, atol=1e-6)
    # Unrouted frames (expert_id == num_experts) must be exactly zero.
    unrouted = (expert_ids == layer.num_experts)
    assert torch.equal(got_precomp[unrouted], torch.zeros_like(got_precomp[unrouted]))


def test_routed_mlp_gradients_match():
    layer = _make_layer()
    B, T, D = 2, 16, 64
    torch.manual_seed(2)
    expert_ids = torch.randint(0, layer.num_experts, (B, T))
    lens = _per_sample_key_lens(expert_ids, layer.num_experts).tolist()

    torch.manual_seed(3)
    x0 = torch.randn(B, T, D)
    grad_seed = torch.randn(B, T, D)  # identical upstream grad for both

    def run(use_new):
        layer.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        out = (layer._routed_mlp(x, expert_ids, lens) if use_new
               else _ref_routed_mlp(layer, x, expert_ids))
        (out * grad_seed).sum().backward()
        return out.detach(), x.grad.clone(), layer.routed_mlps[0].fc1.weight.grad.clone()

    out_ref, gx_ref, gw_ref = run(False)
    out_new, gx_new, gw_new = run(True)

    assert torch.allclose(out_new, out_ref, atol=1e-6)
    assert torch.allclose(gx_new, gx_ref, atol=1e-5), \
        (gx_new - gx_ref).abs().max().item()
    assert torch.allclose(gw_new, gw_ref, atol=1e-5), \
        (gw_new - gw_ref).abs().max().item()


def test_routed_mlp_all_unrouted_is_zero():
    layer = _make_layer()
    B, T, D = 2, 8, 64
    x = torch.randn(B, T, D)
    expert_ids = torch.full((B, T), layer.num_experts)  # all unrouted
    lens = _per_sample_key_lens(expert_ids, layer.num_experts).tolist()
    out = layer._routed_mlp(x, expert_ids, lens)
    assert torch.equal(out, torch.zeros_like(out))
