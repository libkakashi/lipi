"""Static-shape bucket batching tests — pure numpy/torch CPU, fast."""

import numpy as np
import torch

from src.training.dataloader import (
    BucketBatchSampler, collate_moe, compute_bucket_edges,
)


def _widths(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.integers(97, 1281, size=n)
    w[-50:] = 1280  # mass atom at the render cap, like real shards
    return w


def test_edges_cover_max_and_are_4px_aligned():
    w = _widths()
    edges = compute_bucket_edges(w)
    assert edges == sorted(edges)
    assert all(e % 4 == 0 for e in edges)
    assert edges[-1] >= w.max()
    assert len(edges) <= 16


def test_edges_collapse_on_atom():
    w = np.full(100, 512)
    edges = compute_bucket_edges(w)
    assert edges == [512]


def test_edges_optimal_on_bimodal():
    # Two tight clusters: the optimal 2-bucket split has zero waste, and
    # a third bucket gains nothing — DP must find exactly [100, 1000].
    w = np.array([100] * 50 + [1000] * 50)
    assert compute_bucket_edges(w) == [100, 1000]


def test_edges_stop_at_diminishing_gains():
    # Uniform widths: waste falls ~1/K, so K stays small once another
    # bucket saves <0.5% of total pixels.
    rng = np.random.default_rng(1)
    w = rng.integers(400, 1281, size=5000)
    edges = compute_bucket_edges(w)
    assert 2 <= len(edges) <= 16
    e = np.array(edges)
    pad = e[np.searchsorted(e, w)] - w
    assert pad.sum() / w.sum() < 0.20  # bounded padding waste


def _sampler(w, cap=32):
    edges = compute_bucket_edges(w)
    capacities = {e: max(1, cap * edges[-1] // e) for e in edges}
    return BucketBatchSampler(w, edges, capacities), edges, capacities


def test_batches_are_static_shapes():
    w = _widths()
    sampler, edges, capacities = _sampler(w)
    edges_arr = np.array(edges)
    seen = set()
    for batch in sampler:
        # Bucket = smallest edge >= the batch's max width; batch size
        # must be exactly that bucket's capacity (static shape).
        max_w = w[batch].max()
        edge = edges_arr[np.searchsorted(edges_arr, max_w)]
        assert len(batch) == capacities[int(edge)]
        seen.update(batch)
    assert seen == set(range(len(w)))  # every sample visited


def test_composition_reshuffles_across_epochs():
    w = _widths()
    sampler, _, _ = _sampler(w)
    e1 = {tuple(sorted(b)) for b in sampler}
    e2 = {tuple(sorted(b)) for b in sampler}
    assert e1 != e2


def test_weighted_sampling_draws_with_replacement():
    w = _widths(n=200)
    edges = compute_bucket_edges(w)
    capacities = {e: 16 for e in edges}
    weights = np.zeros(len(w))
    weights[:10] = 1.0  # only first 10 samples ever drawn
    sampler = BucketBatchSampler(w, edges, capacities, sample_weights=weights)
    drawn = {i for batch in sampler for i in batch}
    assert drawn <= set(range(10))


def _sample(width):
    return (
        torch.zeros(3, 32, width, dtype=torch.uint8),   # img
        torch.zeros(5, dtype=torch.long),               # target ids
        torch.tensor(5, dtype=torch.long),              # target len
        torch.tensor(0, dtype=torch.long),              # gid
        torch.tensor(0, dtype=torch.long),              # sid
        "label",                                        # label
        torch.zeros(width, dtype=torch.long),           # group labels
        [],                                             # segments
    )


def test_collate_pads_to_bucket_edge():
    out = collate_moe([_sample(100), _sample(130)], pad_to_widths=[128, 256])
    assert out[0].shape == (2, 3, 32, 256)
    assert out[6].shape == (2, 256)  # group labels padded to same width


def test_collate_exact_edge_and_fallback():
    out = collate_moe([_sample(128)], pad_to_widths=[128, 256])
    assert out[0].shape == (1, 3, 32, 128)
    # Wider than the last edge (val-style) → round up to multiple of 4
    out = collate_moe([_sample(300)], pad_to_widths=[128, 256])
    assert out[0].shape == (1, 3, 32, 300)
    # No buckets → old behavior
    out = collate_moe([_sample(130)])
    assert out[0].shape == (1, 3, 32, 132)
