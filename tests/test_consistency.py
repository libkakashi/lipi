"""Two-view consistency loss + collate tests — synthetic tensors, fast."""

import torch

from src.training.dataloader import collate_moe
from src.training.losses import compute_consistency_loss

# 2 groups: group 0 single-script (vocab 10), group 1 two scripts (8, 12)
VOCABS = [[10], [8, 12]]
MAX_VOCAB = 12
B, T, G = 2, 6, 2


def _inputs(identical=True, seed=0):
    torch.manual_seed(seed)
    logits1 = torch.randn(B, T, MAX_VOCAB)
    group_logits1 = torch.randn(B, T, G + 1)
    if identical:
        logits2 = logits1.clone()
        group_logits2 = group_logits1.clone()
    else:
        logits2 = torch.randn(B, T, MAX_VOCAB)
        group_logits2 = torch.randn(B, T, G + 1)
    # frames: sample 0 → flat script 0, sample 1 → flat script 2; one
    # blank frame each
    flat = torch.tensor([[0, 0, 0, 0, 0, -1],
                         [2, 2, 2, 2, 2, -1]])
    gl = torch.tensor([[0, 0, 0, 0, 0, -100],
                       [1, 1, 1, 1, 1, -100]])
    aligned = torch.tensor([True, True])
    return logits1, logits2, group_logits1, group_logits2, flat, gl, aligned


class TestConsistencyLoss:

    def test_identical_views_zero_loss(self):
        l1, l2, g1, g2, flat, gl, aligned = _inputs(identical=True)
        loss = compute_consistency_loss(l1, l2, g1, g2, flat, gl, aligned, VOCABS)
        assert loss.item() < 1e-6

    def test_different_views_positive_loss(self):
        l1, l2, g1, g2, flat, gl, aligned = _inputs(identical=False)
        loss = compute_consistency_loss(l1, l2, g1, g2, flat, gl, aligned, VOCABS)
        assert loss.item() > 0.01

    def test_unaligned_samples_skipped(self):
        l1, l2, g1, g2, flat, gl, _ = _inputs(identical=False)
        aligned = torch.tensor([False, False])
        loss = compute_consistency_loss(l1, l2, g1, g2, flat, gl, aligned, VOCABS)
        assert loss.item() == 0.0

    def test_vocab_slice_ignores_padding_tail(self):
        """Garbage past a script's vocab size must not affect the loss."""
        l1, l2, g1, g2, flat, gl, aligned = _inputs(identical=True)
        # flat script 0 has vocab 10 — poison columns 10: in one view only
        l2[:, :, 10:] += 100.0
        # restrict to sample 0 (script 0); sample 1 uses script 2 (vocab 12)
        aligned = torch.tensor([True, False])
        loss = compute_consistency_loss(l1, l2, g1, g2, flat, gl, aligned, VOCABS)
        assert loss.item() < 1e-6

    def test_none_logits_skips_ctc_term(self):
        _, _, g1, g2, flat, gl, aligned = _inputs(identical=False)
        loss = compute_consistency_loss(None, None, g1, g2, flat, gl,
                                        aligned, VOCABS)
        loss_full = compute_consistency_loss(*_inputs(identical=False)[:2],
                                             g1, g2, flat, gl, aligned, VOCABS)
        assert loss.item() > 0  # LID term still active
        assert loss.item() != loss_full.item()

    def test_emission_slot_logits(self):
        """CTC logits at 2 emission slots per frame (v6 encoder) against
        frame-granularity flat_scripts — the mask must expand to slot
        granularity instead of crashing (regression: smoke run batch 0,
        mask (B, T) indexing logits (B, 2T, V))."""
        _, _, g1, g2, flat, gl, aligned = _inputs(identical=True)
        torch.manual_seed(3)
        l1 = torch.randn(B, 2 * T, MAX_VOCAB)
        # Identical views at 2T → CTC term zero; LID term zero
        loss = compute_consistency_loss(l1, l1.clone(), g1, g2.copy_(g1),
                                        flat, gl, aligned, VOCABS)
        assert loss.item() < 1e-6
        # Different views at 2T → positive, and both slots participate:
        # poisoning only odd slots (frame's 2nd emission) must move it.
        l2 = l1.clone()
        l2[:, 1::2, :8] += 3.0
        loss_odd = compute_consistency_loss(l1, l2, g1, g1.clone(),
                                            flat, gl, aligned, VOCABS)
        assert loss_odd.item() > 0.01


class TestTwoViewCollate:

    def _sample(self, w, two_views):
        img = torch.zeros(3, 32, w, dtype=torch.uint8)
        tids = torch.tensor([1, 2], dtype=torch.long)
        tlen = torch.tensor(2, dtype=torch.long)
        gid = torch.tensor(0, dtype=torch.long)
        sid = torch.tensor(0, dtype=torch.long)
        gl = torch.zeros(w, dtype=torch.long)
        segs = [{"group_id": 0, "script_id": 0, "text": "ab",
                 "offset": 0, "width": w}]
        base = (img, tids, tlen, gid, sid, "ab", gl, segs)
        if not two_views:
            return base
        return base + (torch.ones(3, 32, w, dtype=torch.uint8), True)

    def test_standard_batch_unchanged(self):
        out = collate_moe([self._sample(20, False), self._sample(24, False)])
        assert len(out) == 8
        assert out[0].shape == (2, 3, 32, 24)

    def test_two_view_batch(self):
        out = collate_moe([self._sample(20, True), self._sample(24, True)])
        assert len(out) == 10
        imgs, imgs2, aligned = out[0], out[8], out[9]
        assert imgs.shape == imgs2.shape == (2, 3, 32, 24)
        assert aligned.dtype == torch.bool and aligned.all()
        # Both views padded identically (zeros on the right)
        assert (imgs2[0, :, :, 20:] == 0).all()
