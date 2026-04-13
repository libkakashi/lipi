"""
Tests for LIDCoarse.
"""

import pytest
import torch

from src.model.lid import LIDCoarse, SCRIPTS, GROUPS, NUM_SCRIPTS, NUM_GROUPS


class TestLIDCoarse:

    def test_output_shape(self):
        lid = LIDCoarse(in_channels=64, num_groups=NUM_GROUPS)
        x = torch.randn(4, 64, 8, 32)
        logits = lid(x)
        assert logits.shape == (4, NUM_GROUPS)

    def test_argmax_and_softmax(self):
        lid = LIDCoarse(in_channels=64, num_groups=NUM_GROUPS)
        x = torch.randn(2, 64, 8, 32)
        logits = lid(x)
        group_ids = logits.argmax(dim=-1)
        confidences = logits.softmax(dim=-1).max(dim=-1).values
        assert group_ids.shape == (2,)
        assert confidences.shape == (2,)
        assert (confidences >= 0).all() and (confidences <= 1).all()

    def test_variable_width(self):
        lid = LIDCoarse(in_channels=64)
        for w in [8, 16, 32, 48]:
            logits = lid(torch.randn(1, 64, 8, w))
            assert logits.shape == (1, NUM_GROUPS)

    def test_num_groups(self):
        assert NUM_GROUPS == 13
        assert len(GROUPS) == 13
