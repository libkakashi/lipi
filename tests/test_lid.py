"""
Tests for hierarchical LID (LIDCoarse + LIDFine).
"""

import pytest
import torch

from src.model.lid import LIDCoarse, LIDFine, SCRIPTS, GROUPS, NUM_SCRIPTS, NUM_GROUPS


class TestLIDCoarse:

    def test_output_shape(self):
        lid = LIDCoarse(in_channels=64, num_groups=NUM_GROUPS)
        x = torch.randn(4, 64, 8, 32)
        logits = lid(x)
        assert logits.shape == (4, NUM_GROUPS)

    def test_predict(self):
        lid = LIDCoarse(in_channels=64)
        x = torch.randn(2, 64, 8, 32)
        group_ids, confidences = lid.predict(x)
        assert group_ids.shape == (2,)
        assert confidences.shape == (2,)
        assert (confidences >= 0).all() and (confidences <= 1).all()

    def test_variable_width(self):
        lid = LIDCoarse(in_channels=64)
        for w in [8, 16, 32, 48]:
            logits = lid(torch.randn(1, 64, 8, w))
            assert logits.shape == (1, NUM_GROUPS)

    def test_num_groups(self):
        assert NUM_GROUPS == 8
        assert len(GROUPS) == 8


class TestLIDFine:

    def test_output_shape(self):
        lid = LIDFine(in_dim=288, num_scripts=NUM_SCRIPTS)
        x = torch.randn(4, 128, 288)  # (B, H*W, C)
        logits = lid(x)
        assert logits.shape == (4, NUM_SCRIPTS)

    def test_predict(self):
        lid = LIDFine(in_dim=192)
        x = torch.randn(2, 64, 192)
        script_ids, confidences = lid.predict(x)
        assert script_ids.shape == (2,)
        assert confidences.shape == (2,)

    def test_num_scripts(self):
        assert NUM_SCRIPTS == len(SCRIPTS)

    def test_gradient_flow(self):
        lid = LIDFine(in_dim=192)
        lid.train()
        x = torch.randn(2, 64, 192)
        logits = lid(x)
        loss = logits.sum()
        loss.backward()
        for name, param in lid.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
