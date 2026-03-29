"""
Tests for Micro-LID script classifier.
"""

import pytest
import torch

from src.model.lid import MicroLID, SCRIPT_NAMES, NUM_SCRIPTS


@pytest.fixture
def lid():
    return MicroLID()


class TestMicroLID:

    def test_output_shape(self, lid):
        x = torch.randn(4, 3, 32, 128)
        logits = lid(x)
        assert logits.shape == (4, NUM_SCRIPTS)

    @pytest.mark.parametrize("width", [32, 64, 128, 256, 320])
    def test_variable_width(self, lid, width):
        x = torch.randn(1, 3, 32, width)
        logits = lid(x)
        assert logits.shape == (1, NUM_SCRIPTS)

    def test_num_scripts(self):
        assert NUM_SCRIPTS == 11
        assert len(SCRIPT_NAMES) == 11

    def test_param_count(self, lid):
        params = sum(p.numel() for p in lid.parameters())
        # Should be tiny — well under 100K params
        assert params < 100_000, f"LID too large: {params}"

    def test_gradient_flow(self):
        lid = MicroLID()
        lid.train()
        x = torch.randn(2, 3, 32, 128)
        logits = lid(x)
        loss = logits.sum()
        loss.backward()
        for name, param in lid.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_batch_sizes(self, lid):
        for B in [1, 8, 32]:
            x = torch.randn(B, 3, 32, 128)
            logits = lid(x)
            assert logits.shape == (B, NUM_SCRIPTS)
