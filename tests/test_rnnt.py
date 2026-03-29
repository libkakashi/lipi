"""
Tests for RNN-T components: PredictionNetwork, JointNetwork, decode, loss.
"""

import warnings
warnings.filterwarnings("ignore", message=".*rnnt_loss.*deprecated.*")

import pytest
import torch
import torchaudio

from src.model.prediction_net import PredictionNetwork
from src.model.joint_net import JointNetwork
from src.model.rnnt_model import LipiRNNT
from src.model.decode import greedy_decode, greedy_decode_with_confidence


@pytest.fixture
def pred_net():
    return PredictionNetwork(vocab_size=401)


@pytest.fixture
def joint_net():
    return JointNetwork(enc_dim=384, pred_dim=128, joint_dim=256, vocab_size=401)


@pytest.fixture
def rnnt_model():
    return LipiRNNT(vocab_size=401)


class TestPredictionNetwork:

    def test_output_shape(self, pred_net):
        tokens = torch.tensor([[1, 5, 10]])  # (1, 3)
        output, hidden = pred_net(tokens)
        assert output.shape == (1, 3, 128)
        assert hidden.shape == (1, 1, 128)

    def test_single_step(self, pred_net):
        token = torch.tensor([[0]])  # blank
        out, h = pred_net(token)
        assert out.shape == (1, 1, 128)

        # Step 2 with hidden state
        out2, h2 = pred_net(torch.tensor([[5]]), h)
        assert out2.shape == (1, 1, 128)
        # Hidden state should differ from initial
        assert not torch.allclose(h, h2)

    def test_batch(self, pred_net):
        tokens = torch.randint(0, 401, (8, 5))
        output, hidden = pred_net(tokens)
        assert output.shape == (8, 5, 128)
        assert hidden.shape == (1, 8, 128)

    def test_param_count(self, pred_net):
        params = sum(p.numel() for p in pred_net.parameters())
        # Embedding: 401*128=51K + GRU: ~99K = ~150K total
        assert 0.1e6 < params < 0.2e6, f"Unexpected param count: {params}"


class TestJointNetwork:

    def test_training_lattice(self, joint_net):
        enc = torch.randn(2, 32, 1, 384)
        pred = torch.randn(2, 1, 10, 128)
        logits = joint_net(enc, pred)
        assert logits.shape == (2, 32, 10, 401)

    def test_inference_single(self, joint_net):
        enc = torch.randn(1, 1, 1, 384)
        pred = torch.randn(1, 1, 1, 128)
        logits = joint_net(enc, pred)
        assert logits.shape == (1, 1, 1, 401)

    def test_param_count(self, joint_net):
        params = sum(p.numel() for p in joint_net.parameters())
        # enc_proj: 384*256 + pred_proj: 128*256 + output: 256*401 = ~235K
        assert 0.2e6 < params < 0.3e6, f"Unexpected param count: {params}"


class TestRNNTModel:

    def test_forward_shape(self, rnnt_model):
        images = torch.randn(2, 3, 32, 128)
        targets = torch.randint(1, 401, (2, 8))
        logits, enc_lengths = rnnt_model(images, targets)
        # T=32 (W=128 -> W/4), U+1=9 (8 targets + 1 blank prepend)
        assert logits.shape == (2, 32, 9, 401)
        assert enc_lengths.tolist() == [32, 32]

    def test_backward(self, rnnt_model):
        rnnt_model.train()
        images = torch.randn(2, 3, 32, 128)
        targets = torch.randint(1, 401, (2, 8))
        logits, _ = rnnt_model(images, targets)
        logits.sum().backward()
        for name, p in rnnt_model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestRNNTLoss:

    def test_loss_computes(self):
        B, T, U, V = 2, 32, 9, 401
        logits = torch.randn(B, T, U, V, requires_grad=True)
        targets = torch.randint(1, V, (B, 8)).int()
        logit_lengths = torch.full((B,), T, dtype=torch.int)
        target_lengths = torch.full((B,), 8, dtype=torch.int)

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=0,
            reduction="mean",
            fused_log_softmax=True,
        )

        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_loss_backward(self):
        B, T, U, V = 2, 16, 5, 100
        logits = torch.randn(B, T, U, V, requires_grad=True)
        targets = torch.randint(1, V, (B, 4)).int()
        logit_lengths = torch.full((B,), T, dtype=torch.int)
        target_lengths = torch.full((B,), 4, dtype=torch.int)

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=0,
            reduction="mean",
            fused_log_softmax=True,
        )
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_loss_with_model(self, rnnt_model):
        """End-to-end: model forward -> RNN-T loss -> backward."""
        rnnt_model.train()
        images = torch.randn(2, 3, 32, 64)
        targets = torch.randint(1, 401, (2, 5))

        logits, _ = rnnt_model(images, targets)
        T = logits.shape[1]
        target_lengths = torch.full((2,), 5, dtype=torch.int)
        logit_lengths = torch.full((2,), T, dtype=torch.int)

        loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=targets.int(),
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=0,
            reduction="mean",
            fused_log_softmax=True,
        )

        assert not torch.isnan(loss)
        loss.backward()
        # Verify gradients flow to all components
        assert rnnt_model.encoder.proj1.weight.grad is not None
        assert rnnt_model.prediction_net.embedding.weight.grad is not None
        assert rnnt_model.joint_net.enc_proj.weight.grad is not None


class TestGreedyDecode:

    def test_decode_produces_tokens(self, pred_net, joint_net):
        enc_out = torch.randn(2, 16, 384)
        results = greedy_decode(enc_out, pred_net, joint_net)
        assert len(results) == 2
        for tokens in results:
            assert isinstance(tokens, list)
            assert all(isinstance(t, int) for t in tokens)
            assert all(0 < t < 401 for t in tokens)

    def test_decode_max_tokens(self, pred_net, joint_net):
        enc_out = torch.randn(1, 16, 384)
        results = greedy_decode(enc_out, pred_net, joint_net, max_tokens=5)
        assert len(results[0]) <= 5

    def test_decode_with_confidence(self, pred_net, joint_net):
        enc_out = torch.randn(2, 16, 384)
        results = greedy_decode_with_confidence(enc_out, pred_net, joint_net)
        assert len(results) == 2
        for tokens, confidence in results:
            assert 0.0 <= confidence <= 1.0
