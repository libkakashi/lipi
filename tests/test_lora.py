"""
Tests for LoRA adapter injection.
"""

import pytest
import torch

from src.model.encoder import LipiEncoder
from src.model.lora import inject_lora, get_lora_target_modules, get_lora_param_count


@pytest.fixture
def encoder():
    return LipiEncoder()


class TestLoRATargetModules:

    def test_target_modules_exist(self, encoder):
        targets = get_lora_target_modules()
        modules = dict(encoder.named_modules())
        for t in targets:
            assert t in modules, f"Target module not found: {t}"

    def test_target_count(self):
        targets = get_lora_target_modules(stage2_blocks=4, stage3_blocks=3)
        # 4 modules per block (qkv, proj, fc1, fc2) * (4 + 3) blocks = 28
        assert len(targets) == 28

    def test_no_stage1_targets(self):
        targets = get_lora_target_modules()
        stage1_targets = [t for t in targets if t.startswith("stage1")]
        assert len(stage1_targets) == 0


class TestLoRAInjection:

    def test_inject_and_forward(self, encoder):
        model = inject_lora(encoder, rank=16)
        model.eval()
        x = torch.randn(1, 3, 32, 128)
        with torch.no_grad():
            features, lengths = model(x)
        assert features.shape == (1, 32, 384)

    def test_only_lora_trainable(self, encoder):
        model = inject_lora(encoder, rank=16)
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad, f"LoRA param should be trainable: {name}"
            else:
                assert not param.requires_grad, f"Non-LoRA param should be frozen: {name}"

    def test_stage1_frozen(self, encoder):
        model = inject_lora(encoder, rank=16)
        lora_params = [n for n, _ in model.named_parameters() if "lora_" in n]
        stage1_lora = [n for n in lora_params if "stage1" in n]
        assert len(stage1_lora) == 0, f"Stage 1 should have no LoRA: {stage1_lora}"

    def test_different_ranks(self, encoder):
        for rank in [4, 8, 16, 32]:
            enc = LipiEncoder()
            model = inject_lora(enc, rank=rank)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # Trainable params should scale roughly linearly with rank
            assert trainable > 0
            if rank == 16:
                # ~0.64M for rank 16
                assert 0.5e6 < trainable < 0.8e6

    def test_gradient_flow(self, encoder):
        model = inject_lora(encoder, rank=16)
        model.train()
        x = torch.randn(1, 3, 32, 64)
        features, _ = model(x)
        features.sum().backward()

        # LoRA params should have gradients
        lora_grads = 0
        for name, param in model.named_parameters():
            if "lora_" in name and param.grad is not None:
                lora_grads += 1
                assert not torch.isnan(param.grad).any()
        assert lora_grads > 0, "No LoRA params received gradients"


class TestLoRAParamCount:

    def test_estimated_vs_actual(self):
        encoder = LipiEncoder()
        model = inject_lora(encoder, rank=16)
        actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        estimated = get_lora_param_count(rank=16)
        # Allow some tolerance (PEFT may add a few extra params)
        assert abs(actual_trainable - estimated["total"]) / estimated["total"] < 0.05
