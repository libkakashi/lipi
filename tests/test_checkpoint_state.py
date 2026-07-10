import torch

from src.training.checkpoint import normalize_model_state_keys


def test_normalize_model_state_keys_strips_compile_wrapper_prefix():
    tensor = torch.ones(2)
    normalized, changed = normalize_model_state_keys({"_orig_mod.layer.weight": tensor})
    assert changed
    assert list(normalized) == ["layer.weight"]
    assert normalized["layer.weight"] is tensor


def test_normalize_model_state_keys_leaves_plain_state_unchanged():
    state = {"layer.weight": torch.ones(2)}
    normalized, changed = normalize_model_state_keys(state)
    assert not changed
    assert normalized is state
