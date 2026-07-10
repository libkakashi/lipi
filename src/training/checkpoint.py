"""Checkpoint state helpers shared by training, evaluation, and inference."""

from __future__ import annotations


def normalize_model_state_keys(state):
    """Strip the wrapper prefix emitted by ``torch.compile().state_dict()``."""
    prefix = "_orig_mod."
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items()}, True
    return state, False
