"""
Exponential moving average of model weights.

Evaluating/exporting from the averaged weights is consistently a little
more accurate and much more stable across epochs than the raw weights —
especially on noisy/augmented data — at zero inference cost.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn


class ModelEMA:
    """EMA of model parameters, updated once per optimizer step.

    Keyed on parameter names of the module it was constructed with, so
    construct and update it with the *uncompiled* module (torch.compile
    wrappers prefix names with `_orig_mod.` but share the same tensors).

    Shadow params are fp32 copies on the model's device. Decay warms up
    as (1 + n) / (10 + n) so early averages track the fast-moving young
    model instead of anchoring to the random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.updates = 0
        self.shadow = {n: p.detach().clone().float()
                       for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p.detach().float(), 1.0 - d)

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """Temporarily swap the EMA weights into the model (for eval)."""
        backup = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                backup[n] = p.detach().clone()
                p.copy_(self.shadow[n].to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(backup[n])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {n: t.cpu() for n, t in self.shadow.items()},
        }

    def load_state_dict(self, sd: dict):
        """Load EMA state, skipping shape-mismatched entries (e.g. CTC
        heads after a vocab change) — those keep their fresh-init shadow."""
        self.decay = sd.get("decay", self.decay)
        self.updates = sd.get("updates", 0)
        skipped = []
        for n, t in sd.get("shadow", {}).items():
            cur = self.shadow.get(n)
            if cur is None or cur.shape != t.shape:
                skipped.append(n)
                continue
            self.shadow[n] = t.to(device=cur.device, dtype=cur.dtype)
        if skipped:
            print(f"  [ema] skipped {len(skipped)} mismatched shadow entries")
