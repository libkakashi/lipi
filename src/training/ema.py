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
        # Aligned flat lists for the per-step foreach update: the model has
        # ~900 parameter tensors (tiny MoE expert weights), and one lerp_
        # launch per tensor cost ~4-6 ms/step in kernel-launch overhead.
        self._param_list = [p for _, p in model.named_parameters()]
        self._shadow_list = [self.shadow[n]
                             for n, _ in model.named_parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        if self._param_list and self._param_list[0].dtype == torch.float32:
            torch._foreach_lerp_(self._shadow_list, self._param_list, 1.0 - d)
        else:
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
            # copy_ (not rebind) keeps the foreach update's aligned list
            # pointing at the live shadow tensors.
            cur.copy_(t.to(device=cur.device, dtype=cur.dtype))
        if skipped:
            print(f"  [ema] skipped {len(skipped)} mismatched shadow entries")
