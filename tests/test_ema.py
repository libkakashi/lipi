"""Weight EMA tests — tiny models, CPU, fast."""

import torch
import torch.nn as nn

from src.training.ema import ModelEMA


def _model():
    torch.manual_seed(0)
    return nn.Linear(4, 3)


class TestModelEMA:

    def test_shadow_starts_as_copy(self):
        m = _model()
        ema = ModelEMA(m, decay=0.9)
        for n, p in m.named_parameters():
            assert torch.equal(ema.shadow[n], p.detach().float())

    def test_update_lerps_toward_params(self):
        m = _model()
        ema = ModelEMA(m, decay=0.9)
        old = {n: p.detach().clone() for n, p in m.named_parameters()}
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        ema.update(m)
        # First update: warmup decay = min(0.9, 2/11) = 2/11
        d = 2 / 11
        for n, p in m.named_parameters():
            expected = old[n] * d + p.detach() * (1 - d)
            assert torch.allclose(ema.shadow[n], expected, atol=1e-6)

    def test_average_parameters_swaps_and_restores(self):
        m = _model()
        ema = ModelEMA(m, decay=0.9)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(5.0)
        raw = {n: p.detach().clone() for n, p in m.named_parameters()}
        with ema.average_parameters(m):
            for n, p in m.named_parameters():
                assert torch.equal(p.detach(), ema.shadow[n])
        for n, p in m.named_parameters():
            assert torch.equal(p.detach(), raw[n])

    def test_restore_on_exception(self):
        m = _model()
        ema = ModelEMA(m, decay=0.9)
        raw = {n: p.detach().clone() for n, p in m.named_parameters()}
        try:
            with ema.average_parameters(m):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        for n, p in m.named_parameters():
            assert torch.equal(p.detach(), raw[n])

    def test_state_dict_roundtrip(self):
        m = _model()
        ema = ModelEMA(m, decay=0.9)
        for _ in range(3):
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(0.5)
            ema.update(m)
        sd = ema.state_dict()

        ema2 = ModelEMA(_model(), decay=0.5)
        ema2.load_state_dict(sd)
        assert ema2.decay == 0.9
        assert ema2.updates == 3
        for n in sd["shadow"]:
            assert torch.equal(ema2.shadow[n], ema.shadow[n])

    def test_load_skips_mismatched_shapes(self):
        m = _model()
        ema = ModelEMA(m, decay=0.9)
        sd = ema.state_dict()
        sd["shadow"]["weight"] = torch.zeros(7, 7)  # wrong shape
        before = ema.shadow["weight"].clone()
        ema.load_state_dict(sd)
        assert torch.equal(ema.shadow["weight"], before)  # kept fresh shadow
        assert torch.equal(ema.shadow["bias"], sd["shadow"]["bias"])
