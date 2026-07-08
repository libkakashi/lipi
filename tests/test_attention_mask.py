"""WindowedAttention mask tests (SDPA/CPU path).

Brute-force reference: per-position softmax over allowed keys only —
wrap-consistent (shift mask) and non-pad (width padding mask). The module
must match it, which fails if pad keys leak into real queries' attention.
"""

import torch
import torch.nn.functional as F

from src.model.blocks import WindowedAttention


def reference_windowed_attention(mod, x, h, w):
    B, N, C = x.shape
    H, hd = mod.num_heads, mod.head_dim
    win_w, win_h = mod.window_w, mod.window_h
    shift_w = mod.shift_w if mod.shift else 0

    xg = x.reshape(B, h, w, C)
    pad_w = (win_w - w % win_w) % win_w
    wp = w + pad_w
    xp = F.pad(xg, (0, 0, 0, pad_w))
    if shift_w > 0:
        xp = torch.roll(xp, -shift_w, dims=2)

    qkv = mod.qkv(xp).reshape(B, h, wp, 3, H, hd)
    q, k, v = qkv.unbind(3)
    q = mod.q_norm(q)
    k = mod.k_norm(k)
    scale = hd ** -0.5

    cols = torch.arange(wp)
    pad_flag = ((cols + shift_w) % wp) >= w
    if shift_w > 0:
        wrap_flag = cols >= (wp - shift_w)
    else:
        wrap_flag = torch.zeros(wp, dtype=torch.bool)

    out = torch.zeros(B, h, wp, H, hd)
    for b in range(B):
        for wh in range(h // win_h):
            for wi in range(wp // win_w):
                pos = [(wh * win_h + r, wi * win_w + c)
                       for r in range(win_h) for c in range(win_w)]
                for qi, (rq, cq) in enumerate(pos):
                    for hh in range(H):
                        scores, vals = [], []
                        for ki, (rk, ck) in enumerate(pos):
                            if wrap_flag[cq] != wrap_flag[ck]:
                                continue
                            if pad_flag[ck] and not pad_flag[cq]:
                                continue
                            s = (q[b, rq, cq, hh] @ k[b, rk, ck, hh]) * scale
                            s = s + mod.rel_pos_bias[
                                mod.rel_pos_index[qi, ki], hh]
                            scores.append(s)
                            vals.append(v[b, rk, ck, hh])
                        aw = torch.softmax(torch.stack(scores), 0)
                        out[b, rq, cq, hh] = \
                            (aw[:, None] * torch.stack(vals)).sum(0)

    out = mod.proj(out.reshape(B, h, wp, C))
    if shift_w > 0:
        out = torch.roll(out, shift_w, dims=2)
    return out[:, :, :w, :].reshape(B, h * w, C)


def _check(shift, h, window_h, w, window_w=8, dim=8, heads=2, seed=0):
    torch.manual_seed(seed)
    mod = WindowedAttention(dim=dim, num_heads=heads,
                            window_h=window_h, window_w=window_w, shift=shift)
    mod.eval()
    x = torch.randn(1, h * w, dim)
    with torch.no_grad():
        got = mod(x, h, w)
        want = reference_windowed_attention(mod, x, h, w)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


class TestPadMasking:

    def test_no_shift_with_pad(self):
        _check(shift=False, h=1, window_h=1, w=13)  # pad 3

    def test_shift_with_pad(self):
        _check(shift=True, h=1, window_h=1, w=13)  # pad 3, wrap 4

    def test_shift_no_pad_regression(self):
        _check(shift=True, h=1, window_h=1, w=16)  # exact fit

    def test_no_shift_no_pad_regression(self):
        _check(shift=False, h=1, window_h=1, w=16)

    def test_window_h2_with_pad(self):
        _check(shift=False, h=2, window_h=2, w=11)  # pad 5, 2D window

    def test_shift_window_h2_with_pad(self):
        _check(shift=True, h=2, window_h=2, w=11)

    def test_pad_keys_do_not_leak(self):
        """Direct probe: real-position outputs must be identical whether
        computed with masking (module) or with pad columns physically
        absent — checked via a window-exact width where the last window
        holds only 2 real columns."""
        _check(shift=False, h=1, window_h=1, w=10, window_w=8)  # pad 6
