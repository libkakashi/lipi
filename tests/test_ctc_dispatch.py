"""Equivalence of the grouped CTC dispatch/loss vs the old implementations.

The grouped gather/scatter in _ctc_logits/_ctc_feedback and the single
advanced-index gather in compute_ctc_loss_segments must match the old
per-script boolean-mask / per-segment slice-copy code exactly (forward and
gradients) — the rewrites are purely performance changes: the old code's
backward allocated + accumulated full-size (B, T, max_vocab) buffers
hundreds of times per step and dominated training."""

import torch
import torch.nn.functional as F

import src.training.losses as losses_mod
from src.model.blocks import _per_sample_key_lens
from src.model.encoder import LipiMoEEncoder


def _make_model():
    torch.manual_seed(0)
    return LipiMoEEncoder(
        dim=128,
        num_groups=2,
        group_script_vocab_sizes=[[60], [80, 120]],
        group_script_names=[["s0"], ["s1", "s2"]],
        drop_path_rate=0.0,
    )


def _ref_ctc_logits(model, x, flat_scripts, script_lens_cpu):
    """Old implementation: per-script boolean-mask scatter."""
    B, T, _ = x.shape
    max_vocab = max(m.max_vocab for m in model.ctc_modules)
    logits = torch.zeros(B, T, max_vocab, device=x.device, dtype=x.dtype)
    for (g, s), flat_id in model._flat_script_id.items():
        if not any(script_lens_cpu[b][flat_id] > 0 for b in range(B)):
            continue
        mask = (flat_scripts == flat_id)
        head = model.ctc_modules[g].heads[s]
        vs = head.vocab_size
        head_out = head(x[mask]).to(logits.dtype)
        logits[mask] = F.pad(head_out, (0, max_vocab - vs))
    return logits


def _ref_ctc_feedback(model, inter_logits, flat_scripts, script_lens_cpu):
    """Old implementation: per-script boolean-mask gather + scatter."""
    B, T, _ = inter_logits.shape
    fb = torch.zeros(B, T, model.enc_out_dim,
                     device=inter_logits.device, dtype=inter_logits.dtype)
    for (g, s), flat_id in model._flat_script_id.items():
        if not any(script_lens_cpu[b][flat_id] > 0 for b in range(B)):
            continue
        mask = (flat_scripts == flat_id)
        head = model.ctc_modules[g].heads[s]
        vs = head.vocab_size
        post = inter_logits[mask][:, :vs].softmax(dim=-1)
        fb[mask] = (post @ head.proj.weight.to(post.dtype)).to(fb.dtype)
    return fb


def _routing(model, B, T, seed, blank_frac=0.25):
    torch.manual_seed(seed)
    flat = torch.randint(0, model.total_scripts, (B, T))
    flat[torch.rand(B, T) < blank_frac] = -1  # blank / unrouted
    lens = _per_sample_key_lens(flat, model.total_scripts).tolist()
    return flat, lens


def test_ctc_logits_matches_reference():
    model = _make_model()
    B, T = 3, 24
    flat, lens = _routing(model, B, T, seed=1)
    torch.manual_seed(2)
    x = torch.randn(B, T, 128)

    ref = _ref_ctc_logits(model, x, flat, lens)
    got = model._ctc_logits(x, flat, lens)
    assert torch.allclose(got, ref, atol=1e-6), (got - ref).abs().max().item()
    # Blank frames stay exactly zero
    assert torch.equal(got[flat == -1], torch.zeros_like(got[flat == -1]))


def test_ctc_logits_gradients_match():
    model = _make_model()
    B, T = 2, 16
    flat, lens = _routing(model, B, T, seed=3)
    torch.manual_seed(4)
    x0 = torch.randn(B, T, 128)
    upstream = torch.randn(B, T, max(m.max_vocab for m in model.ctc_modules))

    def run(fn):
        model.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        out = fn(x)
        (out * upstream).sum().backward()
        head = model.ctc_modules[1].heads[0]
        return x.grad.clone(), head.proj.weight.grad.clone()

    gx_ref, gw_ref = run(lambda x: _ref_ctc_logits(model, x, flat, lens))
    gx_new, gw_new = run(lambda x: model._ctc_logits(x, flat, lens))
    assert torch.allclose(gx_new, gx_ref, atol=1e-5)
    assert torch.allclose(gw_new, gw_ref, atol=1e-5)


def test_ctc_feedback_matches_reference():
    model = _make_model()
    B, T = 3, 20
    flat, lens = _routing(model, B, T, seed=5)
    max_vocab = max(m.max_vocab for m in model.ctc_modules)
    torch.manual_seed(6)
    il0 = torch.randn(B, T, max_vocab)
    upstream = torch.randn(B, T, model.enc_out_dim)

    def run(fn):
        model.zero_grad(set_to_none=True)
        il = il0.clone().requires_grad_(True)
        out = fn(il)
        (out * upstream).sum().backward()
        return out.detach(), il.grad.clone()

    out_ref, g_ref = run(lambda il: _ref_ctc_feedback(model, il, flat, lens))
    out_new, g_new = run(lambda il: model._ctc_feedback(il, flat, lens))
    assert torch.allclose(out_new, out_ref, atol=1e-6)
    assert torch.allclose(g_new, g_ref, atol=1e-5)
    assert torch.equal(out_new[flat == -1],
                       torch.zeros_like(out_new[flat == -1]))


def test_ctc_logits_all_blank():
    model = _make_model()
    B, T = 2, 8
    flat = torch.full((B, T), -1)
    lens = _per_sample_key_lens(flat, model.total_scripts).tolist()
    x = torch.randn(B, T, 128)
    out = model._ctc_logits(x, flat, lens)
    assert torch.equal(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# compute_ctc_loss_segments: gather rewrite vs old per-segment copy loop
# ---------------------------------------------------------------------------

def _ref_loss_segments(logits, segments_batch, enc_lengths,
                       group_script_names, group_script_vocabs):
    """Old implementation (per-segment slice-copy into a padded buffer)."""
    device = logits.device
    B, T, _ = logits.shape
    buckets = {}
    for b in range(B):
        for seg in segments_batch[b]:
            text, g, s = seg["text"], seg["group_id"], seg["script_id"]
            offset_px, width_px = seg["offset"], seg["width"]
            if not text or width_px == 0:
                continue
            frame_start = offset_px // 4
            frame_end = min((offset_px + width_px + 3) // 4, T)
            seg_len = frame_end - frame_start
            if seg_len < 1:
                continue
            script_name = group_script_names[g][s]
            ids = list(losses_mod._encode_text_cached(text, script_name))
            if not ids:
                continue
            n_rep = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
            if seg_len < len(ids) + n_rep:
                continue
            vs = group_script_vocabs[g][s]
            buckets.setdefault((g, s), []).append({
                "b": b, "frame_start": frame_start, "frame_end": frame_end,
                "seg_len": seg_len, "ids": ids, "vs": vs,
            })

    total = torch.zeros(1, device=device)
    chars = 0
    for (g, s), segs in buckets.items():
        vs = segs[0]["vs"]
        segs = sorted(segs, key=lambda sg: sg["seg_len"])
        max_T = max(sg["seg_len"] for sg in segs)
        N = len(segs)
        batched = torch.zeros(max_T, N, vs, device=device, dtype=logits.dtype)
        in_l = torch.zeros(N, dtype=torch.long, device=device)
        tg_l = torch.zeros(N, dtype=torch.long, device=device)
        cat = []
        for i, sg in enumerate(segs):
            b, fs, fe = sg["b"], sg["frame_start"], sg["frame_end"]
            batched[:sg["seg_len"], i, :] = logits[b, fs:fe, :vs]
            in_l[i] = sg["seg_len"]
            tg_l[i] = len(sg["ids"])
            cat.extend(sg["ids"])
        lp = batched.float().log_softmax(dim=-1)
        tg = torch.tensor(cat, dtype=torch.long, device=device)
        total = total + F.ctc_loss(lp, tg, in_l, tg_l, blank=0,
                                   reduction="sum", zero_infinity=True)
        chars += len(cat)
    return total / chars if chars else total


def test_ctc_loss_segments_matches_reference(monkeypatch):
    # Isolate tensor assembly from real text encoding.
    monkeypatch.setattr(losses_mod, "_encode_text_cached",
                        lambda text, script: tuple((ord(c) % 40) + 1
                                                   for c in text))
    names = [["s0"], ["s1", "s2"]]
    vocabs = [[60], [80, 120]]
    B, T = 3, 48
    max_vocab = 120
    torch.manual_seed(7)
    logits0 = torch.randn(B, T, max_vocab)

    segments = [
        [  # image 0: two segments, different scripts
            {"group_id": 0, "script_id": 0, "text": "hello", "offset": 0,
             "width": 80},
            {"group_id": 1, "script_id": 1, "text": "worlds", "offset": 84,
             "width": 100},
        ],
        [  # image 1: one segment near the right edge (exercises clamping)
            {"group_id": 1, "script_id": 0, "text": "abc", "offset": 148,
             "width": 44},
        ],
        [  # image 2: same script as image 0 (multi-segment bucket) + skips
            {"group_id": 0, "script_id": 0, "text": "xyzw", "offset": 8,
             "width": 72},
            {"group_id": 0, "script_id": 0, "text": "", "offset": 100,
             "width": 20},           # empty text → skipped
            {"group_id": 0, "script_id": 0, "text": "toolongtext" * 5,
             "offset": 120, "width": 8},  # too long for frames → skipped
        ],
    ]
    enc_lengths = torch.full((B,), T, dtype=torch.long)

    def run(fn):
        lg = logits0.clone().requires_grad_(True)
        loss = fn(lg, segments, enc_lengths, names, vocabs)
        loss.backward()
        return loss.detach(), lg.grad.clone()

    loss_ref, grad_ref = run(_ref_loss_segments)
    loss_new, grad_new = run(losses_mod.compute_ctc_loss_segments)

    assert torch.allclose(loss_new, loss_ref, atol=1e-5), \
        (loss_new.item(), loss_ref.item())
    assert torch.allclose(grad_new, grad_ref, atol=1e-5), \
        (grad_new - grad_ref).abs().max().item()
