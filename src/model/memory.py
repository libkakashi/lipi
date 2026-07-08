"""Measured per-bucket batch sizes for LipiMoEEncoder — zero knobs.

Replaces the old predict-and-extrapolate pixel budget: instead of fitting
a linear memory model from tiny probes and extrapolating 100x, run a
worst-case training step at each static bucket width and measure what
this GPU actually does:

  1. Binary-search the largest batch that survives (memory ceiling).
  2. Time a ladder of batch sizes below it and pick the smallest one
     within _RATE_THRESHOLD of peak throughput — bigger batches waste
     VRAM on CPU-bound per-sample work without adding img/s, and cost
     optimizer steps per epoch.

Training then only ever uses those measured (B, W) shapes, so peak
memory is deterministic and OOM is impossible by construction (modulo
_MEM_HEADROOM, which absorbs eager-vs-compiled differences and long-run
allocator drift).

Results are cached on disk keyed by GPU + model config + a fingerprint
of the model/probe source code, so the search (a few minutes) runs once
per machine/config and re-runs itself when the code changes.
"""

import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.training.losses import compute_lid2_loss

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / ".cache" / "capacities"

# Fraction of the measured memory ceiling to use. The probe is worst-case
# (largest vocab, all frames routed, dense targets) and runs eager;
# compiled kernels usually use less, so this only needs to absorb
# allocator drift across a long run of K interleaved shapes.
_MEM_HEADROOM = 0.9
# Backstop for the doubling search on huge GPUs; the throughput knee
# selects far below this in practice.
_MAX_BATCH = 4096
# Smallest batch within this fraction of peak img/s wins.
_RATE_THRESHOLD = 0.95


def _code_fingerprint() -> str:
    """Hash of the source files the measurement depends on.

    Any change to the model or the probe re-runs the search — no manual
    cache invalidation.
    """
    h = hashlib.sha256()
    files = sorted((_REPO_ROOT / "src" / "model").glob("*.py"))
    files.append(_REPO_ROOT / "src" / "training" / "losses.py")
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _probe_step(model, B: int, W: int,
                group_script_vocabs: list[list[int]],
                compute_ctc: bool, two_views: bool):
    """One worst-case training step at (B, W): forward + losses + backward.

    Worst case means: every frame routed to a random expert (blank frames
    would skip expert compute), scripts spread across all experts,
    scheduled routing sampling active, and a CTC term that decodes every
    frame against the largest script vocab with dense targets (L = T/2)
    — the memory ceiling of an all-CJK batch of dense text.
    """
    device = next(model.parameters()).device
    T = W // model.time_downsample
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                 else torch.float16)

    imgs = torch.randint(0, 256, (B, 3, 32, W), device=device,
                         dtype=torch.uint8)
    counts = torch.tensor([max(len(vs), 1) for vs in group_script_vocabs],
                          device=device)
    gl = torch.randint(0, model.num_groups, (B, T), device=device)
    sl = (torch.rand(B, T, device=device) * counts[gl]).long()
    v_max = max(vs for group in group_script_vocabs for vs in group)

    def _fwd():
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            return model(imgs, group_ids=gl, script_ids=sl,
                         compute_ctc=compute_ctc, route_sample_p=0.25)

    # Two-view consistency keeps both forward graphs alive until backward.
    outs = [_fwd(), _fwd()] if two_views else [_fwd()]

    loss = torch.zeros((), device=device)
    for out in outs:
        glog = out["group_logits"]
        Tm = glog.shape[1]
        loss = loss + F.cross_entropy(
            glog.reshape(-1, glog.shape[-1]), gl[:, :Tm].reshape(-1))
        loss = loss + compute_lid2_loss(
            out.get("lid2_logits_per_group", {}), gl[:, :Tm], sl[:, :Tm])
        if compute_ctc:
            for key in ("logits", "inter_logits"):
                lg = out.get(key)
                if lg is None:
                    continue
                # Mirror compute_ctc_loss_segments' peak footprint: a
                # (T, N, V) copy of the logits in compute dtype (the
                # batched per-segment buffer) + its fp32 log-softmax +
                # CTC alphas. All chunks' buffers stay alive in the
                # autograd graph until backward, so the ceiling is every
                # frame under the largest vocab.
                seg = lg.permute(1, 0, 2)[:, :, :v_max].contiguous()
                log_probs = seg.float().log_softmax(dim=-1)
                Tl = seg.shape[0]
                L = max(Tl // 2, 1)
                targets = ((torch.arange(B * L, device=device) % 2) + 1
                           ).view(B, L)
                loss = loss + F.ctc_loss(
                    log_probs, targets,
                    torch.full((B,), Tl, dtype=torch.long, device=device),
                    torch.full((B,), L, dtype=torch.long, device=device),
                    blank=0, reduction="sum", zero_infinity=True)
    loss.backward()


def _search_buckets(model, widths: list[int],
                    group_script_vocabs: list[list[int]],
                    compute_ctc: bool, two_views: bool, ema: bool) -> dict:
    """Per width: memory ceiling (binary search) + throughput knee."""
    device = next(model.parameters()).device
    # Autotune during measurement so its workspace spikes are included,
    # on exactly the (B, W) shapes training will use.
    torch.backends.cudnn.benchmark = True

    total_b = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable_b = sum(p.numel() * p.element_size()
                      for p in model.parameters() if p.requires_grad)
    # Ballast for memory the probe doesn't allocate but training will:
    # Adam m/v (two fp32 copies of trainable params, lazily created at
    # the first optimizer step) and the EMA shadow (one fp32 copy of all
    # params). Params and grads are real during the probe.
    ballast_bytes = 2 * trainable_b + (total_b if ema else 0)
    ballast = torch.empty(max(ballast_bytes, 1), dtype=torch.uint8,
                          device=device)

    was_training = model.training
    model.train()

    def _fits(B, W):
        try:
            _probe_step(model, B, W, group_script_vocabs,
                        compute_ctc, two_views)
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        except RuntimeError as e:
            # cuBLAS/cudnn workspace allocation failures surface as
            # generic RuntimeError, not OutOfMemoryError.
            msg = str(e).lower()
            if "out of memory" in msg or "alloc" in msg:
                return False
            raise
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    def _step_time(B, W, reps=3):
        """Median seconds per step (first call warms up cudnn autotune)."""
        _probe_step(model, B, W, group_script_vocabs, compute_ctc, two_views)
        model.zero_grad(set_to_none=True)
        times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _probe_step(model, B, W, group_script_vocabs,
                        compute_ctc, two_views)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            model.zero_grad(set_to_none=True)
        return sorted(times)[len(times) // 2]

    buckets: dict[int, dict] = {}
    prev = 1
    try:
        # Widest first: a batch that fits at width W fits at any
        # narrower width, so each ceiling seeds the next search.
        for W in sorted(widths, reverse=True):
            t0 = time.time()
            lo, hi = 0, None
            B = max(prev, 1)
            while hi is None:
                if _fits(B, W):
                    lo = B
                    if B >= _MAX_BATCH:
                        break
                    B = min(B * 2, _MAX_BATCH)
                else:
                    hi = B
            if lo == 0:
                raise RuntimeError(
                    f"Capacity search: batch of 1 at width {W} does not "
                    f"fit on this GPU")
            while hi is not None and hi - lo > 1:
                mid = (lo + hi) // 2
                if _fits(mid, W):
                    lo = mid
                else:
                    hi = mid
            mem_max = lo
            prev = mem_max

            # Throughput knee: smallest B within _RATE_THRESHOLD of peak
            # img/s. Above the knee, extra batch size adds no throughput
            # and costs optimizer steps per epoch.
            b_cap = max(1, int(mem_max * _MEM_HEADROOM))
            ladder = []
            b = b_cap
            while b >= 1 and len(ladder) < 4:
                ladder.append(b)
                b //= 2
            rates = {b: b / _step_time(b, W) for b in sorted(ladder)}
            peak = max(rates.values())
            selected = min(b for b, r in rates.items()
                           if r >= _RATE_THRESHOLD * peak)
            buckets[W] = {
                "mem_max": mem_max,
                "batch": selected,
                "rates": {str(b): round(r, 1) for b, r in rates.items()},
            }
            print(f"    W={W}: mem ceiling B={mem_max}, chose B={selected} "
                  f"({rates[selected]:.0f} img/s, "
                  f"peak {peak:.0f}) [{time.time() - t0:.0f}s]")
    finally:
        del ballast
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
        torch.cuda.empty_cache()

    return buckets


def find_bucket_capacities(
    model,
    bucket_widths: list[int],
    group_script_vocabs: list[list[int]],
    compute_ctc: bool = True,
    two_views: bool = False,
    ema: bool = False,
) -> dict[int, int]:
    """Measured batch size per bucket width. No knobs.

    "Fits" and "fast" are measured directly on this GPU with worst-case
    steps — no memory model, no extrapolation, nothing to tune. Results
    are cached under .cache/capacities keyed by GPU + model config +
    source fingerprint, so any hardware or code change re-measures
    automatically. Delete the cache dir to force a re-measure by hand.
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        print("  Capacity search: non-CUDA device — default ~50k px/batch")
        return {int(w): max(1, 50_000 // int(w)) for w in bucket_widths}

    widths = sorted(int(w) for w in bucket_widths)
    trainable_b = sum(p.numel() * p.element_size()
                      for p in model.parameters() if p.requires_grad)
    key = {
        "gpu": torch.cuda.get_device_name(device),
        "total_mem": torch.cuda.get_device_properties(device).total_memory,
        "torch": torch.__version__,
        "code": _code_fingerprint(),
        "config": model.config,
        "compute_ctc": compute_ctc,
        "two_views": two_views,
        "ema": ema,
        "widths": widths,
        "trainable_bytes": trainable_b,  # freeze mode changes the answer
    }
    tag = hashlib.sha256(
        json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = _CACHE_DIR / f"{tag}.json"

    buckets = None
    if cache_path.exists():
        buckets = {int(w): v for w, v in
                   json.loads(cache_path.read_text())["buckets"].items()}
        print(f"  Capacity table: cached ({cache_path.name})")
    if buckets is None:
        print(f"  Capacity search: measuring {len(widths)} bucket widths "
              f"(one-time, cached until code/GPU/config changes)...")
        buckets = _search_buckets(model, widths, group_script_vocabs,
                                  compute_ctc, two_views, ema)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"key": key, "buckets": buckets}, indent=2))
        print(f"  Capacity table: saved {cache_path}")

    caps = {w: int(buckets[w]["batch"]) for w in widths}
    print("  Batch shapes: "
          + ", ".join(f"{caps[w]}x{w}" for w in widths))
    return caps
