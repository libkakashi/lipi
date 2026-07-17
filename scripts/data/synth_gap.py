#!/usr/bin/env python3
"""
Quantify the synth→real domain gap between two sample dumps.

Computes per-image low-level statistics that a model can exploit as domain
shortcuts even when images look similar to the eye (pixel-level cleanliness,
edge morphology, channel correlation), then reports:

  * per-metric medians/IQRs for both sets and the KS distance between the
    two distributions (0 = identical, 1 = disjoint);
  * a domain-classifier AUC: logistic regression on the per-image feature
    vector, 5-fold CV. AUC ≈ 0.5 means synth is statistically
    indistinguishable from real on these features; AUC ≈ 1.0 means a model
    can tell the domains apart from a glance (and will shortcut on it).

Run against dumps made by dump_samples.py at --scale 1 (native resolution —
upscaling contaminates the statistics):

    python scripts/data/synth_gap.py --synth /tmp/px_synth/audit_samples \
        --real /tmp/px_real/audit_samples
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

METRICS = [
    ("bg_noise", "bg noise floor (std, 8x8 paper blocks)"),
    ("rgb_eq", "exact R==G==B pixel fraction"),
    ("extreme", "pixels at pure extremes (<=5 / >=250)"),
    ("soft_edge", "soft-edge fraction (gradual ramps)"),
    ("chan_dec", "channel decorrelation mean |R-G|"),
    ("hist_ent", "gray histogram entropy (bits)"),
    ("ink_level", "ink level (5th pct luma)"),
    ("paper_level", "paper level (95th pct luma)"),
]


def image_features(path: Path) -> dict | None:
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float64)
    if a.size == 0:
        return None
    g = a.mean(2)
    h, w = g.shape
    f = {}
    thr = np.percentile(g, 75)
    bg = g >= thr
    stds = []
    for y in range(0, h - 8, 8):
        for x in range(0, w - 8, 8):
            if bg[y:y + 8, x:x + 8].all():
                stds.append(g[y:y + 8, x:x + 8].std())
    f["bg_noise"] = float(np.median(stds)) if stds else 0.0
    f["rgb_eq"] = float((np.ptp(a, axis=2) == 0).mean())
    f["extreme"] = float(((g >= 250) | (g <= 5)).mean())
    gx = np.abs(np.diff(g, axis=1))
    moving = gx > 8
    f["soft_edge"] = (float((gx[moving] < 60).mean())
                      if moving.sum() > 50 else 0.5)
    f["chan_dec"] = float(np.abs(a[:, :, 0] - a[:, :, 1]).mean())
    hist, _ = np.histogram(g, bins=64, range=(0, 255))
    ph = hist / max(hist.sum(), 1)
    f["hist_ent"] = float(-(ph[ph > 0] * np.log2(ph[ph > 0])).sum())
    f["ink_level"] = float(np.percentile(g, 5))
    f["paper_level"] = float(np.percentile(g, 95))
    return f


def ks_distance(a: list, b: list) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no scipy needed)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def domain_auc(fs: list[dict], fr: list[dict]) -> float:
    """5-fold CV AUC of a logistic regression separating synth from real.

    Plain numpy implementation (gradient descent on standardized features)
    to avoid a sklearn dependency.
    """
    keys = [k for k, _ in METRICS]
    X = np.array([[f[k] for k in keys] for f in fs + fr])
    y = np.array([0] * len(fs) + [1] * len(fr))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    folds = np.array_split(np.arange(len(y)), 5)
    scores = np.zeros(len(y))
    for k in range(5):
        te = folds[k]
        tr = np.setdiff1d(np.arange(len(y)), te)
        wgt = np.zeros(X.shape[1])
        b = 0.0
        for _ in range(500):
            z = X[tr] @ wgt + b
            p = 1 / (1 + np.exp(-z))
            gwgt = X[tr].T @ (p - y[tr]) / len(tr)
            gb = (p - y[tr]).mean()
            wgt -= 0.5 * gwgt
            b -= 0.5 * gb
        scores[te] = X[te] @ wgt + b
    # AUC via rank statistic
    order = np.argsort(scores)
    ranks = np.empty(len(y))
    ranks[order] = np.arange(1, len(y) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))  # direction-free


def collect(d: str) -> list[dict]:
    out = []
    for p in sorted(Path(d).glob("*.png")):
        f = image_features(p)
        if f is not None:
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--real", required=True)
    args = ap.parse_args()

    fs, fr = collect(args.synth), collect(args.real)
    print(f"synth n={len(fs)}   real n={len(fr)}\n")
    print(f"{'metric':<42}{'SYNTH med [IQR]':<26}{'REAL med [IQR]':<26}{'KS':>5}")
    for k, name in METRICS:
        a = [f[k] for f in fs]
        b = [f[k] for f in fr]

        def fmt(v):
            return (f"{np.median(v):7.3f} "
                    f"[{np.percentile(v, 25):6.3f}-{np.percentile(v, 75):6.3f}]")
        print(f"{name:<42}{fmt(a):<26}{fmt(b):<26}{ks_distance(a, b):5.2f}")

    auc = domain_auc(fs, fr)
    print(f"\nDOMAIN-CLASSIFIER AUC: {auc:.3f}   "
          f"(0.5 = indistinguishable, 1.0 = trivially separable)")


if __name__ == "__main__":
    main()
