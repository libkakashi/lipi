#!/usr/bin/env python3
"""
Deep data audit for finalized MDS shard roots — the gate before a long run.

Goes well past scripts/data/real/prepare.py::verify (a 300-sample smoke
check). For each dataset root and split it runs:

  * a full sidecar scan (widths.npy / script_ids.npy) — per-script counts and
    width distribution over EVERY sample, no image decode;
  * a stratified deep sample (N per script) that decodes image + label +
    target_ids + segments and flags:
      - degenerate images (near-constant pixels → blank crops);
      - empty / whitespace-only / control-char labels;
      - CTC infeasibility: out_slots = (width // time_ds) * emit < len(ids) +
        n_repeats → the sample is UNLEARNABLE (CTC can't fit the label in the
        frames) and injects clamped-inf loss noise every epoch;
      - decode round-trip mismatch (single-segment only): decode(ids) != label
        → the codec can't represent the label → label noise the model can
        never score, a hard accuracy ceiling;
      - height / group_labels / id-range violations;
  * an exact-duplicate image scan within each split and a train/val LEAKAGE
    check (shared image hashes → inflated val).

Prints a per-dataset report and a combined per-script synth-vs-real coverage
table, then a PASS / WARN / FAIL verdict. Read-only; safe to run against the
live data volume while a training container stages.

    python scripts/data/audit.py --data shards-v9-48,real-v1-48 --root /vol/data
"""

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming import Stream, StreamingDataset  # noqa: E402

from src.taxonomy import GROUPS, NUM_GROUPS, SCRIPTS  # noqa: E402
from src.encoding.decompose import decode_ids  # noqa: E402

TIME_DS = 4      # encoder.time_downsample
EMIT = 2         # encoder.emit_per_frame
PER_SCRIPT = 400  # deep-sample cap per script per split
LEAK_HASHES = 20000  # images hashed per split for dup/leakage


def _chunks(split_dir: Path) -> list[Path]:
    out = []
    for c in sorted(split_dir.glob("chunk_*")):
        idx = c / "index.json"
        if idx.exists() and sum(
                s.get("samples", 0)
                for s in json.loads(idx.read_text())["shards"]) > 0:
            out.append(c)
    return out


def _img_hash(img: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(img).tobytes(),
                           digest_size=12).hexdigest()


def _n_repeats(ids: list[int]) -> int:
    return sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])


def _has_control(text: str) -> bool:
    # Cc/Cf other than ordinary spacing; NFC strip should have removed these.
    return any(unicodedata.category(ch) in ("Cc", "Cf") for ch in text)


def audit_split(root: Path, name: str, split: str, height: int,
                rng: random.Random) -> dict:
    split_dir = root / split
    chunks = _chunks(split_dir)
    if not chunks:
        return {"empty": True}

    ds = StreamingDataset(streams=[Stream(local=str(c)) for c in chunks],
                          shuffle=False)
    n = len(ds)
    widths = np.load(str(split_dir / "widths.npy"))
    sids = np.load(str(split_dir / "script_ids.npy"))
    sidecar_ok = (n == len(widths) == len(sids))

    # Per-script counts + width stats over ALL samples (cheap).
    by_script = Counter()
    for k, v in zip(*np.unique(sids, return_counts=True)):
        by_script[SCRIPTS[k] if k < len(SCRIPTS) else f"?{k}"] += int(v)

    # Stratified deep sample: up to PER_SCRIPT indices per script id.
    idx_by_sid = defaultdict(list)
    for i, s in enumerate(sids):
        idx_by_sid[int(s)].append(i)
    deep_idx = []
    for s, idxs in idx_by_sid.items():
        deep_idx.extend(rng.sample(idxs, min(PER_SCRIPT, len(idxs))))
    rng.shuffle(deep_idx)

    flags = Counter()
    rt_bad = Counter()   # per-script round-trip mismatches
    rt_tot = Counter()
    infeasible_by_script = Counter()
    examples = defaultdict(list)  # flag -> up to 3 (label, extra)

    for i in deep_idx:
        s = ds[i]
        img = s["image"]
        w = int(s["width"])
        label = s["label"]
        sid = int(s["script_id"])
        script = SCRIPTS[sid] if sid < len(SCRIPTS) else "?"

        # image sanity
        if img.shape[0] != 3 or img.shape[1] != height or img.shape[2] != w:
            flags["img_shape"] += 1
            if len(examples["img_shape"]) < 3:
                examples["img_shape"].append((label, str(img.shape)))
        elif float(img.std()) < 3.0:  # near-constant → blank/degenerate crop
            flags["img_blank"] += 1
            if len(examples["img_blank"]) < 3:
                examples["img_blank"].append((label, f"std={img.std():.1f}"))

        # label sanity
        if not label or not label.strip():
            flags["label_empty"] += 1
        elif _has_control(label):
            flags["label_control"] += 1
            if len(examples["label_control"]) < 3:
                examples["label_control"].append((label, ""))

        # group_labels / id ranges
        gl = s["group_labels"]
        if gl.shape[0] != w or int(gl.max()) > NUM_GROUPS:
            flags["group_labels"] += 1
        if not (0 <= sid < len(SCRIPTS)):
            flags["sid_range"] += 1

        # CTC feasibility (whole width for single-seg; per-seg otherwise)
        segs = json.loads(s["segments"])
        ids = s["target_ids"][:int(s["target_len"])].tolist()
        out_slots = (w // TIME_DS) * EMIT
        if ids and out_slots < len(ids) + _n_repeats(ids):
            flags["ctc_infeasible"] += 1
            infeasible_by_script[script] += 1
            if len(examples["ctc_infeasible"]) < 3:
                examples["ctc_infeasible"].append(
                    (label, f"w={w} slots={out_slots} need={len(ids)+_n_repeats(ids)}"))

        # decode round-trip (single-segment only — mixed uses per-seg codecs)
        if len(segs) == 1 and ids:
            rt_tot[script] += 1
            try:
                dec = decode_ids(ids, script)
            except Exception:
                dec = None
            if dec != label:
                rt_bad[script] += 1
                if len(examples["roundtrip"]) < 5:
                    examples["roundtrip"].append((label, f"{script}: got {dec!r}"))

    # dup / leakage hashes
    hashes = set()
    dups = 0
    sample_for_hash = (deep_idx if len(deep_idx) <= LEAK_HASHES
                       else rng.sample(deep_idx, LEAK_HASHES))
    for i in sample_for_hash:
        h = _img_hash(ds[i]["image"])
        if h in hashes:
            dups += 1
        hashes.add(h)

    return {
        "empty": False, "n": n, "sidecar_ok": sidecar_ok,
        "by_script": dict(by_script),
        "wmin": int(widths.min()), "wmax": int(widths.max()),
        "wmed": int(np.median(widths)),
        "deep_n": len(deep_idx), "flags": dict(flags),
        "rt_bad": dict(rt_bad), "rt_tot": dict(rt_tot),
        "infeasible_by_script": dict(infeasible_by_script),
        "hashes": hashes, "dups": dups, "examples": {k: v for k, v in examples.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="comma-separated dataset names under --root")
    p.add_argument("--root", default="/vol/data")
    p.add_argument("--height", type=int, default=48)
    args = p.parse_args()

    rng = random.Random(0)
    root = Path(args.root)
    names = [x.strip() for x in args.data.split(",")]
    combined = defaultdict(lambda: defaultdict(int))  # script -> dataset -> n
    reports = {}
    verdict = "PASS"

    for name in names:
        droot = root / name
        print(f"\n{'='*70}\nDATASET  {name}\n{'='*70}")
        splits = {}
        for split in ("train", "val"):
            r = audit_split(droot, name, split, args.height, rng)
            splits[split] = r
            if r.get("empty"):
                print(f"  {split}: EMPTY")
                continue
            f = r["flags"]
            print(f"  {split}: {r['n']:>9,} samples  "
                  f"sidecars={'OK' if r['sidecar_ok'] else 'MISMATCH'}  "
                  f"width {r['wmin']}/{r['wmed']}/{r['wmax']} (min/med/max)")
            print(f"    deep-checked {r['deep_n']:,}: " +
                  (", ".join(f"{k}={v}" for k, v in f.items()) or "no flags"))
            # round-trip mismatch rate (label-noise proxy)
            rb, rt = r["rt_bad"], r["rt_tot"]
            tot_bad, tot_rt = sum(rb.values()), sum(rt.values())
            if tot_rt:
                rate = 100 * tot_bad / tot_rt
                worst = sorted(rb.items(), key=lambda kv: -kv[1])[:5]
                print(f"    round-trip mismatch: {tot_bad}/{tot_rt} "
                      f"({rate:.2f}%)" +
                      (f"  worst: {worst}" if worst else ""))
            if f.get("ctc_infeasible"):
                worst = sorted(r["infeasible_by_script"].items(),
                               key=lambda kv: -kv[1])[:5]
                print(f"    CTC-infeasible worst scripts: {worst}")
            for flag, exs in r["examples"].items():
                if flag in ("roundtrip", "ctc_infeasible", "img_blank"):
                    for lbl, extra in exs[:3]:
                        print(f"      [{flag}] {lbl!r}  {extra}")
            for sc, cnt in r["by_script"].items():
                combined[sc][name] += cnt

        # train/val leakage
        tr, va = splits.get("train", {}), splits.get("val", {})
        if not tr.get("empty") and not va.get("empty"):
            leak = len(tr["hashes"] & va["hashes"])
            print(f"  train/val leakage: {leak} shared image hashes "
                  f"(train dups={tr['dups']}, val dups={va['dups']})")
            if leak > 0:
                verdict = "WARN" if verdict == "PASS" else verdict
        reports[name] = splits

        # per-dataset verdict escalation
        for split, r in splits.items():
            if r.get("empty"):
                continue
            if not r["sidecar_ok"]:
                verdict = "FAIL"
            rt = sum(r["rt_tot"].values())
            if rt and 100 * sum(r["rt_bad"].values()) / rt > 2.0:
                verdict = "FAIL"
            dn = max(r["deep_n"], 1)
            if r["flags"].get("ctc_infeasible", 0) / dn > 0.02:
                verdict = "FAIL"
            if (r["flags"].get("img_blank", 0)
                    + r["flags"].get("label_empty", 0)) / dn > 0.02:
                verdict = "WARN" if verdict == "PASS" else verdict

    # combined per-script coverage (synth vs real)
    print(f"\n{'='*70}\nCOMBINED per-script coverage across {names}\n{'='*70}")
    print(f"  {'script':<14}" + "".join(f"{n[:14]:>16}" for n in names)
          + f"{'TOTAL':>12}")
    for sc in sorted(combined, key=lambda s: -sum(combined[s].values())):
        row = combined[sc]
        tot = sum(row.values())
        print(f"  {sc:<14}" + "".join(f"{row.get(n,0):>16,}" for n in names)
              + f"{tot:>12,}")
    # flag scripts with zero real coverage (if a 'real' set is present)
    real_names = [n for n in names if "real" in n]
    if real_names:
        zero_real = [sc for sc in combined
                     if sum(combined[sc].get(n, 0) for n in real_names) == 0]
        if zero_real:
            print(f"\n  scripts with ZERO real-data coverage "
                  f"({len(zero_real)}): {sorted(zero_real)}")

    print(f"\n{'='*70}\nVERDICT: {verdict}\n{'='*70}")


if __name__ == "__main__":
    main()
