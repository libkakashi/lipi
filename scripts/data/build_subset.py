#!/usr/bin/env python3
"""
Build a small, curated training subset from the full real + synth shards.

Three curation principles (see the discussion that motivated this):

  1. Cross-script balance by EFFECTIVE vocabulary, not raw alphabet size.
     Char frequency is Zipfian, so `vocab_size` over-weights large scripts
     (Han's ~3800 glyphs are mostly rare). We sample each script's actual
     text, count how many distinct characters cover P%=95% of occurrences,
     and weight per-script quota by sqrt(effective_vocab) — then floor + cap
     so nothing dominates and nothing starves.

  2. Domain ratio (printed / handwritten / degraded / synth). Real handwriting
     is the only HW source (synth is font-rendered = printed-domain), and it
     exists for only some scripts — so the ratio is a GLOBAL target, filled
     real-first per domain and synth-backfilled. `--goal printed` biases
     toward scanned-document robustness; `--goal general` keeps more HW.

  3. Cap the redundant giants (Tibetan 2.78M, IIIT Indic) at the per-script
     target; keep all real for scarce scripts; 100% synth for the no-real 10.

Output is a drop-in shard root (chunk_* + split-root widths/script_ids sidecars
+ metadata.pt) that LipiStreamingDataset reads like any other.

    # inspect the plan without writing:
    python scripts/data/build_subset.py --real real-v1-48 --synth shards-v9-48 \
        --out subset-v1 --root /vol/data --total 700000 --goal printed --plan
    # build it:
    python scripts/data/build_subset.py ... (drop --plan)
"""

import argparse
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming import MDSWriter, Stream, StreamingDataset  # noqa: E402

from src.taxonomy import GROUPS, SCRIPTS, TAXONOMY_VERSION  # noqa: E402

MDS_COLUMNS = {
    "image": "ndarray:uint8", "label": "str", "script_id": "int",
    "group_id": "int", "target_ids": "ndarray:int64", "target_len": "int",
    "width": "int", "group_labels": "ndarray:int32", "segments": "str",
}

# Source → domain. Real handwriting is the only HW source; synth is printed.
SOURCE_DOMAIN = {
    # handwriting
    "iiit_uc": "hw", "iiit_indic_hw_words": "hw", "iiit_hw_dev": "hw",
    "iiit_hw_telugu": "hw", "iiit_ma_fix": "hw", "casia_hwdb2": "hw",
    "norhand_v3": "hw", "catmus_modern": "hw", "khatt": "hw", "thai_hw": "hw",
    "hhd_ethiopic": "hw", "digital_peter": "hw", "rimes": "hw",
    "teklia_belfort": "hw", "teklia_popp": "hw", "teklia_himanis": "hw",
    "riks_gbg_polis": "hw", "riks_svea": "hw", "riks_krigshovratt": "hw",
    "riks_bergskollegium": "hw", "riks_trolldom": "hw",
    "riks_frihetstiden": "hw", "riks_alvsborg": "hw",
    "tibetan_cursive": "hw",
    # clean / typeset / camera print
    "mozhi": "printed", "tibetan_gbooks": "printed",
    "tibetan_norbuketaka": "printed", "tibetan_uchan": "printed",
    "tibetan_khyentse": "printed", "tibetan_drutsa": "printed",
    "riks_fraktur": "printed", "burmese_bl": "printed",
    "burmese_bl_pseudo": "printed", "bmod": "printed",
    # scanned / woodblock / degraded
    "heidata_fid4sa": "degraded", "gt4histocr": "degraded",
    "tibetan_lhasakanjur": "degraded", "tibetan_derge": "degraded",
    "tibetan_betsug": "degraded",
}

# Per-domain target fractions of each script's quota (real-first, synth fills
# the remainder — so synth share = 1 - sum(real fractions actually met)).
GOAL_FRACTIONS = {
    "printed": {"printed": 0.32, "degraded": 0.14, "hw": 0.12},
    "general": {"printed": 0.24, "degraded": 0.10, "hw": 0.26},
}
COVERAGE = 0.95        # effective-vocab coverage threshold
FREQ_SAMPLE = 1500     # samples per primary script for the char-freq pass


def _nonempty_chunks(split_dir: Path) -> list[Path]:
    out = []
    for c in sorted(split_dir.glob("chunk_*")):
        idx = c / "index.json"
        if idx.exists() and sum(s.get("samples", 0) for s in
                                json.loads(idx.read_text())["shards"]) > 0:
            out.append(c)
    return out


def _chunk_n(chunk: Path) -> int:
    return sum(s.get("samples", 0) for s in
               json.loads((chunk / "index.json").read_text())["shards"])


def _source_of(chunk: Path) -> str:
    name = chunk.name
    if name.startswith("chunk_r_"):
        return name[len("chunk_r_"):].rsplit("_", 1)[0]
    return name.rsplit("_", 1)[0] if name.startswith("chunk_") else name


def load_index(root: Path, dataset: str, split: str, is_synth: bool):
    """Return (chunks, sids, domains) aligned to LipiStreamingDataset order.

    domains[i] is the source-domain of sample i ('synth' for the synth set).
    """
    split_dir = root / dataset / split
    chunks = _nonempty_chunks(split_dir)
    sids = np.load(str(split_dir / "script_ids.npy"))
    domains = np.empty(len(sids), dtype=object)
    off = 0
    for c in chunks:
        n = _chunk_n(c)
        domains[off:off + n] = ("synth" if is_synth
                                else SOURCE_DOMAIN.get(_source_of(c), "hw"))
        off += n
    assert off == len(sids), f"{dataset}/{split}: {off} != sidecar {len(sids)}"
    return chunks, sids, domains


def effective_vocab(root, real, synth, rng) -> dict:
    """Per-script count of distinct chars covering COVERAGE of occurrences,
    from a stratified sample of segment texts across both datasets."""
    counts = defaultdict(Counter)
    for dataset, is_synth in ((real, False), (synth, True)):
        split_dir = root / dataset / "train"
        chunks = _nonempty_chunks(split_dir)
        sids = np.load(str(split_dir / "script_ids.npy"))
        ds = StreamingDataset(streams=[Stream(local=str(c)) for c in chunks],
                              shuffle=False)
        idx_by = defaultdict(list)
        for i, s in enumerate(sids):
            idx_by[int(s)].append(i)
        for s, idxs in idx_by.items():
            for i in rng.sample(idxs, min(FREQ_SAMPLE, len(idxs))):
                for seg in json.loads(ds[i]["segments"]):
                    sc = SCRIPTS[seg["script_id"]] if seg.get(
                        "script_id", -1) < len(SCRIPTS) else None
                    if sc:
                        for ch in unicodedata.normalize("NFC", seg.get("text", "")):
                            counts[sc][ch] += 1
    eff = {}
    for sc in SCRIPTS:
        c = counts.get(sc)
        if not c:
            eff[sc] = 1
            continue
        freqs = sorted(c.values(), reverse=True)
        total = sum(freqs)
        cum, k = 0, 0
        for f in freqs:
            cum += f
            k += 1
            if cum >= COVERAGE * total:
                break
        eff[sc] = k
    return eff


def plan(eff, avail, total, floor, cap, goal,
         real_cap=45000, synth_cap=22000, synth_min=5000):
    """Real-availability-driven allocation.

    Take real generously where it exists (up to real_cap), preferring printed
    real for the printed goal (printed → degraded → hw fill order), then add a
    modest synth coverage floor (capped at synth_cap). This uses our real data
    instead of drowning it in synth for the real-starved big-vocab scripts.
    `total`/`floor`/`cap`/`goal` still set the effective-vocab coverage target
    N that the synth floor aims for on synth-only scripts.
    """
    w = {s: eff[s] ** 0.5 for s in SCRIPTS}
    wsum = sum(w.values())
    targets, quotas = {}, {}
    # domain priority: printed goal takes printed real first; general takes hw.
    order = (("printed", "degraded", "hw") if goal == "printed"
             else ("hw", "printed", "degraded"))
    for s in SCRIPTS:
        N = int(min(max(total * w[s] / wsum, floor), cap))
        targets[s] = N
        av = avail.get(s, {})
        real_budget = min(real_cap, sum(av.get(d, 0)
                                        for d in ("printed", "degraded", "hw")))
        remaining = real_budget
        q = {"printed": 0, "degraded": 0, "hw": 0}
        for dom in order:
            take = min(av.get(dom, 0), remaining)
            q[dom] = take
            remaining -= take
        # synth floor: fill toward the coverage target, but never balloon.
        q["synth"] = min(av.get("synth", 0), synth_cap,
                         max(N - real_budget, synth_min))
        quotas[s] = q
    return targets, quotas


def build(root, real, synth, out, quotas, val_frac, rng):
    out_root = root / out
    for split in ("train", "val"):
        # scale quotas for val
        sq = quotas if split == "train" else {
            s: {d: max(int(v * val_frac), 0) for d, v in q.items()}
            for s, q in quotas.items()}
        # pick global indices per dataset for this split
        picks = {}  # dataset -> set(global idx)
        for dataset, is_synth in ((real, False), (synth, True)):
            _, sids, domains = load_index(root, dataset, split, is_synth)
            pool = defaultdict(list)
            for i in range(len(sids)):
                pool[(int(sids[i]), domains[i])].append(i)
            chosen = set()
            for s in range(len(SCRIPTS)):
                for dom, want in sq.get(SCRIPTS[s], {}).items():
                    cand = pool.get((s, dom), [])
                    if want and cand:
                        chosen.update(rng.sample(cand, min(want, len(cand))))
            picks[dataset] = chosen

        out_split = out_root / split / "chunk_000"
        if out_split.exists():
            import shutil
            shutil.rmtree(out_split)
        out_split.mkdir(parents=True, exist_ok=True)
        widths, sidl = [], []
        n_written = 0
        with MDSWriter(out=str(out_split), columns=MDS_COLUMNS,
                       size_limit=1 << 26) as w:
            for dataset, is_synth in ((real, False), (synth, True)):
                chosen = picks[dataset]
                if not chosen:
                    continue
                chunks = _nonempty_chunks(root / dataset / split)
                # ONE StreamingDataset over all chunks — creating one per chunk
                # leaks FDs/shared-memory ("Too many open files"). Access the
                # chosen indices in sorted order so each shard is touched once,
                # forward.
                ds = StreamingDataset(
                    streams=[Stream(local=str(c)) for c in chunks],
                    shuffle=False, batch_size=1)
                for g in sorted(chosen):
                    s = ds[g]
                    w.write(s)
                    widths.append(int(s["width"]))
                    sidl.append(int(s["script_id"]))
                    n_written += 1
                del ds
        np.save(str(out_root / split / "widths.npy"),
                np.array(widths, dtype=np.int32))
        np.save(str(out_root / split / "script_ids.npy"),
                np.array(sidl, dtype=np.int32))
        print(f"  {split}: wrote {n_written} samples", flush=True)

    import torch
    torch.save({"active_scripts": list(SCRIPTS), "active_groups": list(GROUPS),
                "taxonomy_version": TAXONOMY_VERSION, "height": 48,
                "max_width": 2048}, out_root / "metadata.pt")
    print(f"built {out_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--synth", required=True)
    p.add_argument("--out", default="subset-v1")
    p.add_argument("--root", default="/vol/data")
    p.add_argument("--total", type=int, default=700000)
    p.add_argument("--floor", type=int, default=15000)
    p.add_argument("--cap", type=int, default=55000)
    p.add_argument("--real-cap", type=int, default=45000,
                   help="max real samples taken per script")
    p.add_argument("--synth-cap", type=int, default=22000,
                   help="max synth coverage samples per script")
    p.add_argument("--synth-min", type=int, default=5000,
                   help="min synth coverage per script")
    p.add_argument("--goal", choices=("printed", "general"), default="printed")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--plan", action="store_true")
    args = p.parse_args()

    rng = random.Random(0)
    root = Path(args.root)

    # Raise the open-file limit — StreamingDataset opens many shard files.
    import resource
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    # availability per (script, domain) from the train sidecars
    avail = defaultdict(lambda: defaultdict(int))
    for dataset, is_synth in ((args.real, False), (args.synth, True)):
        _, sids, domains = load_index(root, dataset, "train", is_synth)
        for i in range(len(sids)):
            avail[SCRIPTS[int(sids[i])]][domains[i]] += 1

    print("computing effective vocabulary (sampled char-freq)...", flush=True)
    eff = effective_vocab(root, args.real, args.synth, rng)
    targets, quotas = plan(eff, avail, args.total, args.floor, args.cap,
                           args.goal, args.real_cap, args.synth_cap,
                           args.synth_min)

    print(f"\n{'script':<12}{'effVoc':>7}{'target':>8}"
          f"{'printed':>9}{'degrad':>8}{'hw':>7}{'synth':>8}{'realAvl':>9}")
    tot = defaultdict(int)
    for s in sorted(SCRIPTS, key=lambda x: -targets[x]):
        q = quotas[s]
        realav = sum(v for d, v in avail[s].items() if d != "synth")
        print(f"{s:<12}{eff[s]:>7}{targets[s]:>8}{q['printed']:>9}"
              f"{q['degraded']:>8}{q['hw']:>7}{q['synth']:>8}{realav:>9}")
        for d, v in q.items():
            tot[d] += v
    gt = sum(tot.values())
    print(f"\n  TOTAL {gt:,}  |  " + "  ".join(
        f"{d}={tot[d]:,} ({100*tot[d]/max(gt,1):.0f}%)" for d in
        ("printed", "degraded", "hw", "synth")))
    real_share = 100 * (gt - tot["synth"]) / max(gt, 1)
    print(f"  real:synth ≈ {real_share:.0f}:{100-real_share:.0f}")

    if args.plan:
        return
    print("\nbuilding subset...", flush=True)
    build(root, args.real, args.synth, args.out, quotas, args.val_frac, rng)


if __name__ == "__main__":
    main()
