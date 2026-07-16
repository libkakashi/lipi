#!/usr/bin/env python3
"""
Dump a source-stratified sample of real crops as viewable PNGs + a label
manifest, for human/agent visual inspection (label<->image pairing, quality,
domain). Reads a finalized MDS root; writes {out}/NNN_{source}.png and
{out}/manifest.jsonl ({file, source, script, label, width}).

Stratifies by chunk source prefix (chunk_r_{source}_N) so every ingestion
source is represented regardless of its share of the corpus. Upscales each
crop (nearest, honest — no interpolation hiding artifacts) so 48px-tall crops
are legible.

    python scripts/data/dump_samples.py --data real-v1-48 --root /vol/data \
        --out /vol/assets/audit_samples --per-source 2 --scale 3
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming import Stream, StreamingDataset  # noqa: E402

from src.taxonomy import SCRIPTS  # noqa: E402

CHUNK_RE = re.compile(r"^chunk_r_(.+)_\d+$")


def _source_of(chunk: Path) -> str:
    m = CHUNK_RE.match(chunk.name)
    return m.group(1) if m else chunk.name


def _nonempty(chunk: Path) -> bool:
    idx = chunk / "index.json"
    return idx.exists() and sum(
        s.get("samples", 0)
        for s in json.loads(idx.read_text())["shards"]) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--root", default="/vol/data")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="/vol/assets/audit_samples")
    p.add_argument("--per-source", type=int, default=2)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--script", default=None,
                   help="if set, sample only this script (by name) across all "
                        "chunks — for datasets not stratified by source prefix "
                        "(e.g. synth shards). Uses the split script_ids.npy.")
    args = p.parse_args()

    rng = random.Random(0)
    split_dir = Path(args.root) / args.data / args.split

    if args.script is not None:
        # Script-filtered mode: pick indices whose split-level script id
        # matches. Accepts a comma-separated list; each script is its own
        # "source" group so per-script samples are drawn separately.
        scripts = [s.strip() for s in args.script.split(",")]
        chunks = [c for c in sorted(split_dir.glob("chunk_*")) if _nonempty(c)]
        sids = np.load(str(split_dir / "script_ids.npy"))
        by_source = {sc: chunks for sc in scripts}
        match_by_source = {sc: np.nonzero(sids == SCRIPTS.index(sc))[0].tolist()
                           for sc in scripts}
    else:
        by_source = defaultdict(list)
        for c in sorted(split_dir.glob("chunk_*")):
            if _nonempty(c):
                by_source[_source_of(c)].append(c)
        match_by_source = None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # clear stale
    for f in out.glob("*.png"):
        f.unlink()
    manifest = (out / "manifest.jsonl").open("w")

    n = 0
    for source in sorted(by_source):
        chunks = by_source[source]
        ds = StreamingDataset(streams=[Stream(local=str(c)) for c in chunks],
                              shuffle=False)
        pool = (match_by_source[source] if match_by_source is not None
                else range(len(ds)))
        picks = rng.sample(list(pool), min(args.per_source, len(pool)))
        for i in picks:
            s = ds[i]
            img = s["image"]  # (3,H,W) uint8
            pil = Image.fromarray(
                np.ascontiguousarray(img.transpose(1, 2, 0)))
            if args.scale > 1:
                pil = pil.resize((pil.width * args.scale,
                                  pil.height * args.scale), Image.NEAREST)
            sid = int(s["script_id"])
            script = SCRIPTS[sid] if sid < len(SCRIPTS) else f"?{sid}"
            fname = f"{n:03d}_{source}.png"
            pil.save(str(out / fname))
            segs = json.loads(s["segments"])
            gl = s["group_labels"]
            manifest.write(json.dumps({
                "file": fname, "source": source, "script": script,
                "label": s["label"], "width": int(s["width"]),
                "segments": [{"script": SCRIPTS[sg["script_id"]]
                              if sg.get("script_id", -1) < len(SCRIPTS) else "?",
                              "text": sg.get("text", ""),
                              "offset": sg.get("offset", 0),
                              "width": sg.get("width", 0)} for sg in segs],
                "seg_width_sum": sum(sg.get("width", 0) for sg in segs),
                "group_labels_len": int(len(gl)),
            }, ensure_ascii=False) + "\n")
            n += 1
    manifest.close()
    print(f"dumped {n} samples from {len(by_source)} sources → {out}")


if __name__ == "__main__":
    main()
