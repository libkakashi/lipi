#!/usr/bin/env python3
"""
Resize an existing MDS shard set to a new input height (aspect-preserving).

Converts 64px shards → 48px (or any target) without re-rendering: each stored
sample was resized to its source height preserving aspect, so a uniform
scale = target/source shrinks BOTH dims (height H→target, width W→round(W*scale)).
Per-column group_labels are nearest-resampled and segment offset/width scaled
so LID supervision stays aligned; target_ids/label are unchanged.

Reads {in}/train and {in}/val, writes {out}/train and {out}/val with the same
chunked MDS layout + per-chunk widths/script_ids sidecars + metadata.pt, so the
result trains exactly like a natively-generated set.

Usage:
    python scripts/data/convert_height.py --in data/real-v1 --out data/real-v1-48 \
        --from-height 64 --to-height 48
    # chunk-sharded (parallel across containers):
    ... --chunk-shard 0:4        # process source chunks where idx % 4 == 0
    python scripts/data/convert_height.py --out data/real-v1-48 --finalize \
        --to-height 48
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming import MDSWriter, StreamingDataset

from src.taxonomy import GROUPS, SCRIPTS, TAXONOMY_VERSION

MDS_COLUMNS = {
    "image": "ndarray:uint8",
    "label": "str",
    "script_id": "int",
    "group_id": "int",
    "target_ids": "ndarray:int64",
    "target_len": "int",
    "width": "int",
    "group_labels": "ndarray:int32",
    "segments": "str",
}


def convert_sample(s: dict, to_h: int) -> dict:
    img = s["image"]                       # (3, H, W) uint8
    _, H, W = img.shape
    scale = to_h / H
    new_w = max(1, round(W * scale))

    pil = Image.fromarray(np.ascontiguousarray(img.transpose(1, 2, 0)))
    pil = pil.resize((new_w, to_h), Image.BILINEAR)
    new_img = np.ascontiguousarray(
        np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1))

    # per-column group labels → nearest-resample to new width
    gl = s["group_labels"]                 # (W,) int32
    if len(gl) != W:                       # defensive; should equal W
        gl = np.resize(gl, W)
    idx = np.clip(np.round(np.arange(new_w) / scale).astype(np.int64), 0, W - 1)
    new_gl = gl[idx].astype(np.int32)

    segs = json.loads(s["segments"])
    for seg in segs:
        seg["offset"] = round(seg.get("offset", 0) * scale)
        seg["width"] = max(1, round(seg.get("width", 0) * scale))

    return {
        "image": new_img,
        "label": s["label"],
        "script_id": int(s["script_id"]),
        "group_id": int(s["group_id"]),
        "target_ids": s["target_ids"].astype(np.int64),
        "target_len": int(s["target_len"]),
        "width": new_w,
        "group_labels": new_gl,
        "segments": json.dumps(segs, ensure_ascii=False),
    }


def _chunks(split_dir: Path) -> list[Path]:
    return sorted(split_dir.glob("chunk_*"))


def convert(in_dir: Path, out_dir: Path, to_h: int,
            shard_index: int, n_shards: int) -> None:
    for split in ("train", "val"):
        src_split = in_dir / split
        if not src_split.is_dir():
            continue
        for ci, chunk in enumerate(_chunks(src_split)):
            if ci % n_shards != (shard_index % n_shards):
                continue
            idx_json = chunk / "index.json"
            if not idx_json.exists():
                continue
            n = sum(sh.get("samples", 0)
                    for sh in json.loads(idx_json.read_text())["shards"])
            if n == 0:
                continue
            out_chunk = out_dir / split / chunk.name
            marker = out_chunk / ".done"
            if marker.exists():
                continue
            if out_chunk.exists():
                import shutil
                shutil.rmtree(out_chunk)
            out_chunk.mkdir(parents=True, exist_ok=True)

            ds = StreamingDataset(local=str(chunk), shuffle=False,
                                  batch_size=1)
            widths, sids = [], []
            with MDSWriter(out=str(out_chunk), columns=MDS_COLUMNS,
                           size_limit=1 << 26) as w:
                for i in range(len(ds)):
                    conv = convert_sample(ds[i], to_h)
                    w.write(conv)
                    widths.append(conv["width"])
                    sids.append(conv["script_id"])
            np.save(str(out_chunk / "widths.npy"),
                    np.array(widths, dtype=np.int32))
            np.save(str(out_chunk / "script_ids.npy"),
                    np.array(sids, dtype=np.int32))
            marker.touch()
            print(f"  {split}/{chunk.name}: {len(ds)} samples → {to_h}px",
                  flush=True)


def finalize(out_dir: Path, to_h: int, max_width: int) -> None:
    import torch
    for split in ("train", "val"):
        split_dir = out_dir / split
        if not split_dir.is_dir():
            continue
        for name in ("widths.npy", "script_ids.npy"):
            parts = []
            for chunk in _chunks(split_dir):
                f = chunk / name
                if f.exists():
                    parts.append(np.load(str(f)))
            arr = (np.concatenate(parts).astype(np.int32)
                   if parts else np.zeros(0, dtype=np.int32))
            np.save(str(split_dir / name), arr)
    src_meta = None
    for cand in (out_dir / "metadata.pt",):
        if cand.exists():
            src_meta = torch.load(cand, weights_only=False)
    meta = dict(src_meta or {})
    meta.update({
        "active_scripts": meta.get("active_scripts", list(SCRIPTS)),
        "active_groups": meta.get("active_groups", list(GROUPS)),
        "taxonomy_version": TAXONOMY_VERSION,
        "height": to_h,
        "max_width": max_width,
    })
    torch.save(meta, out_dir / "metadata.pt")
    print(f"finalized {out_dir} at {to_h}px")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--from-height", type=int, default=64)
    p.add_argument("--to-height", type=int, default=48)
    p.add_argument("--max-width", type=int, default=2048)
    p.add_argument("--chunk-shard", type=str, default="0:1")
    p.add_argument("--finalize", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    if args.finalize:
        finalize(out_dir, args.to_height, args.max_width)
        return
    if not args.inp:
        p.error("--in required unless --finalize")
    si, n = (int(x) for x in args.chunk_shard.split(":"))
    convert(Path(args.inp), out_dir, args.to_height, si, n)


if __name__ == "__main__":
    main()
