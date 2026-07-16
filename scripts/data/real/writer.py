"""
Real-world OCR data → Lipi MDS chunks.

Writes the same chunked MDS layout generate.py produces, so real chunks can
sit next to synth chunks under one root: LipiStreamingDataset opens every
train/chunk_* dir as a stream, and root sidecars are per-chunk sidecars
concatenated in sorted-chunk order. Real chunks are named chunk_r_{source}_N,
which sorts after the synth chunk_NNNN dirs.

v1 ingestion policy: single-script lines only. The trainer supervises LID
from per-pixel group_labels and per-segment pixel offsets; for a real crop
we only know those exactly when the whole crop is one script (one full-width
segment). Mixed-script lines (split_by_script yields >1 run) are dropped and
counted — they come back in v2 via word boxes (born-digital PDFs) or CTC
forced alignment.
"""

import hashlib
import json
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.taxonomy import (
    GROUP_TO_ID, GROUPS, NUM_GROUPS, SCRIPT_TO_GROUP, SCRIPT_TO_ID, SCRIPTS,
    TAXONOMY_VERSION,
)
from src.data.color import rgb_to_input
from src.data.rendering import image_has_ink
from src.data.script_detect import split_by_script
from src.encoding.decompose import decode_ids, encode_text

# Must match scripts/data/generate.py MDS_COLUMNS — the trainer reads every
# column, including group_labels/segments for LID supervision.
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

_MAX_VAL_PER_CHUNK = 500


def prepare_label(text: str) -> str | None:
    """NFC-normalize, drop format controls, collapse whitespace."""
    if not text:
        return None
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text
                   if unicodedata.category(c) != "Cf")
    text = " ".join(text.split())
    return text or None


def build_sample(img: Image.Image, text: str, script: str,
                 height: int, max_width: int) -> tuple[dict | None, str]:
    """Convert one (image, label) pair into a shard sample.

    Returns (sample, "ok") or (None, drop_reason).
    """
    label = prepare_label(text)
    if label is None:
        return None, "empty_label"

    runs = split_by_script(label, script)
    if not runs:
        return None, "empty_label"
    if len(runs) > 1:
        return None, "mixed_script"
    run_text, run_script = runs[0]
    if run_script not in SCRIPT_TO_ID:
        return None, "unknown_script"

    ids = encode_text(run_text, run_script)
    if not ids:
        return None, "unencodable"
    # The stored label is metadata (decoding targets come from target_ids);
    # a failed round-trip means the codec silently dropped characters, so
    # the image would show text the targets don't contain.
    if decode_ids(ids, run_script) != run_text:
        return None, "roundtrip_mismatch"

    if img.width < 4 or img.height < 4:
        return None, "tiny_image"
    # LA/P/CMYK-mode inputs crash image_has_ink's channel indexing.
    if img.mode != "RGB":
        img = img.convert("RGB")
    upscaled = img.height < height
    if img.height != height:
        new_w = max(1, round(img.width * height / img.height))
        img = img.resize((new_w, height), Image.BILINEAR)
    # No width squish: horizontally compressing a long line to fit max_width
    # distorts glyph aspect and starves the CTC frame budget. Keep the natural
    # aspect ratio; only drop crops so wide they'd blow up batch memory
    # (max_width is a hard reject threshold, not a resize target).
    if img.width > max_width:
        return None, "too_wide"
    if not image_has_ink(img):
        return None, "blank_image"

    arr = rgb_to_input(img).numpy()  # (3, H, W) uint8
    w = arr.shape[2]
    sid = SCRIPT_TO_ID[run_script]
    gid = GROUP_TO_ID[SCRIPT_TO_GROUP[run_script]]

    sample = {
        "image": arr,
        "label": label,
        "script_id": sid,
        "group_id": gid,
        "target_ids": np.array(ids, dtype=np.int64),
        "target_len": len(ids),
        "width": w,
        # One full-width segment: the crop is single-script, so unlike the
        # generator we have no inter-segment whitespace to mark as blank.
        "group_labels": np.full(w, gid, dtype=np.int32),
        "segments": json.dumps([{
            "group_id": gid, "script_id": sid,
            "text": label, "width": w, "offset": 0,
        }]),
    }
    return (sample, "upscaled") if upscaled else (sample, "ok")


class RealChunkWriter:
    """Rotating MDS chunk writer with per-chunk sidecars and stats."""

    def __init__(self, out_root: Path, source: str, val_ratio: float = 0.1,
                 samples_per_chunk: int = 50_000):
        self.out_root = Path(out_root)
        self.source = source
        self.val_ratio = val_ratio
        self.samples_per_chunk = samples_per_chunk
        self.stats: Counter = Counter()
        self.script_counts: Counter = Counter()
        self._chunk = -1
        self._in_chunk = 0
        self._val_in_chunk = 0
        self._tw = self._vw = None
        self._sidecars = None
        self._open_next_chunk()

    def _dirs(self):
        name = f"chunk_r_{self.source}_{self._chunk:04d}"
        return (self.out_root / "train" / name, self.out_root / "val" / name)

    def _open_next_chunk(self):
        from streaming import MDSWriter
        self._close_writers()
        self._chunk += 1
        self._in_chunk = 0
        self._val_in_chunk = 0
        t_dir, v_dir = self._dirs()
        # Wipe partial chunks so re-runs (Modal retries) are idempotent.
        for d in (t_dir, v_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        self._tw = MDSWriter(out=str(t_dir), columns=MDS_COLUMNS,
                             size_limit=1 << 26)
        self._vw = MDSWriter(out=str(v_dir), columns=MDS_COLUMNS,
                             size_limit=1 << 26)
        self._sidecars = {"train": ([], []), "val": ([], [])}

    def _close_writers(self):
        if self._tw is None:
            return
        self._tw.finish()
        self._vw.finish()
        t_dir, v_dir = self._dirs()
        for split, d in (("train", t_dir), ("val", v_dir)):
            widths, sids = self._sidecars[split]
            if not widths:
                shutil.rmtree(d)  # empty MDS dirs confuse downstream globs
                continue
            np.save(str(d / "widths.npy"), np.array(widths, dtype=np.int32))
            np.save(str(d / "script_ids.npy"), np.array(sids, dtype=np.int32))
        self._tw = self._vw = None

    def add(self, img: Image.Image, text: str, script: str,
            height: int, max_width: int) -> bool:
        sample, reason = build_sample(img, text, script, height, max_width)
        self.stats[reason] += 1
        if sample is None:
            return False
        if self._in_chunk >= self.samples_per_chunk:
            self._open_next_chunk()

        h = int(hashlib.md5(
            f"{self.source}_{self._chunk}_{self._in_chunk}_{sample['label']}"
            .encode()).hexdigest(), 16)
        is_val = (h % 1000 < int(self.val_ratio * 1000)
                  and self._val_in_chunk < _MAX_VAL_PER_CHUNK)
        writer = self._vw if is_val else self._tw
        split = "val" if is_val else "train"
        writer.write(sample)
        widths, sids = self._sidecars[split]
        widths.append(sample["width"])
        sids.append(sample["script_id"])
        self._in_chunk += 1
        if is_val:
            self._val_in_chunk += 1
        self.script_counts[SCRIPTS[sample["script_id"]]] += 1
        return True

    def close(self) -> dict:
        self._close_writers()
        kept = sum(self.script_counts.values())
        return {"source": self.source, "kept": kept,
                "by_script": dict(self.script_counts),
                "drops": {k: v for k, v in self.stats.items()
                          if k not in ("ok", "upscaled")},
                "upscaled": self.stats.get("upscaled", 0)}


def finalize_root(root: Path, height: int, max_width: int) -> None:
    """Rebuild root sidecars over all chunks and stamp metadata.

    Mirrors generate.py's assemble_sidecars: sorted chunk order is the
    order LipiStreamingDataset streams samples in.
    """
    import torch
    root = Path(root)
    for split in ("train", "val"):
        split_dir = root / split
        # Preempted/killed writers leave dirs without index.json — those
        # were never valid MDS chunks, so drop them. A chunk WITH an
        # index but missing sidecars is inconsistent → still an error.
        for chunk in sorted(split_dir.glob("chunk_*")):
            if not (chunk / "index.json").exists():
                print(f"finalize: removing never-finished chunk {chunk}")
                shutil.rmtree(chunk)
        for name in ("widths.npy", "script_ids.npy"):
            parts = []
            for chunk in sorted(split_dir.glob("chunk_*")):
                f = chunk / name
                if not f.exists():
                    raise FileNotFoundError(
                        f"{f} missing — chunk has no per-chunk sidecars")
                parts.append(np.load(str(f)))
            arr = (np.concatenate(parts).astype(np.int32)
                   if parts else np.zeros(0, dtype=np.int32))
            np.save(str(split_dir / name), arr)

    meta_path = root / "metadata.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, weights_only=False)
        if meta.get("height") != height:
            raise RuntimeError(
                f"height mismatch: metadata says {meta.get('height')}, "
                f"writer ran at {height}")
        if meta.get("taxonomy_version") != TAXONOMY_VERSION:
            raise RuntimeError("taxonomy_version mismatch in metadata.pt")
    else:
        torch.save({
            "active_scripts": list(SCRIPTS),
            "active_groups": list(GROUPS),
            "taxonomy_version": TAXONOMY_VERSION,
            "height": height,
            "max_width": max_width,
        }, meta_path)
