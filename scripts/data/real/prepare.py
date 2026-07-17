#!/usr/bin/env python3
"""
Ingest real OCR datasets into Lipi MDS chunks.

Usage:
    python scripts/data/real/prepare.py --list
    python scripts/data/real/prepare.py --source hhd_ethiopic \
        --out data/real-v1 --height 64
    python scripts/data/real/prepare.py --finalize --out data/real-v1

Each source writes train/chunk_r_{source}_* + val/chunk_r_{source}_* under
--out. Run --finalize once after the last source to rebuild root sidecars
and stamp metadata.pt. The finished root trains standalone, or its chunks
merge with synth shards at stage time (modal train --data "synth,real").
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.data.real.sources import REGISTRY, SourceSpec, detect_columns
from scripts.data.real.writer import RealChunkWriter, finalize_root


def ingest(spec: SourceSpec, out: Path, height: int, max_width: int,
           val_ratio: float, limit: int | None) -> dict:
    from datasets import load_dataset

    writer = RealChunkWriter(out, spec.name, val_ratio=val_ratio)
    t0 = time.time()
    seen = 0
    for config in (spec.configs or (spec.config,)):
        config_script = spec.script
        if config_script is None and config:
            config_script = spec.lang_map.get(config)
            if config_script is None:
                print(f"{spec.name}[{config}]: no script mapping, skipping")
                continue
        for split in spec.splits:
            ds = load_dataset(spec.hub, config, split=split, streaming=True)
            it = iter(ds)
            try:
                first = next(it)
            except StopIteration:
                print(f"{spec.name}[{config}/{split}]: empty, skipping")
                continue
            image_col, text_col, lang_col = detect_columns(first, spec)
            if config_script is None and lang_col is None:
                raise RuntimeError(
                    f"{spec.name}: per-row script source but no language "
                    f"column in {list(first)} — set lang_col in the spec")
            print(f"{spec.name}[{config or ''}/{split}]: image={image_col} "
                  f"text={text_col}"
                  + (f" lang={lang_col}" if lang_col else ""), flush=True)

            for row in itertools.chain([first], it):
                if limit is not None and seen >= limit:
                    break
                seen += 1
                script = config_script
                if script is None:
                    lang = str(row.get(lang_col, "")).strip().lower()
                    script = spec.lang_map.get(lang)
                    if script is None:
                        writer.stats["unmapped_lang"] += 1
                        continue
                img = row[image_col]
                text = row[text_col]
                if img is None or text is None:
                    writer.stats["null_row"] += 1
                    continue
                try:
                    writer.add(img, str(text), script, height, max_width)
                except Exception:
                    # One malformed sample must not kill the source.
                    writer.stats["sample_error"] += 1
                if seen % 20_000 == 0:
                    kept = sum(writer.script_counts.values())
                    print(f"  {seen} rows → {kept} kept "
                          f"({seen / (time.time() - t0):.0f} rows/s)",
                          flush=True)

    report = writer.close()
    report["rows_seen"] = seen
    report["seconds"] = round(time.time() - t0, 1)
    return report


def _parse_iiit_annotations(extract_dir: Path):
    """Yield (image_path, text) from IIIT-style split annotation files.

    Layout: train/val txt files with "relpath<ws>token" lines, where token
    is either the word itself or an index into a vocab/lexicon file.
    test* files are never read (eval quarantine).
    """
    def _rel(f):
        return str(f.relative_to(extract_dir)).lower()

    # Match on the full relative path: split markers sometimes live in a
    # parent directory (train.zip → train_x/annotations.txt), and a test
    # marker anywhere in the path must exclude the file (eval quarantine).
    ann_files = [f for f in extract_dir.rglob("*.txt")
                 if ("train" in _rel(f) or "val" in _rel(f))
                 and "test" not in _rel(f)]
    # Dozens+ of matching txts means per-line GT sidecars (HHD-style
    # xxx.gt.txt), not split lists — signal the caller to use pair mode
    # instead of walking every file.
    if len(ann_files) > 50:
        raise RuntimeError(
            f"{len(ann_files)} annotation candidates — pair layout")
    if not ann_files:
        tree = "\n".join(str(p.relative_to(extract_dir))
                         for p in list(extract_dir.rglob("*"))[:40])
        raise RuntimeError(f"no train/val annotation txt found; tree:\n{tree}")

    vocab = None
    for vf in extract_dir.rglob("*.txt"):
        if any(k in vf.stem.lower() for k in ("vocab", "lexicon")):
            vocab = vf.read_text(encoding="utf-8",
                                 errors="ignore").splitlines()
            break

    # Fallback resolver: index every image by its last two path components
    # ("12/word_00042.jpg") — annotation rel-paths often assume a different
    # root than where extraction landed them. Two components (not one)
    # because bare filenames repeat across train/val/test splits.
    img_index: dict[str, Path] = {}
    for p in extract_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                                ".bmp", ".gif"):
            img_index[f"{p.parent.name}/{p.name}"] = p

    warnings_left = 3
    for ann in ann_files:
        base_dirs = (ann.parent, extract_dir)
        resolved = 0
        unresolved_samples = []
        for line in ann.read_text(encoding="utf-8",
                                  errors="ignore").splitlines():
            raw = line.strip()
            # Two delimiter styles in the wild: "path label..." and
            # CSV-ish "path, label" (IIIT-INDIC-HW-WORDS).
            if "," in raw and " " not in raw.split(",", 1)[0]:
                rel_part, rest = raw.split(",", 1)
                parts = [rel_part.strip()] + rest.split()
            else:
                parts = raw.split()
            if len(parts) < 2:
                continue
            rel = parts[0]
            # Formats seen across IIIT releases: "path word",
            # "path vocab_idx" and "path vocab_idx word".
            if len(parts) >= 3 and parts[1].isdigit():
                text = " ".join(parts[2:])
            elif vocab is not None and parts[1].isdigit():
                idx = int(parts[1])
                if idx >= len(vocab):
                    continue
                text = vocab[idx]
            else:
                text = " ".join(parts[1:])
            rel = rel.lstrip("./")
            img_path = None
            for base in base_dirs:
                cand = base / rel
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                key = "/".join(rel.replace("\\", "/").split("/")[-2:])
                img_path = img_index.get(key)
            if img_path is not None:
                resolved += 1
                yield img_path, text
            elif len(unresolved_samples) < 3:
                unresolved_samples.append(rel)
        if resolved == 0 and warnings_left > 0:
            warnings_left -= 1
            near = []
            for p in extract_dir.rglob("*"):
                near.append(str(p.relative_to(extract_dir)))
                if len(near) >= 8:
                    break
            print(f"  WARNING {ann.name}: 0/{len(unresolved_samples)}+ "
                  f"resolved; sample rels {unresolved_samples}; "
                  f"tree sample {near}", flush=True)


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif")


def _parse_pair_annotations(extract_dir: Path):
    """Yield (image_path, text) from sidecar-pair layouts.

    HHD-Ethiopic-style: each line image has a same-stem text file
    (xxx.png ↔ xxx.gt.txt or xxx.txt). Anything on a path containing
    "test" is skipped (eval quarantine).
    """
    # GT4HistOCR-style variant infixes: 0001.bin.png / 0001.nrm.png /
    # 0001.dew.png all pair with 0001.gt.txt — index each image under both
    # its full stem and its infix-stripped stem so either matches.
    def _norm(stem: str) -> str:
        for infix in (".bin", ".nrm", ".dew"):
            if stem.endswith(infix):
                return stem[: -len(infix)]
        return stem

    imgs: dict[str, Path] = {}
    for p in extract_dir.rglob("*"):
        if p.suffix.lower() in _IMG_EXTS:
            rel = str(p.relative_to(extract_dir)).lower()
            if "test" in rel:
                continue
            stem = p.name[: -len(p.suffix)]
            imgs.setdefault(stem, p)
            imgs.setdefault(_norm(stem), p)
    n_txt = matched = 0
    for t in extract_dir.rglob("*.txt"):
        rel = str(t.relative_to(extract_dir)).lower()
        if "test" in rel:
            continue
        n_txt += 1
        stem = t.name[:-4]
        if stem.endswith(".gt"):
            stem = stem[:-3]
        img = imgs.get(stem) or imgs.get(_norm(stem))
        if img is None:
            continue
        text = t.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            matched += 1
            yield img, text
    if n_txt and matched == 0:
        tree = [str(p.relative_to(extract_dir))
                for p in list(extract_dir.rglob("*"))[:10]]
        print(f"  WARNING pair-parse: 0/{n_txt} txt matched an image; "
              f"tree {tree}", flush=True)


def _parse_iiit_uc(extract_dir: Path):
    """Yield (image_path, text) from IIIT-Indic-HW-UC per-language zips.

    Two sub-layouts coexist inside one zip:
      * Manual: `.../gt.txt` with tab-separated `images/xxx.jpg\tword`,
        images in a sibling `images/` dir.
      * Google: `.../{writer}/text/{id}.txt`, each file holding many
        `name\tword` lines, images in the sibling `{writer}/image/` dir.
    Anything on a path containing "test" is skipped (eval quarantine).
    """
    for gt in extract_dir.rglob("gt.txt"):
        if "test" in str(gt.relative_to(extract_dir)).lower():
            continue
        base = gt.parent
        for line in gt.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" not in line:
                continue
            rel, text = line.split("\t", 1)
            img = base / rel
            if text.strip() and img.exists():
                yield img, text.strip()

    for tf in extract_dir.rglob("*.txt"):
        if tf.parent.name != "text":
            continue
        if "test" in str(tf.relative_to(extract_dir)).lower():
            continue
        img_dir = tf.parent.parent / "image"
        if not img_dir.is_dir():
            continue
        for line in tf.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" not in line:
                continue
            name, text = line.split("\t", 1)
            img = img_dir / name
            if text.strip() and img.exists():
                yield img, text.strip()


_ALTO_NS = "{http://www.loc.gov/standards/alto/ns-v4#}"


def _parse_alto(extract_dir: Path):
    """Yield (cropped PIL line image, text) from ALTO-v4 page XML.

    Layout (heiDATA/BL scanned prints): `.../alto/{page}.xml` alongside the
    page image `.../{page}.jpg`. Each TextLine carries HPOS/VPOS/WIDTH/HEIGHT
    (pixel box); line text is its String CONTENTs joined. Pages open once and
    crop many lines. "test" paths are skipped (eval quarantine).
    """
    import xml.etree.ElementTree as ET

    from PIL import Image

    for xml in extract_dir.rglob("*.xml"):
        rel = str(xml.relative_to(extract_dir)).replace("\\", "/").lower()
        if "/alto/" not in rel or "test" in rel:
            continue
        stem = str(xml).replace("/alto/", "/")[:-4]
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            cand = Path(stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue
        try:
            root = ET.parse(xml).getroot()
            page = Image.open(img_path)
            page.load()
        except Exception:
            continue
        pw, ph = page.size
        for tl in root.iter(f"{_ALTO_NS}TextLine"):
            try:
                hp = int(float(tl.get("HPOS")))
                vp = int(float(tl.get("VPOS")))
                w = int(float(tl.get("WIDTH")))
                h = int(float(tl.get("HEIGHT")))
            except (TypeError, ValueError):
                continue
            parts = [s.get("CONTENT", "")
                     for s in tl.iter(f"{_ALTO_NS}String")]
            text = " ".join(p for p in parts if p).strip()
            if not text or w < 4 or h < 4:
                continue
            # Clamp to page bounds (some ALTO boxes overrun by a few px).
            box = (max(0, hp), max(0, vp), min(pw, hp + w), min(ph, vp + h))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            yield page.crop(box), text


def _parse_bmod(extract_dir: Path):
    """Yield (image_path, text) from B-MOD split-annotation files.

    Labels live in `train.easy` / `train.medium` / `train.hard` (+ valid.*)
    with `relpath<whitespace>transcription` per line; images are line crops
    under `lines/`. `test.*` and `valid.*` splits are skipped (train only for
    supervision; valid/test reserved for eval). Falls back to a filename
    index when the annotation path doesn't resolve directly.
    """
    img_index: dict[str, Path] = {}
    for p in extract_dir.rglob("*.jpg"):
        img_index[p.name] = p

    for ann in extract_dir.rglob("*"):
        if not ann.is_file():
            continue
        if not ann.name.lower().startswith("train."):
            continue
        for line in ann.read_text(encoding="utf-8",
                                  errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)   # path has no spaces; text may
            if len(parts) != 2:
                continue
            rel, text = parts
            text = text.strip()
            if not text:
                continue
            img = extract_dir / rel
            if not img.exists():
                img = img_index.get(Path(rel).name)
            if img is not None:
                yield img, text


def _parse_ndl_tsv(extract: Path):
    """NDL ocr-ndloneline: pre-cropped line JPGs + labeldata_pdm.tsv.

    TSV columns: filename, text, orientation (tate=vertical / yoko=
    horizontal), IIIF URL. The TSV lives in the GitHub repo, not the image
    zip — fetch it here. Vertical (tate) lines are skipped: the model
    reads horizontal lines only.
    """
    import csv
    import requests

    tsv = None
    for branch in ("main", "master"):
        url = ("https://raw.githubusercontent.com/ndl-lab/ocr-ndloneline/"
               f"{branch}/labeldata_pdm.tsv")
        r = requests.get(url, timeout=60)
        if r.ok and r.text.strip():
            tsv = r.text
            break
    if tsv is None:
        raise RuntimeError("could not fetch labeldata_pdm.tsv")

    by_name = {p.name: p for p in extract.rglob("*.jpg")}
    n_tate = 0
    for row in csv.reader(tsv.splitlines(), delimiter="\t"):
        if len(row) < 3:
            continue
        fname, text, orient = row[0], row[1], row[2]
        if orient.strip() == "tate":
            n_tate += 1
            continue
        p = by_name.get(Path(fname).name)
        if p is not None and text.strip():
            yield p, text
    print(f"  [ndl] skipped {n_tate} vertical (tate) lines", flush=True)


def ingest_lmdb_gdrive(spec: SourceSpec, out: Path, height: int,
                       max_width: int, val_ratio: float,
                       limit: int | None) -> dict:
    """FudanVI-style lmdb sets shared as a Google Drive folder.

    Downloads the folder with gdown, then reads every lmdb whose path
    contains one of spec.configs (e.g. 'scene', 'web') and skips
    'document' (synthetic Text-Renderer data — this loader exists
    precisely because the HF mirror can't exclude it) and any 'test'
    split. lmdb layout is the standard STR one: num-samples,
    image-%09d (encoded bytes), label-%09d (utf-8 text).
    """
    import io

    import gdown
    import lmdb
    from PIL import Image

    writer = RealChunkWriter(out, spec.name, val_ratio=val_ratio)
    t0 = time.time()
    seen = 0
    work = Path("/tmp/real-lmdb")
    wanted = tuple(s.lower() for s in spec.configs) or ("scene", "web")
    try:
        work.mkdir(parents=True, exist_ok=True)
        for url, script in spec.urls:
            print(f"{spec.name}: gdown folder {url}", flush=True)
            gdown.download_folder(url=url, output=str(work), quiet=False,
                                  use_cookies=False)
            mdbs = sorted(work.rglob("data.mdb"))
            print(f"  found {len(mdbs)} lmdb dirs: "
                  f"{[str(m.parent.relative_to(work)) for m in mdbs]}",
                  flush=True)
            for mdb in mdbs:
                rel = str(mdb.parent.relative_to(work)).lower()
                if "test" in rel or "document" in rel:
                    continue
                if not any(w in rel for w in wanted):
                    continue
                env = lmdb.open(str(mdb.parent), readonly=True, lock=False,
                                readahead=False, meminit=False)
                with env.begin() as txn:
                    n = int(txn.get(b"num-samples") or b"0")
                    print(f"  [{rel}] {n} samples", flush=True)
                    for i in range(1, n + 1):
                        if limit is not None and seen >= limit:
                            break
                        seen += 1
                        img_b = txn.get(f"image-{i:09d}".encode())
                        lab_b = txn.get(f"label-{i:09d}".encode())
                        if not img_b or not lab_b:
                            writer.stats["missing_kv"] += 1
                            continue
                        try:
                            img = Image.open(io.BytesIO(img_b))
                            img.load()
                            writer.add(img, lab_b.decode("utf-8"), script,
                                       height, max_width)
                        except Exception:
                            writer.stats["sample_error"] += 1
                        if seen % 20_000 == 0:
                            kept = sum(writer.script_counts.values())
                            print(f"  {seen} rows → {kept} kept", flush=True)
                env.close()
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)

    report = writer.close()
    report["rows_seen"] = seen
    report["seconds"] = round(time.time() - t0, 1)
    return report


def ingest_url_zip(spec: SourceSpec, out: Path, height: int, max_width: int,
                   val_ratio: float, limit: int | None) -> dict:
    """Download annotation zips (IIIT-style) and ingest word/line crops."""
    import urllib.request
    import zipfile

    from PIL import Image

    writer = RealChunkWriter(out, spec.name, val_ratio=val_ratio)
    t0 = time.time()
    seen = 0
    work = Path("/tmp/real-zips")
    for url, script in spec.urls:
        if limit is not None and seen >= limit:
            break
        try:
            work.mkdir(parents=True, exist_ok=True)
            # Sanitize: Dataverse URLs carry query params that make ugly
            # (and sometimes invalid) local filenames.
            import re as _re
            safe = _re.sub(r"[^A-Za-z0-9._-]", "_", Path(url).name) or "dl"
            if not safe.endswith((".zip", ".tar")):
                safe += ".zip"
            zpath = work / safe
            extract = work / (zpath.stem + "_x")
            # The CVIT CDN sometimes closes mid-transfer, yielding a
            # truncated file that fails as BadZipFile — verify length
            # against Content-Length and retry.
            last_err = None
            for attempt in range(3):
                try:
                    print(f"{spec.name}: downloading {url} "
                          f"(attempt {attempt + 1})", flush=True)
                    expected = 0
                    if url.startswith("gdrive:"):
                        # Google Drive file: requests can't pass the
                        # large-file confirm interstitial; gdown can.
                        import gdown
                        gdown.download(id=url[len("gdrive:"):],
                                       output=str(zpath), quiet=False)
                    else:
                        # requests follows multi-hop redirects (303→301→
                        # stream) and streams robustly — urllib silently
                        # returned 0 bytes on Dataverse's redirect chain.
                        import requests
                        with requests.get(
                                url, stream=True, timeout=120,
                                allow_redirects=True,
                                headers={"User-Agent": "Mozilla/5.0 (lipi-prep)"}
                        ) as r:
                            r.raise_for_status()
                            expected = int(r.headers.get("Content-Length") or 0)
                            with open(zpath, "wb") as f:
                                for chunk in r.iter_content(1 << 20):
                                    f.write(chunk)
                    got = zpath.stat().st_size
                    if got == 0:
                        raise IOError("downloaded 0 bytes")
                    if expected and got != expected:
                        raise IOError(f"truncated: {got}/{expected} bytes")
                    print(f"  downloaded {got / 1e9:.2f} GB", flush=True)
                    import tarfile as _tf0
                    if zipfile.is_zipfile(zpath):
                        with zipfile.ZipFile(zpath) as zf:
                            zf.extractall(extract)
                    elif _tf0.is_tarfile(zpath):
                        with _tf0.open(zpath) as tf:
                            tf.extractall(extract, filter="data")
                    else:
                        raise IOError("not a zip or tar archive")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"  attempt {attempt + 1} failed: {e!r}",
                          flush=True)
                    zpath.unlink(missing_ok=True)
                    import shutil as _sh3
                    _sh3.rmtree(extract, ignore_errors=True)
            if last_err is not None:
                raise last_err
            zpath.unlink()
            # IIIT wraps the payload in nested archives (bengal.zip →
            # iiit-indic/bn.zip → images + annotations). Unwrap until flat.
            import tarfile
            for _ in range(3):
                inner = (list(extract.rglob("*.zip"))
                         + list(extract.rglob("*.tar.gz"))
                         + list(extract.rglob("*.tgz")))
                if not inner:
                    break
                for arc in inner:
                    # Extract in place: annotation files reference paths
                    # relative to the archive's own top-level dir (train.zip
                    # → train/...), so a renamed dest breaks resolution.
                    dest = arc.parent
                    print(f"  unwrapping {arc.name}", flush=True)
                    if arc.suffix == ".zip":
                        with zipfile.ZipFile(arc) as izf:
                            izf.extractall(dest)
                    else:
                        with tarfile.open(arc) as itf:
                            itf.extractall(dest, filter="data")
                    arc.unlink()

            n_before = sum(writer.script_counts.values())
            # Always log the extracted tree so an unfamiliar layout is
            # diagnosable in one run instead of guessing across many.
            from collections import Counter as _C
            _all = list(extract.rglob("*"))
            _ext = _C(p.suffix.lower() for p in _all if p.is_file())
            _sample = [str(p.relative_to(extract)) for p in _all[:12]]
            print(f"  [{Path(url).name}] extracted {len(_all)} entries, "
                  f"exts={dict(_ext)}; sample={_sample}", flush=True)

            def _split_then_pairs():
                if spec.loader == "ndl_tsv":
                    yield from _parse_ndl_tsv(extract)
                    return
                if spec.loader == "alto_zip":
                    yield from _parse_alto(extract)
                    return
                if spec.loader == "bmod_zip":
                    yield from _parse_bmod(extract)
                    return
                if spec.loader == "uc_zip":
                    yield from _parse_iiit_uc(extract)
                    return
                yielded = False
                try:
                    for item in _parse_iiit_annotations(extract):
                        yielded = True
                        yield item
                except RuntimeError:
                    pass  # no split annotation files — try pair layout
                if not yielded:
                    yield from _parse_pair_annotations(extract)

            for item, text in _split_then_pairs():
                if limit is not None and seen >= limit:
                    break
                seen += 1
                # ALTO yields cropped PIL images directly; other parsers
                # yield file paths to open.
                if isinstance(item, Path):
                    try:
                        img = Image.open(item)
                        img.load()
                    except Exception:
                        writer.stats["bad_image_file"] += 1
                        continue
                else:
                    img = item
                try:
                    writer.add(img, text, script, height, max_width)
                except Exception:
                    writer.stats["sample_error"] += 1
                if seen % 20_000 == 0:
                    kept = sum(writer.script_counts.values())
                    print(f"  {seen} rows → {kept} kept", flush=True)
            print(f"  {Path(url).name} ({script}): "
                  f"+{sum(writer.script_counts.values()) - n_before} kept",
                  flush=True)
        except Exception as e:
            import traceback
            print(f"  FAILED {url}: {e!r}\n{traceback.format_exc()}",
                  flush=True)
            writer.stats[f"zip_failed:{Path(url).name}"] += 1
        finally:
            import shutil as _sh
            _sh.rmtree(work, ignore_errors=True)  # reclaim disk per zip

    report = writer.close()
    report["rows_seen"] = seen
    report["seconds"] = round(time.time() - t0, 1)
    return report


def ingest_hub_url(spec: SourceSpec, out: Path, height: int, max_width: int,
                   val_ratio: float, limit: int | None) -> dict:
    """Hub rows whose image is an http(s) URL (detected per-row —
    some OpenPecha parquets have misaligned column names)."""
    import io
    from concurrent.futures import ThreadPoolExecutor

    import urllib.request
    from datasets import load_dataset
    from PIL import Image

    from scripts.data.real.sources import _PREFERRED_TEXT

    def fetch(url):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read()
        except Exception:
            return None

    writer = RealChunkWriter(out, spec.name, val_ratio=val_ratio)
    t0 = time.time()
    seen = 0
    for split in spec.splits:
        ds = load_dataset(spec.hub, spec.config, split=split, streaming=True)

        def rows():
            nonlocal seen
            for row in ds:
                if limit is not None and seen >= limit:
                    return
                seen += 1
                url = next((v for v in row.values()
                            if isinstance(v, str) and v.startswith("http")),
                           None)
                text = None
                for p in _PREFERRED_TEXT:
                    v = row.get(p)
                    if isinstance(v, str) and not v.startswith("http"):
                        text = v
                        break
                if url is None or text is None:
                    writer.stats["no_url_or_text"] += 1
                    continue
                yield url, text

        with ThreadPoolExecutor(max_workers=32) as ex:
            batch = []
            for item in rows():
                batch.append(item)
                if len(batch) < 256:
                    continue
                for (u, text), data in zip(
                        batch, ex.map(fetch, (u for u, _ in batch))):
                    if data is None:
                        writer.stats["download_failed"] += 1
                        continue
                    try:
                        img = Image.open(io.BytesIO(data))
                        img.load()
                    except Exception:
                        writer.stats["bad_image_file"] += 1
                        continue
                    writer.add(img, text, spec.script, height, max_width)
                kept = sum(writer.script_counts.values())
                print(f"  {seen} rows → {kept} kept "
                      f"({seen / (time.time() - t0):.0f} rows/s)", flush=True)
                batch = []
            for (u, text), data in zip(
                    batch, ex.map(fetch, (u for u, _ in batch))):
                if data is None:
                    writer.stats["download_failed"] += 1
                    continue
                try:
                    img = Image.open(io.BytesIO(data))
                    img.load()
                except Exception:
                    writer.stats["bad_image_file"] += 1
                    continue
                writer.add(img, text, spec.script, height, max_width)

    report = writer.close()
    report["rows_seen"] = seen
    report["seconds"] = round(time.time() - t0, 1)
    return report


def verify(root: Path, height: int, n_samples: int = 300) -> None:
    """Integrity-check a finalized root the way the trainer will read it."""
    import json as _json
    import random

    import numpy as np
    from streaming import Stream, StreamingDataset

    from src.taxonomy import NUM_GROUPS, SCRIPTS
    from src.encoding.decompose import decode_ids

    for split in ("train", "val"):
        split_dir = root / split
        chunks = []
        for c in sorted(split_dir.glob("chunk_*")):
            idx = c / "index.json"
            if idx.exists() and sum(
                    s.get("samples", 0)
                    for s in _json.loads(idx.read_text())["shards"]) > 0:
                chunks.append(c)
        if not chunks:
            raise RuntimeError(f"{split_dir}: no non-empty chunks")
        ds = StreamingDataset(
            streams=[Stream(local=str(c)) for c in chunks], shuffle=False)
        widths = np.load(str(split_dir / "widths.npy"))
        sids = np.load(str(split_dir / "script_ids.npy"))
        if not (len(ds) == len(widths) == len(sids)):
            raise RuntimeError(
                f"{split}: dataset={len(ds)} widths={len(widths)} "
                f"script_ids={len(sids)} — sidecars out of sync")

        rng = random.Random(0)
        for i in rng.sample(range(len(ds)), min(n_samples, len(ds))):
            s = ds[i]
            img = s["image"]
            assert img.shape[0] == 3 and img.shape[1] == height, \
                f"{split}[{i}]: image shape {img.shape}, expected H={height}"
            assert img.shape[2] == s["width"] == widths[i], \
                f"{split}[{i}]: width mismatch"
            assert s["script_id"] == sids[i], f"{split}[{i}]: sid mismatch"
            gl = s["group_labels"]
            assert gl.shape[0] == s["width"], f"{split}[{i}]: group_labels"
            assert gl.max() <= NUM_GROUPS, f"{split}[{i}]: bad group label"
            segs = _json.loads(s["segments"])
            assert segs and all("offset" in g for g in segs)
            # Decode round-trip is only well-defined for single-script lines:
            # mixed-script (multi-segment) targets are encoded per-segment with
            # different codecs, so decoding the whole sequence with one codec
            # is meaningless. Synth keeps mixed lines; real-v1 dropped them.
            if len(segs) == 1:
                script = SCRIPTS[s["script_id"]]
                tids = s["target_ids"][:s["target_len"]].tolist()
                dec = decode_ids(tids, script)
                assert dec == s["label"], \
                    (f"{split}[{i}] ({script}): decode mismatch "
                     f"{dec!r} != {s['label']!r}")

        by_script = {SCRIPTS[k]: int(v) for k, v in
                     zip(*np.unique(sids, return_counts=True))}
        print(f"{split}: {len(ds)} samples OK, widths {widths.min()}-"
              f"{widths.max()}, per-script {by_script}")
    print("verify OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default=None,
                   help="registry name, or comma-separated list")
    p.add_argument("--out", type=str, default="data/real-v1")
    p.add_argument("--height", type=int, default=48, choices=(32, 48, 64))
    # Hard reject threshold (not a squish target): keep long lines at natural
    # aspect; only drop crops wider than this to bound batch memory. ~42:1 at
    # 48px covers full document lines without distortion.
    p.add_argument("--max-width", type=int, default=2048)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None,
                   help="cap rows per source (smoke tests)")
    p.add_argument("--max-tier", type=int, default=2,
                   help="skip sources above this license tier")
    p.add_argument("--list", action="store_true")
    p.add_argument("--finalize", action="store_true",
                   help="rebuild root sidecars + metadata, then exit")
    p.add_argument("--verify", action="store_true",
                   help="integrity-check a finalized root, then exit")
    args = p.parse_args()

    if args.list:
        for s in REGISTRY.values():
            print(f"{s.name:18s} tier{s.license_tier}  {s.hub:42s} "
                  f"{s.script or 'per-row'}  {s.notes}")
        return

    out = Path(args.out)
    if args.finalize:
        finalize_root(out, args.height, args.max_width)
        print(f"finalized {out}")
        return
    if args.verify:
        verify(out, args.height)
        return

    if not args.source:
        p.error("--source, --list, --finalize, or --verify required")
    reports = []
    for name in args.source.split(","):
        name = name.strip()
        spec = REGISTRY.get(name)
        if spec is None:
            p.error(f"unknown source {name!r} — see --list")
        if spec.license_tier > args.max_tier:
            print(f"SKIP {name}: tier {spec.license_tier} > "
                  f"--max-tier {args.max_tier}")
            continue
        loaders = {"hub": ingest, "url_zip": ingest_url_zip,
                   "uc_zip": ingest_url_zip, "alto_zip": ingest_url_zip,
                   "bmod_zip": ingest_url_zip, "ndl_tsv": ingest_url_zip,
                   "lmdb_gdrive": ingest_lmdb_gdrive,
                   "hub_url_image": ingest_hub_url}
        loader_fn = loaders.get(spec.loader)
        if loader_fn is None:
            print(f"SKIP {name}: needs custom loader {spec.loader!r} "
                  f"(not implemented yet)")
            continue
        try:
            report = loader_fn(spec, out, args.height, args.max_width,
                               args.val_ratio, args.limit)
        except Exception as e:
            # One broken hub layout must not kill a multi-source run;
            # the failed source's partial chunks get rewritten on retry.
            print(f"FAILED {name}: {e!r}")
            reports.append({"source": name, "kept": 0, "error": repr(e)})
            continue
        print(json.dumps(report, ensure_ascii=False, indent=2))
        reports.append(report)

    total = sum(r["kept"] for r in reports)
    print(f"\nDone: {total} samples kept across {len(reports)} sources.")
    print("Run --finalize after the last source before training.")


if __name__ == "__main__":
    main()
