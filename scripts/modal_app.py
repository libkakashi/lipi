"""Lipi on Modal: shard generation (48-core CPU box) + training (GPU).

One-time setup
--------------
    pip install modal
    modal setup                                # browser auth
    modal volume create lipi-assets
    modal volume put lipi-assets training_data /training_data   # ~1.5 GB

Smoke-test the full pipeline first (a few dollars, ~30 min)
-----------------------------------------------------------
    modal run scripts/modal_app.py::generate --out smoke \
        --args "--samples-per-script 200 --val-samples-per-script 50"
    modal run scripts/modal_app.py::train --data smoke --run-name smoke \
        --args "--epochs 1 --consistency-weight 0.5"

Real run
--------
    modal run --detach scripts/modal_app.py::generate --out shards-v5 \
        --args "--samples-per-script 30000 --vocab-proportional --include-chars"
    modal run --detach scripts/modal_app.py::train --data shards-v5 \
        --run-name v5 --args "--epochs 40 --consistency-weight 0.5"

Training self-resumes: checkpoints land on the lipi-runs volume, any
invocation with the same --run-name picks up the newest moe_epoch*.pt, and
a run that approaches Modal's 24 h function ceiling stops at a checkpoint
and spawns its own continuation. If anything is ever torn down, re-run the
same command — it resumes. Crashes retry automatically (which is also a
resume, since resume detection happens at function start).

    modal app logs lipi-ocr                                      # follow
    modal volume ls lipi-runs v5                                 # list ckpts
    modal volume get lipi-runs v5/moe_epoch40.pt checkpoints/    # download

GPU defaults to H100; override per launch: LIPI_GPU=A100-80GB modal run ...
(the capacity prober measures each GPU type once and caches on lipi-runs).
Do not pass --out/--data/--save-dir in --args — the wrapper owns those.
--resume is auto-detected; pass it in --args only to start a run from a
specific checkpoint (e.g. fine-tuning).
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

_LOCAL_REPO = Path(__file__).resolve().parents[1]
_REPO = "/root/lipi"
_ASSETS = "/vol/assets"
_DATA = "/vol/data"
_RUNS = "/vol/runs"

# Function config is evaluated on the client at `modal run` time, so a plain
# env var works as a launch knob.
_GPU = os.environ.get("LIPI_GPU", "H100")

# Leave room under Modal's 24 h function ceiling for data staging,
# torch.compile, checkpoint flush, and the final volume commit.
_SOFT_DEADLINE_S = int(22.5 * 3600)

app = modal.App("lipi-ocr")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",   # torch.compile host-side codegen + pillow build
        # libraqm stack: Modal's mirror ships a Pillow wheel without raqm,
        # so complex scripts (Arabic, Indic, Thai...) would render unshaped.
        # Install the system libs and source-build Pillow against them.
        "libraqm-dev", "libharfbuzz-dev", "libfribidi-dev",
        "libfreetype6-dev", "libjpeg-dev", "zlib1g-dev",
        "libfontconfig1-dev", "pkg-config",
        # Source-built Pillow only decodes what it links: without these,
        # compressed TIFF / JPEG2000 / WebP inputs (BDRC scans etc.) fail
        # as UnidentifiedImageError during real-data ingestion.
        "libtiff-dev", "libopenjp2-7-dev", "libwebp-dev",
    )
    .pip_install(
        "torch==2.11.0",
        "torchvision",
        "torchaudio",
        "timm>=1.0.0",
        "mosaicml-streaming>=0.10.0",
        "freetype-py>=2.4.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    )
    # Source-build Pillow so it links libraqm (features.check("raqm") is
    # True). Must come after the wheel Pillow that torchvision pulls in, so
    # this one wins. Verified on Modal: raqm=True, RAQM layout selectable.
    .run_commands(
        "pip install --no-binary :all: --force-reinstall pillow==11.3.0")
    # copy=True bakes the code into the image (a few MB, cheap layer): the
    # container filesystem stays writable for the training_data/.cache
    # symlinks below.
    .add_local_dir(
        _LOCAL_REPO,
        _REPO,
        copy=True,
        ignore=[
            ".git", "**/__pycache__", "*.pyc", ".pytest_cache", ".cache",
            ".venv", "training_data", "data", "checkpoints", "*.pt",
            "*.png", "render_check*", "debug", "assets",
            "lipi_ocr.egg-info", "uv.lock",
        ],
    )
)

assets_vol = modal.Volume.from_name("lipi-assets", create_if_missing=True)
data_vol = modal.Volume.from_name("lipi-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("lipi-runs", create_if_missing=True)

# Real-data ingestion pulls from the HF hub; the training image doesn't
# need those deps, so keep them in a derived layer.
prep_image = image.pip_install("datasets>=3.0.0", "hf_transfer>=0.1.6",
                               "requests>=2.31.0")
_VOLUMES = {_ASSETS: assets_vol, _DATA: data_vol, _RUNS: runs_vol}

_CKPT_RE = re.compile(r"moe_epoch(\d+)\.pt$")


def _prepare_repo():
    """Wire volume-backed assets and caches into the repo layout.

    All code resolves training_data/ and .cache/ relative to the repo root,
    so two symlinks make the container look exactly like a local checkout.
    The .cache link persists bucket-capacity tables and torch.compile
    artifacts across containers.
    """
    td = Path(_REPO, "training_data")
    if not td.is_symlink() and not td.exists():
        src = Path(_ASSETS, "training_data")
        if not src.is_dir():
            raise RuntimeError(
                "lipi-assets volume is empty. Upload assets once with:\n"
                "  modal volume put lipi-assets training_data /training_data")
        td.symlink_to(src)
    cache = Path(_RUNS, ".cache")
    cache.mkdir(parents=True, exist_ok=True)
    repo_cache = Path(_REPO, ".cache")
    if not repo_cache.is_symlink() and not repo_cache.exists():
        repo_cache.symlink_to(cache)


def _child_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO
    env["PYTHONUNBUFFERED"] = "1"
    env["TORCHINDUCTOR_CACHE_DIR"] = f"{_RUNS}/.cache/torchinductor"
    env["TRITON_CACHE_DIR"] = f"{_RUNS}/.cache/triton"
    # The flex_attention Triton kernel needs ~278 KB of shared memory at our
    # window sizes; H100/A100/L40S all cap below that, so every forward pass
    # crashes. Fall through to the SDPA path (same numerics). Verified by the
    # smoke run — not optional on these GPUs.
    env["LIPI_DISABLE_FLEX_ATTENTION"] = "1"
    return env


def _forbid(argv, *flags):
    clash = [f for f in flags if f in argv]
    if clash:
        raise ValueError(
            f"{' '.join(clash)} is owned by the wrapper — see the module "
            f"docstring for the equivalent function parameter")


def _latest_checkpoint(save_dir: Path):
    best = None
    for p in save_dir.glob("moe_epoch*.pt"):
        m = _CKPT_RE.fullmatch(p.name)
        if m:
            key = (int(m.group(1)), p.stat().st_mtime)
            if best is None or key > best[0]:
                best = (key, p)
    return best[1] if best else None


def _stage_data(name: str, copy_local: bool) -> Path:
    """Stage one shard set, or mix several ("synth-v8,real-v1").

    Multi-name staging symlinks every train/val chunk dir into one mix
    root, rebuilds the root sidecars from per-chunk sidecars (same sorted
    order LipiStreamingDataset streams in), and unions active_scripts in
    metadata — no shard bytes are duplicated.
    """
    names = [n.strip() for n in name.split(",") if n.strip()]
    if len(names) == 1:
        return _stage_one(names[0], copy_local)
    if not copy_local:
        raise RuntimeError("mixing datasets requires copy_local=True")
    staged = [_stage_one(n, True) for n in names]
    return _merge_staged(staged, "mix+" + "+".join(names))


def _merge_staged(roots: list[Path], tag: str) -> Path:
    import numpy as np
    import torch

    dst = Path("/tmp/lipi-shards", tag)
    if (dst / ".stage_complete").exists():
        return dst
    if dst.exists():
        shutil.rmtree(dst)

    metas = []
    for r in roots:
        mp = r / "metadata.pt"
        if not mp.exists():
            raise RuntimeError(f"{r} has no metadata.pt — finalize it first")
        metas.append(torch.load(mp, weights_only=False))
    for key in ("height", "taxonomy_version"):
        vals = {m.get(key) for m in metas}
        if len(vals) != 1:
            raise RuntimeError(f"can't mix shard sets with different {key}: "
                               f"{[m.get(key) for m in metas]}")

    for split in ("train", "val"):
        split_dst = dst / split
        split_dst.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(roots):
            for chunk in sorted((r / split).glob("chunk_*")):
                if not (chunk / "widths.npy").exists():
                    raise RuntimeError(
                        f"{chunk} lacks per-chunk sidecars — regenerate "
                        f"with current generate.py / prepare.py")
                link = split_dst / f"chunk_m{i}_{chunk.name[6:]}"
                link.symlink_to(chunk)
        for name in ("widths.npy", "script_ids.npy"):
            parts = [np.load(str(c / name))
                     for c in sorted(split_dst.glob("chunk_*"))]
            arr = (np.concatenate(parts).astype(np.int32)
                   if parts else np.zeros(0, dtype=np.int32))
            np.save(str(split_dst / name), arr)

    # Union scripts across sets, preserving taxonomy order.
    sys.path.insert(0, _REPO)
    from src.taxonomy import SCRIPT_TO_GROUP, SCRIPTS
    active = set()
    for m in metas:
        active.update(m.get("active_scripts", []))
    scripts = [s for s in SCRIPTS if s in active]
    groups, seen = [], set()
    for s in scripts:
        g = SCRIPT_TO_GROUP[s]
        if g not in seen:
            groups.append(g)
            seen.add(g)
    meta = dict(metas[0])
    meta["active_scripts"] = scripts
    meta["active_groups"] = groups
    torch.save(meta, dst / "metadata.pt")

    (dst / ".stage_complete").touch()
    print(f"Mixed {len(roots)} shard sets at {dst}")
    return dst


def _stage_one(name: str, copy_local: bool) -> Path:
    """Copy shards from the volume to container-local disk, in parallel.

    StreamingDataset does per-sample random reads; local NVMe serves those
    far better than the FUSE-backed volume, so a GPU run is compute-bound
    only when the shards are staged locally (reading the volume in place
    starves the GPU — measured ~30-70 img/s vs the ~300-1000 img/s compute
    ceiling). A serial shutil.copytree of ~85 GB over FUSE took 20+ min;
    the volume is object-storage-backed, so parallel file copies get far
    higher aggregate bandwidth. copy_local=False reads in place (use only
    if the shards outgrow the container disk).
    """
    src = Path(_DATA, name)
    if not (src / "train").is_dir():
        raise RuntimeError(
            f"no shards at lipi-data:/{name} — run the generate function "
            f"first (modal run scripts/modal_app.py::generate --out {name})")
    if not copy_local:
        return src

    dst = Path("/tmp/lipi-shards", name)
    marker = dst / ".stage_complete"
    if marker.exists():
        return dst  # already fully staged in this container
    if dst.exists():
        shutil.rmtree(dst)  # partial stage from an interrupted attempt

    t0 = time.time()
    files = [p for p in src.rglob("*") if p.is_file()]
    size = sum(f.stat().st_size for f in files)
    print(f"Staging {size / 1e9:.1f} GB ({len(files)} files) to local disk "
          f"(parallel)...")

    def _copy(f: Path):
        d = dst / f.relative_to(src)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, d)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=32) as ex:
        for _ in ex.map(_copy, files):
            pass
    marker.touch()
    print(f"  staged in {time.time() - t0:.0f}s")
    return dst


@app.function(
    image=image,
    cpu=32.0,
    memory=64 * 1024,
    timeout=24 * 3600,
    volumes=_VOLUMES,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
)
def convert_height(inp: str, out: str, to_height: int = 48,
                   from_height: int = 64, shards: int = 1,
                   shard_index: int = -1):
    """Resize an MDS shard set to a new height in place on lipi-data.

    Reads lipi-data:/{inp}, writes lipi-data:/{out} at --to-height (aspect
    preserving). shards>1 fans conversion across containers by source-chunk
    index; the coordinator finalizes sidecars+metadata after all finish.
    Chunk writes are idempotent (a .done marker skips completed chunks), so
    retries resume.
    """
    _prepare_repo()
    in_dir, out_dir = f"{_DATA}/{inp}", f"{_DATA}/{out}"
    base = [sys.executable, "scripts/data/convert_height.py",
            "--in", in_dir, "--out", out_dir,
            "--from-height", str(from_height), "--to-height", str(to_height)]

    if shards > 1 and shard_index < 0:
        print(f"Fanning conversion to {shards} containers (this=shard 0)...")
        handles = [convert_height.spawn(inp=inp, out=out, to_height=to_height,
                                        from_height=from_height, shards=shards,
                                        shard_index=i)
                   for i in range(1, shards)]
        subprocess.run(base + ["--chunk-shard", f"0:{shards}"],
                       cwd=_REPO, env=_child_env(), check=True)
        data_vol.commit()
        for h in handles:
            h.get()
        data_vol.reload()
        subprocess.run([sys.executable, "scripts/data/convert_height.py",
                        "--out", out_dir, "--to-height", str(to_height),
                        "--finalize"], cwd=_REPO, env=_child_env(), check=True)
        data_vol.commit()
        return f"converted lipi-data:/{inp} → /{out} @ {to_height}px"

    cmd = base[:]
    if shard_index >= 0:
        cmd += ["--chunk-shard", f"{shard_index}:{shards}"]
    else:
        cmd += ["--chunk-shard", "0:1"]
    subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
    if shard_index <= 0:
        subprocess.run([sys.executable, "scripts/data/convert_height.py",
                        "--out", out_dir, "--to-height", str(to_height),
                        "--finalize"], cwd=_REPO, env=_child_env(), check=True)
    data_vol.commit()
    return f"converted lipi-data:/{inp} → /{out} @ {to_height}px"


@app.function(
    image=image,
    cpu=16.0,
    memory=64 * 1024,
    timeout=2 * 3600,
    volumes=_VOLUMES,
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
)
def audit(data: str, height: int = 48):
    """Deep data audit of one or more shard roots on lipi-data (read-only).

    Samples the shards directly (no staging) and reports label-noise rate,
    CTC width-feasibility, image degeneracy, train/val leakage, and per-script
    synth-vs-real coverage, then a PASS/WARN/FAIL verdict. See
    scripts/data/audit.py for the checks.
    """
    _prepare_repo()
    cmd = [sys.executable, "scripts/data/audit.py",
           "--data", data, "--root", _DATA, "--height", str(height)]
    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
    return f"audit complete for {data}"


@app.function(
    image=image,
    cpu=8.0,
    memory=32 * 1024,
    timeout=1 * 3600,
    volumes=_VOLUMES,
)
def dump_samples(data: str, split: str = "train", per_source: int = 2,
                 scale: int = 3, script: str = ""):
    """Dump a source-stratified sample of crops + label manifest to
    lipi-assets:/audit_samples for visual pairing inspection. Read-only on
    the data volume; download with `modal volume get lipi-assets audit_samples`.
    script (optional) restricts to one script by name (for synth shards not
    stratified by source prefix).
    """
    _prepare_repo()
    out = f"{_ASSETS}/audit_samples"
    cmd = [sys.executable, "scripts/data/dump_samples.py",
           "--data", data, "--root", _DATA, "--split", split,
           "--out", out, "--per-source", str(per_source), "--scale", str(scale)]
    if script:
        cmd += ["--script", script]
    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
    assets_vol.commit()
    return f"dumped samples to lipi-assets:/audit_samples"


@app.function(
    image=image,
    cpu=16.0,
    memory=96 * 1024,
    timeout=4 * 3600,
    volumes=_VOLUMES,
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
)
def build_subset(real: str, synth: str, out: str = "subset-v1",
                 total: int = 700000, goal: str = "printed",
                 floor: int = 15000, cap: int = 55000,
                 real_cap: int = 45000, synth_cap: int = 22000,
                 synth_min: int = 5000,
                 val_frac: float = 0.05, plan: bool = False):
    """Curate a small training subset (effective-vocab balance + domain ratio)
    from the full real+synth shards, writing a drop-in root to lipi-data:/{out}.
    plan=True prints the quota table without writing. See build_subset.py.
    """
    _prepare_repo()
    cmd = [sys.executable, "scripts/data/build_subset.py",
           "--real", real, "--synth", synth, "--out", out, "--root", _DATA,
           "--total", str(total), "--goal", goal, "--floor", str(floor),
           "--cap", str(cap), "--real-cap", str(real_cap),
           "--synth-cap", str(synth_cap), "--synth-min", str(synth_min),
           "--val-frac", str(val_frac)]
    if plan:
        cmd.append("--plan")
    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
    if not plan:
        data_vol.commit()
    return f"{'planned' if plan else 'built'} subset {out}"


def _commit_loop(stop: threading.Event, interval_s: int = 300):
    """Flush checkpoint writes to the volume while training runs."""
    while not stop.wait(interval_s):
        try:
            runs_vol.commit()
        except Exception as e:  # noqa: BLE001 — keep training alive
            print(f"volume commit failed (will retry): {e}")


@app.function(
    image=image,
    cpu=48.0,
    memory=128 * 1024,
    timeout=24 * 3600,
    volumes=_VOLUMES,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
)
def generate(out: str = "shards-v5", args: str = "", shards: int = 1,
             shard_index: int = -1):
    """Render shards on 48-core boxes, writing to the lipi-data volume.

    shards=N fans generation out across N containers: chunk numbering is
    deterministic and chunks are idempotent, so each container takes the
    chunks where idx % N == its index (--chunk-shard) and skips sidecar
    assembly (--no-finalize); the coordinator waits for all of them, then
    runs one finalize pass (all chunks exist → assembly + metadata only).
    Wall-clock divides by ~N for the render phase.

    Retries resume: generate.py skips chunks that already exist in --out.
    """
    _prepare_repo()
    argv = shlex.split(args)
    _forbid(argv, "--out", "--chunk-shard", "--no-finalize")
    out_dir = f"{_DATA}/{out}"

    if shards > 1 and shard_index < 0:
        # Coordinator works shard 0 itself instead of idling on a full
        # 48-CPU reservation while the children render.
        print(f"Fanning out to {shards} generation containers "
              f"(this one takes shard 0)...")
        handles = [generate.spawn(out=out, args=args, shards=shards,
                                  shard_index=i)
                   for i in range(1, shards)]
        cmd = [sys.executable, "scripts/data/generate.py", "--out", out_dir,
               *argv, "--chunk-shard", f"0:{shards}", "--no-finalize"]
        print("Running:", shlex.join(cmd))
        subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
        data_vol.commit()
        for h in handles:
            h.get()  # propagate any shard failure
        # See sibling containers' committed chunks, then finalize.
        data_vol.reload()
        cmd = [sys.executable, "scripts/data/generate.py", "--out", out_dir,
               *argv]
        print("Finalizing:", shlex.join(cmd))
        subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
        data_vol.commit()
        listing = ", ".join(sorted(p.name for p in Path(out_dir).iterdir()))
        return f"shards at lipi-data:/{out}: {listing}"

    cmd = [sys.executable, "scripts/data/generate.py", "--out", out_dir,
           *argv]
    if shard_index >= 0:
        cmd += ["--chunk-shard", f"{shard_index}:{shards}", "--no-finalize"]
    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
    data_vol.commit()
    listing = ", ".join(sorted(p.name for p in Path(out_dir).iterdir()))
    return f"shards at lipi-data:/{out}: {listing}"


@app.function(
    image=prep_image,
    cpu=8.0,
    memory=32 * 1024,
    # Sources stream from the HF hub (no full-dataset staging), so the
    # default container disk is enough; explicit ephemeral_disk can't go
    # below 512 GiB on Modal, which would be pure over-provisioning here.
    timeout=24 * 3600,
    volumes=_VOLUMES,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
)
def prepare_real(source: str = "", out: str = "real-v1", args: str = "",
                 hf_token: str = "", finalize: bool = False,
                 verify: bool = False):
    """Ingest real OCR datasets into MDS chunks on the lipi-data volume.

    Usage:
        modal run scripts/modal_app.py::prepare_real \
            --source hhd_ethiopic,norhand_v3 --out real-v1
        modal run scripts/modal_app.py::prepare_real --out real-v1 --finalize

    Sources are registered in scripts/data/real/sources.py. Chunk writes
    are idempotent (re-runs wipe and rewrite that source's chunks), so
    Modal retries are safe. Run --finalize once after the last source.
    Train on the mix with: ::train --data "shards-v8,real-v1".
    """
    _prepare_repo()
    argv = shlex.split(args)
    _forbid(argv, "--out", "--source", "--finalize", "--verify")

    env = _child_env()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["HF_HOME"] = "/tmp/hf-cache"
    if hf_token:
        env["HF_TOKEN"] = hf_token

    cmd = [sys.executable, "scripts/data/real/prepare.py",
           "--out", f"{_DATA}/{out}", *argv]
    if finalize:
        cmd.append("--finalize")
    elif verify:
        cmd.append("--verify")
    elif source:
        cmd += ["--source", source]
    else:
        raise RuntimeError("pass --source, --finalize, or --verify")
    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, cwd=_REPO, env=env, check=True)
    data_vol.commit()
    listing = ", ".join(sorted(p.name for p in Path(f"{_DATA}/{out}",
                                                    "train").glob("chunk_*")))
    return f"real chunks at lipi-data:/{out}: {listing or '(none yet)'}"


@app.function(
    image=image,
    gpu=_GPU,
    # On-the-fly augmentation is CPU-heavy per sample; with too few cores the
    # dataloader can't feed the GPU and it idles. 32 cores keeps ~28 loader
    # workers busy so the H100 stays compute-bound. Cheap relative to the GPU.
    cpu=32.0,
    memory=96 * 1024,
    # Shards stage to container-local NVMe. real-v1 alone measured 692 GB; a
    # synth+real MIX (shards-v9 585 GB + real-v1 692 GB ≈ 1.28 TB) needs
    # ~1400 GiB. But an oversized disk request forces scarcer instance types
    # that queue/preempt more (a 1400 GiB real-only run stuck 50+ min in
    # staging). Scale per run via LIPI_DISK_GIB: default 1000 fits a single
    # real set; export LIPI_DISK_GIB=1400 for a mix.
    ephemeral_disk=int(os.environ.get("LIPI_DISK_GIB", "1000")) * 1024,
    timeout=24 * 3600,
    volumes=_VOLUMES,
    retries=modal.Retries(max_retries=3, initial_delay=60.0),
)
def train(data: str = "shards-v5", run_name: str = "v5", args: str = "",
          copy_local: bool = True, profile: str = ""):
    """One ≤24 h training segment; spawns its own continuation if needed.

    profile sets LIPI_PROFILE_STEP: "1" = per-section ms breakdown for the
    first steps; "2" = torch.profiler top-ops dump for a few steps, then
    exit. Empty = normal run.
    """
    _prepare_repo()
    argv = shlex.split(args)
    _forbid(argv, "--data", "--save-dir")

    data_dir = _stage_data(data, copy_local)
    save_dir = Path(_RUNS) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "scripts/train/train.py",
           "--data", str(data_dir), "--save-dir", str(save_dir), *argv]
    # Default to eager: torch.compile's Inductor value-range analysis trips
    # on the SDPA fallback's boolean mask algebra ("A Boolean argument can
    # only be used in Eq and Ne") under dynamic shapes. Compile is a speed
    # optimization, not correctness — a crashing default is worse than a
    # working one. Pass --compile in --args once the SDPA path is made
    # compile-clean to recover the ~1.5–2x speedup.
    if "--compile" in argv:
        argv.remove("--compile")
        cmd = [c for c in cmd if c != "--compile"]
    elif "--no-compile" not in argv:
        cmd.append("--no-compile")
    # Use the cores we pay for: this container is cpu=32 but train.py
    # defaults to 12 dataloader workers, so on-the-fly augmentation (CPU-heavy,
    # 2 ops/sample) starves the GPU. Match ~28 workers to the 32 cores unless
    # the caller overrides.
    if "--num-workers" not in argv:
        cmd += ["--num-workers", "28"]
    if "--resume" not in argv:
        latest = _latest_checkpoint(save_dir)
        if latest:
            cmd += ["--resume", str(latest)]
            print(f"Auto-resuming from {latest.name}")

    stop = threading.Event()
    committer = threading.Thread(target=_commit_loop, args=(stop,),
                                 daemon=True)
    committer.start()

    env = _child_env()
    if profile:
        env["LIPI_PROFILE_STEP"] = profile
    print("Running:", shlex.join(cmd))
    proc = subprocess.Popen(cmd, cwd=_REPO, env=env)
    try:
        code = proc.wait(timeout=_SOFT_DEADLINE_S)
    except subprocess.TimeoutExpired:
        # train.py checkpoints every 500 batches and saves atomically, so
        # killing here loses at most 500 batches of the current epoch.
        print(f"Soft deadline ({_SOFT_DEADLINE_S / 3600:.1f} h) hit — "
              f"stopping to re-chain")
        proc.terminate()
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        code = None
    finally:
        stop.set()
        committer.join(timeout=60)
        runs_vol.commit()

    if code == 0:
        return f"training complete — checkpoints in lipi-runs:/{run_name}"
    if code is None:
        handle = train.spawn(data=data, run_name=run_name, args=args,
                             copy_local=copy_local)
        return f"continuing as {handle.object_id}"
    raise RuntimeError(f"train.py exited with code {code}")
