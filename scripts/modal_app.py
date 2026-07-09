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
    """Copy shards from the volume to container-local disk.

    StreamingDataset does per-sample random reads; local NVMe serves those
    far better than the FUSE-backed volume. copy_local=False reads the
    volume in place (use if shards outgrow the container disk).
    """
    src = Path(_DATA, name)
    if not (src / "train").is_dir():
        raise RuntimeError(
            f"no shards at lipi-data:/{name} — run the generate function "
            f"first (modal run scripts/modal_app.py::generate --out {name})")
    if not copy_local:
        return src
    dst = Path("/tmp/lipi-shards", name)
    if not dst.exists():
        t0 = time.time()
        size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
        print(f"Staging {size / 1e9:.1f} GB of shards to local disk...")
        shutil.copytree(src, dst)
        print(f"  staged in {time.time() - t0:.0f}s")
    return dst


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
def generate(out: str = "shards-v5", args: str = ""):
    """Render shards on a 48-core box, writing to the lipi-data volume.

    Retries resume: generate.py skips chunks that already exist in --out.
    """
    _prepare_repo()
    argv = shlex.split(args)
    _forbid(argv, "--out")
    out_dir = f"{_DATA}/{out}"
    cmd = [sys.executable, "scripts/data/generate.py", "--out", out_dir,
           *argv]
    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
    data_vol.commit()
    listing = ", ".join(sorted(p.name for p in Path(out_dir).iterdir()))
    return f"shards at lipi-data:/{out}: {listing}"


@app.function(
    image=image,
    gpu=_GPU,
    cpu=16.0,
    memory=64 * 1024,
    timeout=24 * 3600,
    volumes=_VOLUMES,
    retries=modal.Retries(max_retries=3, initial_delay=60.0),
)
def train(data: str = "shards-v5", run_name: str = "v5", args: str = "",
          copy_local: bool = True):
    """One ≤24 h training segment; spawns its own continuation if needed."""
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
    if "--resume" not in argv:
        latest = _latest_checkpoint(save_dir)
        if latest:
            cmd += ["--resume", str(latest)]
            print(f"Auto-resuming from {latest.name}")

    stop = threading.Event()
    committer = threading.Thread(target=_commit_loop, args=(stop,),
                                 daemon=True)
    committer.start()

    print("Running:", shlex.join(cmd))
    proc = subprocess.Popen(cmd, cwd=_REPO, env=_child_env())
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
