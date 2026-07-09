"""Lipi smoke test on Modal — fully isolated from the real app.

Everything here is prefixed lipi-smoke-* so it can be nuked without touching
the real lipi-ocr app or its volumes:

    modal app stop lipi-smoke
    modal volume delete lipi-smoke-assets
    modal volume delete lipi-smoke-data
    modal volume delete lipi-smoke-runs

Run once end-to-end (a few dollars, ~30 min once assets are up):
    modal volume create lipi-smoke-assets
    modal volume put lipi-smoke-assets training_data /training_data
    modal run scripts/modal_smoke.py::generate
    modal run scripts/modal_smoke.py::train
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

_GPU = os.environ.get("LIPI_GPU", "H100")
_SOFT_DEADLINE_S = int(22.5 * 3600)

app = modal.App("lipi-smoke")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        # libraqm stack — Modal's mirror Pillow lacks raqm; complex scripts
        # need it. Source-build Pillow against these (see modal_app.py).
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
    .run_commands(
        "pip install --no-binary :all: --force-reinstall pillow==11.3.0")
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

assets_vol = modal.Volume.from_name("lipi-smoke-assets", create_if_missing=True)
data_vol = modal.Volume.from_name("lipi-smoke-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("lipi-smoke-runs", create_if_missing=True)
_VOLUMES = {_ASSETS: assets_vol, _DATA: data_vol, _RUNS: runs_vol}

_CKPT_RE = re.compile(r"moe_epoch(\d+)\.pt$")


def _prepare_repo():
    td = Path(_REPO, "training_data")
    if not td.is_symlink() and not td.exists():
        src = Path(_ASSETS, "training_data")
        if not src.is_dir():
            raise RuntimeError(
                "lipi-smoke-assets volume is empty. Upload once with:\n"
                "  modal volume put lipi-smoke-assets training_data "
                "/training_data")
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
    # flex_attention codegen exceeds H100's shared memory budget for our
    # window sizes; fall through to the SDPA path with an explicit mask.
    env["LIPI_DISABLE_FLEX_ATTENTION"] = "1"
    return env


def _latest_checkpoint(save_dir: Path):
    best = None
    for p in save_dir.glob("moe_epoch*.pt"):
        m = _CKPT_RE.fullmatch(p.name)
        if m:
            key = (int(m.group(1)), p.stat().st_mtime)
            if best is None or key > best[0]:
                best = (key, p)
    return best[1] if best else None


@app.function(
    image=image,
    cpu=16.0,          # smoke uses a smaller CPU box than the real generator
    memory=32 * 1024,
    timeout=2 * 3600,  # 200 samples/script * 27 scripts renders in minutes
    volumes=_VOLUMES,
)
def generate():
    """Tiny render: 200 train + 50 val per script (~30 min on 16 cores)."""
    _prepare_repo()
    out_dir = f"{_DATA}/smoke"
    cmd = [
        sys.executable, "scripts/data/generate.py",
        "--out", out_dir,
        "--samples-per-script", "200",
        "--val-samples-per-script", "50",
        "--workers", "16",
    ]
    print("Running:", shlex.join(cmd))
    subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
    data_vol.commit()
    return (f"smoke shards at lipi-smoke-data:/smoke: "
            + ", ".join(sorted(p.name for p in Path(out_dir).iterdir())))


@app.function(
    image=image,
    gpu=_GPU,
    cpu=16.0,
    memory=64 * 1024,
    timeout=2 * 3600,  # smoke = 1 epoch; if it can't finish an epoch, no point retrying
    volumes=_VOLUMES,
)
def train():
    """One epoch on the smoke shards; verifies capacity search, EMA, self-cond
    CTC, two-view consistency, checkpoint save/resume path end-to-end.
    """
    _prepare_repo()
    src = Path(_DATA, "smoke")
    if not (src / "train").is_dir():
        raise RuntimeError(
            "no smoke shards — run generate first "
            "(modal run scripts/modal_smoke.py::generate)")

    dst = Path("/tmp/smoke-shards")
    if not dst.exists():
        t0 = time.time()
        size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
        print(f"Staging {size / 1e9:.2f} GB to local disk...")
        shutil.copytree(src, dst)
        print(f"  staged in {time.time() - t0:.0f}s")

    save_dir = Path(_RUNS) / "smoke"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/train/train.py",
        "--data", str(dst),
        "--save-dir", str(save_dir),
        "--epochs", "1",
        "--consistency-weight", "0.5",
        # Eager, matching the real-run default (modal_app.py). torch.compile
        # is now non-crashing but not worth it on this architecture: the
        # data-dependent MoE routing, per-script CTC dispatch and per-layer
        # DropPath fragment the graph and thrash recompiles (measured ~3
        # img/s in warmup vs ~41 eager). Pass --compile to exercise it.
        "--no-compile",
    ]
    latest = _latest_checkpoint(save_dir)
    if latest:
        cmd += ["--resume", str(latest)]
        print(f"Resuming from {latest.name}")

    stop = threading.Event()

    def _commit_loop():
        while not stop.wait(120):
            try:
                runs_vol.commit()
            except Exception as e:  # noqa: BLE001
                print(f"volume commit failed (will retry): {e}")

    committer = threading.Thread(target=_commit_loop, daemon=True)
    committer.start()

    print("Running:", shlex.join(cmd))
    try:
        subprocess.run(cmd, cwd=_REPO, env=_child_env(), check=True)
    finally:
        stop.set()
        committer.join(timeout=60)
        runs_vol.commit()

    ckpts = sorted(p.name for p in save_dir.glob("moe_epoch*.pt"))
    return f"smoke training OK — checkpoints: {ckpts}"
