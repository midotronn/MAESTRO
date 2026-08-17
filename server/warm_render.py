"""Warm Blender render pool for the editor when it runs ON the GPU pod.

A cold Y-Bot render pays ~8s for Blender startup + scene load every time. This module keeps a small
pool of persistent Blender daemons (``scripts/blender_daemon.py``) alive with the scene preloaded, so
a render is just: write a poses ``.npz`` + a request, poll for the daemon's done marker, then ffmpeg
the frames. Because the hosted editor is co-located with the pod, everything here is local file I/O
and local subprocesses (no ssh/scp round-trip either).

The compare render (before vs after) submits its two passes to two different daemons so they render
in parallel. If the pool is unavailable (editor not on the pod, no cached scene, daemons not up),
callers fall back to the existing cold-render path in :mod:`server.rendering`.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

WS = os.environ.get("AGENTLODGE_POD_WS", "/workspace")
POOL_SIZE = int(os.environ.get("AGENTLODGE_WARM_POOL", "2"))
PROTOCOL_VERSION = 3
DAEMON_ROOT = Path(WS) / "blend_daemon"
REPO_ROOT = Path(__file__).resolve().parents[1]
_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
_HB_STALE = 30          # seconds; a daemon whose heartbeat is older than this is considered dead
_START_LOCK = threading.Lock()


def _blender() -> Path:
    return Path(WS) / "blender" / "blender"


def _scene() -> Path:
    return Path(WS) / "ybot_scene.blend"


def _ybot() -> Path:
    return Path(WS) / "EDGE" / "SMPL-to-FBX" / "ybot.fbx"


def _daemon_script() -> Path:
    return REPO_ROOT / "scripts" / "blender_daemon.py"


def on_pod() -> bool:
    """True when the editor is co-located with the pod (hosted mode) and the render assets exist, so
    we can drive local warm daemons instead of ssh + cold Blender."""
    host = (os.environ.get("AGENTLODGE_POD_HOST") or "").strip().lower()
    if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        return False
    return _blender().exists() and _scene().exists() and _ybot().exists() and _daemon_script().exists()


def _dir(i: int) -> Path:
    return DAEMON_ROOT / f"d{i}"


def _pid_alive(d: Path) -> bool:
    try:
        pid = int((d / "daemon.pid").read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _protocol_ready(d: Path) -> bool:
    try:
        return (d / "daemon.ready").read_text().strip() == str(PROTOCOL_VERSION)
    except OSError:
        return False


def _alive(d: Path) -> bool:
    hb = d / "daemon.hb"
    try:
        return (
            _pid_alive(d)
            and _protocol_ready(d)
            and hb.exists()
            and (time.time() - hb.stat().st_mtime) < _HB_STALE
        )
    except OSError:
        return False


def _stop_daemon(d: Path, timeout: float = 5.0) -> None:
    try:
        pid = int((d / "daemon.pid").read_text().strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.warning("could not stop incompatible Blender daemon %s: %s", pid, exc)
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(d):
            return
        time.sleep(0.05)
    logger.warning("incompatible Blender daemon %s did not stop within %.1fs", pid, timeout)


def _start_daemon(i: int, *, width: int, height: int, samples: int) -> None:
    d = _dir(i)
    d.mkdir(parents=True, exist_ok=True)
    for name in ("daemon.ready", "daemon.hb", "daemon.pid"):
        try:
            (d / name).unlink()
        except OSError:
            pass
    cmd = [
        str(_blender()),
        "-b",
        str(_scene()),
        "-noaudio",
        "-P",
        str(_daemon_script()),
        "--",
        "--ybot",
        str(_ybot()),
        "--requests-dir",
        str(d),
        "--width",
        str(width),
        "--height",
        str(height),
        "--samples",
        str(samples),
        "--idle-exit",
        "0",
    ]
    env = os.environ.copy()
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = _EGL
    log = (d / "daemon.log").open("wb")
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        env=env,
    )
    log.close()
    (d / "daemon.pid").write_text(str(proc.pid))


def ensure_pool(*, width: int = 448, height: int = 448, samples: int = 8, wait_ready: float = 0.0) -> int:
    """Start any dead daemons in the pool. Returns the number of daemons currently alive (optionally
    after waiting up to ``wait_ready`` seconds for freshly-started ones to load the scene)."""
    if not on_pod():
        return 0
    with _START_LOCK:
        for i in range(POOL_SIZE):
            d = _dir(i)
            if _pid_alive(d) and (d / "daemon.ready").exists() and not _protocol_ready(d):
                _stop_daemon(d)
            if not _alive(d) and not _pid_alive(d):
                try:
                    _start_daemon(i, width=width, height=height, samples=samples)
                except Exception as exc:  # noqa: BLE001 - best-effort warm-up
                    logger.warning("warm daemon %d failed to start: %s", i, exc)
    if wait_ready > 0:
        deadline = time.time() + wait_ready
        while time.time() < deadline and sum(_alive(_dir(i)) for i in range(POOL_SIZE)) < POOL_SIZE:
            time.sleep(1)
    return sum(_alive(_dir(i)) for i in range(POOL_SIZE))


def available() -> bool:
    return on_pod() and any(_alive(_dir(i)) for i in range(POOL_SIZE))


def warm_render(poses_npz: str, frames_dir: str, *, daemon: int, samples: int = 8,
                width: int = 448, height: int = 448, timeout: float = 600.0,
                engine: str = "eevee", denoise: int = 1,
                rig_metrics: str = "", fast: bool = False, stride: int = 1,
                projection_only: bool = False, batch_render: bool = False,
                video_path: str = "") -> bool:
    """Submit one render and optionally capture exact projected rig metrics. Returns True on success."""
    d = _dir(daemon % POOL_SIZE)
    if not _alive(d):
        return False
    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_path.glob("frame_*"):
        try:
            old_frame.unlink()
        except OSError:
            return False
    if video_path:
        try:
            Path(video_path).unlink(missing_ok=True)
        except OSError:
            return False
    rid = "r" + uuid.uuid4().hex[:10]
    done, fail = d / f"{rid}.done", d / f"{rid}.fail"
    req = {"id": rid, "poses": str(poses_npz), "frames_dir": str(frames_dir),
           "width": width, "height": height, "samples": samples,
           "engine": str(engine), "denoise": int(denoise), "fast": bool(fast),
           "stride": max(1, int(stride)), "rig_metrics": str(rig_metrics),
           "projection_only": bool(projection_only), "batch_render": bool(batch_render),
           "video_path": str(video_path)}
    tmp = d / f"{rid}.req.tmp"
    tmp.write_text(json.dumps(req))
    tmp.rename(d / f"{rid}.req")          # atomic publish so the daemon never reads a half-written file
    deadline = time.time() + timeout
    while time.time() < deadline:
        if done.exists():
            return not video_path or Path(video_path).is_file()
        if fail.exists():
            logger.warning("warm render %s failed: %s", rid, fail.read_text()[-300:])
            return False
        if not _alive(d):                  # daemon died mid-render
            return False
        time.sleep(0.05)
    return False
