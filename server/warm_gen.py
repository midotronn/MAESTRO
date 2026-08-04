"""Warm LODGE/EDGE generation pool for the editor when it runs ON the GPU pod.

A cold live regeneration reloads the diffusion checkpoints every seed (LODGE ~150s, EDGE ~72s). This
module keeps ONE persistent ``scripts/gen_daemon.py`` per backbone alive with its model preloaded, so
a fresh seed is submitted as a request file and read back once the daemon writes the ``.npy`` -- all
local file I/O (the editor is co-located with the pod), no ssh. Each backbone runs in its own process
(``gen_daemon/<backbone>``) so LODGE and EDGE code paths never mix. If a daemon is unavailable,
callers fall back to the per-call ``gen_take.py`` path in :class:`server.processing.PodTakeProvider`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

WS = os.environ.get("AGENTLODGE_POD_WS", "/workspace")
DAEMON_ROOT = Path(WS) / "gen_daemon"
_HB_STALE = 45          # seconds; a daemon whose heartbeat is older than this is considered dead
_START_LOCK = threading.Lock()


def _py() -> str:
    return os.environ.get("AGENTLODGE_POD_PYTHON", "python")


def _daemon_script() -> Path:
    return Path(WS) / "AgentLODGE" / "scripts" / "gen_daemon.py"


def _dir(backbone: str) -> Path:
    return DAEMON_ROOT / str(backbone)


def _feats(sid: str, backbone: str) -> Path:
    return Path(WS) / (f"lodge_fd_{sid}_feats.npy" if backbone == "lodge" else f"edge{sid}_slices.npy")


def on_pod() -> bool:
    host = (os.environ.get("AGENTLODGE_POD_HOST") or "").strip().lower()
    return host in ("127.0.0.1", "localhost", "0.0.0.0") and _daemon_script().exists()


def available(sid: str, backbone: str) -> bool:
    """True when the warm daemon can serve ``(sid, backbone)`` (co-located + preprocessed for gen)."""
    return backbone in ("lodge", "edge") and on_pod() and _feats(sid, backbone).exists()


def _alive(backbone: str) -> bool:
    d = _dir(backbone)
    hb = d / "daemon.hb"
    try:
        return (d / "daemon.ready").exists() and hb.exists() \
            and (time.time() - hb.stat().st_mtime) < _HB_STALE
    except OSError:
        return False


def ensure_daemon(backbone: str) -> bool:
    if not on_pod():
        return False
    d = _dir(backbone)
    with _START_LOCK:
        if _alive(backbone):
            return True
        d.mkdir(parents=True, exist_ok=True)
        for name in ("daemon.ready", "daemon.hb"):
            try:
                (d / name).unlink()
            except OSError:
                pass
        cmd = (f"WORKSPACE={WS} {_py()} {_daemon_script()} --backbone {backbone} "
               f"--requests-dir {d} > {d}/daemon.log 2>&1")
        try:
            subprocess.Popen(["setsid", "bash", "-c", cmd], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
        except Exception as exc:  # noqa: BLE001 - best-effort warm-up
            logger.warning("warm %s gen daemon failed to start: %s", backbone, exc)
            return False
    deadline = time.time() + 20                             # daemon writes daemon.ready before loading models
    while time.time() < deadline and not (d / "daemon.ready").exists():
        time.sleep(0.5)
    return (d / "daemon.ready").exists()


def warm_generate(sid: str, backbone: str, seed: int, a: int, b: int, *,
                  timeout: float = 600.0) -> Path | None:
    """Submit a warm window generation and wait for it. Returns the bank ``.npy`` path or None.

    The FIRST request to a backbone pays its one-time model load; subsequent seeds are just the
    diffusion. Any failure returns None so the caller falls back to the per-call path.
    """
    if not available(sid, backbone):
        return None
    if not ensure_daemon(backbone):
        return None
    d = _dir(backbone)
    rid = "g" + uuid.uuid4().hex[:10]
    req = {"id": rid, "sid": sid, "backbone": backbone, "seed": int(seed), "a": int(a), "b": int(b)}
    tmp = d / f"{rid}.req.tmp"
    tmp.write_text(json.dumps(req))
    tmp.rename(d / f"{rid}.req")                            # atomic publish
    done, fail = d / f"{rid}.done", d / f"{rid}.fail"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if done.exists():
            try:
                path = Path(done.read_text().split()[0])
            except Exception:  # noqa: BLE001
                return None
            return path if path.exists() else None
        if fail.exists():
            logger.warning("warm %s gen %s failed: %s", backbone, rid, fail.read_text()[-300:])
            return None
        if not _alive(backbone) and not done.exists():     # daemon died mid-request
            return None
        time.sleep(0.5)
    return None
