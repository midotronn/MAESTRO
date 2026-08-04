"""Warm LODGE generation pool for the editor when it runs ON the GPU pod.

A cold live regeneration reloads the LODGE checkpoints every seed (~150s). This module keeps a single
persistent ``scripts/gen_daemon.py`` alive with the models preloaded, so a fresh seed is submitted as
a request file and read back once the daemon writes the ``.npy`` -- all local file I/O (the editor is
co-located with the pod), no ssh. If the daemon is unavailable, callers fall back to the existing
per-call ``gen_take.py`` path in :class:`server.processing.PodTakeProvider`.
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
DAEMON_DIR = Path(WS) / "gen_daemon"
_HB_STALE = 45          # seconds; a daemon whose heartbeat is older than this is considered dead
_START_LOCK = threading.Lock()


def _py() -> str:
    return os.environ.get("AGENTLODGE_POD_PYTHON", "python")


def _daemon_script() -> Path:
    return Path(WS) / "AgentLODGE" / "scripts" / "gen_daemon.py"


def on_pod() -> bool:
    host = (os.environ.get("AGENTLODGE_POD_HOST") or "").strip().lower()
    return host in ("127.0.0.1", "localhost", "0.0.0.0") and _daemon_script().exists()


def available(sid: str) -> bool:
    """True when the warm LODGE daemon can serve ``sid`` (co-located + song preprocessed for gen)."""
    return on_pod() and (Path(WS) / f"lodge_fd_{sid}_feats.npy").exists()


def _alive() -> bool:
    hb = DAEMON_DIR / "daemon.hb"
    try:
        return (DAEMON_DIR / "daemon.ready").exists() and hb.exists() \
            and (time.time() - hb.stat().st_mtime) < _HB_STALE
    except OSError:
        return False


def ensure_daemon() -> bool:
    if not on_pod():
        return False
    with _START_LOCK:
        if _alive():
            return True
        DAEMON_DIR.mkdir(parents=True, exist_ok=True)
        for name in ("daemon.ready", "daemon.hb"):
            try:
                (DAEMON_DIR / name).unlink()
            except OSError:
                pass
        cmd = (f"WORKSPACE={WS} {_py()} {_daemon_script()} --requests-dir {DAEMON_DIR} "
               f"> {DAEMON_DIR}/daemon.log 2>&1")
        try:
            subprocess.Popen(["setsid", "bash", "-c", cmd], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, close_fds=True)
        except Exception as exc:  # noqa: BLE001 - best-effort warm-up
            logger.warning("warm gen daemon failed to start: %s", exc)
            return False
    # wait briefly for the daemon to write daemon.ready (it does so before loading models)
    deadline = time.time() + 20
    while time.time() < deadline and not (DAEMON_DIR / "daemon.ready").exists():
        time.sleep(0.5)
    return (DAEMON_DIR / "daemon.ready").exists()


def warm_generate(sid: str, backbone: str, seed: int, a: int, b: int, *,
                  timeout: float = 600.0) -> Path | None:
    """Submit a warm LODGE window generation and wait for it. Returns the bank ``.npy`` path or None.

    The FIRST request pays the one-time model load (~150s); subsequent seeds are just the diffusion.
    Only LODGE is served warm; EDGE and any failure return None so the caller falls back.
    """
    if str(backbone) != "lodge" or not available(sid):
        return None
    if not ensure_daemon():
        return None
    rid = "g" + uuid.uuid4().hex[:10]
    req = {"id": rid, "sid": sid, "backbone": "lodge", "seed": int(seed), "a": int(a), "b": int(b)}
    tmp = DAEMON_DIR / f"{rid}.req.tmp"
    tmp.write_text(json.dumps(req))
    tmp.rename(DAEMON_DIR / f"{rid}.req")                    # atomic publish
    done, fail = DAEMON_DIR / f"{rid}.done", DAEMON_DIR / f"{rid}.fail"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if done.exists():
            try:
                path = Path(done.read_text().split()[0])
            except Exception:  # noqa: BLE001
                return None
            return path if path.exists() else None
        if fail.exists():
            logger.warning("warm gen %s failed: %s", rid, fail.read_text()[-300:])
            return None
        if not _alive() and not done.exists():              # daemon died mid-request
            return None
        time.sleep(0.5)
    return None
