"""On-demand Blender render of the *current edited* motion, dispatched to the GPU pod.

After edits, the metrics/agent-log update immediately but the preview video still shows the base take
(rendering needs Blender on the pod). This module renders the session's current motion -- either just
the edited window (fast) or the full song with music -- as the canonical gray Y-Bot, and pulls the
mp4 back into ``server/media/<sid>/edited.mp4`` so the UI can swap it in. Job status is polled by the
UI, mirroring :mod:`server.processing`.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

from server.processing import REPO, _scp_from, _scp_to, _ssh, pod_config

_RJOBS: dict[str, dict] = {}
_RLOCK = threading.Lock()


def get_render_job(sid: str) -> dict:
    with _RLOCK:
        return dict(_RJOBS.get(sid, {"status": "idle", "message": "", "progress": 0}))


def _scp_many(cfg, locals_list, remote_dir: str, timeout: int = 300):
    """scp several local files to one remote dir in a SINGLE connection (one handshake, not N)."""
    import subprocess
    if not locals_list:
        return None
    return subprocess.run(
        ["scp", "-P", cfg.port, "-i", cfg.key, *[str(p) for p in locals_list],
         f"{cfg.target}:{remote_dir}/"],
        capture_output=True, text=True, timeout=timeout)


# scripts change rarely; upload them once per (server process, host) so repeat renders skip ~4 scp
# handshakes. Restarting the server re-sends them, so edits to the pod scripts still propagate.
_SCRIPTS_SENT: set = set()
_PREWARMED: set = set()


def prewarm_pod() -> None:
    """Best-effort: on server start, page-cache the pod venv's torch/scipy (so the FIRST render's
    forward-kinematics pays ~9s, not the ~15-25s cold import off the network volume) AND upload the
    render scripts (so the first render skips ~4 scp handshakes). Fire-and-forget; no-op if
    unconfigured or already done this process."""
    cfg = pod_config()
    if not cfg.host or cfg.host in _PREWARMED:
        return
    _PREWARMED.add(cfg.host)
    al_py = os.environ.get("AGENTLODGE_POD_PYTHON", f"{cfg.ws}/AgentLODGE/.venv/bin/python")

    def _w() -> None:
        try:
            _upload_scripts(cfg)
            _ssh(cfg, f"CUDA_VISIBLE_DEVICES= {al_py} -c 'import torch, numpy, scipy.signal' "
                      f">/dev/null 2>&1; echo warmed", timeout=180)
        except Exception:  # noqa: BLE001 - warming is best-effort
            pass

    threading.Thread(target=_w, daemon=True).start()


def _upload_scripts(cfg) -> None:
    """Upload the render scripts once per (server process, host), batched into a single scp."""
    if cfg.host in _SCRIPTS_SENT:
        return
    _ssh(cfg, f"mkdir -p {cfg.ws}/AgentLODGE/scripts")
    scripts = [REPO / "scripts" / s for s in
               ("render_one_ybot.sh", "render_blender_dance.py", "blender_render_ybot.py",
                "blender_studio.py")]
    scripts = [p for p in scripts if p.exists()]
    r = _scp_many(cfg, scripts, f"{cfg.ws}/AgentLODGE/scripts")
    if r is not None and r.returncode == 0:
        _SCRIPTS_SENT.add(cfg.host)


def _set(sid: str, **kw) -> None:
    with _RLOCK:
        _RJOBS.setdefault(sid, {}).update(kw)


def start_render(sid: str, motion: np.ndarray, media_dir: Path, *, scope: str = "window",
                 a: int | None = None, b: int | None = None) -> None:
    _set(sid, status="queued", message="queued", progress=3, scope=scope, started=time.time())
    threading.Thread(target=_render, args=(sid, np.asarray(motion), media_dir, scope, a, b),
                     daemon=True).start()


def _render(sid: str, motion: np.ndarray, media_dir: Path, scope: str,
            a: int | None, b: int | None) -> None:
    cfg = pod_config()
    if not cfg.host:
        _set(sid, status="error", progress=0,
             message="No GPU pod configured (set AGENTLODGE_POD_HOST). Rendering needs the pod's Blender.")
        return
    # window render is fast + silent; full render carries the song audio.
    with_audio = scope == "full"
    if scope == "window" and a is not None and b is not None:
        motion = motion[int(a):int(b)]
    if motion.shape[0] < 2:
        _set(sid, status="error", progress=0, message="nothing to render (empty window).")
        return
    try:
        _set(sid, status="rendering", progress=8, message="checking the GPU pod\u2026")
        # the pod's SSH occasionally has a slow banner exchange; retry the reachability probe
        reachable = False
        for _ in range(4):
            try:
                if _ssh(cfg, "echo ok", timeout=30).returncode == 0:
                    reachable = True
                    break
            except Exception:  # noqa: BLE001 - transient connect timeout: retry
                pass
            time.sleep(5)
        if not reachable:
            _set(sid, status="error", progress=0,
                 message=f"can't reach the GPU pod at {cfg.host}:{cfg.port} (retried).")
            return
        ws = cfg.ws
        media_dir.mkdir(parents=True, exist_ok=True)
        local_npy = media_dir / f"_render_{scope}.npy"
        np.save(local_npy, motion.astype(np.float32))

        _set(sid, progress=16, message="uploading the edited motion\u2026")
        _upload_scripts(cfg)                                 # once per process (pre-done at startup)
        remote_npy = f"{ws}/edit_render_{sid}.npy"
        if _scp_to(cfg, str(local_npy), remote_npy).returncode != 0:
            _set(sid, status="error", progress=0, message="upload of the motion failed.")
            return

        frames = int(motion.shape[0])
        base = f"{ws}/edit_render_{sid}"
        audio_sid = sid if with_audio else ""
        # Render FK runs in the persistent CUDA venv (the old /root/al_venv is wiped on pod restart).
        al_py = os.environ.get("AGENTLODGE_POD_PYTHON", f"{ws}/AgentLODGE/.venv/bin/python")
        # Fast preview vs high-quality export. The window preview drops resolution/samples AND runs the
        # SMPL forward-kinematics on CPU (CUDA_VISIBLE_DEVICES= empty) -- torch's CUDA context init alone
        # costs ~12s and EEVEE renders through EGL/OpenGL, not CUDA, so the GPU render is unaffected.
        # Measured: 180-frame window in ~36s at 480x480/16 (well under a minute) vs ~65s with GPU-FK/540.
        if scope == "window":
            rw = os.environ.get("AGENTLODGE_RENDER_WIN_W", "448")
            rh = os.environ.get("AGENTLODGE_RENDER_WIN_H", "448")
            rs = os.environ.get("AGENTLODGE_RENDER_WIN_SAMPLES", "8")
            render_env = f"RENDER_W={rw} RENDER_H={rh} RENDER_SAMPLES={rs} CUDA_VISIBLE_DEVICES="
            est_sec = 25 + int(frames * 0.25)
        else:
            rw = os.environ.get("AGENTLODGE_RENDER_FULL_W", "1080")
            rh = os.environ.get("AGENTLODGE_RENDER_FULL_H", "1080")
            rs = os.environ.get("AGENTLODGE_RENDER_FULL_SAMPLES", "96")
            render_env = f"RENDER_W={rw} RENDER_H={rh} RENDER_SAMPLES={rs}"
            est_sec = 45 + int(frames * 1.6)
        # Launch the render in the BACKGROUND with done/fail markers, then poll -- a blocking ssh over
        # a multi-minute render tends to hang its channel even after the render finishes.
        launch = _ssh(
            cfg,
            f"cd {ws}/AgentLODGE && sed -i 's/\\r$//' scripts/render_one_ybot.sh; "
            f"rm -f {base}.mp4 {base}.done {base}.fail {base}.log; "
            f"setsid bash -c 'AL_PY={al_py} WORKSPACE={ws} {render_env} bash scripts/render_one_ybot.sh "
            f"{remote_npy} {base}.mp4 {audio_sid} >> {base}.log 2>&1 && touch {base}.done || touch {base}.fail' "
            f"</dev/null >/dev/null 2>&1 & echo LAUNCHED",
            timeout=40)
        if "LAUNCHED" not in (launch.stdout or ""):
            _set(sid, status="error", progress=0, message="could not start the render on the pod.")
            return
        _set(sid, progress=24, frames=frames,
             message=f"rendering {frames} frames on the GPU "
                     f"({'~' + str(est_sec) + 's' if est_sec < 90 else '~' + str(max(1, est_sec // 60)) + ' min'})\u2026")

        deadline = time.time() + 60 * 45
        while time.time() < deadline:
            time.sleep(3)                                    # tight poll: the fast window render is ~35-50s
            try:
                chk = _ssh(cfg, f"if [ -f {base}.done ]; then echo DONE; "
                                f"elif [ -f {base}.fail ]; then echo FAIL; else echo RUN; fi", timeout=25)
                state = ((chk.stdout or "").strip().splitlines()[-1:] or [""])[0]
            except Exception:  # noqa: BLE001 - transient ssh hiccup: keep polling (render runs on pod)
                state = ""
            if state == "DONE":
                break
            if state == "FAIL":
                try:
                    tail = _ssh(cfg, f"tail -c 400 {base}.log 2>/dev/null", timeout=25).stdout or ""
                except Exception:  # noqa: BLE001
                    tail = ""
                _set(sid, status="error", progress=0, message=f"render failed: {tail.strip()[-280:]}")
                return
            frac = min(0.95, (time.time() - _RJOBS[sid].get("started", time.time())) / est_sec)
            _set(sid, progress=int(24 + 64 * frac))
        else:
            _set(sid, status="error", progress=0, message="render timed out on the pod.")
            return

        _set(sid, progress=90, message="downloading the rendered video\u2026")
        dst = media_dir / "edited.mp4"
        ok = False
        for _ in range(3):                                   # scp can also hit a transient hiccup
            try:
                if _scp_from(cfg, f"{base}.mp4", str(dst)).returncode == 0:
                    ok = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(5)
        if not ok:
            _set(sid, status="error", progress=0, message="could not fetch the rendered video.")
            return
        _set(sid, status="done", progress=100, message="ready", video="edited.mp4",
             elapsed=round(time.time() - _RJOBS[sid].get("started", time.time())))
    except Exception as exc:  # noqa: BLE001
        _set(sid, status="error", progress=0, message=f"render error: {exc}")
