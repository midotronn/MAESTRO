"""On-demand Blender render of the *current edited* motion, dispatched to the GPU pod.

After edits, the metrics/agent-log update immediately but the preview video still shows the base take
(rendering needs Blender on the pod). This module renders the session's current motion -- either just
the edited window (fast) or the full song with music -- as the canonical gray Y-Bot, and pulls the
mp4 back into ``server/media/<sid>/edited.mp4`` so the UI can swap it in. Job status is polled by the
UI, mirroring :mod:`server.processing`.
"""

from __future__ import annotations

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
        _ssh(cfg, f"mkdir -p {ws}/AgentLODGE/scripts")
        for s in ("render_one_ybot.sh", "render_blender_dance.py", "blender_render_ybot.py",
                  "blender_studio.py"):
            p = REPO / "scripts" / s
            if p.exists():
                _scp_to(cfg, str(p), f"{ws}/AgentLODGE/scripts/{s}")
        remote_npy = f"{ws}/edit_render_{sid}.npy"
        if _scp_to(cfg, str(local_npy), remote_npy).returncode != 0:
            _set(sid, status="error", progress=0, message="upload of the motion failed.")
            return

        frames = int(motion.shape[0])
        est_sec = max(60, int(frames * 1.5))                 # observed ~1.5s/frame incl. FK + startup
        base = f"{ws}/edit_render_{sid}"
        audio_sid = sid if with_audio else ""
        # Launch the render in the BACKGROUND with done/fail markers, then poll -- a blocking ssh over
        # a multi-minute render tends to hang its channel even after the render finishes.
        launch = _ssh(
            cfg,
            f"cd {ws}/AgentLODGE && sed -i 's/\\r$//' scripts/render_one_ybot.sh; "
            f"rm -f {base}.mp4 {base}.done {base}.fail {base}.log; "
            f"setsid bash -c 'WORKSPACE={ws} bash scripts/render_one_ybot.sh {remote_npy} {base}.mp4 "
            f"{audio_sid} >> {base}.log 2>&1 && touch {base}.done || touch {base}.fail' "
            f"</dev/null >/dev/null 2>&1 & echo LAUNCHED",
            timeout=40)
        if "LAUNCHED" not in (launch.stdout or ""):
            _set(sid, status="error", progress=0, message="could not start the render on the pod.")
            return
        _set(sid, progress=24, frames=frames,
             message=f"rendering {frames} frames on the GPU (~{max(1, est_sec // 60)} min)\u2026")

        deadline = time.time() + 60 * 45
        while time.time() < deadline:
            time.sleep(10)
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
