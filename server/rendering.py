"""On-demand Blender render of the *current edited* motion, dispatched to the GPU pod.

After edits, the metrics/agent-log update immediately but the preview video still shows the base take
(rendering needs Blender on the pod). This module renders the session's current motion -- either just
the edited window (fast) or the full song with music -- as the canonical gray Y-Bot, and pulls the
mp4 back into ``server/media/<sid>/edited.mp4`` so the UI can swap it in. Job status is polled by the
UI, mirroring :mod:`server.processing`.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from server.processing import REPO, _scp_from, _scp_to, _ssh, pod_config

logger = logging.getLogger(__name__)

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
            # The SMPL-X-derived FK joint template is licence gated (not bundled); fetch it locally so
            # server-side FK works. If it never arrives, rendering falls back to the pod's torch FK.
            tmpl = REPO / "server" / "data" / "smplx_neu_J_1.npy"
            if not tmpl.exists():
                tmpl.parent.mkdir(parents=True, exist_ok=True)
                _scp_from(cfg, f"{cfg.ws}/LODGE/data/smplx_neu_J_1.npy", str(tmpl))
            _upload_scripts(cfg)
            egl = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
            bb = f"{cfg.ws}/blender/blender"
            # Build the cached Blender scene once (rig + studio baked) so renders skip the FBX import.
            _ssh(cfg, f"test -f {cfg.ws}/ybot_scene.blend || (cd {cfg.ws}/AgentLODGE && "
                      f"__EGL_VENDOR_LIBRARY_FILENAMES={egl} {bb} -b -noaudio -P "
                      f"scripts/blender_render_ybot.py -- --build-scene {cfg.ws}/ybot_scene.blend "
                      f"--ybot {cfg.ws}/EDGE/SMPL-to-FBX/ybot.fbx --width 448 --height 448 --samples 8 "
                      f">/dev/null 2>&1)", timeout=150)
            # Warm torch too, for the pod-torch-FK fallback path only.
            _ssh(cfg, f"CUDA_VISIBLE_DEVICES= {al_py} -c 'import torch, numpy, scipy.signal' "
                      f">/dev/null 2>&1; echo warmed", timeout=180)
            # When the editor runs ON the pod, bring up the warm Blender daemon pool so compare (and
            # future renders) skip the ~8s Blender startup entirely.
            try:
                from server import warm_render
                warm_render.ensure_pool()
            except Exception:  # noqa: BLE001 - warm pool is best-effort
                pass
        except Exception:  # noqa: BLE001 - warming is best-effort
            pass

    threading.Thread(target=_w, daemon=True).start()


def _upload_scripts(cfg) -> None:
    """Upload the render scripts once per (server process, host), batched into a single scp."""
    if cfg.host in _SCRIPTS_SENT:
        return
    _ssh(cfg, f"mkdir -p {cfg.ws}/AgentLODGE/scripts")
    scripts = [REPO / "scripts" / s for s in
               ("render_one_ybot.sh", "render_poses_ybot.sh", "render_blender_dance.py",
                "blender_render_ybot.py", "blender_studio.py")]
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

        # Server-side FK: compute the render poses locally (pure numpy, validated identical to the pod's
        # torch FK) and upload the small npz. The pod then renders with NO torch import at all -- killing
        # the single biggest source of render-time variance (~12-24s cold torch import off the network
        # volume). Falls back to uploading the raw motion + the torch-FK script if anything goes wrong.
        use_warm = os.environ.get("AGENTLODGE_RENDER_WARM", "1") != "0"
        warm_ok = False
        remote_in = f"{ws}/edit_render_{sid}.npy"
        render_script = "render_one_ybot.sh"
        _set(sid, progress=16, message="preparing the render\u2026")
        _upload_scripts(cfg)                                 # once per process (pre-done at startup)
        if use_warm:
            try:
                from server import fk
                local_poses = media_dir / f"_render_{scope}_poses.npz"
                fk.save_poses_npz(motion, str(local_poses))
                remote_poses = f"{ws}/edit_render_{sid}_poses.npz"
                if _scp_to(cfg, str(local_poses), remote_poses).returncode == 0:
                    remote_in, render_script, warm_ok = remote_poses, "render_poses_ybot.sh", True
            except Exception as exc:  # noqa: BLE001 - fall back to the pod torch FK
                logger.warning("server-side FK failed (%s); falling back to pod torch FK", exc)
        if not warm_ok:
            local_npy = media_dir / f"_render_{scope}.npy"
            np.save(local_npy, motion.astype(np.float32))
            if _scp_to(cfg, str(local_npy), remote_in).returncode != 0:
                _set(sid, status="error", progress=0, message="upload of the motion failed.")
                return

        frames = int(motion.shape[0])
        base = f"{ws}/edit_render_{sid}"
        audio_sid = sid if with_audio else ""
        # Render FK runs in the persistent CUDA venv (the old /root/al_venv is wiped on pod restart).
        al_py = os.environ.get("AGENTLODGE_POD_PYTHON", f"{ws}/AgentLODGE/.venv/bin/python")
        # Fast preview vs high-quality export. The window preview drops resolution/samples; with warm
        # (server-FK) mode the pod render is Blender-only. Full export keeps 1080/96.
        if scope == "window":
            rw = os.environ.get("AGENTLODGE_RENDER_WIN_W", "448")
            rh = os.environ.get("AGENTLODGE_RENDER_WIN_H", "448")
            rs = os.environ.get("AGENTLODGE_RENDER_WIN_SAMPLES", "8")
            # CUDA_VISIBLE_DEVICES= only matters for the pod-torch-FK fallback (EEVEE renders via EGL).
            render_env = f"RENDER_W={rw} RENDER_H={rh} RENDER_SAMPLES={rs} CUDA_VISIBLE_DEVICES="
            est_sec = (15 if warm_ok else 25) + int(frames * 0.2)
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
            f"cd {ws}/AgentLODGE && sed -i 's/\\r$//' scripts/{render_script}; "
            f"rm -f {base}.mp4 {base}.done {base}.fail {base}.log; "
            f"setsid bash -c 'AL_PY={al_py} WORKSPACE={ws} {render_env} bash scripts/{render_script} "
            f"{remote_in} {base}.mp4 {audio_sid} >> {base}.log 2>&1 && touch {base}.done || touch {base}.fail' "
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


# =========================================================================== before/after compare
# Renders the edited window twice -- the pre-edit ("before") and current ("after") motion -- as two
# short window clips, launched in PARALLEL on the pod so the wall time stays close to a single render.
# The UI plays them side by side, synced, with the before/after window metrics.
_CJOBS: dict[str, dict] = {}


def get_compare_job(sid: str) -> dict:
    with _RLOCK:
        return dict(_CJOBS.get(sid, {"status": "idle", "message": "", "progress": 0}))


def _cset(sid: str, **kw) -> None:
    with _RLOCK:
        _CJOBS.setdefault(sid, {}).update(kw)


def start_compare_render(sid: str, before_motion: np.ndarray, after_motion: np.ndarray,
                         media_dir: Path, *, metrics: dict | None = None) -> None:
    _cset(sid, status="queued", message="queued", progress=3, started=time.time(),
          metrics=metrics or {}, before_video=None, after_video=None)
    threading.Thread(
        target=_compare_render,
        args=(sid, np.asarray(before_motion), np.asarray(after_motion), media_dir),
        daemon=True).start()


def _launch_window_render(cfg, sid: str, tag: str, motion: np.ndarray, media_dir: Path,
                          ws: str) -> str | None:
    """Server-FK a window motion, upload it, and launch a background Blender window render on the pod.

    Returns the remote base path (``.../cmp_<tag>_<sid>``) whose ``.mp4``/``.done``/``.fail`` markers to
    poll, or ``None`` on failure. Mirrors the warm (server-FK) path of :func:`_render`.
    """
    al_py = os.environ.get("AGENTLODGE_POD_PYTHON", f"{ws}/AgentLODGE/.venv/bin/python")
    base = f"{ws}/cmp_{tag}_{sid}"
    remote_in = f"{base}.npy"
    render_script = "render_one_ybot.sh"
    warm_ok = False
    if os.environ.get("AGENTLODGE_RENDER_WARM", "1") != "0":
        try:
            from server import fk
            local_poses = media_dir / f"_cmp_{tag}_poses.npz"
            fk.save_poses_npz(motion, str(local_poses))
            remote_poses = f"{base}_poses.npz"
            if _scp_to(cfg, str(local_poses), remote_poses).returncode == 0:
                remote_in, render_script, warm_ok = remote_poses, "render_poses_ybot.sh", True
        except Exception as exc:  # noqa: BLE001 - fall back to the pod torch FK
            logger.warning("compare FK failed (%s); falling back to pod torch FK", exc)
    if not warm_ok:
        local_npy = media_dir / f"_cmp_{tag}.npy"
        np.save(local_npy, motion.astype(np.float32))
        if _scp_to(cfg, str(local_npy), remote_in).returncode != 0:
            return None
    rw = os.environ.get("AGENTLODGE_RENDER_WIN_W", "448")
    rh = os.environ.get("AGENTLODGE_RENDER_WIN_H", "448")
    rs = os.environ.get("AGENTLODGE_RENDER_WIN_SAMPLES", "8")
    render_env = f"RENDER_W={rw} RENDER_H={rh} RENDER_SAMPLES={rs} CUDA_VISIBLE_DEVICES="
    launch = _ssh(
        cfg,
        f"cd {ws}/AgentLODGE && sed -i 's/\\r$//' scripts/{render_script}; "
        f"rm -f {base}.mp4 {base}.done {base}.fail {base}.log; "
        f"setsid bash -c 'AL_PY={al_py} WORKSPACE={ws} {render_env} bash scripts/{render_script} "
        f"{remote_in} {base}.mp4 \"\" >> {base}.log 2>&1 && touch {base}.done || touch {base}.fail' "
        f"</dev/null >/dev/null 2>&1 & echo LAUNCHED",
        timeout=40)
    return base if "LAUNCHED" in (launch.stdout or "") else None


def _ffmpeg_frames(frames_dir: str, out_mp4: Path, fps: int = 30) -> bool:
    """Encode ``frames_dir/frame_%05d.png`` (contiguous from 0) to an mp4 locally on the pod."""
    import glob
    import subprocess
    if not sorted(glob.glob(f"{frames_dir}/frame_*.png")):
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", f"{frames_dir}/frame_%05d.png", "-c:v", "libx264", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", str(out_mp4)],
            capture_output=True, timeout=180)
        return r.returncode == 0 and out_mp4.exists()
    except Exception:  # noqa: BLE001
        return False


def _compare_warm(sid: str, before_motion: np.ndarray, after_motion: np.ndarray,
                  media_dir: Path) -> bool:
    """Fast compare via the warm Blender pool: server-FK both windows, render before/after in
    PARALLEL on two persistent daemons (no 8s startup, no ssh/scp), then ffmpeg locally. Returns
    True on success; False (with no side effects on the job) if the pool can't serve it."""
    from server import warm_render as wr
    from server import fk
    if not wr.available():
        return False
    media_dir.mkdir(parents=True, exist_ok=True)
    frames = int(max(before_motion.shape[0], after_motion.shape[0]))
    _cset(sid, status="rendering", progress=25, frames=frames,
          message=f"rendering before & after ({frames} frames each) on the warm GPU\u2026")
    try:
        specs = [
            ("before", str(media_dir / "_cmp_before_poses.npz"), str(media_dir / "_cmp_before_frames"), 0),
            ("after", str(media_dir / "_cmp_after_poses.npz"), str(media_dir / "_cmp_after_frames"), 1),
        ]
        fk.save_poses_npz(before_motion, specs[0][1])
        fk.save_poses_npz(after_motion, specs[1][1])
    except Exception as exc:  # noqa: BLE001 - server-FK unavailable -> let the caller fall back
        logger.warning("warm compare server-FK failed (%s)", exc)
        return False
    results: dict[str, bool] = {}

    def _run(tag: str, npz: str, frames_dir: str, didx: int) -> None:
        results[tag] = wr.warm_render(npz, frames_dir, daemon=didx, samples=8)

    threads = [threading.Thread(target=_run, args=s) for s in specs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not (results.get("before") and results.get("after")):
        return False
    _cset(sid, progress=90, message="encoding before & after\u2026")
    enc: dict[str, bool] = {}

    def _enc(tag: str, frames_dir: str, out: Path) -> None:
        enc[tag] = _ffmpeg_frames(frames_dir, out)

    ethreads = [
        threading.Thread(target=_enc, args=("before", specs[0][2], media_dir / "cmp_before.mp4")),
        threading.Thread(target=_enc, args=("after", specs[1][2], media_dir / "cmp_after.mp4")),
    ]
    for t in ethreads:
        t.start()
    for t in ethreads:
        t.join()
    return bool(enc.get("before") and enc.get("after"))


def _compare_render(sid: str, before_motion: np.ndarray, after_motion: np.ndarray,
                    media_dir: Path) -> None:
    cfg = pod_config()
    if not cfg.host:
        _cset(sid, status="error", progress=0,
              message="No GPU pod configured (set AGENTLODGE_POD_HOST).")
        return
    if before_motion.shape[0] < 2 or after_motion.shape[0] < 2:
        _cset(sid, status="error", progress=0, message="nothing to compare (empty window).")
        return
    started = _CJOBS.get(sid, {}).get("started", time.time())
    # Fast path: warm Blender pool (editor co-located on the pod). Falls through to the cold ssh
    # render below if the pool is unavailable or fails.
    try:
        if _compare_warm(sid, before_motion, after_motion, media_dir):
            _cset(sid, status="done", progress=100, message="ready",
                  before_video="cmp_before.mp4", after_video="cmp_after.mp4",
                  elapsed=round(time.time() - started))
            return
    except Exception as exc:  # noqa: BLE001 - never let the warm path break compare; fall back
        logger.warning("warm compare path errored (%s); using cold render", exc)
    try:
        _cset(sid, status="rendering", progress=8, message="checking the GPU pod\u2026")
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
            _cset(sid, status="error", progress=0,
                  message=f"can't reach the GPU pod at {cfg.host}:{cfg.port} (retried).")
            return
        ws = cfg.ws
        media_dir.mkdir(parents=True, exist_ok=True)
        _cset(sid, progress=16, message="preparing before + after renders\u2026")
        _upload_scripts(cfg)
        base_b = _launch_window_render(cfg, sid, "before", before_motion, media_dir, ws)
        base_a = _launch_window_render(cfg, sid, "after", after_motion, media_dir, ws)
        if not base_b or not base_a:
            _cset(sid, status="error", progress=0, message="could not start the compare render on the pod.")
            return
        frames = int(max(before_motion.shape[0], after_motion.shape[0]))
        est_sec = 20 + int(frames * 0.28)                    # two parallel window renders
        _cset(sid, progress=24, frames=frames,
              message=f"rendering before & after ({frames} frames each) on the GPU "
                      f"(~{est_sec}s)\u2026")

        deadline = time.time() + 60 * 45
        done_b = done_a = False
        while time.time() < deadline:
            time.sleep(3)
            try:
                chk = _ssh(
                    cfg,
                    f"st(){{ if [ -f $1.done ]; then echo D; elif [ -f $1.fail ]; then echo F; else echo R; fi; }}; "
                    f"echo B=$(st {base_b}) A=$(st {base_a})",
                    timeout=25)
                line = ((chk.stdout or "").strip().splitlines()[-1:] or [""])[0]
            except Exception:  # noqa: BLE001 - transient ssh hiccup: keep polling
                line = ""
            sb = "R"
            sa = "R"
            for tok in line.split():
                if tok.startswith("B="):
                    sb = tok[2:]
                elif tok.startswith("A="):
                    sa = tok[2:]
            if sb == "F" or sa == "F":
                bad = base_b if sb == "F" else base_a
                try:
                    tail = _ssh(cfg, f"tail -c 400 {bad}.log 2>/dev/null", timeout=25).stdout or ""
                except Exception:  # noqa: BLE001
                    tail = ""
                _cset(sid, status="error", progress=0,
                      message=f"compare render failed: {tail.strip()[-260:]}")
                return
            done_b = done_b or sb == "D"
            done_a = done_a or sa == "D"
            if done_b and done_a:
                break
            frac = min(0.95, (time.time() - _CJOBS[sid].get("started", time.time())) / est_sec)
            _cset(sid, progress=int(24 + 64 * frac))
        else:
            _cset(sid, status="error", progress=0, message="compare render timed out on the pod.")
            return

        _cset(sid, progress=90, message="downloading before + after\u2026")
        outs = {"before": (base_b, media_dir / "cmp_before.mp4"),
                "after": (base_a, media_dir / "cmp_after.mp4")}
        for _tag, (base, dst) in outs.items():
            ok = False
            for _ in range(3):
                try:
                    if _scp_from(cfg, f"{base}.mp4", str(dst)).returncode == 0:
                        ok = True
                        break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(5)
            if not ok:
                _cset(sid, status="error", progress=0,
                      message=f"could not fetch the {_tag} video.")
                return
        _cset(sid, status="done", progress=100, message="ready",
              before_video="cmp_before.mp4", after_video="cmp_after.mp4",
              elapsed=round(time.time() - _CJOBS[sid].get("started", time.time())))
    except Exception as exc:  # noqa: BLE001
        _cset(sid, status="error", progress=0, message=f"compare error: {exc}")
