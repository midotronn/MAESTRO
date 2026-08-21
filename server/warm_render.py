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

import copy
import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from server.distributed.render_contract import (
    RENDER_CONTRACT_VERSION,
    file_sha256,
)

logger = logging.getLogger(__name__)

WS = os.environ.get("AGENTLODGE_POD_WS", "/workspace")
POOL_SIZE = int(os.environ.get("AGENTLODGE_WARM_POOL", "6"))
PROTOCOL_VERSION = 6
SELECTOR_VERSION = 2
DAEMON_ROOT = Path(
    os.environ.get(
        "AGENTLODGE_RENDER_DAEMON_ROOT",
        str(Path(WS) / "blend_daemon"),
    )
).resolve()
REPO_ROOT = Path(__file__).resolve().parents[1]
_EGL = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
_HB_STALE = 30          # seconds; a daemon whose heartbeat is older than this is considered dead
_START_LOCK = threading.Lock()
_PROVENANCE_CACHE_KEY: tuple | None = None
_PROVENANCE_CACHE: dict | None = None


def _blender() -> Path:
    return Path(WS) / "blender" / "blender"


def _scene() -> Path:
    return Path(WS) / "ybot_scene.blend"


def _ybot() -> Path:
    return Path(WS) / "EDGE" / "SMPL-to-FBX" / "ybot.fbx"


def _daemon_script() -> Path:
    return REPO_ROOT / "scripts" / "blender_daemon.py"


def _renderer_files() -> tuple[Path, ...]:
    scripts = REPO_ROOT / "scripts"
    return (
        scripts / "blender_daemon.py",
        scripts / "blender_render_ybot.py",
        scripts / "blender_studio.py",
        scripts / "render_root_motion.py",
    )


def daemon_root() -> Path:
    configured = os.environ.get("AGENTLODGE_RENDER_DAEMON_ROOT", "").strip()
    return Path(configured).resolve() if configured else DAEMON_ROOT


def on_pod() -> bool:
    """True when the editor is co-located with the pod (hosted mode) and the render assets exist, so
    we can drive local warm daemons instead of ssh + cold Blender."""
    host = (os.environ.get("AGENTLODGE_POD_HOST") or "").strip().lower()
    if host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        return False
    return _blender().exists() and _scene().exists() and _ybot().exists() and _daemon_script().exists()


def _dir(i: int) -> Path:
    return daemon_root() / f"d{i}"


def _selector_shim() -> Path | None:
    multi_gpu = os.environ.get(
        "AGENTLODGE_RENDER_MULTI_GPU",
        "",
    ).strip().lower()
    if multi_gpu not in {"1", "true", "yes", "on"}:
        return None
    gpu_index = os.environ.get("AGENTLODGE_GPU_INDEX", "").strip()
    if not gpu_index.isdigit():
        raise RuntimeError(
            "multi-GPU render requires a non-negative AGENTLODGE_GPU_INDEX"
        )
    selector = os.environ.get(
        "AGENTLODGE_EGL_SELECTOR_SHIM",
        "",
    ).strip()
    if not selector:
        raise RuntimeError(
            "multi-GPU render requires AGENTLODGE_EGL_SELECTOR_SHIM"
        )
    selector_path = Path(selector).resolve()
    if not selector_path.is_file() or selector_path.stat().st_size == 0:
        raise RuntimeError(
            f"multi-GPU EGL selector shim is missing: {selector_path}"
        )
    return selector_path


def _selector_identity() -> dict | None:
    selector = _selector_shim()
    if selector is None:
        return None
    source = REPO_ROOT / "scripts" / "egl_cuda_device_selector.c"
    return {
        "version": SELECTOR_VERSION,
        "build_id": f"sha256:{file_sha256(source)}",
        "binary_sha256": file_sha256(selector),
        "requested_cuda_index": int(os.environ["AGENTLODGE_GPU_INDEX"]),
        "selected_cuda_index": int(os.environ["AGENTLODGE_GPU_INDEX"]),
    }


def _blender_version() -> str:
    result = subprocess.run(
        [str(_blender()), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            f"could not attest Blender version: {result.stderr[-400:]}"
        )
    first_line = result.stdout.splitlines()[0].strip()
    return first_line.removeprefix("Blender ").strip()


def render_provenance() -> dict:
    global _PROVENANCE_CACHE_KEY, _PROVENANCE_CACHE

    scene = _scene().resolve()
    ybot = _ybot().resolve()
    renderer_files = _renderer_files()
    missing = [
        path
        for path in (scene, ybot, *renderer_files)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "render provenance files are missing: "
            + ", ".join(str(path) for path in missing)
        )
    selector_path = _selector_shim()
    identity_paths = (scene, ybot, *renderer_files, _blender().resolve())
    if selector_path is not None:
        identity_paths = (*identity_paths, selector_path.resolve())
    try:
        cache_key = (
            RENDER_CONTRACT_VERSION,
            PROTOCOL_VERSION,
            tuple(
                (
                    str(path),
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                    path.stat().st_ctime_ns,
                )
                for path in identity_paths
            ),
            os.environ.get("AGENTLODGE_RENDER_MULTI_GPU", ""),
            os.environ.get("AGENTLODGE_GPU_INDEX", ""),
        )
    except OSError as exc:
        raise RuntimeError("could not stat render provenance files") from exc
    if cache_key == _PROVENANCE_CACHE_KEY and _PROVENANCE_CACHE is not None:
        return copy.deepcopy(_PROVENANCE_CACHE)
    selector = _selector_identity()
    provenance = {
        "render_contract_version": RENDER_CONTRACT_VERSION,
        "daemon_protocol_version": PROTOCOL_VERSION,
        "scene": {
            "blend_sha256": file_sha256(scene),
            "ybot_sha256": file_sha256(ybot),
        },
        "renderer": {
            "blender_version": _blender_version(),
            "blender_daemon_sha256": file_sha256(renderer_files[0]),
            "blender_render_ybot_sha256": file_sha256(renderer_files[1]),
            "blender_studio_sha256": file_sha256(renderer_files[2]),
            "render_root_motion_sha256": file_sha256(renderer_files[3]),
        },
        "selector": (
            None
            if selector is None
            else {
                "version": selector["version"],
                "build_id": selector["build_id"],
                "binary_sha256": selector["binary_sha256"],
            }
        ),
    }
    _PROVENANCE_CACHE_KEY = cache_key
    _PROVENANCE_CACHE = copy.deepcopy(provenance)
    return provenance


def _daemon_environment(
    selector_attestation_path: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = _EGL
    selector = _selector_shim()
    if selector is not None:
        if selector_attestation_path is None:
            raise RuntimeError(
                "multi-GPU render requires a selector attestation path"
            )
        existing = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = (
            f"{selector}{os.pathsep}{existing}" if existing else str(selector)
        )
        env["AGENTLODGE_EGL_ATTESTATION_PATH"] = str(
            selector_attestation_path.resolve()
        )
    return env


def _pid_alive(d: Path) -> bool:
    try:
        pid = int((d / "daemon.pid").read_text().strip())
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().split()[2] == "Z":
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def _protocol_ready(d: Path) -> bool:
    try:
        return (d / "daemon.ready").read_text().strip() == str(PROTOCOL_VERSION)
    except OSError:
        return False


def _read_attestation(d: Path) -> dict | None:
    try:
        raw = json.loads(
            (d / "daemon.attestation.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _quality_contract(
    *,
    width: int,
    height: int,
    samples: int,
    engine: str,
    denoise: int,
    frame_format: str,
) -> dict:
    return {
        "width": max(1, int(width)),
        "height": max(1, int(height)),
        "samples": max(1, int(samples)),
        "engine": str(engine).lower(),
        "denoise": int(denoise),
        "frame_format": str(frame_format).lower().lstrip("."),
    }


def _attestation_matches(
    d: Path,
    *,
    width: int,
    height: int,
    samples: int,
    engine: str,
    denoise: int,
    frame_format: str,
    provenance: dict | None = None,
) -> bool:
    attestation = _read_attestation(d)
    if attestation is None:
        return False
    try:
        pid = int((d / "daemon.pid").read_text().strip())
        provenance = provenance or render_provenance()
        if (
            attestation.get("schema_version") != 1
            or attestation.get("pid") != pid
            or attestation.get("render_contract_version")
            != RENDER_CONTRACT_VERSION
            or attestation.get("daemon_protocol_version")
            != PROTOCOL_VERSION
            or attestation.get("scene") != provenance["scene"]
            or attestation.get("renderer") != provenance["renderer"]
            or attestation.get("quality")
            != _quality_contract(
                width=width,
                height=height,
                samples=samples,
                engine=engine,
                denoise=denoise,
                frame_format=frame_format,
            )
        ):
            return False
        gpu = attestation.get("gpu")
        if (
            not isinstance(gpu, dict)
            or not str(gpu.get("uuid") or "").startswith("GPU-")
            or not str(gpu.get("pci_bus_id") or "")
        ):
            return False
        expected_selector = _selector_identity()
        actual_selector = attestation.get("selector")
        if expected_selector is None:
            if actual_selector is not None:
                return False
            if gpu.get("selection_mode") != "single-visible-gpu":
                return False
            resolved = os.environ.get(
                "AGENTLODGE_RESOLVED_GPU_INDEX",
                "",
            ).strip()
            if resolved and int(gpu.get("cuda_index")) != int(resolved):
                return False
        else:
            if not isinstance(actual_selector, dict):
                return False
            for key in (
                "version",
                "build_id",
                "binary_sha256",
                "requested_cuda_index",
                "selected_cuda_index",
            ):
                if actual_selector.get(key) != expected_selector.get(key):
                    return False
            if (
                gpu.get("selection_mode") != "egl-cuda-device-nv"
                or int(gpu.get("cuda_index"))
                != expected_selector["selected_cuda_index"]
            ):
                return False
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _alive(d: Path) -> bool:
    hb = d / "daemon.hb"
    attestation = _read_attestation(d)
    try:
        return (
            _pid_alive(d)
            and _protocol_ready(d)
            and isinstance(attestation, dict)
            and attestation.get("daemon_protocol_version") == PROTOCOL_VERSION
            and attestation.get("render_contract_version")
            == RENDER_CONTRACT_VERSION
            and attestation.get("pid")
            == int((d / "daemon.pid").read_text().strip())
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


def _start_daemon(
    i: int,
    *,
    width: int,
    height: int,
    samples: int,
    engine: str = "eevee",
    denoise: int = 1,
    frame_format: str = "png",
) -> None:
    d = _dir(i)
    d.mkdir(parents=True, exist_ok=True)
    for name in (
        "daemon.ready",
        "daemon.hb",
        "daemon.pid",
        "daemon.attestation.json",
        "egl-selector.attestation.json",
    ):
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
        "--engine",
        str(engine),
        "--denoise",
        str(int(denoise)),
        "--frame-format",
        str(frame_format),
        "--idle-exit",
        "0",
    ]
    env = _daemon_environment(d / "egl-selector.attestation.json")
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


def ensure_pool(
    *,
    width: int = 448,
    height: int = 448,
    samples: int = 8,
    engine: str = "eevee",
    denoise: int = 1,
    frame_format: str = "png",
    wait_ready: float = 0.0,
) -> int:
    """Start any dead daemons in the pool. Returns the number of daemons currently alive (optionally
    after waiting up to ``wait_ready`` seconds for freshly-started ones to load the scene)."""
    if not on_pod():
        return 0
    quality = _quality_contract(
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        denoise=denoise,
        frame_format=frame_format,
    )
    provenance: dict | None = None

    def ready(d: Path) -> bool:
        nonlocal provenance
        if not _alive(d):
            return False
        if provenance is None:
            provenance = render_provenance()
        return _attestation_matches(d, provenance=provenance, **quality)

    with _START_LOCK:
        for i in range(POOL_SIZE):
            d = _dir(i)
            if _pid_alive(d) and (d / "daemon.ready").exists() and not _protocol_ready(d):
                _stop_daemon(d)
            elif (
                _pid_alive(d)
                and (d / "daemon.ready").exists()
                and not ready(d)
            ):
                _stop_daemon(d)
            if not ready(d) and not _pid_alive(d):
                try:
                    _start_daemon(i, **quality)
                except Exception as exc:  # noqa: BLE001 - best-effort warm-up
                    logger.warning("warm daemon %d failed to start: %s", i, exc)
    if wait_ready > 0:
        deadline = time.time() + wait_ready
        while (
            time.time() < deadline
            and sum(ready(_dir(i)) for i in range(POOL_SIZE)) < POOL_SIZE
        ):
            time.sleep(1)
    return sum(ready(_dir(i)) for i in range(POOL_SIZE))


def ensure_configured_pool(*, wait_ready: float = 0.0) -> int:
    """Warm the production full-render quality when Filament owns the SLA path."""
    backend = os.environ.get(
        "AGENTLODGE_FULL_RENDER_BACKEND",
        "blender",
    ).strip().lower()
    if backend == "filament":
        return ensure_pool(
            width=max(
                1,
                int(os.environ.get("AGENTLODGE_RENDER_FULL_W", "1080")),
            ),
            height=max(
                1,
                int(os.environ.get("AGENTLODGE_RENDER_FULL_H", "1080")),
            ),
            samples=max(
                1,
                int(os.environ.get("AGENTLODGE_RENDER_FULL_SAMPLES", "96")),
            ),
            engine=os.environ.get("AGENTLODGE_RENDER_ENGINE", "eevee"),
            denoise=int(os.environ.get("AGENTLODGE_RENDER_DENOISE", "1")),
            frame_format="tga",
            wait_ready=wait_ready,
        )
    return ensure_pool(wait_ready=wait_ready)


def ready_daemons() -> list[int]:
    """Return the concrete daemon slots that are currently ready for requests."""
    return [index for index in range(POOL_SIZE) if _alive(_dir(index))]


def available() -> bool:
    return on_pod() and any(_alive(_dir(i)) for i in range(POOL_SIZE))


def daemon_attestation(
    daemon: int,
    *,
    width: int,
    height: int,
    samples: int,
    engine: str,
    denoise: int,
    frame_format: str,
) -> dict:
    d = _dir(daemon % POOL_SIZE)
    quality = _quality_contract(
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        denoise=denoise,
        frame_format=frame_format,
    )
    if not _alive(d) or not _attestation_matches(d, **quality):
        raise RuntimeError(
            f"Blender daemon {daemon} failed render provenance attestation"
        )
    attestation = _read_attestation(d)
    assert attestation is not None
    return attestation


def warm_render(poses_npz: str, frames_dir: str, *, daemon: int, samples: int = 8,
                width: int = 448, height: int = 448, timeout: float = 600.0,
                engine: str = "eevee", denoise: int = 1,
                rig_metrics: str = "", fast: bool = False,
                foot_grounding: bool = False, stride: int = 1,
                projection_only: bool = False, batch_render: bool = False,
                video_path: str = "", export_glb: str = "", frame_start: int = 0,
                frame_end: int | None = None, clear_frames: bool = True,
                frame_format: str = "png") -> bool:
    """Submit one render and optionally capture exact projected rig metrics. Returns True on success."""
    d = _dir(daemon % POOL_SIZE)
    quality = _quality_contract(
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        denoise=denoise,
        frame_format=frame_format,
    )
    if not _alive(d) or not _attestation_matches(d, **quality):
        return False
    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)
    if clear_frames:
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
    if export_glb:
        try:
            Path(export_glb).unlink(missing_ok=True)
        except OSError:
            return False
    rid = "r" + uuid.uuid4().hex[:10]
    done, fail = d / f"{rid}.done", d / f"{rid}.fail"
    req = {"id": rid, "poses": str(poses_npz), "frames_dir": str(frames_dir),
           "width": width, "height": height, "samples": samples,
           "engine": str(engine), "denoise": int(denoise), "fast": bool(fast),
           "foot_grounding": bool(foot_grounding),
           "stride": max(1, int(stride)), "rig_metrics": str(rig_metrics),
           "projection_only": bool(projection_only), "batch_render": bool(batch_render),
           "video_path": str(video_path), "export_glb": str(export_glb),
           "frame_start": max(0, int(frame_start)),
           "frame_end": -1 if frame_end is None else int(frame_end),
           "clear_frames": bool(clear_frames), "frame_format": str(frame_format)}
    tmp = d / f"{rid}.req.tmp"
    tmp.write_text(json.dumps(req))
    tmp.rename(d / f"{rid}.req")          # atomic publish so the daemon never reads a half-written file
    deadline = time.time() + timeout
    while time.time() < deadline:
        if done.exists():
            return (
                (not video_path or Path(video_path).is_file())
                and (not export_glb or Path(export_glb).is_file())
            )
        if fail.exists():
            logger.warning("warm render %s failed: %s", rid, fail.read_text()[-300:])
            return False
        if not _alive(d):                  # daemon died mid-render
            return False
        time.sleep(0.05)
    return False
