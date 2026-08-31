"""On-demand Blender render of the *current edited* motion, dispatched to the GPU pod.

After edits, the metrics/agent-log update immediately but the preview video still shows the base take
(rendering needs Blender on the pod). This module renders the session's current motion -- either just
the edited window (fast) or the full song with music -- as the canonical gray Y-Bot, and pulls the
mp4 back into ``server/media/<sid>/edited.mp4`` so the UI can swap it in. Job status is polled by the
UI, mirroring :mod:`server.processing`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np

from server.distributed.render_contract import (
    RENDER_CONTRACT_VERSION,
    RGB_DIGEST_VERSION,
    WORKER_SHARD_VALIDATION_VERSION,
    canonical_render_identity,
    inspect_ffv1_shard,
    render_identity_digest,
)
from server.distributed.runtime import capability_enabled, distributed_transport
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
                warm_render.ensure_configured_pool()
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
                "blender_render_ybot.py", "blender_studio.py",
                "render_root_motion.py")]
    scripts = [p for p in scripts if p.exists()]
    r = _scp_many(cfg, scripts, f"{cfg.ws}/AgentLODGE/scripts")
    if r is not None and r.returncode == 0:
        _SCRIPTS_SENT.add(cfg.host)


def _set(sid: str, **kw) -> None:
    with _RLOCK:
        _RJOBS.setdefault(sid, {}).update(kw)


def _quality_render_settings(scope: str) -> tuple[int, int, int, str, int]:
    """Return the same render settings used by the established cold quality path."""
    if scope == "full":
        width = max(1, int(os.environ.get("AGENTLODGE_RENDER_FULL_W", "1080")))
        height = max(1, int(os.environ.get("AGENTLODGE_RENDER_FULL_H", "1080")))
        samples = max(1, int(os.environ.get("AGENTLODGE_RENDER_FULL_SAMPLES", "96")))
    else:
        width = max(1, int(os.environ.get("AGENTLODGE_RENDER_WIN_W", "448")))
        height = max(1, int(os.environ.get("AGENTLODGE_RENDER_WIN_H", "448")))
        samples = max(1, int(os.environ.get("AGENTLODGE_RENDER_WIN_SAMPLES", "8")))
    engine = os.environ.get("AGENTLODGE_RENDER_ENGINE", "eevee")
    denoise = int(os.environ.get("AGENTLODGE_RENDER_DENOISE", "1"))
    return width, height, samples, engine, denoise


def _distributed_enabled() -> bool:
    return capability_enabled("render.frames")


def _render_ranges(frame_count: int, worker_count: int) -> list[tuple[int, int]]:
    """Split every source frame into contiguous, non-overlapping render ranges."""
    if frame_count < 1:
        return []
    workers = max(1, min(int(worker_count), frame_count))
    base, remainder = divmod(frame_count, workers)
    ranges = []
    start = 0
    for index in range(workers):
        end = start + base + (1 if index < remainder else 0)
        ranges.append((start, end))
        start = end
    return ranges


def _validate_render_ranges(
    ranges: list[tuple[int, int]],
    frame_count: int,
) -> None:
    cursor = 0
    for start, end in ranges:
        if start != cursor or end <= start:
            raise RuntimeError(
                f"distributed render ranges are not contiguous: {ranges}"
            )
        cursor = end
    if cursor != frame_count:
        raise RuntimeError(
            f"distributed render ranges cover {cursor}/{frame_count} frames"
        )


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_render_output(
    output: dict,
    *,
    start: int,
    end: int,
    width: int,
    height: int,
    samples: int,
    engine: str,
    denoise: int,
    frame_format: str,
    fps: int,
    render_provenance: Mapping[str, object],
) -> tuple[str, str, str]:
    expected = {
        "frame_start": start,
        "frame_end": end,
        "frames": end - start,
        "width": width,
        "height": height,
        "samples": samples,
        "engine": str(engine).lower(),
        "denoise": denoise,
        "frame_format": str(frame_format).lower().lstrip("."),
        "fps": fps,
        "transport": "ffv1",
        "render_contract_version": RENDER_CONTRACT_VERSION,
    }
    actual = {
        key: (
            str(output.get(key)).lower().lstrip(".")
            if key in {"engine", "frame_format"}
            else output.get(key)
        )
        for key in expected
    }
    for key in {
        "frame_start",
        "frame_end",
        "frames",
        "width",
        "height",
        "samples",
        "denoise",
        "fps",
    }:
        try:
            actual[key] = int(actual[key])
        except (TypeError, ValueError):
            pass
    if actual != expected:
        raise RuntimeError(
            f"distributed render returned invalid range/quality metadata: {output}"
        )
    returned_provenance = output.get("render_provenance")
    attestation = output.get("daemon_attestation")
    if (
        not isinstance(returned_provenance, Mapping)
        or dict(returned_provenance) != dict(render_provenance)
        or not isinstance(attestation, Mapping)
    ):
        raise RuntimeError("distributed render provenance is missing or mismatched")
    expected_identity = render_identity_digest(
        render_provenance,
        {
            "width": width,
            "height": height,
            "samples": samples,
            "engine": engine,
            "denoise": denoise,
            "frame_format": frame_format,
            "fps": fps,
        },
    )
    if output.get("render_identity_digest") != expected_identity:
        raise RuntimeError("distributed render identity digest is missing or mismatched")
    attested_provenance = {
        key: attestation.get(key)
        for key in (
            "render_contract_version",
            "daemon_protocol_version",
            "scene",
            "renderer",
        )
    }
    selector = attestation.get("selector")
    attested_provenance["selector"] = (
        None
        if selector is None
        else {
            key: selector.get(key)
            for key in ("version", "build_id", "binary_sha256")
        }
    )
    gpu = attestation.get("gpu")
    if (
        attested_provenance != dict(render_provenance)
        or attestation.get("quality")
        != {
            "width": width,
            "height": height,
            "samples": samples,
            "engine": str(engine).lower(),
            "denoise": denoise,
            "frame_format": str(frame_format).lower().lstrip("."),
        }
        or not isinstance(gpu, Mapping)
        or not str(gpu.get("uuid") or "").startswith("GPU-")
        or not str(gpu.get("pci_bus_id") or "")
        or not isinstance(gpu.get("cuda_index"), int)
    ):
        raise RuntimeError("distributed render daemon attestation is invalid")
    if selector is None:
        if gpu.get("selection_mode") != "single-visible-gpu":
            raise RuntimeError("distributed render GPU selection is not attested")
    elif (
        not isinstance(selector, Mapping)
        or gpu.get("selection_mode") != "egl-cuda-device-nv"
        or selector.get("selected_cuda_index") != gpu.get("cuda_index")
        or selector.get("requested_cuda_index") != gpu.get("cuda_index")
        or not isinstance(selector.get("egl_device_index"), int)
    ):
        raise RuntimeError("distributed render EGL selection is not attested")
    source_hash = str(output.get("source_frames_sha256") or "").lower()
    shard_hash = str(output.get("shard_sha256") or "").lower()
    source_rgb_hash = str(
        output.get("source_decoded_rgb_sha256") or ""
    ).lower()
    shard_rgb_hash = str(
        output.get("shard_decoded_rgb_sha256") or ""
    ).lower()
    if (
        re.fullmatch(r"[a-f0-9]{64}", source_hash) is None
        or re.fullmatch(r"[a-f0-9]{64}", shard_hash) is None
        or re.fullmatch(r"[a-f0-9]{64}", source_rgb_hash) is None
        or re.fullmatch(r"[a-f0-9]{64}", shard_rgb_hash) is None
        or source_rgb_hash != shard_rgb_hash
    ):
        raise RuntimeError(
            f"distributed render shard hashes are invalid: {output}"
        )
    shard_validation = output.get("shard_validation")
    if not isinstance(shard_validation, Mapping):
        raise RuntimeError("distributed render is missing shard validation metadata")
    reported_validation = {
        "codec": str(shard_validation.get("codec") or "").lower(),
        "width": shard_validation.get("width"),
        "height": shard_validation.get("height"),
        "fps": shard_validation.get("fps"),
        "frames": shard_validation.get("frames"),
        "decoded_rgb_digest_version": shard_validation.get(
            "decoded_rgb_digest_version"
        ),
        "decoded_rgb_sha256": str(
            shard_validation.get("decoded_rgb_sha256") or ""
        ).lower(),
    }
    for key in {"width", "height", "fps", "frames"}:
        try:
            reported_validation[key] = int(reported_validation[key])
        except (TypeError, ValueError):
            pass
    expected_validation = {
        "codec": "ffv1",
        "width": width,
        "height": height,
        "fps": fps,
        "frames": end - start,
        "decoded_rgb_digest_version": RGB_DIGEST_VERSION,
        "decoded_rgb_sha256": shard_rgb_hash,
    }
    if (
        output.get("decoded_rgb_digest_version") != RGB_DIGEST_VERSION
        or reported_validation != expected_validation
        or shard_validation.get("worker_validation_version")
        != WORKER_SHARD_VALIDATION_VERSION
        or shard_validation.get("worker_shard_full_decode") is not False
    ):
        raise RuntimeError(
            "distributed render returned invalid decoded shard metadata"
        )
    return source_hash, shard_hash, shard_rgb_hash


def _worker_metadata(worker) -> dict:
    metadata = dict(getattr(worker, "metadata", {}) or {})
    heartbeat = getattr(worker, "heartbeat", None)
    if callable(heartbeat):
        try:
            metadata.update(dict(heartbeat().get("metadata") or {}))
        except Exception:  # noqa: BLE001 - worker health was already checked
            pass
    return metadata


def _worker_render_provenance(worker, metadata: Mapping[str, object] | None = None) -> dict:
    metadata = dict(metadata or _worker_metadata(worker))
    provenance = metadata.get("render_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("render_contract_version")
        != RENDER_CONTRACT_VERSION
    ):
        raise RuntimeError(
            f"render worker {getattr(worker, 'worker_id', '?')} "
            "is missing compatible render provenance"
        )
    return dict(provenance)


def _render_quality_contract(
    *,
    width: int,
    height: int,
    samples: int,
    engine: str,
    denoise: int,
    frame_format: str,
    fps: int = 30,
) -> dict[str, object]:
    return {
        "width": int(width),
        "height": int(height),
        "samples": int(samples),
        "engine": str(engine).lower(),
        "denoise": int(denoise),
        "frame_format": str(frame_format).lower().lstrip("."),
        "fps": int(fps),
    }


def _select_render_worker_cohort(
    workers,
    *,
    requested_workers: int,
    quality: Mapping[str, object],
) -> tuple[list, tuple[str, ...], dict, str]:
    groups: dict[str, list[tuple[object, dict, str]]] = {}
    for worker in workers:
        metadata = _worker_metadata(worker)
        provenance = _worker_render_provenance(worker, metadata)
        advertised_quality = metadata.get("quality")
        if isinstance(advertised_quality, Mapping):
            expected_worker_quality = {
                key: quality[key]
                for key in (
                    "width",
                    "height",
                    "samples",
                    "engine",
                    "denoise",
                    "frame_format",
                )
            }
            normalized_advertised = {
                key: (
                    str(advertised_quality.get(key)).lower().lstrip(".")
                    if key in {"engine", "frame_format"}
                    else int(advertised_quality.get(key))
                )
                for key in expected_worker_quality
            }
            if normalized_advertised != expected_worker_quality:
                raise RuntimeError(
                    f"render worker {getattr(worker, 'worker_id', '?')} "
                    "advertises incompatible quality"
                )
        identity = canonical_render_identity(provenance, quality)
        canonical = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        advertised_digest = metadata.get("render_identity_digest")
        if advertised_digest not in {None, "", digest}:
            raise RuntimeError(
                f"render worker {getattr(worker, 'worker_id', '?')} "
                "advertises a mismatched render identity"
            )
        groups.setdefault(canonical, []).append((worker, provenance, digest))
    if not groups:
        raise RuntimeError("no compatible render worker cohort is available")
    canonical, records = min(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
        ),
    )
    del canonical
    records.sort(key=lambda item: str(getattr(item[0], "worker_id", "")))
    limit = max(1, int(requested_workers))
    required = min(limit, len(workers))
    if len(records) < required:
        raise RuntimeError(
            "render workers have mixed provenance/quality and no homogeneous "
            f"cohort can satisfy {required} requested assignments"
        )
    selected = [record[0] for record in records[:limit]]
    eligible_worker_ids = tuple(
        str(getattr(record[0], "worker_id")) for record in records
    )
    return selected, eligible_worker_ids, records[0][1], records[0][2]


def _count_frame_files(frames_dir: Path, frame_format: str) -> int:
    """Count completed render frames while a warm Blender request is running."""
    suffix = "." + frame_format.lower().lstrip(".")
    try:
        with os.scandir(frames_dir) as entries:
            return sum(
                1
                for entry in entries
                if entry.is_file()
                and entry.name.startswith("frame_")
                and entry.name.lower().endswith(suffix)
            )
    except OSError:
        return 0


def start_render(sid: str, motion: np.ndarray, media_dir: Path, *, scope: str = "window",
                 a: int | None = None, b: int | None = None,
                 audio_wav: str | None = None) -> None:
    with _RLOCK:
        current = _RJOBS.get(sid, {})
        if current.get("status") in {"queued", "rendering"}:
            return
        _RJOBS.setdefault(sid, {}).update(
            status="queued",
            message="queued",
            progress=3,
            scope=scope,
            started=time.time(),
        )
    threading.Thread(target=_render, args=(sid, np.asarray(motion), media_dir, scope, a, b),
                     kwargs={"audio_wav": audio_wav},
                     daemon=True).start()


def _render_warm_local(sid: str, motion: np.ndarray, media_dir: Path, scope: str,
                       *, audio_wav: str | None = None) -> bool:
    """Render full-quality frames through the resident Blender process when hosted on the pod."""
    from server import filament_render

    if filament_render.enabled(scope):
        return filament_render.render_full_motion(
            sid,
            motion,
            media_dir,
            audio_wav=audio_wav,
            update=lambda **fields: _set(sid, **fields),
        )
    from server import fk
    from server import warm_render as wr
    distributed = _distributed_enabled() and scope == "full"
    if not distributed and not wr.on_pod():
        return False
    width, height, samples, engine, denoise = _quality_render_settings(scope)
    frame_format = (
        os.environ.get("AGENTLODGE_RENDER_FRAME_FORMAT", "tga").lower()
        if scope == "full"
        else "png"
    )
    if frame_format not in {"png", "tga"}:
        frame_format = "tga"
    media_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = media_dir / ".render_cache"
    cache_dir.mkdir(exist_ok=True)
    requested_workers = (
        max(1, int(os.environ.get("AGENTLODGE_FULL_RENDER_WORKERS", "6")))
        if scope == "full"
        else 1
    )
    render_quality = _render_quality_contract(
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        denoise=denoise,
        frame_format=frame_format,
        fps=30,
    )
    coordinator = None
    transport = "filesystem"
    render_workers = []
    daemon_ids = []
    eligible_worker_ids: tuple[str, ...] = ()
    selected_render_provenance: dict
    if distributed:
        transport = distributed_transport("render.frames")
        heartbeat_max_age = float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
        )
        if transport == "http":
            from server.distributed import HttpTaskCoordinator

            coordinator = HttpTaskCoordinator.from_env()
            candidates = coordinator.require_workers(
                "render.frames",
                max_age_seconds=heartbeat_max_age,
            )
        else:
            from server.distributed import FileTaskCoordinator, WorkerRegistry

            registry = WorkerRegistry.from_env()
            candidates = registry.require(
                "render.frames",
                max_age_seconds=heartbeat_max_age,
            )
            coordinator = FileTaskCoordinator(
                registry,
                heartbeat_max_age=heartbeat_max_age,
            )
        (
            render_workers,
            eligible_worker_ids,
            selected_render_provenance,
            render_identity,
        ) = _select_render_worker_cohort(
            candidates,
            requested_workers=requested_workers,
            quality=render_quality,
        )
        worker_count = min(len(render_workers), int(motion.shape[0]))
        render_workers = render_workers[:worker_count]
        worker_ids = [worker.worker_id for worker in render_workers]
    else:
        selected_render_provenance = wr.render_provenance()
        render_identity = render_identity_digest(
            selected_render_provenance,
            render_quality,
        )
    audio_context = ""
    if scope == "full" and audio_wav:
        try:
            stat = Path(audio_wav).stat()
            audio_context = f":audio:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            audio_context = ":audio:missing"
    cache_key = _render_cache_key(
        motion,
        width=width,
        height=height,
        samples=samples,
        stride=1,
        engine=engine,
        denoise=denoise,
        context=(
            f"single:{scope}:frames:{frame_format}:"
            f"{RENDER_CONTRACT_VERSION}:identity:{render_identity}"
            f"{audio_context}"
        ),
    )
    cached_video = cache_dir / f"{cache_key}.mp4"
    output_video = media_dir / "edited.mp4"
    if cached_video.is_file() and cached_video.stat().st_size > 0:
        shutil.copyfile(cached_video, output_video)
        _set(
            sid,
            status="rendering",
            progress=96,
            frames=int(motion.shape[0]),
            rendered_frames=int(motion.shape[0]),
            workers=0,
            worker_ids=[],
            message="reusing the exact cached render\u2026",
        )
        return True

    if not distributed:
        wr.ensure_pool(
            width=width,
            height=height,
            samples=samples,
            engine=engine,
            denoise=denoise,
            frame_format=frame_format,
            wait_ready=60,
        )
        daemon_ids = wr.ready_daemons()[:requested_workers]
        worker_count = min(len(daemon_ids), int(motion.shape[0]))
        daemon_ids = daemon_ids[:worker_count]
        worker_ids = [f"blender-daemon-{daemon}" for daemon in daemon_ids]
    if worker_count < 1:
        return False

    duration = motion.shape[0] / 30.0
    if distributed and transport == "filesystem":
        shared_temp = (
            os.environ.get("AGENTLODGE_DISTRIBUTED_TMP", "").strip()
            or os.environ.get("AGENTLODGE_SHARED_ROOT", "").strip()
        )
        if not shared_temp:
            raise RuntimeError(
                "distributed rendering requires AGENTLODGE_SHARED_ROOT "
                "or AGENTLODGE_DISTRIBUTED_TMP"
            )
        temp_parent = Path(shared_temp)
    elif distributed:
        assert coordinator is not None
        temp_parent = coordinator.scratch_root
    else:
        temp_parent = Path(
            os.environ.get("AGENTLODGE_RENDER_TMP", tempfile.gettempdir())
        )
    temp_parent.mkdir(parents=True, exist_ok=True)
    if distributed and transport == "http":
        assert coordinator is not None
        render_root = coordinator.create_scratch_dir(prefix=f"maestro-{sid}")
    else:
        render_root = Path(
            tempfile.mkdtemp(prefix=f"maestro-{sid}-", dir=temp_parent)
        )
    poses_path = render_root / "poses.npz"
    frames_dir = render_root / "frames"
    frames_dir.mkdir()
    shards_dir = render_root / "shards"
    if distributed:
        shards_dir.mkdir()
    try:
        fk.save_poses_npz(motion, poses_path)
        ranges = _render_ranges(int(motion.shape[0]), worker_count)
        _validate_render_ranges(ranges, int(motion.shape[0]))
        _set(
            sid,
            status="rendering",
            progress=24,
            frames=int(motion.shape[0]),
            workers=worker_count,
            worker_ids=worker_ids,
            rendered_frames=0,
            message=(
                f"full-quality {'dance' if scope == 'full' else 'window'} render "
                f"on {worker_count} worker{'s' if worker_count != 1 else ''} "
                f"({duration:.1f}s of motion)\u2026"
            ),
        )
        results: dict[int, bool] = {}
        result_lock = threading.Lock()
        distributed_progress = []
        distributed_results = []
        expected_artifacts = {}
        expected_provenances = {}

        def _record_progress() -> None:
            if distributed:
                rendered_frames = min(
                    total_frames,
                    sum(
                        end - start
                        for handle, start, end in distributed_progress
                        if coordinator.is_complete(handle)
                    ),
                )
            else:
                rendered_frames = min(
                    total_frames,
                    _count_frame_files(frames_dir, frame_format),
                )
            _set(
                sid,
                progress=min(
                    88,
                    24 + int(64 * rendered_frames / max(1, total_frames)),
                ),
                rendered_frames=rendered_frames,
                workers=worker_count,
                message=(
                    f"rendered {rendered_frames}/{total_frames} full-quality "
                    f"frame{'s' if total_frames != 1 else ''} on "
                    f"{worker_count} worker{'s' if worker_count != 1 else ''}\u2026"
                ),
            )

        def _render_range(index: int, daemon: int, start: int, end: int) -> None:
            shard_duration = (end - start) / 30.0
            ok = wr.warm_render(
                str(poses_path),
                str(frames_dir),
                daemon=daemon,
                samples=samples,
                width=width,
                height=height,
                engine=engine,
                denoise=denoise,
                fast=False,
                stride=1,
                batch_render=True,
                frame_start=start,
                frame_end=end,
                clear_frames=False,
                frame_format=frame_format,
                timeout=(
                    max(900.0, shard_duration * 45)
                    if scope == "full"
                    else max(600.0, shard_duration * 8)
                ),
            )
            with result_lock:
                results[index] = ok

        total_frames = int(motion.shape[0])
        if distributed:
            handles = []
            assert coordinator is not None
            poses_artifact = None
            if transport == "http":
                poses_sha256, poses_size = _file_sha256(poses_path)
                poses_artifact = coordinator.upload_input(
                    poses_path,
                    artifact_key=(
                        f"render-source:{poses_sha256}:{poses_size}"
                    ),
                )
            for worker, (start, end) in zip(render_workers, ranges):
                render_provenance = selected_render_provenance
                shard_duration = (end - start) / 30.0
                shard_path = (
                    shards_dir / f"shard_{start:06d}_{end:06d}.mkv"
                ).resolve()
                payload = {
                    "frame_start": start,
                    "frame_end": end,
                    "width": width,
                    "height": height,
                    "samples": samples,
                    "engine": engine,
                    "denoise": denoise,
                    "frame_format": frame_format,
                    "fps": 30,
                    "timeout": max(900.0, shard_duration * 45),
                    "render_contract_version": RENDER_CONTRACT_VERSION,
                    "render_provenance": render_provenance,
                    "render_identity_digest": render_identity,
                }
                task_id = None
                if transport == "http":
                    from server.distributed import (
                        ARTIFACT_TRANSPORT,
                        PROTOCOL_VERSION,
                        deterministic_task_id,
                    )

                    assert poses_artifact is not None
                    logical_payload = {
                        **payload,
                        "task_protocol_version": PROTOCOL_VERSION,
                        "artifact_transport": ARTIFACT_TRANSPORT,
                        "poses_sha256": poses_artifact.sha256,
                        "poses_size": poses_artifact.size,
                    }
                    task_id = deterministic_task_id(
                        "render.frames",
                        logical_payload,
                    )
                    output_artifact = coordinator.reserve_output(
                        artifact_key=f"render-shard:{task_id}",
                        task_id=task_id,
                    )
                    payload.update(
                        {
                            "task_protocol_version": PROTOCOL_VERSION,
                            "artifact_transport": ARTIFACT_TRANSPORT,
                            "poses_artifact": poses_artifact.to_dict(),
                            "shard_artifact": output_artifact.to_dict(),
                        }
                    )
                    expected_artifacts[task_id] = output_artifact
                else:
                    payload.update(
                        {
                            "poses": str(poses_path.resolve()),
                            "shard_output": str(shard_path),
                        }
                    )
                submit_options = {
                    "worker": worker,
                    "task_id": task_id,
                    "eligible_worker_ids": eligible_worker_ids,
                }
                if transport == "http":
                    submit_options["retry_failed"] = True
                handles.append(
                    coordinator.submit(
                        "render.frames",
                        payload,
                        **submit_options,
                    )
                )
                expected_provenances[
                    handles[-1].request.task_id
                ] = render_provenance
                distributed_progress.append((handles[-1], start, end))
            try:
                distributed_results = coordinator.wait_many(
                    handles,
                    timeout=max(
                        900.0,
                        max((end - start) / 30.0 for start, end in ranges) * 45,
                    ),
                    on_poll=_record_progress,
                )
                results = {index: True for index in range(worker_count)}
            except Exception as exc:  # noqa: BLE001 - render failure is reported below
                raise RuntimeError(f"distributed render failed: {exc}") from exc
        else:
            threads = [
                threading.Thread(
                    target=_render_range,
                    args=(index, daemon, start, end),
                )
                for index, (daemon, (start, end)) in enumerate(
                    zip(daemon_ids, ranges)
                )
            ]
            for thread in threads:
                thread.start()
            last_count = -1
            while any(thread.is_alive() for thread in threads):
                rendered_frames = (
                    total_frames
                    if distributed
                    else min(
                        total_frames,
                        _count_frame_files(frames_dir, frame_format),
                    )
                )
                if rendered_frames != last_count:
                    last_count = rendered_frames
                    _record_progress()
                time.sleep(1)
            for thread in threads:
                thread.join()
        rendered_frames = (
            total_frames
            if distributed
            else min(
                total_frames,
                _count_frame_files(frames_dir, frame_format),
            )
        )
        _set(
            sid,
            progress=min(
                88,
                24 + int(64 * rendered_frames / max(1, total_frames)),
            ),
            rendered_frames=rendered_frames,
            workers=worker_count,
            message=f"rendered {rendered_frames}/{total_frames} full-quality frames\u2026",
        )
        if len(results) != worker_count or not all(results.values()):
            if distributed:
                raise RuntimeError("one or more distributed render shards failed")
            return False
        shard_paths = []
        source_frame_hashes = []
        decoded_rgb_hashes = []
        shard_hashes = []
        if distributed:
            if len(distributed_results) != len(ranges):
                raise RuntimeError(
                    "distributed render returned the wrong number of shards"
                )
            for result, (start, end) in zip(distributed_results, ranges):
                output = result.output
                render_provenance = expected_provenances.get(result.task_id)
                if render_provenance is None:
                    raise RuntimeError(
                        "distributed render returned an unknown task provenance"
                    )
                source_hash, shard_hash, decoded_rgb_hash = _validate_render_output(
                    output,
                    start=start,
                    end=end,
                    width=width,
                    height=height,
                    samples=samples,
                    engine=engine,
                    denoise=denoise,
                    frame_format=frame_format,
                    fps=30,
                    render_provenance=render_provenance,
                )
                shard_path = (
                    shards_dir / f"shard_{start:06d}_{end:06d}.mkv"
                ).resolve()
                if transport == "http":
                    from server.distributed import (
                        ARTIFACT_TRANSPORT,
                        ArtifactRef,
                    )

                    artifact = ArtifactRef.from_dict(
                        output.get("shard_artifact") or {},
                        require_complete=True,
                    )
                    expected_artifact = expected_artifacts.get(
                        result.task_id
                    )
                    if (
                        expected_artifact is None
                        or artifact.artifact_id
                        != expected_artifact.artifact_id
                        or artifact.sha256 != shard_hash
                        or output.get("artifact_transport")
                        != ARTIFACT_TRANSPORT
                    ):
                        raise RuntimeError(
                            "distributed render returned a mismatched shard artifact"
                        )
                    coordinator.download_output(artifact, shard_path)
                else:
                    reported_path = Path(
                        str(output.get("shard_output") or "")
                    ).resolve()
                    if reported_path != shard_path:
                        raise RuntimeError(
                            "distributed render returned an unexpected shard path"
                        )
                if not shard_path.is_file() or shard_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"distributed render shard is missing: {shard_path}"
                    )
                actual_hash, _actual_size = _file_sha256(shard_path)
                if actual_hash != shard_hash:
                    raise RuntimeError(
                        f"distributed render shard hash mismatch: {shard_path}"
                    )
                actual_validation = inspect_ffv1_shard(
                    shard_path,
                    frame_start=start,
                    frame_end=end,
                    width=width,
                    height=height,
                    fps=30,
                )
                if actual_validation["decoded_rgb_sha256"] != decoded_rgb_hash:
                    raise RuntimeError(
                        "distributed render shard decoded RGB hash mismatch: "
                        f"{shard_path}"
                    )
                shard_paths.append(shard_path)
                source_frame_hashes.append(source_hash)
                decoded_rgb_hashes.append(decoded_rgb_hash)
                shard_hashes.append(shard_hash)
            _set(
                sid,
                render_shards=[str(path) for path in shard_paths],
                render_source_frame_hashes=source_frame_hashes,
                render_decoded_rgb_sha256=decoded_rgb_hashes,
                render_shard_sha256=shard_hashes,
            )
        else:
            sequence = _frame_sequence(str(frames_dir))
            if sequence is None or sequence[1] != 0 or sequence[2] != motion.shape[0]:
                logger.warning(
                    "sharded render produced an incomplete frame sequence: %s",
                    sequence,
                )
                return False
        _set(sid, progress=90, message="encoding the full-quality frames\u2026")
        encoded = (
            _ffmpeg_shards(
                shard_paths,
                cached_video,
                frame_count=total_frames,
                audio_wav=audio_wav if scope == "full" else None,
            )
            if distributed
            else _ffmpeg_frames(
                str(frames_dir),
                cached_video,
                audio_wav=audio_wav if scope == "full" else None,
            )
        )
        if not encoded:
            if distributed:
                raise RuntimeError("distributed render encoding failed")
            return False
        shutil.copyfile(cached_video, output_video)
        return True
    finally:
        shutil.rmtree(render_root, ignore_errors=True)


def _render(sid: str, motion: np.ndarray, media_dir: Path, scope: str,
            a: int | None, b: int | None, *, audio_wav: str | None = None) -> None:
    from server import filament_render

    cfg = pod_config()
    filament_required = filament_render.enabled(scope)
    # window render is fast + silent; full render carries the song audio.
    with_audio = scope == "full"
    if scope == "window" and a is not None and b is not None:
        motion = motion[int(a):int(b)]
    if motion.shape[0] < 2:
        _set(sid, status="error", progress=0, message="nothing to render (empty window).")
        return
    try:
        accelerated = _render_warm_local(
            sid,
            motion,
            media_dir,
            scope,
            audio_wav=audio_wav,
        )
        if accelerated:
            _set(
                sid,
                status="done",
                progress=100,
                message="ready",
                video="edited.mp4",
                elapsed=round(time.time() - _RJOBS[sid].get("started", time.time())),
            )
            return
        if filament_required:
            _set(
                sid,
                status="error",
                progress=0,
                message="Filament render did not produce a validated final video",
            )
            return
        if _distributed_enabled() and scope == "full":
            _set(
                sid,
                status="error",
                progress=0,
                message="distributed render failed validation",
            )
            return
    except Exception as exc:  # noqa: BLE001 - retain the proven cold-render fallback
        if filament_required:
            logger.exception("Filament full render failed")
            _set(
                sid,
                status="error",
                progress=0,
                message=f"Filament render failed: {exc}",
            )
            return
        if _distributed_enabled() and scope == "full":
            _set(
                sid,
                status="error",
                progress=0,
                message=f"distributed render failed: {exc}",
            )
            return
        logger.warning("accelerated local render failed (%s); using the cold path", exc)
    if not cfg.host:
        _set(
            sid,
            status="error",
            progress=0,
            message=(
                "No GPU pod configured (set AGENTLODGE_POD_HOST). "
                "Rendering needs local Blender, HTTP render workers, or an SSH pod."
            ),
        )
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
# The UI presents the clips side by side and uses the exact after-render joint projection only to
# summarize which body regions changed.
_CJOBS: dict[str, dict] = {}
_BODY_PART_JOINTS = {
    "torso": ("Torso", (0, 3, 6, 9, 12, 15)),
    "left_arm": ("Left arm", (13, 16, 18, 20)),
    "right_arm": ("Right arm", (14, 17, 19, 21)),
    "left_leg": ("Left leg", (1, 4, 7, 10)),
    "right_leg": ("Right leg", (2, 5, 8, 11)),
}


def get_compare_job(sid: str) -> dict:
    with _RLOCK:
        return dict(_CJOBS.get(sid, {"status": "idle", "message": "", "progress": 0}))


def _cset(sid: str, **kw) -> None:
    with _RLOCK:
        _CJOBS.setdefault(sid, {}).update(kw)


def start_compare_render(sid: str, before_motion: np.ndarray, after_motion: np.ndarray,
                         media_dir: Path, *, metrics: dict | None = None,
                         audio_wav: str | None = None, audio_start: float = 0.0,
                         audio_dur: float = 0.0, comparison_id: str | None = None,
                         head_id: str | None = None, before_id: str | None = None) -> None:
    _cset(sid, status="queued", message="queued", progress=3, started=time.time(),
          metrics=metrics or {}, before_video=None, after_video=None, audio=None, highlight=None,
          comparison_id=comparison_id, head_id=head_id, before_id=before_id)
    threading.Thread(
        target=_compare_render,
        args=(sid, np.asarray(before_motion), np.asarray(after_motion), media_dir),
        kwargs={"audio_wav": audio_wav, "audio_start": audio_start, "audio_dur": audio_dur},
        daemon=True).start()


def _extract_window_audio(wav: str | None, start: float, dur: float, out_mp4: Path) -> str | None:
    """Slice ``[start, start+dur]`` of ``wav`` into a small AAC clip next to the compare videos so the
    comparison plays the window's music. Returns the file name (served from the media dir) or None if
    there is no wav or ffmpeg fails. Best-effort: a missing clip just means a silent comparison."""
    if not wav or dur <= 0 or not Path(wav).exists():
        return None
    import subprocess
    try:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}",
             "-t", f"{dur:.3f}", "-i", str(wav), "-vn", "-c:a", "aac", "-b:a", "160k", str(out_mp4)],
            capture_output=True, timeout=60)
        return out_mp4.name if (r.returncode == 0 and out_mp4.exists()) else None
    except Exception:  # noqa: BLE001 - audio is a nicety; never let it break the compare
        return None


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


def _frame_sequence(frames_dir: str) -> tuple[str, int, int] | None:
    """Return an ffmpeg pattern/start pair for contiguous Blender image numbering."""
    frames = sorted(Path(frames_dir).glob("frame_*.*"))
    parsed: list[tuple[Path, int, int, str]] = []
    for frame in frames:
        match = re.fullmatch(r"frame_(\d+)\.(png|tga)", frame.name)
        if match:
            parsed.append((frame, int(match.group(1)), len(match.group(1)), match.group(2)))
    if not parsed:
        return None
    numbers = [item[1] for item in parsed]
    widths = {item[2] for item in parsed}
    extensions = {item[3] for item in parsed}
    if (
        len(widths) != 1
        or len(extensions) != 1
        or numbers != list(range(numbers[0], numbers[0] + len(numbers)))
    ):
        return None
    digits = widths.pop()
    extension = extensions.pop()
    return (
        str(Path(frames_dir) / f"frame_%0{digits}d.{extension}"),
        numbers[0],
        len(numbers),
    )


def _ffmpeg_frames(frames_dir: str, out_mp4: Path, fps: int = 30,
                   *, audio_wav: str | None = None) -> bool:
    """Encode lossless Blender frames using the established cold-path codec settings."""
    import subprocess
    sequence = _frame_sequence(frames_dir)
    if sequence is None:
        return False
    pattern, start_number, frame_count = sequence
    have_audio = bool(audio_wav and Path(audio_wav).is_file())
    silent = (
        out_mp4.with_name(f".{out_mp4.stem}.silent{out_mp4.suffix}")
        if have_audio
        else out_mp4
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_mp4.unlink(missing_ok=True)
    silent.unlink(missing_ok=True)
    timeout = max(180, int(frame_count * 0.2 + 60))
    color_filter = ["-vf", "format=rgb24"] if pattern.endswith(".tga") else []
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                "-start_number", str(start_number), "-i", pattern,
                *color_filter,
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(silent),
            ],
            capture_output=True, timeout=timeout)
        if r.returncode != 0 or not silent.is_file() or silent.stat().st_size == 0:
            return False
        if not have_audio:
            return True
        muxed = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(silent),
                "-i", str(audio_wav), "-shortest", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                str(out_mp4),
            ],
            capture_output=True,
            timeout=timeout,
        )
        return (
            muxed.returncode == 0
            and out_mp4.is_file()
            and out_mp4.stat().st_size > 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        if have_audio:
            silent.unlink(missing_ok=True)


def _ffmpeg_shards(
    shards: list[Path],
    out_mp4: Path,
    *,
    frame_count: int,
    fps: int = 30,
    audio_wav: str | None = None,
) -> bool:
    """Decode ordered lossless FFV1 shards into the established final H.264/audio pipeline."""
    import subprocess

    if not shards or frame_count < 1 or any(
        not shard.is_file() or shard.stat().st_size == 0
        for shard in shards
    ):
        return False
    have_audio = bool(audio_wav and Path(audio_wav).is_file())
    silent = (
        out_mp4.with_name(f".{out_mp4.stem}.silent{out_mp4.suffix}")
        if have_audio
        else out_mp4
    )
    concat_file = out_mp4.with_name(
        f".{out_mp4.stem}.{uuid.uuid4().hex}.concat.txt"
    )
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_mp4.unlink(missing_ok=True)
    silent.unlink(missing_ok=True)
    lines = []
    for shard in shards:
        escaped = str(shard.resolve()).replace("\\", "/").replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    timeout = max(180, int(frame_count * 0.2 + 60))
    try:
        encoded = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-frames:v",
                str(frame_count),
                "-vf",
                "format=rgb24",
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(silent),
            ],
            capture_output=True,
            timeout=timeout,
        )
        if (
            encoded.returncode != 0
            or not silent.is_file()
            or silent.stat().st_size == 0
        ):
            return False
        if not have_audio:
            return True
        muxed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(silent),
                "-i",
                str(audio_wav),
                "-shortest",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ],
            capture_output=True,
            timeout=timeout,
        )
        return (
            muxed.returncode == 0
            and out_mp4.is_file()
            and out_mp4.stat().st_size > 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    finally:
        concat_file.unlink(missing_ok=True)
        if have_audio:
            silent.unlink(missing_ok=True)


def _render_cache_key(motion: np.ndarray, *, width: int, height: int,
                      samples: int, stride: int, engine: str = "eevee",
                      denoise: int = 1, context: str = "") -> str:
    h = hashlib.sha256()
    h.update(b"maestro-quality-render-v3")
    h.update(
        f"{width}:{height}:{samples}:{stride}:{engine}:{int(denoise)}".encode("ascii")
    )
    h.update(context.encode("utf-8"))
    h.update(np.ascontiguousarray(motion, dtype=np.float32).tobytes())
    return h.hexdigest()[:24]


def _smooth(values: np.ndarray, radius: int = 3) -> np.ndarray:
    if values.size < 3 or radius <= 0:
        return values
    width = min(values.size, 2 * radius + 1)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _build_change_highlight(before_poses: str, after_poses: str,
                            after_rig_metrics: str, *, fps: int = 30) -> dict | None:
    """Build exact screen-space body-part halos from FK deltas and the rendered Y-Bot projection."""
    with np.load(before_poses) as data:
        before = data["fk_joints"].astype(np.float64)
    with np.load(after_poses) as data:
        after = data["fk_joints"].astype(np.float64)
    with np.load(after_rig_metrics) as data:
        projected = data["projected"].astype(np.float64)
        rendered_frames = (
            data["rendered_frames"].astype(np.int64)
            if "rendered_frames" in data.files
            else None
        )
    n = min(before.shape[0], after.shape[0], projected.shape[0])
    if n < 2 or before.shape[1] < 22 or after.shape[1] < 22 or projected.shape[1] < 22:
        return None

    before = before[:n]
    after = after[:n]
    projected = projected[:n]
    if rendered_frames is not None:
        rendered_frames = rendered_frames[
            (rendered_frames >= 0) & (rendered_frames < n)
        ]
        if 0 < rendered_frames.size < n:
            target = np.arange(n, dtype=np.float64)
            dense = np.full_like(projected, np.nan)
            for joint in range(min(22, projected.shape[1])):
                for axis in range(3):
                    values = projected[rendered_frames, joint, axis]
                    valid = np.isfinite(values)
                    if valid.any():
                        dense[:, joint, axis] = np.interp(
                            target,
                            rendered_frames[valid].astype(np.float64),
                            values[valid],
                        )
            projected = dense
    before_rel = before - before[:, :1]
    after_rel = after - after[:, :1]
    joint_delta = np.linalg.norm(after_rel - before_rel, axis=-1)

    scores: dict[str, np.ndarray] = {}
    for part, (_label, joints) in _BODY_PART_JOINTS.items():
        values = np.sqrt(np.mean(np.square(joint_delta[:, joints]), axis=1))
        scores[part] = _smooth(values)

    root_path = (after[:, 0] - after[:1, 0]) - (before[:, 0] - before[:1, 0])
    scores["travel"] = _smooth(np.linalg.norm(root_path, axis=1))
    aggregates = {
        part: float(np.percentile(values, 90))
        for part, values in scores.items()
    }
    peak = max(aggregates.values(), default=0.0)
    if not np.isfinite(peak) or peak < 0.006:
        return None

    gate = max(0.008, peak * 0.28)
    selected = [part for part, value in aggregates.items() if value >= gate]
    if not selected:
        selected = [max(aggregates, key=aggregates.get)]

    frames: list[list[dict]] = []
    combined = np.zeros(n, dtype=np.float64)
    for frame in range(n):
        markers = []
        for part in selected:
            denom = max(aggregates[part], 1e-8)
            strength = float(np.clip(scores[part][frame] / denom, 0.0, 1.0))
            if strength < 0.16:
                continue
            if part == "travel":
                label = "Travel / lower body"
                joints = (0, 1, 2, 4, 5, 7, 8, 10, 11)
            else:
                label, joints = _BODY_PART_JOINTS[part]
            points = projected[frame, joints]
            valid = (
                np.isfinite(points).all(axis=1)
                & (points[:, 2] > 0)
                & (points[:, 0] > -0.2)
                & (points[:, 0] < 1.2)
                & (points[:, 1] > -0.2)
                & (points[:, 1] < 1.2)
            )
            if not valid.any():
                continue
            xy = points[valid, :2].copy()
            xy[:, 1] = 1.0 - xy[:, 1]
            low = xy.min(axis=0)
            high = xy.max(axis=0)
            center = 0.5 * (low + high)
            pad = 0.05 if part in ("torso", "travel") else 0.04
            radius = np.maximum(0.5 * (high - low) + pad, (0.065, 0.085))
            radius = np.minimum(radius, (0.30, 0.34))
            markers.append({
                "part": part,
                "label": label,
                "x": round(float(np.clip(center[0], 0.0, 1.0)), 4),
                "y": round(float(np.clip(center[1], 0.0, 1.0)), 4),
                "rx": round(float(radius[0]), 4),
                "ry": round(float(radius[1]), 4),
                "strength": round(strength, 3),
            })
            combined[frame] += strength
        frames.append(markers)

    labels = [
        ("Travel / lower body" if part == "travel" else _BODY_PART_JOINTS[part][0])
        for part in selected
    ]
    return {
        "fps": int(fps),
        "parts": labels,
        "peak_frame": int(np.argmax(combined)),
        "frames": frames,
    }


def _compare_warm(sid: str, before_motion: np.ndarray, after_motion: np.ndarray,
                  media_dir: Path) -> bool:
    """Render full-quality before/after streams in parallel through warm Blender daemons."""
    from server import warm_render as wr
    from server import fk
    width, height, samples, engine, denoise = _quality_render_settings("window")
    stride = 1
    if not wr.on_pod():
        return False
    media_dir.mkdir(parents=True, exist_ok=True)
    frames = int(max(before_motion.shape[0], after_motion.shape[0]))
    duration = frames / 30.0
    _cset(sid, status="rendering", progress=25, frames=frames,
          message=f"rendering a full-quality before & after preview ({frames} frames)\u2026")
    try:
        after_rig = media_dir / "_cmp_after_rig.npz"
        after_rig.unlink(missing_ok=True)
        specs = [
            {
                "tag": "before",
                "npz": str(media_dir / "_cmp_before_poses.npz"),
                "frames_dir": str(media_dir / "_cmp_before_frames"),
                "daemon": 0,
                "rig": "",
                "motion": before_motion,
            },
            {
                "tag": "after",
                "npz": str(media_dir / "_cmp_after_poses.npz"),
                "frames_dir": str(media_dir / "_cmp_after_frames"),
                "daemon": 1,
                "rig": str(after_rig),
                "motion": after_motion,
            },
        ]
        fk.save_poses_npz(before_motion, specs[0]["npz"])
        fk.save_poses_npz(after_motion, specs[1]["npz"])
    except Exception as exc:  # noqa: BLE001 - server-FK unavailable -> let the caller fall back
        logger.warning("warm compare server-FK failed (%s)", exc)
        return False
    cache_dir = media_dir / ".render_cache"
    cache_dir.mkdir(exist_ok=True)
    results: dict[str, bool] = {}
    missing: list[dict] = []
    for spec in specs:
        cache_key = _render_cache_key(
            spec["motion"],
            width=width,
            height=height,
            samples=samples,
            stride=stride,
            engine=engine,
            denoise=denoise,
            context=f"compare:{spec['tag']}",
        )
        spec["cache_video"] = cache_dir / f"{cache_key}.mp4"
        spec["cache_rig"] = cache_dir / f"{cache_key}.rig.npz"
        spec["final"] = media_dir / f"cmp_{spec['tag']}.mp4"
        cache_video = Path(spec["cache_video"])
        cache_ready = cache_video.is_file() and cache_video.stat().st_size > 0
        if spec["rig"]:
            cache_rig = Path(spec["cache_rig"])
            cache_ready = (
                cache_ready
                and cache_rig.is_file()
                and cache_rig.stat().st_size > 0
            )
        if cache_ready:
            shutil.copyfile(spec["cache_video"], spec["final"])
            if spec["rig"]:
                shutil.copyfile(spec["cache_rig"], after_rig)
            results[spec["tag"]] = True
            continue
        missing.append(spec)

    needed = min(len(missing), wr.POOL_SIZE)
    if needed:
        wr.ensure_pool(
            width=width,
            height=height,
            samples=samples,
            wait_ready=60,
        )
        daemon_ids = wr.ready_daemons()
        if len(daemon_ids) < needed:
            return False
        for spec, daemon in zip(missing, daemon_ids):
            spec["daemon"] = daemon
    missing_tags = {spec["tag"] for spec in missing}
    for spec in missing:
        frames_dir = Path(spec["frames_dir"])
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

    def _run(spec: dict) -> None:
        results[spec["tag"]] = wr.warm_render(
            spec["npz"],
            spec["frames_dir"],
            daemon=spec["daemon"],
            samples=samples,
            width=width,
            height=height,
            timeout=max(600.0, duration * 4),
            engine=engine,
            denoise=denoise,
            rig_metrics=spec["rig"],
            fast=False,
            stride=stride,
            projection_only=bool(spec["rig"]),
            batch_render=True,
        )

    threads = [
        *(threading.Thread(target=_run, args=(spec,)) for spec in missing),
    ]
    for t in threads:
        t.start()
    total_frames = sum(int(spec["motion"].shape[0]) for spec in specs)
    cached_frames = sum(
        int(spec["motion"].shape[0])
        for spec in specs
        if spec["tag"] not in missing_tags
    )
    last_count = -1
    while any(t.is_alive() for t in threads):
        rendered_frames = cached_frames + sum(
            min(
                int(spec["motion"].shape[0]),
                _count_frame_files(Path(spec["frames_dir"]), "png"),
            )
            for spec in missing
        )
        if rendered_frames != last_count:
            last_count = rendered_frames
            _cset(
                sid,
                progress=min(
                    88,
                    25 + int(63 * rendered_frames / max(1, total_frames)),
                ),
                rendered_frames=rendered_frames,
                message=(
                    f"rendered {rendered_frames}/{total_frames} full-quality "
                    "comparison frames\u2026"
                ),
            )
        time.sleep(0.5)
    for t in threads:
        t.join()
    rendered_frames = cached_frames + sum(
        min(
            int(spec["motion"].shape[0]),
            _count_frame_files(Path(spec["frames_dir"]), "png"),
        )
        for spec in missing
    )
    _cset(
        sid,
        progress=min(88, 25 + int(63 * rendered_frames / max(1, total_frames))),
        rendered_frames=rendered_frames,
        message=f"rendered {rendered_frames}/{total_frames} comparison frames\u2026",
    )
    if not all(results.get(spec["tag"]) for spec in specs):
        return False
    _cset(sid, progress=90, message="encoding before & after\u2026")
    enc: dict[str, bool] = {
        spec["tag"]: True for spec in specs if spec["tag"] not in missing_tags
    }
    enc_lock = threading.Lock()

    def _highlight() -> None:
        try:
            highlight = _build_change_highlight(
                specs[0]["npz"],
                specs[1]["npz"],
                str(after_rig),
            )
            if highlight:
                _cset(sid, highlight=highlight)
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("compare body-part highlight metadata failed: %s", exc)

    def _enc(spec: dict) -> None:
        out = spec["final"]
        enc[spec["tag"]] = _ffmpeg_frames(spec["frames_dir"], out)
        if enc[spec["tag"]]:
            shutil.copyfile(out, spec["cache_video"])
            if spec["rig"] and after_rig.is_file():
                shutil.copyfile(after_rig, spec["cache_rig"])
        with enc_lock:
            completed = sum(bool(enc.get(item["tag"])) for item in specs)
            _cset(
                sid,
                progress=min(98, 90 + int(8 * completed / max(1, len(specs)))),
                message=f"encoded {completed}/{len(specs)} comparison videos\u2026",
            )

    ethreads = [
        threading.Thread(target=_highlight),
        *(threading.Thread(target=_enc, args=(spec,)) for spec in missing),
    ]
    for t in ethreads:
        t.start()
    for t in ethreads:
        t.join()
    for spec in missing:
        if enc.get(spec["tag"]):
            shutil.rmtree(spec["frames_dir"], ignore_errors=True)
    return bool(enc.get("before") and enc.get("after"))


def _compare_render(sid: str, before_motion: np.ndarray, after_motion: np.ndarray,
                    media_dir: Path, *, audio_wav: str | None = None,
                    audio_start: float = 0.0, audio_dur: float = 0.0) -> None:
    cfg = pod_config()
    if not cfg.host:
        _cset(sid, status="error", progress=0,
              message="No GPU pod configured (set AGENTLODGE_POD_HOST).")
        return
    if before_motion.shape[0] < 2 or after_motion.shape[0] < 2:
        _cset(sid, status="error", progress=0, message="nothing to compare (empty window).")
        return
    started = _CJOBS.get(sid, {}).get("started", time.time())
    # Slice the window's music so the comparison plays with sound (best-effort; silent if unavailable).
    audio_result: dict[str, str | None] = {"name": None}

    def _extract_audio() -> None:
        audio_result["name"] = _extract_window_audio(
            audio_wav,
            audio_start,
            audio_dur,
            media_dir / "cmp_audio.m4a",
        )

    audio_thread = threading.Thread(target=_extract_audio, daemon=True)
    audio_thread.start()
    # Fast path: warm Blender pool (editor co-located on the pod). Falls through to the cold ssh
    # render below if the pool is unavailable or fails.
    try:
        if _compare_warm(
            sid,
            before_motion,
            after_motion,
            media_dir,
        ):
            audio_thread.join()
            _cset(sid, status="done", progress=100, message="ready",
                  before_video="cmp_before.mp4", after_video="cmp_after.mp4",
                  audio=audio_result["name"], elapsed=round(time.time() - started))
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
        audio_thread.join()
        _cset(sid, status="done", progress=100, message="ready",
              before_video="cmp_before.mp4", after_video="cmp_after.mp4", audio=audio_result["name"],
              elapsed=round(time.time() - _CJOBS[sid].get("started", time.time())))
    except Exception as exc:  # noqa: BLE001
        _cset(sid, status="error", progress=0, message=f"compare error: {exc}")
