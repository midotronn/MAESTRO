"""Feature-flagged full-song Filament renderer for the warm SLA path."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

BACKEND_VERSION = "filament-full-v1"
DEFAULT_ROOT = Path("/workspace/maestro-filament-poc")


def _elapsed_seconds(started: float) -> float:
    elapsed = float(time.perf_counter() - started)
    return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else 0.0


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.partial"
    )
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.partial"
    )
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def enabled(scope: str) -> bool:
    return (
        scope == "full"
        and os.environ.get("AGENTLODGE_FULL_RENDER_BACKEND", "blender").strip().lower()
        == "filament"
    )


def _required_path(name: str, default: Path, *, executable: bool = False) -> Path:
    path = Path(os.environ.get(name, str(default))).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{name} is missing: {path}")
    if executable and not os.access(path, os.X_OK):
        raise RuntimeError(f"{name} is not executable: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_indices() -> list[int]:
    configured = os.environ.get("AGENTLODGE_FILAMENT_GPU_INDICES", "").strip()
    if configured:
        try:
            indices = [int(value.strip()) for value in configured.split(",") if value.strip()]
        except ValueError as exc:
            raise RuntimeError(
                "AGENTLODGE_FILAMENT_GPU_INDICES must be comma-separated integers"
            ) from exc
    else:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"unable to enumerate Filament GPUs: {result.stderr.strip()}")
        try:
            indices = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        except ValueError as exc:
            raise RuntimeError("nvidia-smi returned an invalid GPU index") from exc
    if not indices or any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise RuntimeError(f"invalid Filament GPU indices: {indices}")
    limit = max(1, int(os.environ.get("AGENTLODGE_FILAMENT_GPUS", str(len(indices)))))
    if limit > len(indices):
        raise RuntimeError(
            f"requested {limit} Filament GPUs, but only {len(indices)} are available"
        )
    return indices[:limit]


def _ranges(frame_count: int, worker_count: int) -> list[tuple[int, int]]:
    workers = max(1, min(frame_count, worker_count))
    base, remainder = divmod(frame_count, workers)
    result = []
    start = 0
    for index in range(workers):
        end = start + base + (1 if index < remainder else 0)
        result.append((start, end))
        start = end
    return result


def _worker_assignments(gpu_indices: list[int]) -> list[tuple[int, int]]:
    try:
        workers_per_gpu = int(
            os.environ.get("AGENTLODGE_FILAMENT_WORKERS_PER_GPU", "1")
        )
    except ValueError as exc:
        raise RuntimeError(
            "AGENTLODGE_FILAMENT_WORKERS_PER_GPU must be an integer"
        ) from exc
    if not 1 <= workers_per_gpu <= 8:
        raise RuntimeError(
            "AGENTLODGE_FILAMENT_WORKERS_PER_GPU must be between 1 and 8"
        )
    return [
        (gpu_index, slot)
        for slot in range(workers_per_gpu)
        for gpu_index in gpu_indices
    ]


def _probe_payload(path: Path, *, count_frames: bool) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
    command.extend(
        [
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames:"
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _probe_video(path: Path) -> dict:
    payload = _probe_payload(path, count_frames=False)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}")
    stream = streams[0]
    raw_frames = stream.get("nb_frames")
    try:
        frames = int(raw_frames)
    except (TypeError, ValueError):
        payload = _probe_payload(path, count_frames=True)
        streams = payload.get("streams") or []
        if len(streams) != 1:
            raise RuntimeError(f"expected one video stream in {path}")
        stream = streams[0]
        raw_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
        try:
            frames = int(raw_frames)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"ffprobe did not report a frame count for {path}") from exc
    return {
        "codec": str(stream.get("codec_name") or ""),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "frame_rate": str(stream.get("avg_frame_rate") or ""),
        "frames": frames,
        "duration": float((payload.get("format") or {}).get("duration") or 0.0),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_probe(path: Path, probe: dict, *, expected_frames: int) -> None:
    if (
        probe["codec"] != "h264"
        or probe["width"] != 1080
        or probe["height"] != 1080
        or probe["frame_rate"] != "30/1"
        or probe["frames"] != expected_frames
    ):
        raise RuntimeError(f"Filament video validation failed for {path}: {probe}")


def _run(command: list[str], *, timeout: float) -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{Path(command[0]).name} failed: {detail[-1000:]}")


def _concat_shards(
    shards: list[Path],
    output: Path,
    *,
    audio_wav: str | None,
    timeout: float,
) -> None:
    concat_file = output.with_suffix(".concat.txt")
    silent = output.with_suffix(".silent.mp4")
    try:
        lines = []
        for path in shards:
            escaped = str(path.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'\n")
        concat_file.write_text(
            "".join(lines),
            encoding="utf-8",
        )
        _run(
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
                "-c",
                "copy",
                str(silent),
            ],
            timeout=timeout,
        )
        if audio_wav:
            audio = Path(audio_wav)
            if not audio.is_file():
                raise RuntimeError(f"full-render audio is missing: {audio}")
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(silent),
                    "-i",
                    str(audio),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-shortest",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                timeout=timeout,
            )
        else:
            shutil.move(silent, output)
    finally:
        concat_file.unlink(missing_ok=True)
        silent.unlink(missing_ok=True)


def _settings() -> dict[str, str]:
    defaults = {
        "MAESTRO_FILAMENT_WORLD_LINEAR": "0.08",
        "MAESTRO_FILAMENT_INDIRECT_LUX": "25000",
        "MAESTRO_FILAMENT_KEY_LUX": "0",
        "MAESTRO_FILAMENT_FILL_LUX": "0",
        "MAESTRO_FILAMENT_ASSET_LIGHT_SCALE": "15",
        "MAESTRO_FILAMENT_ASSET_LIGHT_FALLOFF": "20",
        "MAESTRO_FILAMENT_SHADOW_TYPE": "pcss",
        "MAESTRO_FILAMENT_AO_ENABLED": "1",
        "MAESTRO_FILAMENT_ASYNC_RING": "8",
        "AGENTLODGE_FILAMENT_FAST_GROUNDING": "0",
        "AGENTLODGE_FILAMENT_FOOT_GROUNDING": "0",
    }
    return {name: os.environ.get(name, value) for name, value in defaults.items()}


def _cache_key(
    motion: np.ndarray,
    *,
    binary: Path,
    static_glb: Path,
    ibl: Path,
    selector: Path,
    real_vulkan: Path,
    nvidia_vk_icd: Path,
    settings: dict[str, str],
    worker_count: int,
    audio_wav: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(BACKEND_VERSION.encode("ascii"))
    digest.update(np.ascontiguousarray(motion, dtype=np.float32).tobytes())
    for path in (binary, static_glb, ibl, selector, real_vulkan, nvidia_vk_icd):
        digest.update(_sha256(path).encode("ascii"))
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    digest.update(str(worker_count).encode("ascii"))
    if audio_wav:
        stat = Path(audio_wav).stat()
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()[:24]


def render_full_motion(
    sid: str,
    motion: np.ndarray,
    media_dir: Path,
    *,
    audio_wav: str | None = None,
    update: Callable[..., None] | None = None,
) -> bool:
    """Export one animated GLB, render contiguous GPU shards, and publish one MP4."""
    from server import fk
    from server import warm_render as wr

    width = max(1, int(os.environ.get("AGENTLODGE_RENDER_FULL_W", "1080")))
    height = max(1, int(os.environ.get("AGENTLODGE_RENDER_FULL_H", "1080")))
    if (width, height) != (1080, 1080):
        raise RuntimeError(
            "the validated Filament binary requires AGENTLODGE_RENDER_FULL_W/H=1080"
        )
    if not wr.on_pod():
        raise RuntimeError("Filament rendering requires the editor and Blender exporter on the Pod")

    root = Path(os.environ.get("AGENTLODGE_FILAMENT_ROOT", str(DEFAULT_ROOT))).resolve()
    binary = _required_path(
        "AGENTLODGE_FILAMENT_BINARY", root / "filament_bench", executable=True
    )
    static_glb = _required_path(
        "AGENTLODGE_FILAMENT_STATIC_GLB", root / "ybot_visible_static.glb"
    )
    ibl = _required_path(
        "MAESTRO_FILAMENT_IBL",
        root
        / "filament"
        / "bin"
        / "assets"
        / "ibl"
        / "lightroom_14b"
        / "lightroom_14b_ibl.ktx",
    )
    selector_dir = Path(
        os.environ.get("AGENTLODGE_FILAMENT_VULKAN_SELECTOR_DIR", root / "vulkan-selector")
    ).resolve()
    selector = _required_path(
        "AGENTLODGE_FILAMENT_VULKAN_SELECTOR", selector_dir / "libvulkan.so.1"
    )
    real_vulkan = _required_path(
        "MAESTRO_VK_REAL_LIBRARY", selector_dir / "libvulkan.real.so.1"
    )
    nvidia_vk_icd = _required_path(
        "AGENTLODGE_NVIDIA_VK_ICD",
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
    )
    gpu_indices = _gpu_indices()
    worker_assignments = _worker_assignments(gpu_indices)
    frame_count = int(motion.shape[0])
    if frame_count < 1:
        raise RuntimeError("Filament received no source frames")
    ranges = _ranges(frame_count, len(worker_assignments))
    worker_assignments = worker_assignments[: len(ranges)]
    settings = _settings()

    media_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = media_dir / ".render_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = _cache_key(
        motion,
        binary=binary,
        static_glb=static_glb,
        ibl=ibl,
        selector=selector,
        real_vulkan=real_vulkan,
        nvidia_vk_icd=nvidia_vk_icd,
        settings=settings,
        worker_count=len(worker_assignments),
        audio_wav=audio_wav,
    )
    cached_video = cache_dir / f"filament-{cache_key}.mp4"
    output_video = media_dir / "edited.mp4"
    cache_enabled = os.environ.get("AGENTLODGE_FILAMENT_DISABLE_CACHE", "0") != "1"
    if cache_enabled and cached_video.is_file() and cached_video.stat().st_size > 0:
        try:
            _validate_probe(
                cached_video,
                _probe_video(cached_video),
                expected_frames=frame_count,
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            cached_video.unlink(missing_ok=True)
        else:
            _atomic_link_or_copy(cached_video, output_video)
            if update:
                update(
                    status="rendering",
                    progress=96,
                    frames=frame_count,
                    rendered_frames=frame_count,
                    workers=0,
                    worker_ids=[],
                    message="reusing the cached Filament render\u2026",
                )
            return True

    temp_parent = Path(os.environ.get("AGENTLODGE_RENDER_TMP", tempfile.gettempdir()))
    temp_parent.mkdir(parents=True, exist_ok=True)
    render_root = Path(tempfile.mkdtemp(prefix=f"maestro-filament-{sid}-", dir=temp_parent))
    candidate_video = render_root / "edited.mp4"
    success = False
    started = time.perf_counter()
    try:
        poses = render_root / "poses.npz"
        animated_glb = render_root / "animated.glb"
        export_frames = render_root / "export"
        export_frames.mkdir()
        pose_started = time.perf_counter()
        fk.save_poses_npz(motion, poses)
        pose_serialization_fk_seconds = _elapsed_seconds(pose_started)

        if update:
            update(
                status="rendering",
                progress=24,
                frames=frame_count,
                workers=len(worker_assignments),
                worker_ids=[
                    f"filament-gpu{gpu_index}-slot{slot}"
                    for gpu_index, slot in worker_assignments
                ],
                rendered_frames=0,
                message="exporting the exact animated scene\u2026",
            )
        warm_pool_started = time.perf_counter()
        wr.ensure_pool(
            width=1080,
            height=1080,
            samples=max(1, int(os.environ.get("AGENTLODGE_RENDER_FULL_SAMPLES", "96"))),
            engine=os.environ.get("AGENTLODGE_RENDER_ENGINE", "eevee"),
            denoise=int(os.environ.get("AGENTLODGE_RENDER_DENOISE", "1")),
            frame_format="tga",
            wait_ready=60,
        )
        ready = wr.ready_daemons()
        if not ready:
            raise RuntimeError("no warm Blender daemon is available for GLB export")
        warm_pool_ensure_attestation_seconds = _elapsed_seconds(warm_pool_started)
        export_started = time.perf_counter()
        exported = wr.warm_render(
            str(poses),
            str(export_frames),
            daemon=ready[0],
            samples=max(1, int(os.environ.get("AGENTLODGE_RENDER_FULL_SAMPLES", "96"))),
            width=1080,
            height=1080,
            engine=os.environ.get("AGENTLODGE_RENDER_ENGINE", "eevee"),
            denoise=int(os.environ.get("AGENTLODGE_RENDER_DENOISE", "1")),
            batch_render=True,
            export_glb=str(animated_glb),
            fast=settings["AGENTLODGE_FILAMENT_FAST_GROUNDING"] == "1",
            foot_grounding=(
                settings["AGENTLODGE_FILAMENT_FOOT_GROUNDING"] == "1"
            ),
            frame_start=0,
            frame_end=frame_count,
            timeout=max(180.0, frame_count / 30.0 * 2.0),
            frame_format="tga",
        )
        if not exported or not animated_glb.is_file() or animated_glb.stat().st_size == 0:
            raise RuntimeError("warm Blender GLB export failed")
        export_seconds = _elapsed_seconds(export_started)
        blender_request_export_seconds = export_seconds

        if update:
            update(
                progress=36,
                message=(
                    f"rendering {frame_count} frames with {len(worker_assignments)} "
                    f"workers on {len(gpu_indices)} GPUs\u2026"
                ),
            )
        timeout = max(
            600.0,
            float(os.environ.get("AGENTLODGE_FILAMENT_TIMEOUT_SECONDS", "600")),
        )
        workers = []
        worker_launch_started = time.perf_counter()
        try:
            for worker_index, ((gpu_index, slot), (start, end)) in enumerate(
                zip(worker_assignments, ranges)
            ):
                worker_dir = (
                    render_root / f"worker-{worker_index}-gpu-{gpu_index}-slot-{slot}"
                )
                worker_dir.mkdir()
                shard = worker_dir / f"shard-{start:06d}-{end:06d}.mp4"
                log = worker_dir / "filament.log"
                environment = os.environ.copy()
                environment.update(settings)
                environment.update(
                    {
                        "CUDA_VISIBLE_DEVICES": str(gpu_index),
                        "LD_LIBRARY_PATH": (
                            str(selector_dir)
                            + (
                                os.pathsep + environment["LD_LIBRARY_PATH"]
                                if environment.get("LD_LIBRARY_PATH")
                                else ""
                            )
                        ),
                        "MAESTRO_VK_REAL_LIBRARY": str(real_vulkan),
                        "MAESTRO_VK_DEVICE_INDEX": str(gpu_index),
                        "VK_ICD_FILENAMES": str(nvidia_vk_icd),
                        "MAESTRO_FILAMENT_IBL": str(ibl),
                        "MAESTRO_FILAMENT_JOB_ONLY": "1",
                        "MAESTRO_FILAMENT_FRAME_OFFSET": str(start),
                        "MAESTRO_FILAMENT_ASYNC_FRAMES": str(end - start),
                        "MAESTRO_FILAMENT_ASYNC_VIDEO_PATH": str(shard),
                        "MAESTRO_FILAMENT_ASYNC_ENCODER": "h264_nvenc",
                        "MAESTRO_FILAMENT_WRITE_ASYNC_SAMPLES": "0",
                    }
                )
                log_handle = log.open("wb")
                try:
                    process = subprocess.Popen(
                        [str(binary), str(static_glb), str(animated_glb), str(worker_dir)],
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=environment,
                    )
                except Exception:
                    log_handle.close()
                    raise
                workers.append(
                    {
                        "index": worker_index,
                        "gpu_index": gpu_index,
                        "slot": slot,
                        "start": start,
                        "end": end,
                        "shard": shard,
                        "log": log,
                        "log_handle": log_handle,
                        "process": process,
                        "started": time.perf_counter(),
                    }
                )
        except Exception:
            for worker in workers:
                worker["process"].terminate()
                worker["log_handle"].close()
            raise
        worker_process_launch_seconds = _elapsed_seconds(worker_launch_started)

        worker_wait_started = time.perf_counter()
        try:
            pending = list(workers)
            while pending:
                now = time.perf_counter()
                for worker in list(pending):
                    return_code = worker["process"].poll()
                    if return_code is None:
                        if now - worker["started"] >= timeout:
                            raise RuntimeError(
                                f"Filament GPU {worker['gpu_index']} timed out"
                            )
                        continue
                    worker["seconds"] = max(0.0, float(now - worker["started"]))
                    worker["log_handle"].close()
                    pending.remove(worker)
                    if return_code != 0:
                        detail = worker["log"].read_text(errors="replace")[-2000:]
                        raise RuntimeError(
                            f"Filament GPU {worker['gpu_index']} failed with code "
                            f"{return_code}: {detail}"
                        )
                if pending:
                    time.sleep(0.02)
        except Exception:
            for worker in workers:
                if worker["process"].poll() is None:
                    worker["process"].terminate()
            for worker in workers:
                if worker["process"].poll() is None:
                    try:
                        worker["process"].wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker["process"].kill()
                if not worker["log_handle"].closed:
                    worker["log_handle"].close()
            raise
        concurrent_worker_wait_seconds = _elapsed_seconds(worker_wait_started)

        worker_reports = []
        shards = []
        shard_validation_started = time.perf_counter()
        for worker in workers:
            shard = worker["shard"]
            if not shard.is_file() or shard.stat().st_size == 0:
                raise RuntimeError(f"Filament shard is missing: {shard}")
            probe = _probe_video(shard)
            expected_frames = worker["end"] - worker["start"]
            _validate_probe(shard, probe, expected_frames=expected_frames)
            log_text = worker["log"].read_text(errors="replace")
            selector_marker = (
                "MAESTRO_VK_SELECTOR selected Vulkan physical device index "
                f"{worker['gpu_index']} of "
            )
            if selector_marker not in log_text:
                raise RuntimeError(
                    f"Filament GPU {worker['gpu_index']} did not attest its "
                    "selected Vulkan physical device"
                )
            shards.append(shard)
            worker_reports.append(
                {
                    "gpu_index": worker["gpu_index"],
                    "slot": worker["slot"],
                    "frame_start": worker["start"],
                    "frame_end": worker["end"],
                    "seconds": worker["seconds"],
                    "probe": probe,
                }
            )
        shard_probe_validation_seconds = _elapsed_seconds(shard_validation_started)

        if update:
            update(
                progress=90,
                rendered_frames=frame_count,
                message="joining the Filament shards and song audio\u2026",
            )
        concat_started = time.perf_counter()
        _concat_shards(
            shards,
            candidate_video,
            audio_wav=audio_wav,
            timeout=max(300.0, frame_count / 30.0 * 2.0),
        )
        concat_seconds = _elapsed_seconds(concat_started)
        concat_mux_seconds = concat_seconds
        finalization_started = time.perf_counter()
        final_probe = _probe_video(candidate_video)
        _validate_probe(candidate_video, final_probe, expected_frames=frame_count)
        if cache_enabled:
            _atomic_link_or_copy(candidate_video, cached_video)
            _atomic_link_or_copy(cached_video, output_video)
        else:
            _atomic_link_or_copy(candidate_video, output_video)
        final_probe_cache_publication_seconds = _elapsed_seconds(finalization_started)
        report = {
            "backend_version": BACKEND_VERSION,
            "frames": frame_count,
            "gpu_indices": gpu_indices,
            "worker_gpu_indices": [
                gpu_index for gpu_index, _slot in worker_assignments
            ],
            "workers_per_gpu": len(worker_assignments) // len(gpu_indices),
            "settings": settings,
            "pose_serialization_fk_seconds": pose_serialization_fk_seconds,
            "warm_pool_ensure_attestation_seconds": warm_pool_ensure_attestation_seconds,
            "blender_request_export_seconds": blender_request_export_seconds,
            "worker_process_launch_seconds": worker_process_launch_seconds,
            "concurrent_worker_wait_seconds": concurrent_worker_wait_seconds,
            "shard_probe_validation_seconds": shard_probe_validation_seconds,
            "concat_mux_seconds": concat_mux_seconds,
            "final_probe_cache_publication_seconds": final_probe_cache_publication_seconds,
            "export_seconds": export_seconds,
            "concat_seconds": concat_seconds,
            "total_seconds": _elapsed_seconds(started),
            "animated_glb_bytes": animated_glb.stat().st_size,
            "animated_glb_sha256": _sha256(animated_glb),
            "workers": worker_reports,
            "final": final_probe,
        }
        (media_dir / "filament_render_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        success = True
        return True
    finally:
        candidate_video.unlink(missing_ok=True)
        keep = os.environ.get("AGENTLODGE_FILAMENT_KEEP_SCRATCH", "0") == "1"
        if success and not keep:
            shutil.rmtree(render_root, ignore_errors=True)
