#!/usr/bin/env python3
"""Compare direct Blender frames with the distributed FFV1 render path."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import warm_render  # noqa: E402
from server.distributed.handlers import (  # noqa: E402
    RenderFramesHandler,
    _ffmpeg_executable,
    _frame_digest,
)


def decoded_rgb_hash(
    source: Path,
    *,
    frame_count: int,
    frame_start: int = 0,
    sequence_format: str | None = None,
) -> str:
    command = [_ffmpeg_executable(), "-v", "error"]
    if sequence_format:
        command.extend(
            [
                "-start_number",
                str(frame_start),
                "-i",
                str(source / f"frame_%04d.{sequence_format}"),
            ]
        )
    else:
        command.extend(["-i", str(source)])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    digest = hashlib.sha256()
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    assert process.stderr is not None
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg RGB decode failed ({return_code}): {stderr.strip()}"
        )
    return digest.hexdigest()


def validate_render_equivalence(
    poses_path: Path,
    output_dir: Path,
    *,
    shared_root: Path,
    frame_start: int = 0,
    frame_end: int = 12,
    reference_daemon: int = 0,
    worker_daemon: int = 1,
    width: int = 1080,
    height: int = 1080,
    samples: int = 96,
    engine: str = "eevee",
    denoise: int = 1,
    frame_format: str = "tga",
    fps: int = 30,
    timeout: float = 900.0,
) -> dict:
    poses_path = poses_path.resolve()
    shared_root = shared_root.resolve()
    output_dir = output_dir.resolve()
    try:
        poses_path.relative_to(shared_root)
        output_dir.relative_to(shared_root)
    except ValueError as exc:
        raise ValueError("poses and output_dir must be inside shared_root") from exc
    if not poses_path.is_file():
        raise FileNotFoundError(poses_path)
    if frame_end <= frame_start:
        raise ValueError("frame_end must be greater than frame_start")

    reference_frames = output_dir / "reference_frames"
    shard_path = output_dir / "distributed_ffv1.mkv"
    worker_tmp = output_dir / "worker_tmp"
    shutil.rmtree(reference_frames, ignore_errors=True)
    shutil.rmtree(worker_tmp, ignore_errors=True)
    shard_path.unlink(missing_ok=True)
    reference_frames.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if warm_render.ensure_pool(
        width=width,
        height=height,
        samples=samples,
        wait_ready=120,
    ) < 1:
        raise RuntimeError("no warm Blender daemon is available")
    started_at = time.time()
    if not warm_render.warm_render(
        str(poses_path),
        str(reference_frames),
        daemon=reference_daemon,
        samples=samples,
        width=width,
        height=height,
        engine=engine,
        denoise=denoise,
        fast=False,
        stride=1,
        batch_render=True,
        frame_start=frame_start,
        frame_end=frame_end,
        clear_frames=True,
        frame_format=frame_format,
        timeout=timeout,
    ):
        raise RuntimeError("reference Blender render failed")
    reference_seconds = time.time() - started_at

    handler = RenderFramesHandler(
        shared_root=shared_root,
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        denoise=denoise,
        frame_format=frame_format,
        daemon=worker_daemon,
        local_tmp=worker_tmp,
    )
    handler.preload()
    worker_started_at = time.time()
    worker_result = handler(
        {
            "poses": str(poses_path),
            "shard_output": str(shard_path),
            "frame_start": frame_start,
            "frame_end": frame_end,
            "width": width,
            "height": height,
            "samples": samples,
            "engine": engine,
            "denoise": denoise,
            "frame_format": frame_format,
            "fps": fps,
            "timeout": timeout,
        }
    )
    worker_seconds = time.time() - worker_started_at

    frame_count = frame_end - frame_start
    reference_source_hash = _frame_digest(
        reference_frames,
        frame_start,
        frame_end,
        frame_format,
    )
    reference_rgb_hash = decoded_rgb_hash(
        reference_frames,
        frame_count=frame_count,
        frame_start=frame_start,
        sequence_format=frame_format,
    )
    shard_rgb_hash = decoded_rgb_hash(
        shard_path,
        frame_count=frame_count,
    )
    source_match = (
        reference_source_hash == worker_result["source_frames_sha256"]
    )
    rgb_match = reference_rgb_hash == shard_rgb_hash
    report = {
        "schema_version": 1,
        "status": "passed" if source_match and rgb_match else "failed",
        "poses": str(poses_path),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frames": frame_count,
        "quality": {
            "width": width,
            "height": height,
            "samples": samples,
            "engine": engine,
            "denoise": denoise,
            "frame_format": frame_format,
            "fps": fps,
            "render_every_frame": True,
        },
        "timings_seconds": {
            "reference_render": round(reference_seconds, 3),
            "worker_render_and_package": round(worker_seconds, 3),
        },
        "reference_source_frames_sha256": reference_source_hash,
        "worker_source_frames_sha256": worker_result[
            "source_frames_sha256"
        ],
        "reference_decoded_rgb_sha256": reference_rgb_hash,
        "ffv1_decoded_rgb_sha256": shard_rgb_hash,
        "source_frame_bytes_match": source_match,
        "decoded_rgb_match": rgb_match,
        "shard_sha256": worker_result["shard_sha256"],
        "shard_path": str(shard_path),
    }
    (output_dir / "equivalence_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "passed":
        raise RuntimeError(
            "render equivalence failed; inspect equivalence_report.json"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=Path("/workspace"),
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=12)
    parser.add_argument("--reference-daemon", type=int, default=0)
    parser.add_argument("--worker-daemon", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    report = validate_render_equivalence(
        args.poses,
        args.output_dir,
        shared_root=args.shared_root,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        reference_daemon=args.reference_daemon,
        worker_daemon=args.worker_daemon,
        timeout=args.timeout,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
