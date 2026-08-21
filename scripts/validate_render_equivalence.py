#!/usr/bin/env python3
"""Validate exact FFV1 transport and bounded EEVEE rerender variation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import warm_render  # noqa: E402
from server.distributed.handlers import (  # noqa: E402
    RenderFramesHandler,
    _package_ffv1,
    _ffmpeg_executable,
    _frame_digest,
)
from server.distributed.render_contract import render_identity_digest  # noqa: E402

DEFAULT_MAX_CHANGED_CHANNEL_FRACTION = 0.0005
DEFAULT_MAX_MEAN_ABSOLUTE_ERROR = 0.001
DEFAULT_MAX_CHANNEL_ERROR = 8


def _decoded_rgb_command(
    source: Path,
    *,
    frame_count: int,
    frame_start: int = 0,
    sequence_format: str | None = None,
) -> list[str]:
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
    return command


def _decoded_rgb_bytes(
    source: Path,
    *,
    frame_count: int,
    frame_start: int = 0,
    sequence_format: str | None = None,
) -> bytes:
    process = subprocess.run(
        _decoded_rgb_command(
            source,
            frame_count=frame_count,
            frame_start=frame_start,
            sequence_format=sequence_format,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg RGB decode failed ({process.returncode}): {stderr.strip()}"
        )
    return process.stdout


def decoded_rgb_hash(
    source: Path,
    *,
    frame_count: int,
    frame_start: int = 0,
    sequence_format: str | None = None,
) -> str:
    process = subprocess.Popen(
        _decoded_rgb_command(
            source,
            frame_count=frame_count,
            frame_start=frame_start,
            sequence_format=sequence_format,
        ),
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


def decoded_rgb_difference(
    reference: Path,
    candidate: Path,
    *,
    frame_count: int,
    reference_frame_start: int = 0,
    reference_sequence_format: str | None = None,
    candidate_frame_start: int = 0,
    candidate_sequence_format: str | None = None,
) -> dict:
    reference_bytes = _decoded_rgb_bytes(
        reference,
        frame_count=frame_count,
        frame_start=reference_frame_start,
        sequence_format=reference_sequence_format,
    )
    candidate_bytes = _decoded_rgb_bytes(
        candidate,
        frame_count=frame_count,
        frame_start=candidate_frame_start,
        sequence_format=candidate_sequence_format,
    )
    if len(reference_bytes) != len(candidate_bytes):
        raise RuntimeError(
            "decoded RGB byte counts differ: "
            f"reference={len(reference_bytes)}, candidate={len(candidate_bytes)}"
        )
    reference_values = np.frombuffer(reference_bytes, dtype=np.uint8).astype(
        np.int16
    )
    candidate_values = np.frombuffer(candidate_bytes, dtype=np.uint8).astype(
        np.int16
    )
    differences = np.abs(reference_values - candidate_values)
    changed_channels = int(np.count_nonzero(differences))
    channels = int(differences.size)
    mean_absolute_error = float(np.mean(differences))
    mean_squared_error = float(
        np.mean(np.square(differences, dtype=np.float64))
    )
    psnr = (
        math.inf
        if mean_squared_error == 0.0
        else 10.0 * math.log10((255.0**2) / mean_squared_error)
    )
    return {
        "channels": channels,
        "changed_channels": changed_channels,
        "changed_channel_fraction": round(changed_channels / channels, 9),
        "identical_channel_fraction": round(
            (channels - changed_channels) / channels,
            9,
        ),
        "mean_absolute_error_8bit": round(mean_absolute_error, 9),
        "max_absolute_error_8bit": int(np.max(differences)),
        "psnr_db": None if math.isinf(psnr) else round(psnr, 3),
    }


def render_difference_within_tolerance(
    difference: dict,
    *,
    max_changed_channel_fraction: float,
    max_mean_absolute_error: float,
    max_channel_error: int,
) -> bool:
    return (
        float(difference["changed_channel_fraction"])
        <= max_changed_channel_fraction
        and float(difference["mean_absolute_error_8bit"])
        <= max_mean_absolute_error
        and int(difference["max_absolute_error_8bit"]) <= max_channel_error
    )


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
    max_changed_channel_fraction: float = (
        DEFAULT_MAX_CHANGED_CHANNEL_FRACTION
    ),
    max_mean_absolute_error: float = DEFAULT_MAX_MEAN_ABSOLUTE_ERROR,
    max_channel_error: int = DEFAULT_MAX_CHANNEL_ERROR,
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
    worker_frames = output_dir / "worker_source_frames"
    shard_path = output_dir / "distributed_ffv1.mkv"
    shutil.rmtree(reference_frames, ignore_errors=True)
    shutil.rmtree(worker_frames, ignore_errors=True)
    shard_path.unlink(missing_ok=True)
    reference_frames.mkdir(parents=True, exist_ok=True)
    worker_frames.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if warm_render.ensure_pool(
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        denoise=denoise,
        frame_format=frame_format,
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
    )
    handler.preload()
    render_provenance = handler.render_provenance()
    worker_payload = {
        "poses": str(poses_path),
        "frames_dir": str(worker_frames),
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
        "render_contract_version": render_provenance[
            "render_contract_version"
        ],
        "render_provenance": render_provenance,
    }
    worker_payload["render_identity_digest"] = render_identity_digest(
        render_provenance,
        {
            key: worker_payload[key]
            for key in (
                "width",
                "height",
                "samples",
                "engine",
                "denoise",
                "frame_format",
                "fps",
            )
        },
    )
    worker_started_at = time.time()
    worker_result = handler(worker_payload)
    worker_render_seconds = time.time() - worker_started_at

    frame_count = frame_end - frame_start
    packaging_started_at = time.time()
    _package_ffv1(
        worker_frames,
        shard_path,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_format=frame_format,
        fps=fps,
    )
    packaging_seconds = time.time() - packaging_started_at
    reference_source_hash = _frame_digest(
        reference_frames,
        frame_start,
        frame_end,
        frame_format,
    )
    worker_source_hash = _frame_digest(
        worker_frames,
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
    worker_rgb_hash = decoded_rgb_hash(
        worker_frames,
        frame_count=frame_count,
        frame_start=frame_start,
        sequence_format=frame_format,
    )
    shard_rgb_hash = decoded_rgb_hash(
        shard_path,
        frame_count=frame_count,
    )
    reference_worker_difference = decoded_rgb_difference(
        reference_frames,
        worker_frames,
        frame_count=frame_count,
        reference_frame_start=frame_start,
        reference_sequence_format=frame_format,
        candidate_frame_start=frame_start,
        candidate_sequence_format=frame_format,
    )
    reference_within_tolerance = render_difference_within_tolerance(
        reference_worker_difference,
        max_changed_channel_fraction=max_changed_channel_fraction,
        max_mean_absolute_error=max_mean_absolute_error,
        max_channel_error=max_channel_error,
    )
    worker_source_integrity_match = (
        worker_source_hash == worker_result["source_frames_sha256"]
    )
    transport_rgb_match = worker_rgb_hash == shard_rgb_hash
    shard_digest = hashlib.sha256()
    with shard_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            shard_digest.update(chunk)
    passed = (
        worker_source_integrity_match
        and transport_rgb_match
        and reference_within_tolerance
    )
    report = {
        "schema_version": 2,
        "status": "passed" if passed else "failed",
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
            "worker_render": round(worker_render_seconds, 3),
            "lossless_package": round(packaging_seconds, 3),
        },
        "reference_source_frames_sha256": reference_source_hash,
        "worker_source_frames_sha256": worker_source_hash,
        "worker_reported_source_frames_sha256": worker_result[
            "source_frames_sha256"
        ],
        "worker_source_integrity_match": worker_source_integrity_match,
        "reference_source_frame_bytes_match": (
            reference_source_hash == worker_source_hash
        ),
        "reference_decoded_rgb_sha256": reference_rgb_hash,
        "worker_decoded_rgb_sha256": worker_rgb_hash,
        "ffv1_decoded_rgb_sha256": shard_rgb_hash,
        "transport_decoded_rgb_match": transport_rgb_match,
        "reference_worker_pixel_difference": reference_worker_difference,
        "reference_worker_tolerance": {
            "max_changed_channel_fraction": (
                max_changed_channel_fraction
            ),
            "max_mean_absolute_error_8bit": max_mean_absolute_error,
            "max_absolute_error_8bit": max_channel_error,
        },
        "reference_worker_within_tolerance": reference_within_tolerance,
        "render_repeatability_note": (
            "Independent EEVEE GPU renders are not byte-deterministic. "
            "The transport comparison is exact; the independent render "
            "comparison is bounded by strict 8-bit pixel thresholds."
        ),
        "shard_sha256": shard_digest.hexdigest(),
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
    parser.add_argument(
        "--max-changed-channel-fraction",
        type=float,
        default=DEFAULT_MAX_CHANGED_CHANNEL_FRACTION,
    )
    parser.add_argument(
        "--max-mean-absolute-error",
        type=float,
        default=DEFAULT_MAX_MEAN_ABSOLUTE_ERROR,
    )
    parser.add_argument(
        "--max-channel-error",
        type=int,
        default=DEFAULT_MAX_CHANNEL_ERROR,
    )
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
        max_changed_channel_fraction=args.max_changed_channel_fraction,
        max_mean_absolute_error=args.max_mean_absolute_error,
        max_channel_error=args.max_channel_error,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
