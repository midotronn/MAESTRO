"""Versioned render provenance and lossless shard validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

RENDER_CONTRACT_VERSION = "render.frames-ffv1-v3"
RGB_DIGEST_VERSION = "rgb24-global-frame-v1"
WORKER_SHARD_VALIDATION_VERSION = "source-rgb-digest+ffprobe-v1"


def canonical_render_identity(
    provenance: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the render-affecting provenance and quality in canonical form."""
    if not isinstance(provenance, Mapping):
        raise ValueError("render provenance must be an object")
    required_quality = {
        "width": int(quality["width"]),
        "height": int(quality["height"]),
        "samples": int(quality["samples"]),
        "engine": str(quality["engine"]).lower(),
        "denoise": int(quality["denoise"]),
        "frame_format": str(quality["frame_format"]).lower().lstrip("."),
        "fps": int(quality["fps"]),
    }
    if (
        min(
            required_quality["width"],
            required_quality["height"],
            required_quality["samples"],
            required_quality["fps"],
        )
        < 1
    ):
        raise ValueError("render quality dimensions, samples, and fps must be positive")
    return {
        "schema_version": 1,
        "provenance": dict(provenance),
        "quality": required_quality,
    }


def render_identity_digest(
    provenance: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> str:
    canonical = json.dumps(
        canonical_render_identity(provenance, quality),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_executable() -> str:
    configured = os.environ.get("AGENTLODGE_FFMPEG", "").strip()
    if configured:
        return configured
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise FileNotFoundError("ffmpeg is required for render validation") from exc


def _ffprobe_executable() -> str:
    configured = os.environ.get("AGENTLODGE_FFPROBE", "").strip()
    if configured:
        return configured
    executable = shutil.which("ffprobe")
    if executable:
        return executable
    ffmpeg = Path(_ffmpeg_executable())
    sibling = ffmpeg.with_name(
        "ffprobe.exe" if ffmpeg.name.lower().endswith(".exe") else "ffprobe"
    )
    if sibling.is_file():
        return str(sibling)
    raise FileNotFoundError(
        "ffprobe is required for fail-closed FFV1 shard validation"
    )


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decoded_rgb_sha256(
    command: list[str],
    *,
    frame_start: int,
    frame_count: int,
    width: int,
    height: int,
) -> str:
    if frame_count < 1 or width < 1 or height < 1:
        raise ValueError("decoded RGB validation requires positive dimensions and frames")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    digest.update(RGB_DIGEST_VERSION.encode("ascii") + b"\0")
    digest.update(struct.pack(">QQII", frame_start, frame_count, width, height))
    frame_bytes = width * height * 3
    try:
        for offset in range(frame_count):
            pixels = _read_exact(process.stdout, frame_bytes)
            if len(pixels) != frame_bytes:
                process.kill()
                _stdout, stderr = process.communicate()
                detail = stderr.decode("utf-8", errors="replace")[-800:]
                raise RuntimeError(
                    "decoded shard ended before its declared frame range"
                    + (f": {detail}" if detail else "")
                )
            digest.update(struct.pack(">Q", frame_start + offset))
            digest.update(pixels)
        if process.stdout.read(1):
            process.kill()
            process.communicate()
            raise RuntimeError("decoded shard contains more frames than declared")
        _stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"decoded RGB validation failed: {detail}")
    return digest.hexdigest()


def source_sequence_rgb_sha256(
    frames_dir: Path,
    *,
    frame_start: int,
    frame_end: int,
    frame_format: str,
    width: int,
    height: int,
    fps: int,
) -> str:
    frame_count = frame_end - frame_start
    pattern = Path(frames_dir) / f"frame_%04d.{frame_format}"
    command = [
        _ffmpeg_executable(),
        "-v",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        str(frame_start),
        "-i",
        str(pattern),
        "-frames:v",
        str(frame_count),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    return _decoded_rgb_sha256(
        command,
        frame_start=frame_start,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def _parse_fraction(value: Any) -> Fraction:
    text = str(value or "")
    try:
        parsed = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"invalid shard frame rate: {text!r}") from exc
    if parsed <= 0:
        raise RuntimeError(f"invalid shard frame rate: {text!r}")
    return parsed


def probe_ffv1_shard(
    path: Path,
    *,
    frame_start: int,
    frame_end: int,
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    """Validate FFV1 structure and timing without decoding full RGB frames."""
    shard = Path(path).resolve()
    frame_count = frame_end - frame_start
    probe = subprocess.run(
        [
            _ffprobe_executable(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_packets",
            "-show_entries",
            (
                "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
                "nb_read_packets,pix_fmt:"
                "packet=pts_time,dts_time,duration_time,flags"
            ),
            "-of",
            "json",
            str(shard),
        ],
        capture_output=True,
        text=True,
        timeout=max(120, int(frame_count * 0.2 + 30)),
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"ffprobe rejected render shard: {probe.stderr[-800:]}"
        )
    try:
        metadata = json.loads(probe.stdout)
        streams = metadata["streams"]
        packets = metadata["packets"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffprobe returned invalid render shard metadata") from exc
    if not isinstance(streams, list) or len(streams) != 1:
        raise RuntimeError("render shard must contain exactly one video stream")
    stream = streams[0]
    if str(stream.get("codec_name") or "").lower() != "ffv1":
        raise RuntimeError("render shard codec is not FFV1")
    if int(stream.get("width") or 0) != width or int(stream.get("height") or 0) != height:
        raise RuntimeError("render shard dimensions do not match the task")
    rate_value = stream.get("avg_frame_rate")
    if str(rate_value or "") in {"", "0/0"}:
        rate_value = stream.get("r_frame_rate")
    rate = _parse_fraction(rate_value)
    if rate != Fraction(fps, 1):
        raise RuntimeError(
            f"render shard frame rate {rate} does not match requested {fps}"
        )
    counted = stream.get("nb_read_packets")
    try:
        counted_packets = int(counted)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("render shard packet count is unavailable") from exc
    if (
        counted_packets != frame_count
        or not isinstance(packets, list)
        or len(packets) != frame_count
    ):
        raise RuntimeError(
            f"render shard has {counted_packets} packets; expected {frame_count}"
        )
    timestamps: list[float] = []
    try:
        timestamps = [
            float(packet.get("pts_time") or packet.get("dts_time"))
            for packet in packets
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("render shard packet timestamps are unavailable") from exc
    tolerance = max(0.002, 0.1 / fps)
    for index, timestamp in enumerate(timestamps):
        expected = index / fps
        if abs(timestamp - expected) > tolerance:
            raise RuntimeError(
                "render shard packets are not contiguous and ordered at the requested fps"
            )
    return {
        "codec": "ffv1",
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_count,
        "pixel_format": str(stream.get("pix_fmt") or ""),
    }


def inspect_ffv1_shard(
    path: Path,
    *,
    frame_start: int,
    frame_end: int,
    width: int,
    height: int,
    fps: int,
) -> dict[str, Any]:
    """Probe and independently decode a shard for coordinator validation."""
    validation = probe_ffv1_shard(
        path,
        frame_start=frame_start,
        frame_end=frame_end,
        width=width,
        height=height,
        fps=fps,
    )
    decoded_sha256 = _decoded_rgb_sha256(
        [
            _ffmpeg_executable(),
            "-v",
            "error",
            "-i",
            str(Path(path).resolve()),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        frame_start=frame_start,
        frame_count=frame_end - frame_start,
        width=width,
        height=height,
    )
    return {
        **validation,
        "decoded_rgb_digest_version": RGB_DIGEST_VERSION,
        "decoded_rgb_sha256": decoded_sha256,
    }
