"""Build deterministic, independently editable Dynamite media sections.

The source media directory is read-only. Each section is assembled and validated in a hidden
sibling staging directory, then published with directory renames. A content-addressed manifest
makes unchanged reruns no-ops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "experiments" / "user_study" / "music" / "dynamite_editor_segments.json"
)
TOOL_ID = "dynamite-editor-segments-v1"
MANIFEST_NAME = "segment_manifest.json"
_SID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class BuildError(RuntimeError):
    """Raised when a source or generated media contract is violated."""


@dataclass(frozen=True)
class WaveInfo:
    channels: int
    sample_width: int
    sample_rate: int
    samples: int
    compression: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _array_content_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({"dtype": value.dtype.str, "shape": value.shape}))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _find_repository(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "scripts").is_dir() and (candidate / "server").is_dir():
            return candidate
    raise BuildError(f"could not locate repository root above {path}")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise BuildError("segment config schema_version must be 1")
    source = config.get("source")
    if not isinstance(source, dict):
        raise BuildError("segment config is missing source metadata")
    sid = str(source.get("sid", ""))
    if not _SID_RE.fullmatch(sid):
        raise BuildError(f"invalid source sid {sid!r}")
    fps = int(source.get("fps", 0))
    frames = int(source.get("frames", 0))
    sample_rate = int(source.get("audio_sample_rate", 0))
    if fps <= 0 or frames <= 0 or sample_rate <= 0 or sample_rate % fps:
        raise BuildError("source frames, fps, and frame-aligned audio rate are required")

    bounds = config.get("duration_bounds_seconds", {})
    minimum = float(bounds.get("minimum", 0))
    maximum = float(bounds.get("maximum", 0))
    if minimum <= 0 or maximum < minimum:
        raise BuildError("invalid segment duration bounds")

    segments = config.get("segments")
    if not isinstance(segments, list) or not segments:
        raise BuildError("segment config must contain at least one segment")
    expected_start = 0
    seen: set[str] = set()
    for expected_order, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise BuildError("each segment must be an object")
        segment_sid = str(segment.get("sid", ""))
        if not _SID_RE.fullmatch(segment_sid) or segment_sid in seen:
            raise BuildError(f"invalid or duplicate segment sid {segment_sid!r}")
        seen.add(segment_sid)
        start = int(segment.get("start_frame", -1))
        end = int(segment.get("end_frame", -1))
        order = int(segment.get("order", -1))
        if start != expected_start or end <= start:
            raise BuildError("segments must be contiguous, ordered, and non-empty")
        if order != expected_order:
            raise BuildError("segment orders must start at 1 and remain contiguous")
        duration = (end - start) / fps
        if duration < minimum - 1e-9 or duration > maximum + 1e-9:
            raise BuildError(
                f"{segment_sid} duration {duration:.3f}s is outside "
                f"{minimum:.3f}-{maximum:.3f}s"
            )
        expected_start = end
    if expected_start != frames:
        raise BuildError(
            f"segments cover {expected_start} frames, expected source length {frames}"
        )

    catalog = config.get("catalog")
    if not isinstance(catalog, list) or len(catalog) < len(segments):
        raise BuildError("catalog must include every segment")
    ordered_catalog = sorted(catalog, key=lambda entry: int(entry["order"]))
    if [int(entry["order"]) for entry in ordered_catalog] != list(
        range(1, len(ordered_catalog) + 1)
    ):
        raise BuildError("catalog orders must start at 1 and remain contiguous")
    if len({str(entry["sid"]) for entry in ordered_catalog}) != len(ordered_catalog):
        raise BuildError("catalog sids must be unique")
    for segment, entry in zip(segments, ordered_catalog):
        if (
            entry.get("kind") != "segment"
            or entry.get("sid") != segment.get("sid")
            or entry.get("name") != segment.get("name")
            or int(entry["order"]) != int(segment["order"])
        ):
            raise BuildError("the catalog must begin with the configured segments in order")
    policy = config.get("interview_policy", {})
    if policy.get("full_song_visible") is False and sid in {
        str(entry["sid"]) for entry in catalog
    }:
        raise BuildError("hidden full-song sid must not appear in the interview catalog")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"could not load segment config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise BuildError("segment config root must be an object")
    try:
        _validate_config(config)
    except BuildError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise BuildError(f"invalid segment config: {exc}") from exc
    evidence_entries = config.get("evidence", [])
    repository = _find_repository(path) if evidence_entries else None
    for evidence in evidence_entries:
        assert repository is not None
        evidence_path = (repository / str(evidence["path"])).resolve()
        if repository.resolve() not in evidence_path.parents:
            raise BuildError(f"evidence path escapes the repository: {evidence_path}")
        if not evidence_path.is_file():
            raise BuildError(f"missing pinned evidence {evidence_path}")
        if evidence.get("hash_mode") == "canonical_json":
            try:
                actual = _canonical_sha256(
                    json.loads(evidence_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise BuildError(f"could not parse pinned evidence {evidence_path}") from exc
        else:
            actual = _sha256(evidence_path)
        if actual != evidence.get("sha256"):
            raise BuildError(
                f"pinned evidence changed: {evidence_path} "
                f"(expected {evidence.get('sha256')}, got {actual})"
            )
    return config


def _wave_info(path: Path) -> WaveInfo:
    try:
        with wave.open(str(path), "rb") as source:
            return WaveInfo(
                channels=source.getnchannels(),
                sample_width=source.getsampwidth(),
                sample_rate=source.getframerate(),
                samples=source.getnframes(),
                compression=source.getcomptype(),
            )
    except (OSError, wave.Error) as exc:
        raise BuildError(f"could not read WAV {path}: {exc}") from exc


def _wav_pcm_sha256(path: Path, start_sample: int, samples: int) -> str:
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as source:
        if start_sample < 0 or start_sample + samples > source.getnframes():
            raise BuildError(f"audio slice is outside {path}")
        source.setpos(start_sample)
        remaining = samples
        while remaining:
            count = min(remaining, 65536)
            payload = source.readframes(count)
            if len(payload) != count * source.getnchannels() * source.getsampwidth():
                raise BuildError(f"short PCM read from {path}")
            digest.update(payload)
            remaining -= count
    return digest.hexdigest()


def _slice_wav(source_path: Path, output_path: Path, start_sample: int, samples: int) -> None:
    with wave.open(str(source_path), "rb") as source:
        if start_sample < 0 or start_sample + samples > source.getnframes():
            raise BuildError(f"audio slice is outside {source_path}")
        params = source.getparams()
        source.setpos(start_sample)
        with wave.open(str(output_path), "wb") as output:
            output.setparams(params)
            remaining = samples
            while remaining:
                count = min(remaining, 65536)
                payload = source.readframes(count)
                if len(payload) != count * params.nchannels * params.sampwidth:
                    raise BuildError(f"short PCM read from {source_path}")
                output.writeframesraw(payload)
                remaining -= count


def _parse_rate(value: str | None) -> Fraction:
    try:
        rate = Fraction(value or "0")
    except (ValueError, ZeroDivisionError) as exc:
        raise BuildError(f"invalid video frame rate {value!r}") from exc
    if rate <= 0:
        raise BuildError(f"invalid video frame rate {value!r}")
    return rate


def _optional_rate(value: str | None) -> Fraction | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    return _parse_rate(value)


def _validate_video_timing(
    probe: dict[str, Any],
    *,
    expected_frames: int,
    expected_fps: int,
    label: str,
) -> None:
    rate = Fraction(probe["fps_numerator"], probe["fps_denominator"])
    if probe["video_frames"] != expected_frames or rate != expected_fps:
        raise BuildError(
            f"{label} has {probe['video_frames']} frames at {float(rate):.6f} FPS; "
            f"expected {expected_frames} at {expected_fps}"
        )
    duration = probe["video_duration_seconds"]
    if duration is None:
        duration = probe["format_duration_seconds"]
    expected_duration = expected_frames / expected_fps
    if duration is None or abs(duration - expected_duration) > 1 / expected_fps:
        raise BuildError(
            f"{label} duration is {duration}, expected {expected_duration:.9f}s "
            f"within one {expected_fps} FPS frame"
        )


def _probe_preview(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=index,codec_type,avg_frame_rate,r_frame_rate,nb_frames,"
            "nb_read_frames,sample_rate,channels,duration,duration_ts,time_base:"
            "format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise BuildError(f"could not run ffprobe {ffprobe!r}: {exc}") from exc
    if result.returncode:
        raise BuildError(f"ffprobe failed for {path}: {result.stderr[-500:]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise BuildError(f"ffprobe returned invalid JSON for {path}") from exc
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise BuildError(f"{path} has no video stream")
    raw_frames = video.get("nb_read_frames") or video.get("nb_frames")
    try:
        frames = int(raw_frames)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"ffprobe could not count frames in {path}") from exc
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)

    def stream_duration(stream: dict[str, Any] | None) -> float | None:
        if stream is None:
            return None
        if stream.get("duration") not in (None, "N/A"):
            return float(stream["duration"])
        if (
            stream.get("duration_ts") not in (None, "N/A")
            and stream.get("time_base") not in (None, "N/A")
        ):
            return float(int(stream["duration_ts"]) * Fraction(stream["time_base"]))
        return None

    video_duration = stream_duration(video)
    nominal_rate = _optional_rate(video.get("r_frame_rate"))
    average_rate = _optional_rate(video.get("avg_frame_rate"))
    rate = nominal_rate or average_rate
    rate_source = "r_frame_rate" if nominal_rate is not None else "avg_frame_rate"
    if (
        nominal_rate is not None
        and average_rate is not None
        and video_duration is not None
        and abs(float(nominal_rate - average_rate) * video_duration) > 1
    ):
        rate = average_rate
        rate_source = "avg_frame_rate"
    if rate is None:
        raise BuildError(f"ffprobe returned no usable video frame rate for {path}")

    return {
        "video_frames": frames,
        "fps_numerator": rate.numerator,
        "fps_denominator": rate.denominator,
        "fps_source": rate_source,
        "nominal_fps_numerator": nominal_rate.numerator if nominal_rate else None,
        "nominal_fps_denominator": nominal_rate.denominator if nominal_rate else None,
        "average_fps_numerator": average_rate.numerator if average_rate else None,
        "average_fps_denominator": average_rate.denominator if average_rate else None,
        "has_audio": audio is not None,
        "audio_sample_rate": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        "audio_channels": int(audio["channels"]) if audio and audio.get("channels") else None,
        "audio_duration_seconds": stream_duration(audio),
        "video_duration_seconds": video_duration,
        "format_duration_seconds": (
            float(payload.get("format", {}).get("duration"))
            if payload.get("format", {}).get("duration") not in (None, "N/A")
            else None
        ),
    }


def _count_decoded_audio_samples(
    path: Path,
    ffmpeg: str,
    *,
    sample_rate: int,
    channels: int,
) -> int:
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise BuildError(f"could not run ffmpeg {ffmpeg!r}: {exc}") from exc
    if result.returncode:
        raise BuildError(
            f"could not decode preview audio from {path}: "
            f"{result.stderr.decode('utf-8', 'replace')[-500:]}"
        )
    bytes_per_sample = channels * 2
    if len(result.stdout) % bytes_per_sample:
        raise BuildError(f"decoded preview audio from {path} is not whole PCM samples")
    return len(result.stdout) // bytes_per_sample


def _ffmpeg_preview_command(
    ffmpeg: str,
    source_video: Path,
    audio_path: Path,
    output_video: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: int,
    audio_samples: int,
    sample_rate: int,
    channels: int,
) -> list[str]:
    frames = end_frame - start_frame
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_video),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=N/({fps}*TB)",
        "-af",
        f"atrim=start_sample=0:end_sample={audio_samples},asetpts=PTS-STARTPTS",
        "-frames:v",
        str(frames),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        str(output_video),
    ]


def _slice_preview(
    source_video: Path,
    output_video: Path,
    audio_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: int,
    audio_samples: int,
    sample_rate: int,
    channels: int,
    ffmpeg: str,
) -> None:
    command = _ffmpeg_preview_command(
        ffmpeg,
        source_video,
        audio_path,
        output_video,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        audio_samples=audio_samples,
        sample_rate=sample_rate,
        channels=channels,
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise BuildError(f"could not run ffmpeg {ffmpeg!r}: {exc}") from exc
    if result.returncode:
        raise BuildError(f"ffmpeg failed for {output_video}: {result.stderr[-1000:]}")


def _audio_candidates(source_dir: Path, sid: str) -> list[Path]:
    candidates = [source_dir / f"{sid}.wav", source_dir / "audio.wav"]
    workspace = os.environ.get("WORKSPACE")
    if workspace:
        candidates.append(
            Path(workspace) / "LODGE" / "data" / "finedance" / "music_wav" / f"{sid}.wav"
        )
    for parent in source_dir.parents:
        if parent.name == "AgentLODGE":
            candidates.append(
                parent.parent
                / "LODGE"
                / "data"
                / "finedance"
                / "music_wav"
                / f"{sid}.wav"
            )
            break
    return candidates


def _resolve_audio(source_dir: Path, sid: str, explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else _audio_candidates(source_dir, sid)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates if path is not None)
    raise BuildError(f"could not locate canonical WAV; checked: {rendered}")


def _rename_bank_path(relative: Path, source_sid: str, segment_sid: str) -> Path:
    prefix = f"bank_{source_sid}_"
    name = relative.name
    if name.startswith(prefix):
        name = f"bank_{segment_sid}_{name[len(prefix):]}"
    return relative.with_name(name)


def _source_inventory(
    source_dir: Path,
    audio_path: Path,
    config: dict[str, Any],
    *,
    ffprobe: str,
) -> dict[str, Any]:
    source = config["source"]
    expected_frames = int(source["frames"])
    fps = int(source["fps"])
    required = {
        "base_motion.npy": source_dir / "base_motion.npy",
        "beats.npy": source_dir / "beats.npy",
        "preview.mp4": source_dir / "preview.mp4",
        "meta.json": source_dir / "meta.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise BuildError(f"source media is missing: {', '.join(missing)}")

    try:
        base = np.load(required["base_motion.npy"], mmap_mode="r", allow_pickle=False)
        beats = np.load(required["beats.npy"], allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise BuildError(f"could not load source motion/beat arrays: {exc}") from exc
    if base.ndim < 2 or base.shape[0] != expected_frames:
        raise BuildError(
            f"base_motion.npy shape {base.shape} does not match {expected_frames} frames"
        )
    if beats.ndim != 1 or not np.issubdtype(beats.dtype, np.number):
        raise BuildError("beats.npy must be a numeric one-dimensional array")
    if not np.all(np.isfinite(beats)) or np.any(beats < 0) or np.any(beats > expected_frames):
        raise BuildError("beats.npy contains invalid or out-of-range frame positions")

    strengths_path = source_dir / "beat_strengths.npy"
    strengths = None
    if strengths_path.is_file():
        try:
            strengths = np.load(strengths_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise BuildError(f"could not load {strengths_path}: {exc}") from exc
        if strengths.ndim != 1 or strengths.shape != beats.shape:
            raise BuildError("beat_strengths.npy must have one value per beat")
        if not np.all(np.isfinite(strengths)):
            raise BuildError("beat_strengths.npy contains non-finite values")

    try:
        metadata = json.loads(required["meta.json"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"could not load source metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise BuildError("source meta.json must contain an object")
    if source.get("require_front_facing") and metadata.get("front_facing") is not True:
        raise BuildError("approved Dynamite source must be marked front_facing=true")

    audio = _wave_info(audio_path)
    expected_audio = {
        "sample_rate": int(source["audio_sample_rate"]),
        "channels": int(source["audio_channels"]),
        "sample_width": int(source["audio_sample_width_bytes"]),
    }
    if (
        audio.compression != "NONE"
        or audio.sample_rate != expected_audio["sample_rate"]
        or audio.channels != expected_audio["channels"]
        or audio.sample_width != expected_audio["sample_width"]
    ):
        raise BuildError(
            f"canonical WAV contract mismatch: got {audio}, expected {expected_audio}"
        )
    required_samples = expected_frames * audio.sample_rate // fps
    if audio.samples < required_samples:
        raise BuildError(
            f"canonical WAV has {audio.samples} samples, needs at least {required_samples}"
        )
    expected_audio_hash = source.get("audio_sha256")
    actual_audio_hash = _sha256(audio_path)
    if expected_audio_hash and actual_audio_hash != expected_audio_hash:
        raise BuildError(
            f"canonical WAV hash mismatch (expected {expected_audio_hash}, "
            f"got {actual_audio_hash})"
        )

    preview_probe = _probe_preview(required["preview.mp4"], ffprobe)
    _validate_video_timing(
        preview_probe,
        expected_frames=expected_frames,
        expected_fps=fps,
        label="source preview",
    )

    compatible_banks: list[dict[str, Any]] = []
    skipped_banks: list[dict[str, Any]] = []
    bank_dir = source_dir / "bank"
    standard_bank = re.compile(
        rf"^bank_{re.escape(str(source['sid']))}_(?:lodge|edge)_seed\d+\.npy$"
    )
    if bank_dir.is_dir():
        output_names: set[str] = set()
        for path in sorted(bank_dir.rglob("*.npy")):
            relative = path.relative_to(bank_dir)
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise BuildError(f"could not load bank array {path}: {exc}") from exc
            compatible = (
                array.ndim == base.ndim
                and array.shape[0] == expected_frames
                and array.shape[1:] == base.shape[1:]
                and np.issubdtype(array.dtype, np.number)
            )
            if not compatible:
                if standard_bank.fullmatch(path.name):
                    raise BuildError(
                        f"editor bank {path.name} shape {array.shape} is incompatible "
                        f"with base motion {base.shape}"
                    )
                skipped_banks.append(
                    {
                        "path": relative.as_posix(),
                        "shape": list(array.shape),
                        "dtype": array.dtype.str,
                    }
                )
                continue
            output_relative = _rename_bank_path(
                relative,
                str(source["sid"]),
                "{segment_sid}",
            )
            output_key = output_relative.as_posix()
            if output_key in output_names:
                raise BuildError(f"bank output collision for {output_key}")
            output_names.add(output_key)
            compatible_banks.append(
                {
                    "path": path,
                    "relative": relative,
                    "shape": tuple(array.shape),
                    "dtype": array.dtype.str,
                }
            )

    fingerprint_paths: dict[str, Path] = {
        **{name: path for name, path in required.items()},
        "audio.wav": audio_path,
    }
    if strengths is not None:
        fingerprint_paths["beat_strengths.npy"] = strengths_path
    for bank in compatible_banks:
        fingerprint_paths[f"bank/{bank['relative'].as_posix()}"] = bank["path"]
    hashes = {name: _sha256(path) for name, path in sorted(fingerprint_paths.items())}
    return {
        "base": base,
        "beats": beats,
        "strengths": strengths,
        "metadata": metadata,
        "audio": audio,
        "audio_path": audio_path,
        "preview_path": required["preview.mp4"],
        "preview_probe": preview_probe,
        "compatible_banks": compatible_banks,
        "skipped_banks": skipped_banks,
        "fingerprint_paths": fingerprint_paths,
        "hashes": hashes,
    }


def _build_key(
    config_hash: str,
    source_hashes: dict[str, str],
    segment: dict[str, Any],
) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "tool": TOOL_ID,
                "config_sha256": config_hash,
                "source_hashes": source_hashes,
                "segment": segment,
            }
        )
    ).hexdigest()


def _output_is_current(directory: Path, build_key: str) -> bool:
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("tool") != TOOL_ID or manifest.get("build_key") != build_key:
        return False
    files = manifest.get("files")
    if not isinstance(files, dict):
        return False
    for relative, expected in files.items():
        path = directory / Path(relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(expected.get("bytes", -1))
            or _sha256(path) != expected.get("sha256")
        ):
            return False
    return True


def _file_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        relative = path.relative_to(directory).as_posix()
        detail: dict[str, Any] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            detail.update(
                {
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "content_sha256": _array_content_sha256(array),
                }
            )
        elif path.suffix == ".wav":
            info = _wave_info(path)
            detail.update(
                {
                    "samples": info.samples,
                    "sample_rate": info.sample_rate,
                    "channels": info.channels,
                    "sample_width": info.sample_width,
                    "pcm_sha256": _wav_pcm_sha256(path, 0, info.samples),
                }
            )
        files[relative] = detail
    return files


def _build_segment(
    staging_dir: Path,
    segment: dict[str, Any],
    config: dict[str, Any],
    inventory: dict[str, Any],
    *,
    config_hash: str,
    build_key: str,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, Any]:
    source = config["source"]
    fps = int(source["fps"])
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    frames = end - start
    audio: WaveInfo = inventory["audio"]
    samples_per_frame = audio.sample_rate // fps
    start_sample = start * samples_per_frame
    audio_samples = frames * samples_per_frame

    staging_dir.mkdir(parents=True)
    base_slice = np.ascontiguousarray(inventory["base"][start:end])
    np.save(staging_dir / "base_motion.npy", base_slice)

    beats = inventory["beats"]
    beat_mask = (beats >= start) & (beats < end)
    segment_beats = np.asarray(beats[beat_mask] - start, dtype=beats.dtype)
    np.save(staging_dir / "beats.npy", segment_beats)
    if inventory["strengths"] is not None:
        np.save(
            staging_dir / "beat_strengths.npy",
            np.asarray(inventory["strengths"][beat_mask]),
        )

    segment_audio = staging_dir / f"{segment['sid']}.wav"
    _slice_wav(inventory["audio_path"], segment_audio, start_sample, audio_samples)
    output_audio = _wave_info(segment_audio)
    if output_audio.samples != audio_samples:
        raise BuildError(
            f"{segment['sid']} WAV has {output_audio.samples} samples, expected {audio_samples}"
        )
    if _wav_pcm_sha256(segment_audio, 0, audio_samples) != _wav_pcm_sha256(
        inventory["audio_path"],
        start_sample,
        audio_samples,
    ):
        raise BuildError(f"{segment['sid']} WAV samples do not match the source slice")

    bank_outputs: list[str] = []
    if inventory["compatible_banks"]:
        bank_output_dir = staging_dir / "bank"
        for bank in inventory["compatible_banks"]:
            relative = _rename_bank_path(
                bank["relative"],
                str(source["sid"]),
                str(segment["sid"]),
            )
            output = bank_output_dir / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            array = np.load(bank["path"], mmap_mode="r", allow_pickle=False)
            bank_slice = np.ascontiguousarray(array[start:end])
            np.save(output, bank_slice)
            if _array_content_sha256(np.load(output, mmap_mode="r", allow_pickle=False)) != (
                _array_content_sha256(bank_slice)
            ):
                raise BuildError(f"bank slice validation failed for {output}")
            bank_outputs.append(output.relative_to(staging_dir).as_posix())

    metadata = {
        "artist": source["artist"],
        "front_facing": True,
        "interview": True,
        "name": segment["name"],
        "order": int(segment["order"]),
        "segment": {
            "boundary_basis": segment["boundary_basis"],
            "duration_seconds": frames / fps,
            "end_frame": end,
            "end_seconds": end / fps,
            "rationale": segment["rationale"],
            "roles": segment["roles"],
            "source_sid": source["sid"],
            "start_frame": start,
            "start_seconds": start / fps,
        },
    }
    if "front_facing_yaw" in inventory["metadata"]:
        metadata["front_facing_yaw"] = inventory["metadata"]["front_facing_yaw"]
    _write_json(staging_dir / "meta.json", metadata)

    preview = staging_dir / "preview.mp4"
    _slice_preview(
        inventory["preview_path"],
        preview,
        segment_audio,
        start_frame=start,
        end_frame=end,
        fps=fps,
        audio_samples=audio_samples,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        ffmpeg=ffmpeg,
    )
    preview_probe = _probe_preview(preview, ffprobe)
    _validate_video_timing(
        preview_probe,
        expected_frames=frames,
        expected_fps=fps,
        label=f"{segment['sid']} preview",
    )
    if (
        not preview_probe["has_audio"]
        or preview_probe["audio_sample_rate"] != audio.sample_rate
        or preview_probe["audio_channels"] != audio.channels
    ):
        raise BuildError(
            f"{segment['sid']} preview contract mismatch: {preview_probe}"
        )
    expected_duration = frames / fps
    audio_duration = preview_probe["audio_duration_seconds"]
    if (
        audio_duration is None
        or abs(audio_duration - expected_duration) > 1 / audio.sample_rate
    ):
        raise BuildError(
            f"{segment['sid']} preview audio timeline is {audio_duration}, "
            f"expected {expected_duration:.9f}s"
        )
    decoded_audio_samples = _count_decoded_audio_samples(
        preview,
        ffmpeg,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
    )
    decoded_padding = decoded_audio_samples - audio_samples
    if decoded_padding < 0 or decoded_padding >= 1024:
        raise BuildError(
            f"{segment['sid']} preview decodes to {decoded_audio_samples} audio samples, "
            f"expected {audio_samples} plus at most 1023 AAC padding samples"
        )

    saved_base = np.load(staging_dir / "base_motion.npy", mmap_mode="r", allow_pickle=False)
    if _array_content_sha256(saved_base) != _array_content_sha256(base_slice):
        raise BuildError(f"{segment['sid']} base motion does not match its source slice")
    saved_beats = np.load(staging_dir / "beats.npy", allow_pickle=False)
    if _array_content_sha256(saved_beats) != _array_content_sha256(segment_beats):
        raise BuildError(f"{segment['sid']} beat rebasing validation failed")

    files = _file_manifest(staging_dir)
    manifest = {
        "schema_version": 1,
        "tool": TOOL_ID,
        "build_key": build_key,
        "config_sha256": config_hash,
        "source": {
            "sid": source["sid"],
            "hashes": inventory["hashes"],
        },
        "segment": {
            "sid": segment["sid"],
            "name": segment["name"],
            "order": int(segment["order"]),
            "start_frame": start,
            "end_frame": end,
            "frames": frames,
            "start_seconds": start / fps,
            "end_seconds": end / fps,
            "duration_seconds": frames / fps,
            "audio_start_sample": start_sample,
            "audio_samples": audio_samples,
            "rebased_beats": int(segment_beats.size),
        },
        "preview": {
            **preview_probe,
            "decoded_audio_samples": decoded_audio_samples,
            "decoded_audio_padding_samples": decoded_padding,
        },
        "bank_outputs": bank_outputs,
        "skipped_incompatible_bank_arrays": inventory["skipped_banks"],
        "files": files,
    }
    _write_json(staging_dir / MANIFEST_NAME, manifest)
    return manifest


def _publish_directories(
    staged: dict[str, Path],
    output_root: Path,
    transaction_root: Path,
) -> None:
    backups_dir = transaction_root / "backups"
    rollback_dir = transaction_root / "rollback"
    installed: list[Path] = []
    backups: dict[Path, Path] = {}
    try:
        for sid, staged_dir in staged.items():
            destination = output_root / sid
            if destination.exists():
                backups_dir.mkdir(exist_ok=True)
                backup = backups_dir / sid
                os.replace(destination, backup)
                backups[destination] = backup
            try:
                os.replace(staged_dir, destination)
            except Exception:
                backup = backups.pop(destination, None)
                if backup is not None and backup.exists():
                    os.replace(backup, destination)
                raise
            installed.append(destination)
    except Exception:
        rollback_dir.mkdir(exist_ok=True)
        for destination in reversed(installed):
            if destination.exists():
                os.replace(destination, rollback_dir / destination.name)
            backup = backups.get(destination)
            if backup is not None and backup.exists():
                os.replace(backup, destination)
        raise
    for backup in backups.values():
        if backup.exists():
            shutil.rmtree(backup)


def build_segments(
    source_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    audio_path: str | Path | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    force: bool = False,
) -> dict[str, Any]:
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise BuildError(f"source media directory does not exist: {source_dir}")
    output_root = (
        Path(output_root).resolve() if output_root is not None else source_dir.parent.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root == source_dir or source_dir in output_root.parents:
        raise BuildError("output root must not be the source directory or one of its children")

    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    source = config["source"]
    if source_dir.name != source["sid"]:
        raise BuildError(
            f"source directory name {source_dir.name!r} does not match {source['sid']!r}"
        )
    destinations = [output_root / str(segment["sid"]) for segment in config["segments"]]
    if any(destination.resolve() == source_dir for destination in destinations):
        raise BuildError("a segment destination would replace the full-song source")

    resolved_audio = _resolve_audio(
        source_dir,
        str(source["sid"]),
        Path(audio_path).resolve() if audio_path is not None else None,
    )
    inventory = _source_inventory(
        source_dir,
        resolved_audio,
        config,
        ffprobe=ffprobe,
    )
    config_hash = _canonical_sha256(config)
    keys = {
        str(segment["sid"]): _build_key(config_hash, inventory["hashes"], segment)
        for segment in config["segments"]
    }
    reused = [
        str(segment["sid"])
        for segment in config["segments"]
        if not force
        and _output_is_current(output_root / str(segment["sid"]), keys[str(segment["sid"])])
    ]
    reused_set = set(reused)
    pending = [
        segment for segment in config["segments"] if str(segment["sid"]) not in reused_set
    ]
    if not pending:
        return {
            "source_sid": source["sid"],
            "source_unchanged": True,
            "built": [],
            "reused": reused,
            "output_root": str(output_root),
        }

    transaction_root = output_root / f".{source['sid']}.segments-{uuid.uuid4().hex}.staging"
    transaction_root.mkdir()
    staged: dict[str, Path] = {}
    manifests: dict[str, dict[str, Any]] = {}
    try:
        for segment in pending:
            sid = str(segment["sid"])
            staged[sid] = transaction_root / sid
            manifests[sid] = _build_segment(
                staged[sid],
                segment,
                config,
                inventory,
                config_hash=config_hash,
                build_key=keys[sid],
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )
        after_hashes = {
            name: _sha256(path)
            for name, path in sorted(inventory["fingerprint_paths"].items())
        }
        if after_hashes != inventory["hashes"]:
            raise BuildError("source assets changed during the build; no outputs were published")
        _publish_directories(staged, output_root, transaction_root)
    finally:
        if transaction_root.exists():
            shutil.rmtree(transaction_root)

    return {
        "source_sid": source["sid"],
        "source_unchanged": True,
        "built": [
            {
                "sid": sid,
                "start_frame": manifests[sid]["segment"]["start_frame"],
                "end_frame": manifests[sid]["segment"]["end_frame"],
                "frames": manifests[sid]["segment"]["frames"],
            }
            for sid in staged
        ],
        "reused": reused,
        "output_root": str(output_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="full Dynamite editor media directory")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="destination media root (default: source directory's parent)",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audio", type=Path, help="override the canonical Dynamite WAV")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    parser.add_argument("--ffprobe", default=os.environ.get("FFPROBE", "ffprobe"))
    parser.add_argument("--force", action="store_true", help="rebuild validated current outputs")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = build_segments(
            args.source_dir,
            output_root=args.output_root,
            config_path=args.config,
            audio_path=args.audio,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            force=args.force,
        )
    except BuildError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
