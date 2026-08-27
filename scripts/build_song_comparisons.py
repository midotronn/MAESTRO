#!/usr/bin/env python3
"""Build synchronized front-facing LODGE/EDGE/MAESTRO full-song composites on the Pod."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentlodge.dance.transition import (  # noqa: E402
    retime,
    root_facing_yaw,
    stabilize_root_facing,
)
from agentlodge.evaluation.adapters import convert_agentlodge_motion  # noqa: E402
from server.filament_render import render_full_motion  # noqa: E402


METHODS = ("lodge", "edge", "maestro")
LABELS = {"lodge": "LODGE", "edge": "EDGE", "maestro": "MAESTRO"}
SAFE_VALUE = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream["avg_frame_rate"])),
        "frames": int(stream["nb_read_frames"]),
        "duration_seconds": float(payload["format"]["duration"]),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_video(
    path: Path,
    *,
    frames: int,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, object]:
    probe = _probe(path)
    if probe["frames"] != frames or abs(float(probe["fps"]) - 30.0) > 0.01:
        raise RuntimeError(f"invalid frame timing for {path}: {probe}")
    if width is not None and probe["width"] != width:
        raise RuntimeError(f"invalid width for {path}: {probe}")
    if height is not None and probe["height"] != height:
        raise RuntimeError(f"invalid height for {path}: {probe}")
    return probe


def _motion_sources(workspace: Path, sid: str) -> dict[str, Path]:
    return {
        "lodge": workspace / f"lodge_fd_{sid}_full.npy",
        "edge": workspace / f"edge_fd_{sid}_full.npy",
        "maestro": workspace / f"fd_{sid}_STORY_bestofk.npy",
    }


def _prepare_motions(
    workspace: Path,
    sid: str,
    frames: int,
    output: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    sources = _motion_sources(workspace, sid)
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing comparison motion inputs: {missing}")

    raw = {method: np.load(path) for method, path in sources.items()}
    converted = {
        method: convert_agentlodge_motion(motion, method=method)
        for method, motion in raw.items()
    }
    loaded = {method: result.motion for method, result in converted.items()}
    for method, motion in loaded.items():
        if motion.ndim != 2 or motion.shape[1] != 139:
            raise RuntimeError(f"{method} motion has invalid shape {motion.shape}")

    target_yaw = root_facing_yaw(loaded["maestro"])
    motion_dir = output / "motions"
    motion_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, Path] = {}
    report: dict[str, object] = {
        "target_yaw": target_yaw,
        "source_shapes": {
            method: list(motion.shape) for method, motion in raw.items()
        },
        "source_frames": {
            method: int(motion.shape[0]) for method, motion in loaded.items()
        },
        "conversions": {
            method: result.report for method, result in converted.items()
        },
        "motions": {},
    }
    for method in METHODS:
        motion = stabilize_root_facing(
            retime(loaded[method], frames),
            target_yaw=target_yaw,
        )
        path = motion_dir / f"{method}.npy"
        np.save(path, motion.astype(np.float32))
        prepared[method] = path
        report["motions"][method] = {
            "frames": int(motion.shape[0]),
            "sha256": _sha256(path),
        }
    return prepared, report


def _render_lanes(
    sid: str,
    motions: dict[str, Path],
    audio: Path,
    frames: int,
    output: Path,
    *,
    resume: bool,
) -> tuple[dict[str, Path], dict[str, object]]:
    videos: dict[str, Path] = {}
    reports: dict[str, object] = {}
    for method in METHODS:
        render_dir = output / "lanes" / method
        video = render_dir / "edited.mp4"
        if not (resume and video.is_file()):
            motion = np.load(motions[method])

            def update(**fields: object) -> None:
                message = fields.get("message")
                if message:
                    print(f"{sid} {method}: {message}", flush=True)

            render_full_motion(
                f"comparison-{sid}-{method}",
                motion,
                render_dir,
                audio_wav=str(audio),
                update=update,
            )
        videos[method] = video
        reports[method] = _validate_video(
            video,
            frames=frames,
            width=1080,
            height=1080,
        )
    return videos, reports


def _compose(
    videos: dict[str, Path],
    audio: Path,
    output: Path,
    frames: int,
    *,
    resume: bool,
) -> dict[str, object]:
    if resume and output.is_file():
        return _validate_video(output, frames=frames, width=1440, height=480)

    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if not font.is_file():
        raise RuntimeError(f"comparison label font is missing: {font}")
    filters = []
    for index, method in enumerate(METHODS):
        filters.append(
            f"[{index}:v]scale=400:400:flags=lanczos,setsar=1,"
            "pad=480:480:40:40:color=0x171b22,"
            f"drawtext=fontfile={font}:text='{LABELS[method]}':"
            "fontcolor=white:fontsize=22:x=(w-text_w)/2:y=9"
            f"[{method}]"
        )
    filters.append("[lodge][edge][maestro]hstack=inputs=3[video]")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(videos["lodge"]),
            "-i",
            str(videos["edge"]),
            "-i",
            str(videos["maestro"]),
            "-i",
            str(audio),
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[video]",
            "-map",
            "3:a:0",
            "-frames:v",
            str(frames),
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return _validate_video(output, frames=frames, width=1440, height=480)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/approved-comparisons"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reports = []
    for song in manifest["songs"]:
        sid = str(song["sid"])
        slug = str(song["slug"])
        if not SAFE_VALUE.fullmatch(sid) or not SAFE_VALUE.fullmatch(slug):
            raise RuntimeError(f"unsafe sid or slug: {sid!r}, {slug!r}")
        frames = int(song["frames"])
        if frames < 300:
            raise RuntimeError(f"invalid frame target for {sid}: {frames}")
        audio = args.workspace / "LODGE" / "data" / "finedance" / "music_wav" / f"{sid}.wav"
        if not audio.is_file():
            raise RuntimeError(f"audio is missing for {sid}: {audio}")

        song_output = args.output / slug
        print(f"{song['name']}: preparing {frames} frames", flush=True)
        motions, motion_report = _prepare_motions(
            args.workspace,
            sid,
            frames,
            song_output,
        )
        videos, lane_report = _render_lanes(
            sid,
            motions,
            audio,
            frames,
            song_output,
            resume=args.resume,
        )
        composite = args.output / "composites" / f"story_{slug}_3way.mp4"
        composite_report = _compose(
            videos,
            audio,
            composite,
            frames,
            resume=args.resume,
        )
        reports.append(
            {
                **song,
                **motion_report,
                "lanes": lane_report,
                "composite": {
                    "path": str(composite),
                    **composite_report,
                },
            }
        )
        print(f"{song['name']}: {composite}", flush=True)

    report_path = args.output / "render-report.json"
    report_path.write_text(
        json.dumps(
            {
                "protocol": manifest.get("protocol"),
                "source_layout": ["LODGE", "EDGE", "MAESTRO"],
                "songs": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
