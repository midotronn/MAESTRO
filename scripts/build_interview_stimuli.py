"""Build the frozen blind-comparison clips used by the MAESTRO expert study.

The source videos are synchronized LODGE/EDGE/MAESTRO composites. This script extracts the
capability-focused windows recorded in ``experiments/user_study/stimuli/selection.json``, removes
the source labels and story timeline, and preserves the canonical three-lane order. The study
player reorders those lanes according to the private participant assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path


def _ffmpeg() -> str:
    configured = os.environ.get("FFMPEG")
    if configured:
        return configured
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("ffmpeg is required; install it or set FFMPEG") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_video(ffmpeg: str, path: Path, expected_frames: int) -> None:
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            os.devnull,
            "-progress",
            "pipe:1",
            "-nostats",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frames = [
        int(line.partition("=")[2])
        for line in result.stdout.splitlines()
        if line.startswith("frame=")
    ]
    if not frames or abs(frames[-1] - expected_frames) > 1:
        raise RuntimeError(
            f"{path.name} decoded to {frames[-1] if frames else 0} frames; "
            f"expected {expected_frames}"
        )


def build(repo: Path, selection_path: Path) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    player = selection_path.parent / "player"
    output_dir = player / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()
    generated = []

    for excerpt in selection["excerpts"]:
        source = repo / excerpt["source_video"]
        excerpt_dir = output_dir / excerpt["id"]
        excerpt_dir.mkdir(parents=True, exist_ok=True)
        duration = float(excerpt["duration_seconds"])
        expected_frames = round(duration * float(selection["output"]["fps"]))
        permutations = {}
        for order in itertools.permutations(range(3)):
            code = "".join(str(lane) for lane in order)
            output = excerpt_dir / f"{code}.mp4"
            filters = "".join(
                f"[0:v]crop=480:400:{lane * 480}:40,setsar=1[lane{lane}];"
                for lane in range(3)
            )
            filters += "".join(f"[lane{lane}]" for lane in order) + "hstack=inputs=3[video]"
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-ss",
                    str(excerpt["start_seconds"]),
                    "-t",
                    str(duration),
                    "-filter_complex",
                    filters,
                    "-map",
                    "[video]",
                    "-map",
                    "0:a:0",
                    "-r",
                    str(selection["output"]["fps"]),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    "-y",
                    str(output),
                ],
                check=True,
            )
            _decode_video(ffmpeg, output, expected_frames)
            permutations[code] = {
                "output_video": str(output.relative_to(selection_path.parent)).replace("\\", "/"),
                "output_sha256": _sha256(output),
                "output_bytes": output.stat().st_size,
            }
        stale = output_dir / f"{excerpt['id']}.mp4"
        if stale.exists():
            stale.unlink()
        generated.append(
            {
                **excerpt,
                "source_sha256": _sha256(source),
                "frames": expected_frames,
                "permutations": permutations,
            }
        )
        total_bytes = sum(item["output_bytes"] for item in permutations.values())
        print(f"{excerpt['id']}: 6 orders ({total_bytes / 1024 / 1024:.1f} MiB)")

    manifest = {
        "protocol": selection["protocol"],
        "selection_type": selection["selection_type"],
        "blind_lane_order": selection["source_layout"]["lanes"],
        "output": selection["output"],
        "excerpts": generated,
    }
    manifest_path = selection_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    selection = args.selection or (
        repo / "experiments" / "user_study" / "stimuli" / "selection.json"
    )
    build(repo, selection.resolve())


if __name__ == "__main__":
    main()
