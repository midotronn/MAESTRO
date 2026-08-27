"""Build the frozen blind-comparison clips used by the MAESTRO expert study.

The source videos are synchronized LODGE/EDGE/MAESTRO composites. This script extracts the
capability-focused windows recorded in ``experiments/user_study/stimuli/selection.json``, removes
the source labels and story timeline, and renders the one fixed balanced lane order assigned to
each excerpt in ``player/assignments.json``.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _fixed_orders(
    selection_path: Path,
    selection: dict,
) -> tuple[dict[str, tuple[int, ...]], list[dict[str, object]]]:
    assignments_path = selection_path.parent / "player" / "assignments.json"
    assignments = json.loads(assignments_path.read_text(encoding="utf-8"))
    if assignments.get("protocol") != selection.get("protocol"):
        raise ValueError("selection and assignment protocols do not match")

    sequence = assignments.get("sequence")
    if not isinstance(sequence, list):
        raise ValueError("assignments sequence must be a list")

    orders = {}
    for comparison in sequence:
        if not isinstance(comparison, dict):
            raise ValueError(f"invalid fixed-sequence entry: {comparison}")
        excerpt_id = comparison.get("excerpt")
        raw_lanes = comparison.get("lanes")
        if not isinstance(excerpt_id, str) or not isinstance(raw_lanes, list):
            raise ValueError(f"invalid fixed-sequence entry: {comparison}")
        lanes = tuple(raw_lanes)
        if excerpt_id in orders:
            raise ValueError(f"duplicate assignment for {excerpt_id}")
        if (
            any(isinstance(lane, bool) or not isinstance(lane, int) for lane in lanes)
            or sorted(lanes) != [0, 1, 2]
        ):
            raise ValueError(f"invalid lane order for {excerpt_id}: {lanes}")
        orders[excerpt_id] = lanes

    excerpt_ids = [excerpt["id"] for excerpt in selection["excerpts"]]
    if set(orders) != set(excerpt_ids) or len(sequence) != len(excerpt_ids):
        raise ValueError("fixed sequence must assign every selected excerpt exactly once")
    return orders, sequence


def build(repo: Path, selection_path: Path) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    player = selection_path.parent / "player"
    output_dir = player / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()
    fixed_orders, sequence = _fixed_orders(selection_path, selection)
    generated = []

    for excerpt in selection["excerpts"]:
        source = repo / excerpt["source_video"]
        excerpt_dir = output_dir / excerpt["id"]
        excerpt_dir.mkdir(parents=True, exist_ok=True)
        duration = float(excerpt["duration_seconds"])
        expected_frames = round(duration * float(selection["output"]["fps"]))
        order = fixed_orders[excerpt["id"]]
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
        permutations = {
            code: {
                "output_video": str(
                    output.relative_to(selection_path.parent)
                ).replace("\\", "/"),
                "output_sha256": _sha256(output),
                "output_bytes": output.stat().st_size,
            }
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
        print(
            f"{excerpt['id']}: order {code} "
            f"({output.stat().st_size / 1024 / 1024:.1f} MiB)"
        )

    selected_ids = {excerpt["id"] for excerpt in selection["excerpts"]}
    for excerpt_dir in output_dir.iterdir():
        if excerpt_dir.is_dir() and excerpt_dir.name not in selected_ids:
            shutil.rmtree(excerpt_dir)
            continue
        if excerpt_dir.is_dir():
            selected_code = "".join(str(lane) for lane in fixed_orders[excerpt_dir.name])
            for stale in excerpt_dir.glob("*.mp4"):
                if stale.name != f"{selected_code}.mp4":
                    stale.unlink()

    manifest = {
        "protocol": selection["protocol"],
        "selection_type": selection["selection_type"],
        "blind_lane_order": selection["source_layout"]["lanes"],
        "output": selection["output"],
        "fixed_sequence": sequence,
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
