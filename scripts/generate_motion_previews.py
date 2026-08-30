#!/usr/bin/env python3
"""Render deterministic per-motion editor examples through the canonical Y-Bot pod renderer.

The examples use ``MotionBank.apply`` on a neutral host, not the raw clips, so they show the same
composition and root-residual behavior a researcher sees after an editor action. One reel is
rendered and then split into 19 MP4 files to avoid paying Blender startup for every motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.editor.motion_bank import MotionBank, _ease_join  # noqa: E402

FPS = 30
HOLD_FRAMES = 9
BRIDGE_FRAMES = 18
PREVIEW_SCHEMA = 1


def _preview_window(bank: MotionBank, motion_id: str) -> tuple[np.ndarray, dict]:
    spec = bank.resolve(motion_id)
    source = bank.load_clip(spec)
    frame_count = max(150, spec.frames + 75)
    neutral = np.repeat(source[:1], frame_count, axis=0).astype(np.float32)
    example, report = bank.apply(
        neutral,
        spec.id,
        anchor="center",
        direction=spec.canonical_direction,
        repeats=1,
    )
    held = np.concatenate(
        [
            np.repeat(example[:1], HOLD_FRAMES, axis=0),
            example,
            np.repeat(example[-1:], HOLD_FRAMES, axis=0),
        ],
        axis=0,
    )
    return held.astype(np.float32), report


def build_preview_reel(bank: MotionBank | None = None) -> tuple[np.ndarray, list[dict]]:
    bank = bank or MotionBank()
    reel = np.zeros((0, 139), dtype=np.float32)
    spans: list[dict] = []
    for spec in bank.specs:
        example, report = _preview_window(bank, spec.id)
        if reel.shape[0]:
            bridge = np.repeat(example[:1], BRIDGE_FRAMES, axis=0)
            reel = _ease_join(reel, bridge, BRIDGE_FRAMES)
        start = int(reel.shape[0])
        reel = np.concatenate([reel, example], axis=0)
        spans.append(
            {
                "id": spec.id,
                "name": spec.name,
                "start": start,
                "end": int(reel.shape[0]),
                "action_range": [int(x + HOLD_FRAMES) for x in report["action_range"]],
                "event_frame": int(report["event_frame"] + HOLD_FRAMES),
                "direction": report.get("direction"),
                "repeats": int(report["repeats"]),
            }
        )
    return np.ascontiguousarray(reel, dtype=np.float32), spans


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("ffmpeg is required to split motion preview videos") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_reel(reel_path: Path, output: Path, *, width: int, height: int, samples: int) -> None:
    command = [
        "bash",
        str(ROOT / "scripts" / "render_one_ybot.sh"),
        str(reel_path),
        str(output),
        "",
    ]
    env = dict(os.environ)
    env.setdefault("AGENTLODGE_ROOT", str(ROOT))
    env["AL_PY"] = sys.executable
    env["RENDER_W"] = str(width)
    env["RENDER_H"] = str(height)
    env["RENDER_SAMPLES"] = str(samples)
    env["RENDER_FIXED_CAMERA"] = "1"
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(f"Y-Bot renderer did not produce a usable reel: {output}")


def _split_reel(ffmpeg: str, reel: Path, output_dir: Path, spans: list[dict]) -> None:
    for span in spans:
        output = output_dir / f"{span['id']}.mp4"
        start = span["start"] / FPS
        duration = (span["end"] - span["start"]) / FPS
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(reel),
                "-t",
                f"{duration:.6f}",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
        if not output.is_file() or output.stat().st_size < 1024:
            raise RuntimeError(f"ffmpeg did not produce a usable preview: {output}")


def verify_previews(output_dir: Path, bank: MotionBank | None = None) -> list[str]:
    bank = bank or MotionBank()
    failures = []
    for spec in bank.specs:
        path = output_dir / f"{spec.id}.mp4"
        if not path.is_file() or path.stat().st_size < 1024:
            failures.append(spec.id)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "assets" / "motion_bank" / "previews",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--keep-build", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    bank = MotionBank()
    if args.check:
        failures = verify_previews(output_dir, bank)
        if failures:
            print("Missing or invalid motion previews: " + ", ".join(failures), file=sys.stderr)
            return 1
        print(f"Verified {len(bank.specs)} motion previews in {output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir / ".preview-build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    reel_path = build_dir / "motion_previews.npy"
    reel_video = build_dir / "motion_previews.mp4"
    reel, spans = build_preview_reel(bank)
    np.save(reel_path, reel)
    (build_dir / "motion_previews.json").write_text(
        json.dumps({"fps": FPS, "frames": len(reel), "spans": spans}, indent=2),
        encoding="utf-8",
    )

    _render_reel(
        reel_path,
        reel_video,
        width=max(128, args.width),
        height=max(128, args.height),
        samples=max(1, args.samples),
    )
    _split_reel(_ffmpeg(), reel_video, output_dir, spans)
    failures = verify_previews(output_dir, bank)
    if failures:
        raise RuntimeError("preview generation incomplete: " + ", ".join(failures))

    manifest = {
        "schema_version": PREVIEW_SCHEMA,
        "bank_version": bank.version,
        "fps": FPS,
        "generator": "MotionBank.apply -> canonical Y-Bot renderer -> ffmpeg split",
        "motions": [
            {
                **span,
                "file": f"{span['id']}.mp4",
                "sha256": _sha256(output_dir / f"{span['id']}.mp4"),
                "bytes": (output_dir / f"{span['id']}.mp4").stat().st_size,
            }
            for span in spans
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if not args.keep_build:
        shutil.rmtree(build_dir)
    print(f"Generated {len(spans)} editor motion previews in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
