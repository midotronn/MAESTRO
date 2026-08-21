#!/usr/bin/env python3
"""Dispatch exact song timing and editor beat analysis to the resident worker."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed import FileTaskCoordinator, WorkerRegistry


def _source(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: dispatch_beat_tracking.py <sid>", file=sys.stderr)
        return 2
    sid = sys.argv[1]
    workspace = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
    wav = workspace / "LODGE/data/finedance/music_wav" / f"{sid}.wav"
    if not wav.is_file():
        raise FileNotFoundError(wav)
    outputs = {
        "metadata_output": workspace / f"audio_timing_{sid}.json",
        "beats_output": workspace / f"beats_{sid}.npy",
        "strengths_output": workspace / f"beat_strengths_{sid}.npy",
    }
    registry = WorkerRegistry.from_env()
    workers = registry.require(
        "audio.beats",
        max_age_seconds=float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
        ),
    )
    coordinator = FileTaskCoordinator(
        registry,
        heartbeat_max_age=float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
        ),
    )
    handle = coordinator.submit(
        "audio.beats",
        {
            "sid": sid,
            "wav": str(wav),
            **{name: str(path) for name, path in outputs.items()},
            "source": _source(wav),
        },
        worker=workers[0],
    )
    result = coordinator.wait(
        handle,
        timeout=float(
            os.environ.get("AGENTLODGE_BEAT_TRACKING_TIMEOUT", "300")
        ),
    )
    missing = [
        str(path)
        for path in outputs.values()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "audio.beats worker did not produce: " + ", ".join(missing)
        )
    print(
        f"BEAT_TRACKING_{sid}_DONE "
        f"metadata_beats={int(result.output.get('metadata_beats') or 0)} "
        f"editor_beats={int(result.output.get('editor_beats') or 0)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
