#!/usr/bin/env python3
"""Benchmark exact-quality distributed render shards without provisioning resources."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed import FileTaskCoordinator, WorkerRegistry  # noqa: E402
from server.rendering import _render_ranges  # noqa: E402


def _frame_count(poses_path: Path) -> int:
    with np.load(poses_path) as data:
        if "fk_joints" in data:
            return int(data["fk_joints"].shape[0])
        for key in data.files:
            value = data[key]
            if getattr(value, "ndim", 0) >= 1:
                return int(value.shape[0])
    raise ValueError(f"could not infer frame count from {poses_path}")


def benchmark_render_workers(
    poses_path: Path,
    output_dir: Path,
    *,
    registry: WorkerRegistry,
    frame_count: int | None = None,
    worker_limit: int | None = None,
    width: int = 1080,
    height: int = 1080,
    samples: int = 96,
    engine: str = "eevee",
    denoise: int = 1,
    frame_format: str = "tga",
    fps: int = 30,
    timeout: float = 1800.0,
    target_frames: int = 5400,
    render_budget_seconds: float = 23.0,
) -> dict:
    poses_path = poses_path.resolve()
    if not poses_path.is_file():
        raise FileNotFoundError(poses_path)
    total_frames = int(frame_count or _frame_count(poses_path))
    if total_frames < 1:
        raise ValueError("frame_count must be positive")
    workers = registry.require(
        "render.frames",
        max_age_seconds=float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
        ),
    )
    if worker_limit is not None:
        workers = workers[: max(1, int(worker_limit))]
    workers = workers[:total_frames]
    ranges = _render_ranges(total_frames, len(workers))
    run_id = f"render-scale-{uuid.uuid4().hex[:12]}"
    run_dir = output_dir.resolve() / run_id
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=False)

    coordinator = FileTaskCoordinator(
        registry,
        heartbeat_max_age=float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
        ),
    )
    started_at = time.time()
    assignments = []
    handles = []
    for worker, (start, end) in zip(workers, ranges):
        shard_path = shard_dir / f"shard_{start:06d}_{end:06d}.mkv"
        handle = coordinator.submit(
            "render.frames",
            {
                "poses": str(poses_path),
                "shard_output": str(shard_path),
                "frame_start": start,
                "frame_end": end,
                "width": int(width),
                "height": int(height),
                "samples": int(samples),
                "engine": str(engine),
                "denoise": int(denoise),
                "frame_format": str(frame_format),
                "fps": int(fps),
                "timeout": float(timeout),
            },
            worker=worker,
        )
        handles.append(handle)
        assignments.append(
            {
                "assigned_worker_id": worker.worker_id,
                "frame_start": start,
                "frame_end": end,
                "frames": end - start,
                "task_id": handle.request.task_id,
            }
        )

    results = coordinator.wait_many(handles, timeout=timeout)
    finished_at = time.time()
    per_worker = []
    for assignment, result in zip(assignments, results):
        output = result.output
        if (
            int(output.get("frame_start", -1)) != assignment["frame_start"]
            or int(output.get("frame_end", -1)) != assignment["frame_end"]
            or int(output.get("frames", -1)) != assignment["frames"]
            or output.get("transport") != "ffv1"
        ):
            raise RuntimeError(
                f"worker {result.worker_id} returned invalid shard metadata: {output}"
            )
        shard_path = Path(str(output.get("shard_output") or ""))
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            raise RuntimeError(
                f"worker {result.worker_id} did not produce {shard_path}"
            )
        duration = max(0.001, result.finished_at - result.started_at)
        per_worker.append(
            {
                **assignment,
                "worker_id": result.worker_id,
                "duration_seconds": round(duration, 3),
                "frames_per_second": round(assignment["frames"] / duration, 3),
                "source_frames_sha256": output.get("source_frames_sha256"),
                "shard_sha256": output.get("shard_sha256"),
                "shard_path": str(shard_path),
                "shard_bytes": shard_path.stat().st_size,
            }
        )

    wall_seconds = max(0.001, finished_at - started_at)
    per_gpu_rates = sorted(item["frames_per_second"] for item in per_worker)
    median_rate = statistics.median(per_gpu_rates)
    required_workers = math.ceil(
        target_frames / (median_rate * render_budget_seconds)
    )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(wall_seconds, 3),
        "frames": total_frames,
        "worker_count": len(workers),
        "aggregate_frames_per_second": round(total_frames / wall_seconds, 3),
        "median_worker_frames_per_second": round(median_rate, 3),
        "quality": {
            "width": int(width),
            "height": int(height),
            "samples": int(samples),
            "engine": str(engine),
            "denoise": int(denoise),
            "frame_format": str(frame_format),
            "fps": int(fps),
            "render_every_frame": True,
            "transport": "ffv1",
        },
        "capacity_projection": {
            "target_frames": int(target_frames),
            "render_budget_seconds": float(render_budget_seconds),
            "ideal_workers_at_median_rate": required_workers,
            "note": "Add p95 and coordination headroom only after measured scaling runs.",
        },
        "workers": per_worker,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--worker-limit", type=int)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--engine", default="eevee")
    parser.add_argument("--denoise", type=int, default=1)
    parser.add_argument("--frame-format", default="tga")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--target-frames", type=int, default=5400)
    parser.add_argument("--render-budget-seconds", type=float, default=23)
    parser.add_argument("--report")
    args = parser.parse_args()

    report = benchmark_render_workers(
        Path(args.poses),
        Path(args.output_dir),
        registry=WorkerRegistry.from_env(),
        frame_count=args.frames,
        worker_limit=args.worker_limit,
        width=args.width,
        height=args.height,
        samples=args.samples,
        engine=args.engine,
        denoise=args.denoise,
        frame_format=args.frame_format,
        fps=args.fps,
        timeout=args.timeout,
        target_frames=args.target_frames,
        render_budget_seconds=args.render_budget_seconds,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
