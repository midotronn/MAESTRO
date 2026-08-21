#!/usr/bin/env python3
"""Benchmark exact-quality distributed render shards without provisioning resources."""

from __future__ import annotations

import argparse
from collections import Counter
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

from server.distributed import (  # noqa: E402
    ARTIFACT_TRANSPORT,
    ArtifactRef,
    FileTaskCoordinator,
    HttpTaskCoordinator,
    PROTOCOL_VERSION,
    WorkerRegistry,
    deterministic_task_id,
    distributed_transport,
    sha256_file,
)
from server.rendering import (  # noqa: E402
    _render_quality_contract,
    _render_ranges,
    _select_render_worker_cohort,
    _validate_render_output,
    _validate_render_ranges,
)
from server.distributed.render_contract import (  # noqa: E402
    RENDER_CONTRACT_VERSION,
    inspect_ffv1_shard,
)


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
    registry: WorkerRegistry | None = None,
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
    transport = distributed_transport("render.frames")
    heartbeat_max_age = float(
        os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
    )
    if transport == "http":
        coordinator = HttpTaskCoordinator.from_env()
        workers = coordinator.require_workers(
            "render.frames",
            max_age_seconds=heartbeat_max_age,
        )
    else:
        registry = registry or WorkerRegistry.from_env()
        workers = registry.require(
            "render.frames",
            max_age_seconds=heartbeat_max_age,
        )
        coordinator = FileTaskCoordinator(
            registry,
            heartbeat_max_age=heartbeat_max_age,
        )
    requested_workers = (
        len(workers)
        if worker_limit is None
        else max(1, int(worker_limit))
    )
    render_quality = _render_quality_contract(
        width=width,
        height=height,
        samples=samples,
        engine=engine,
        denoise=denoise,
        frame_format=frame_format,
        fps=fps,
    )
    (
        workers,
        eligible_worker_ids,
        render_provenance,
        render_identity,
    ) = _select_render_worker_cohort(
        workers,
        requested_workers=requested_workers,
        quality=render_quality,
    )
    workers = workers[:total_frames]
    worker_metadata = {}
    for worker in workers:
        metadata = dict(worker.metadata)
        heartbeat = getattr(worker, "heartbeat", None)
        if callable(heartbeat):
            try:
                metadata.update(dict(heartbeat().get("metadata") or {}))
            except Exception:  # noqa: BLE001 - health was already validated
                pass
        worker_metadata[worker.worker_id] = metadata
    ranges = _render_ranges(total_frames, len(workers))
    _validate_render_ranges(ranges, total_frames)
    run_id = f"render-scale-{uuid.uuid4().hex[:12]}"
    run_dir = output_dir.resolve() / run_id
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=False)

    started_at = time.time()
    assignments = []
    handles = []
    expected_artifacts = {}
    expected_provenances = {}
    poses_artifact = None
    if transport == "http":
        poses_sha256, poses_size = sha256_file(poses_path)
        poses_artifact = coordinator.upload_input(
            poses_path,
            artifact_key=f"render-source:{poses_sha256}:{poses_size}",
        )
    for worker, (start, end) in zip(workers, ranges):
        shard_path = shard_dir / f"shard_{start:06d}_{end:06d}.mkv"
        payload = {
            "benchmark_run_id": run_id,
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
            "render_contract_version": RENDER_CONTRACT_VERSION,
            "render_provenance": render_provenance,
            "render_identity_digest": render_identity,
        }
        task_id = None
        if transport == "http":
            assert poses_artifact is not None
            task_id = deterministic_task_id(
                "render.frames",
                {
                    **payload,
                    "task_protocol_version": PROTOCOL_VERSION,
                    "artifact_transport": ARTIFACT_TRANSPORT,
                    "poses_sha256": poses_artifact.sha256,
                    "poses_size": poses_artifact.size,
                },
            )
            output_artifact = coordinator.reserve_output(
                artifact_key=f"render-shard:{task_id}",
                task_id=task_id,
            )
            payload.update(
                {
                    "task_protocol_version": PROTOCOL_VERSION,
                    "artifact_transport": ARTIFACT_TRANSPORT,
                    "poses_artifact": poses_artifact.to_dict(),
                    "shard_artifact": output_artifact.to_dict(),
                }
            )
            expected_artifacts[task_id] = output_artifact
        else:
            payload.update(
                {
                    "poses": str(poses_path),
                    "shard_output": str(shard_path),
                }
            )
        submit_options = {
            "worker": worker,
            "task_id": task_id,
            "eligible_worker_ids": eligible_worker_ids,
        }
        if transport == "http":
            submit_options["retry_failed"] = True
        handle = coordinator.submit(
            "render.frames",
            payload,
            **submit_options,
        )
        handles.append(handle)
        expected_provenances[handle.request.task_id] = render_provenance
        assignments.append(
            {
                "assigned_worker_id": worker.worker_id,
                "assigned_gpu_index": str(
                    worker_metadata.get(worker.worker_id, {}).get(
                        "gpu_index",
                        "",
                    )
                ),
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
        render_provenance = expected_provenances[result.task_id]
        source_hash, shard_hash, decoded_rgb_hash = _validate_render_output(
            output,
            start=assignment["frame_start"],
            end=assignment["frame_end"],
            width=int(width),
            height=int(height),
            samples=int(samples),
            engine=str(engine),
            denoise=int(denoise),
            frame_format=str(frame_format),
            fps=int(fps),
            render_provenance=render_provenance,
        )
        shard_path = shard_dir / (
            f"shard_{assignment['frame_start']:06d}_"
            f"{assignment['frame_end']:06d}.mkv"
        )
        artifact_id = ""
        if transport == "http":
            artifact = ArtifactRef.from_dict(
                output.get("shard_artifact") or {},
                require_complete=True,
            )
            expected_artifact = expected_artifacts.get(result.task_id)
            if (
                expected_artifact is None
                or artifact.artifact_id != expected_artifact.artifact_id
                or artifact.sha256 != shard_hash
                or output.get("artifact_transport") != ARTIFACT_TRANSPORT
            ):
                raise RuntimeError(
                    f"worker {result.worker_id} returned a mismatched artifact"
                )
            coordinator.download_output(artifact, shard_path)
            artifact_id = artifact.artifact_id
        else:
            reported_path = Path(str(output.get("shard_output") or "")).resolve()
            if reported_path != shard_path.resolve():
                raise RuntimeError(
                    f"worker {result.worker_id} returned an unexpected shard path"
                )
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            raise RuntimeError(
                f"worker {result.worker_id} did not produce {shard_path}"
            )
        actual_hash, actual_size = sha256_file(shard_path)
        if actual_hash != shard_hash:
            raise RuntimeError(
                f"worker {result.worker_id} shard failed SHA-256 verification"
            )
        actual_validation = inspect_ffv1_shard(
            shard_path,
            frame_start=assignment["frame_start"],
            frame_end=assignment["frame_end"],
            width=int(width),
            height=int(height),
            fps=int(fps),
        )
        if actual_validation["decoded_rgb_sha256"] != decoded_rgb_hash:
            raise RuntimeError(
                f"worker {result.worker_id} shard failed decoded RGB verification"
            )
        duration = max(0.001, result.finished_at - result.started_at)
        daemon_attestation = dict(output["daemon_attestation"])
        attested_gpu = dict(daemon_attestation["gpu"])
        actual_gpu_index = str(attested_gpu["cuda_index"])
        per_worker.append(
            {
                **assignment,
                "worker_id": result.worker_id,
                "gpu_index": actual_gpu_index,
                "gpu_uuid": attested_gpu["uuid"],
                "gpu_pci_bus_id": attested_gpu["pci_bus_id"],
                "gpu_selection_mode": attested_gpu["selection_mode"],
                "duration_seconds": round(duration, 3),
                "frames_per_second": round(assignment["frames"] / duration, 3),
                "source_frames_sha256": source_hash,
                "decoded_rgb_sha256": decoded_rgb_hash,
                "shard_sha256": shard_hash,
                "artifact_id": artifact_id,
                "shard_path": str(shard_path),
                "shard_bytes": actual_size,
            }
        )

    wall_seconds = max(0.001, finished_at - started_at)
    per_gpu_rates = sorted(item["frames_per_second"] for item in per_worker)
    median_rate = statistics.median(per_gpu_rates)
    required_workers = math.ceil(
        target_frames / (median_rate * render_budget_seconds)
    )
    workers_per_gpu = Counter(
        item["gpu_uuid"] or "unknown"
        for item in per_worker
    )
    workers_per_cuda_index = Counter(
        item["gpu_index"] or "unknown"
        for item in per_worker
    )
    report = {
        "schema_version": 2,
        "run_id": run_id,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(wall_seconds, 3),
        "frames": total_frames,
        "worker_count": len(workers),
        "gpu_count": len(
            [gpu for gpu in workers_per_gpu if gpu != "unknown"]
        ),
        "workers_per_gpu": dict(sorted(workers_per_gpu.items())),
        "workers_per_cuda_index": dict(
            sorted(workers_per_cuda_index.items())
        ),
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
        registry=(
            None
            if distributed_transport("render.frames") == "http"
            else WorkerRegistry.from_env()
        ),
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
