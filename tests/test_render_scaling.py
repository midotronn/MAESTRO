import json
import threading
import time
from pathlib import Path

import numpy as np


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def test_render_scaling_report_uses_measured_worker_throughput(tmp_path):
    from scripts.benchmark_render_scaling import benchmark_render_workers
    from server.distributed.registry import WorkerRegistry, WorkerSpec
    from server.distributed.worker import FileTaskWorker

    poses = tmp_path / "poses.npz"
    np.savez(poses, fk_joints=np.zeros((10, 22, 3), dtype=np.float32))
    specs = [
        WorkerSpec(
            worker_id=f"render-{index}",
            capabilities=("render.frames",),
            task_dir=tmp_path / f"worker-{index}",
        )
        for index in range(2)
    ]
    workers = []
    threads = []
    for spec in specs:
        def render(payload):
            shard = Path(payload["shard_output"])
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_bytes(b"ffv1")
            return {
                "frame_start": payload["frame_start"],
                "frame_end": payload["frame_end"],
                "frames": payload["frame_end"] - payload["frame_start"],
                "transport": "ffv1",
                "source_frames_sha256": "a" * 64,
                "shard_sha256": "b" * 64,
                "shard_output": str(shard),
            }

        worker = FileTaskWorker(
            spec,
            {"render.frames": render},
            poll_interval=0.01,
            heartbeat_interval=0.05,
        )
        thread = threading.Thread(target=worker.run_forever, daemon=True)
        thread.start()
        workers.append(worker)
        threads.append(thread)
    for spec in specs:
        _wait_for(spec.is_healthy)

    report = benchmark_render_workers(
        poses,
        tmp_path / "results",
        registry=WorkerRegistry(specs),
        frame_count=10,
        timeout=5,
        target_frames=100,
        render_budget_seconds=10,
    )

    assert report["status"] == "completed"
    assert report["frames"] == 10
    assert report["worker_count"] == 2
    assert len(report["workers"]) == 2
    assert report["aggregate_frames_per_second"] > 0
    assert report["capacity_projection"]["ideal_workers_at_median_rate"] >= 1
    persisted = (
        tmp_path / "results" / report["run_id"] / "report.json"
    )
    assert json.loads(persisted.read_text())["run_id"] == report["run_id"]

    for worker in workers:
        worker.stop()
    for thread in threads:
        thread.join(timeout=2)
