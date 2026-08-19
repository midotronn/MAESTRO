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


def test_render_scaling_reports_actual_failover_worker(
    tmp_path,
    monkeypatch,
):
    from scripts.benchmark_render_scaling import benchmark_render_workers
    from server.distributed.registry import WorkerRegistry, WorkerSpec
    from server.distributed.tasks import PROTOCOL_VERSION
    from server.distributed.worker import FileTaskWorker

    poses = tmp_path / "poses.npz"
    np.savez(poses, fk_joints=np.zeros((1, 22, 3), dtype=np.float32))
    stale = WorkerSpec(
        worker_id="a-stale",
        capabilities=("render.frames",),
        task_dir=tmp_path / "a-stale",
    )
    healthy = WorkerSpec(
        worker_id="b-healthy",
        capabilities=("render.frames",),
        task_dir=tmp_path / "b-healthy",
    )
    stale.task_dir.mkdir()
    stale.heartbeat_path.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "worker_id": stale.worker_id,
                "capabilities": ["render.frames"],
                "status": "ready",
                "updated_at": time.time(),
            }
        )
    )

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
        healthy,
        {"render.frames": render},
        poll_interval=0.01,
        heartbeat_interval=0.05,
    )
    worker_thread = threading.Thread(target=worker.run_forever, daemon=True)
    worker_thread.start()
    _wait_for(healthy.is_healthy)
    monkeypatch.setenv("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "1")

    outcome = {}

    def benchmark():
        try:
            outcome["report"] = benchmark_render_workers(
                poses,
                tmp_path / "results",
                registry=WorkerRegistry([stale, healthy]),
                frame_count=1,
                timeout=5,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            outcome["error"] = exc

    benchmark_thread = threading.Thread(target=benchmark)
    benchmark_thread.start()
    _wait_for(lambda: bool(list((stale.task_dir / "requests").glob("*.json"))))
    heartbeat = json.loads(stale.heartbeat_path.read_text())
    heartbeat["updated_at"] = time.time() - 30
    stale.heartbeat_path.write_text(json.dumps(heartbeat))
    benchmark_thread.join(timeout=5)

    assert "error" not in outcome
    report = outcome["report"]
    assert report["workers"][0]["assigned_worker_id"] == "a-stale"
    assert report["workers"][0]["worker_id"] == "b-healthy"
    worker.stop()
    worker_thread.join(timeout=2)
