import json
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def test_task_ids_are_deterministic_and_payload_sensitive():
    from server.distributed.tasks import deterministic_task_id

    first = deterministic_task_id("echo.task", {"value": 1, "nested": {"b": 2}})
    reordered = deterministic_task_id(
        "echo.task",
        {"nested": {"b": 2}, "value": 1},
    )
    changed = deterministic_task_id("echo.task", {"value": 2, "nested": {"b": 2}})

    assert first == reordered
    assert first != changed


def test_distributed_capabilities_support_staged_rollout(monkeypatch):
    from server.distributed.runtime import capability_enabled

    monkeypatch.delenv("AGENTLODGE_DISTRIBUTED", raising=False)
    assert not capability_enabled("jukebox.extract")

    monkeypatch.setenv("AGENTLODGE_DISTRIBUTED", "1")
    monkeypatch.setenv(
        "AGENTLODGE_DISTRIBUTED_CAPABILITIES",
        "jukebox.extract,lodge",
    )
    assert capability_enabled("jukebox.extract")
    assert capability_enabled("lodge.generate")
    assert not capability_enabled("edge.generate")
    assert not capability_enabled("render.frames")


def test_runpod_launcher_forwards_full_render_quality_contract():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "start_runpod_worker.sh"
    ).read_text(encoding="utf-8")

    for option, environment in (
        ("--render-width", "AGENTLODGE_RENDER_FULL_W"),
        ("--render-height", "AGENTLODGE_RENDER_FULL_H"),
        ("--render-samples", "AGENTLODGE_RENDER_FULL_SAMPLES"),
        ("--render-engine", "AGENTLODGE_RENDER_ENGINE"),
        ("--render-denoise", "AGENTLODGE_RENDER_DENOISE"),
        ("--render-frame-format", "AGENTLODGE_RENDER_FRAME_FORMAT"),
    ):
        assert option in launcher
        assert environment in launcher

    assert "AGENTLODGE_GPU_INDEX" in launcher
    assert 'export CUDA_VISIBLE_DEVICES="$GPU_INDEX"' in launcher
    assert "NVIDIA EGL ignores process-level CUDA_VISIBLE_DEVICES" in launcher


def test_jukebox_preload_executes_representative_inference(
    tmp_path,
    monkeypatch,
):
    from server.distributed.handlers import JukeboxExtractHandler

    edge = tmp_path / "edge"
    shared = tmp_path / "shared"
    edge.mkdir()
    shared.mkdir()
    calls = []

    def fake_extract(path):
        with wave.open(path, "rb") as audio:
            calls.append(
                {
                    "channels": audio.getnchannels(),
                    "sample_width": audio.getsampwidth(),
                    "sample_rate": audio.getframerate(),
                    "frames": audio.getnframes(),
                }
            )
        return np.ones((2, 3), dtype=np.float32), "unused"

    monkeypatch.setitem(
        sys.modules,
        "jukemirlib",
        SimpleNamespace(
            VQVAE=object(),
            TOP_PRIOR=object(),
            setup_models=lambda: pytest.fail("models are already loaded"),
        ),
    )
    handler = JukeboxExtractHandler(
        edge_root=edge,
        shared_root=shared,
        extractor=fake_extract,
        preload_audio_seconds=0.1,
    )

    handler.preload()

    assert calls == [
        {
            "channels": 1,
            "sample_width": 2,
            "sample_rate": 44_100,
            "frames": 4_410,
        }
    ]


def test_file_worker_executes_and_reuses_idempotent_result(tmp_path):
    from server.distributed.coordinator import FileTaskCoordinator
    from server.distributed.registry import WorkerRegistry, WorkerSpec
    from server.distributed.worker import FileTaskWorker

    calls = []
    spec = WorkerSpec(
        worker_id="worker-1",
        capabilities=("echo.task",),
        task_dir=tmp_path / "worker-1",
    )

    def echo(payload):
        calls.append(payload["value"])
        return {"value": payload["value"]}

    worker = FileTaskWorker(
        spec,
        {"echo.task": echo},
        poll_interval=0.01,
        heartbeat_interval=0.05,
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    _wait_for(spec.is_healthy)

    coordinator = FileTaskCoordinator(
        WorkerRegistry([spec]),
        poll_interval=0.01,
    )
    first = coordinator.submit("echo.task", {"value": 7})
    result = coordinator.wait(first, timeout=2)
    assert result.output == {"value": 7}

    repeated = coordinator.submit("echo.task", {"value": 7})
    repeated_result = coordinator.wait(repeated, timeout=2)
    assert repeated_result.output == {"value": 7}
    assert calls == [7]
    with pytest.raises(ValueError, match="different request"):
        coordinator.submit(
            "echo.task",
            {"value": 8},
            task_id=first.request.task_id,
        )

    worker.stop()
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_separate_coordinators_deduplicate_one_task_across_workers(tmp_path):
    from server.distributed.coordinator import FileTaskCoordinator
    from server.distributed.registry import WorkerRegistry, WorkerSpec
    from server.distributed.worker import FileTaskWorker

    calls = []
    call_lock = threading.Lock()
    specs = [
        WorkerSpec(
            worker_id=f"worker-{index}",
            capabilities=("echo.task",),
            task_dir=tmp_path / f"worker-{index}",
        )
        for index in range(2)
    ]
    workers = []
    threads = []
    for spec in specs:
        def echo(payload, worker_id=spec.worker_id):
            with call_lock:
                calls.append((worker_id, payload["value"]))
            return {"value": payload["value"]}

        worker = FileTaskWorker(
            spec,
            {"echo.task": echo},
            poll_interval=0.01,
            heartbeat_interval=0.05,
        )
        thread = threading.Thread(target=worker.run_forever, daemon=True)
        thread.start()
        workers.append(worker)
        threads.append(thread)
    for spec in specs:
        _wait_for(spec.is_healthy)

    registry = WorkerRegistry(specs)
    coordinators = [
        FileTaskCoordinator(registry, poll_interval=0.01)
        for _ in range(2)
    ]
    barrier = threading.Barrier(3)
    handles = []

    def submit(coordinator):
        barrier.wait()
        handles.append(coordinator.submit("echo.task", {"value": 17}))

    submitters = [
        threading.Thread(target=submit, args=(coordinator,))
        for coordinator in coordinators
    ]
    for thread in submitters:
        thread.start()
    barrier.wait()
    for thread in submitters:
        thread.join(timeout=2)

    results = [
        coordinator.wait(handle, timeout=3)
        for coordinator, handle in zip(coordinators, handles)
    ]
    assert len(calls) == 1
    assert {result.worker_id for result in results} == {calls[0][0]}
    for worker in workers:
        worker.stop()
    for thread in threads:
        thread.join(timeout=2)


def test_shared_load_balancing_uses_all_healthy_workers(tmp_path):
    from collections import Counter

    from server.distributed.coordinator import FileTaskCoordinator
    from server.distributed.registry import WorkerRegistry, WorkerSpec
    from server.distributed.worker import FileTaskWorker

    release = threading.Event()
    specs = [
        WorkerSpec(
            worker_id=f"worker-{index}",
            capabilities=("echo.task",),
            task_dir=tmp_path / f"worker-{index}",
        )
        for index in range(2)
    ]
    workers = []
    threads = []
    for spec in specs:
        def echo(payload):
            release.wait(timeout=3)
            return {"value": payload["value"]}

        worker = FileTaskWorker(
            spec,
            {"echo.task": echo},
            poll_interval=0.01,
            heartbeat_interval=0.05,
        )
        thread = threading.Thread(target=worker.run_forever, daemon=True)
        thread.start()
        workers.append(worker)
        threads.append(thread)
    for spec in specs:
        _wait_for(spec.is_healthy)

    registry = WorkerRegistry(specs)
    handles = [
        FileTaskCoordinator(registry, poll_interval=0.01).submit(
            "echo.task",
            {"value": value},
        )
        for value in range(8)
    ]
    assignments = Counter(handle.worker.worker_id for handle in handles)
    assert set(assignments) == {"worker-0", "worker-1"}
    assert abs(assignments["worker-0"] - assignments["worker-1"]) <= 1

    release.set()
    FileTaskCoordinator(registry, poll_interval=0.01).wait_many(
        handles,
        timeout=5,
    )
    for worker in workers:
        worker.stop()
    for thread in threads:
        thread.join(timeout=2)


def test_worker_requeues_claim_when_result_write_exhausts_retries(
    tmp_path,
    monkeypatch,
):
    import server.distributed.worker as worker_module
    from server.distributed.coordinator import FileTaskCoordinator
    from server.distributed.registry import WorkerRegistry, WorkerSpec

    calls = []
    spec = WorkerSpec(
        worker_id="worker-1",
        capabilities=("echo.task",),
        task_dir=tmp_path / "worker-1",
    )

    def echo(payload):
        calls.append(payload["value"])
        return {"value": payload["value"]}

    worker = worker_module.FileTaskWorker(
        spec,
        {"echo.task": echo},
        poll_interval=0.01,
        heartbeat_interval=0.05,
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    _wait_for(spec.is_healthy)

    original_atomic_json = worker_module._atomic_json
    failures = {"remaining": 5}

    def flaky_atomic_json(path, payload):
        if path.parent.name == "results" and failures["remaining"] > 0:
            failures["remaining"] -= 1
            raise OSError("transient shared-volume failure")
        return original_atomic_json(path, payload)

    monkeypatch.setattr(worker_module, "_atomic_json", flaky_atomic_json)
    coordinator = FileTaskCoordinator(
        WorkerRegistry([spec]),
        poll_interval=0.01,
    )
    result = coordinator.wait(
        coordinator.submit("echo.task", {"value": 23}),
        timeout=5,
    )

    assert result.output == {"value": 23}
    assert calls == [23]
    assert thread.is_alive()
    worker.stop()
    thread.join(timeout=2)


def test_worker_recovers_claimed_task_after_restart(tmp_path):
    from server.distributed.registry import WorkerSpec
    from server.distributed.tasks import TaskRequest
    from server.distributed.worker import FileTaskWorker

    spec = WorkerSpec(
        worker_id="worker-1",
        capabilities=("echo.task",),
        task_dir=tmp_path / "worker-1",
    )
    claimed = spec.task_dir / "claimed"
    claimed.mkdir(parents=True)
    request = TaskRequest.create("echo.task", {"value": 9})
    (claimed / f"{request.task_id}.json").write_text(
        json.dumps(request.to_dict())
    )

    worker = FileTaskWorker(spec, {"echo.task": lambda payload: payload})

    assert not list(claimed.glob("*.json"))
    assert (
        spec.task_dir / "requests" / f"{request.task_id}.json"
    ).is_file()
    worker.stop()


def test_coordinator_reassigns_task_from_stale_worker(tmp_path):
    from server.distributed.coordinator import FileTaskCoordinator
    from server.distributed.registry import WorkerRegistry, WorkerSpec
    from server.distributed.tasks import PROTOCOL_VERSION
    from server.distributed.worker import FileTaskWorker

    stale = WorkerSpec(
        worker_id="stale",
        capabilities=("echo.task",),
        task_dir=tmp_path / "stale",
    )
    healthy = WorkerSpec(
        worker_id="healthy",
        capabilities=("echo.task",),
        task_dir=tmp_path / "healthy",
    )
    stale.task_dir.mkdir()
    stale.heartbeat_path.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "worker_id": stale.worker_id,
                "capabilities": ["echo.task"],
                "status": "ready",
                "updated_at": time.time(),
            }
        )
    )
    worker = FileTaskWorker(
        healthy,
        {"echo.task": lambda payload: {"value": payload["value"]}},
        poll_interval=0.01,
        heartbeat_interval=0.05,
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    _wait_for(healthy.is_healthy)
    coordinator = FileTaskCoordinator(
        WorkerRegistry([stale, healthy]),
        poll_interval=0.01,
        heartbeat_max_age=1,
    )
    handle = coordinator.submit(
        "echo.task",
        {"value": 11},
        worker=stale,
    )
    heartbeat = json.loads(stale.heartbeat_path.read_text())
    heartbeat["updated_at"] = time.time() - 30
    stale.heartbeat_path.write_text(json.dumps(heartbeat))

    result = coordinator.wait(handle, timeout=3)

    assert result.worker_id == "healthy"
    assert result.output == {"value": 11}
    worker.stop()
    thread.join(timeout=2)


def test_coordinator_does_not_duplicate_a_claimed_task_on_stale_worker(
    tmp_path,
):
    from server.distributed.coordinator import (
        FileTaskCoordinator,
        TaskTimeoutError,
    )
    from server.distributed.registry import WorkerRegistry, WorkerSpec
    from server.distributed.tasks import PROTOCOL_VERSION
    from server.distributed.worker import FileTaskWorker

    stale = WorkerSpec(
        worker_id="stale",
        capabilities=("echo.task",),
        task_dir=tmp_path / "stale",
    )
    healthy = WorkerSpec(
        worker_id="healthy",
        capabilities=("echo.task",),
        task_dir=tmp_path / "healthy",
    )
    stale.task_dir.mkdir()
    stale.heartbeat_path.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "worker_id": stale.worker_id,
                "capabilities": ["echo.task"],
                "status": "ready",
                "updated_at": time.time(),
            }
        )
    )
    calls = []
    worker = FileTaskWorker(
        healthy,
        {"echo.task": lambda payload: calls.append(payload) or payload},
        poll_interval=0.01,
        heartbeat_interval=0.05,
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    _wait_for(healthy.is_healthy)
    coordinator = FileTaskCoordinator(
        WorkerRegistry([stale, healthy]),
        poll_interval=0.01,
        heartbeat_max_age=1,
    )
    handle = coordinator.submit("echo.task", {"value": 29}, worker=stale)
    request_path = (
        stale.task_dir / "requests" / f"{handle.request.task_id}.json"
    )
    claimed_path = (
        stale.task_dir / "claimed" / f"{handle.request.task_id}.json"
    )
    claimed_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.replace(claimed_path)
    heartbeat = json.loads(stale.heartbeat_path.read_text())
    heartbeat["updated_at"] = time.time() - 30
    stale.heartbeat_path.write_text(json.dumps(heartbeat))

    with pytest.raises(TaskTimeoutError):
        coordinator.wait(handle, timeout=0.25)

    assert claimed_path.is_file()
    assert calls == []
    worker.stop()
    thread.join(timeout=2)


def test_registry_requires_current_matching_heartbeat(tmp_path):
    from server.distributed.registry import WorkerRegistry, WorkerRegistryError, WorkerSpec
    from server.distributed.tasks import PROTOCOL_VERSION

    spec = WorkerSpec(
        worker_id="juke-0",
        capabilities=("jukebox.extract",),
        task_dir=tmp_path / "juke-0",
    )
    spec.task_dir.mkdir()
    spec.heartbeat_path.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "worker_id": "juke-0",
                "capabilities": ["jukebox.extract"],
                "status": "ready",
                "updated_at": time.time() - 60,
            }
        )
    )
    registry = WorkerRegistry([spec])

    with pytest.raises(WorkerRegistryError, match="no healthy workers"):
        registry.require("jukebox.extract", max_age_seconds=5)

    heartbeat = json.loads(spec.heartbeat_path.read_text())
    heartbeat["updated_at"] = time.time()
    spec.heartbeat_path.write_text(json.dumps(heartbeat))
    assert registry.require("jukebox.extract", max_age_seconds=5) == [spec]


def test_jukebox_handler_preserves_slice_outputs_and_cache(tmp_path):
    from server.distributed.handlers import JukeboxExtractHandler

    shared = tmp_path / "shared"
    edge = tmp_path / "edge"
    shared.mkdir()
    edge.mkdir()
    wav = shared / "song_slice0.wav"
    wav.write_bytes(b"wav")
    output = shared / "cache" / "song_slice0.npy"
    calls = []

    def fake_extract(path):
        calls.append(path)
        return np.arange(12, dtype=np.float32).reshape(3, 4), "unused"

    handler = JukeboxExtractHandler(
        edge_root=edge,
        shared_root=shared,
        extractor=fake_extract,
    )
    payload = {"items": [{"wav": str(wav), "output": str(output)}]}

    first = handler(payload)
    second = handler(payload)

    np.testing.assert_array_equal(
        np.load(output),
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    assert first["cached"] == 0
    assert second["cached"] == 1
    assert calls == [str(wav.resolve())]


def test_jukebox_handler_rejects_paths_outside_shared_root(tmp_path):
    from server.distributed.handlers import JukeboxExtractHandler

    shared = tmp_path / "shared"
    edge = tmp_path / "edge"
    shared.mkdir()
    edge.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"wav")
    handler = JukeboxExtractHandler(
        edge_root=edge,
        shared_root=shared,
        extractor=lambda _path: (np.zeros((1, 1), dtype=np.float32), ""),
    )

    with pytest.raises(ValueError, match="outside the shared root"):
        handler(
            {
                "items": [
                    {
                        "wav": str(outside),
                        "output": str(shared / "out.npy"),
                    }
                ]
            }
        )


def test_lodge_handler_preload_loads_models_before_ready(
    tmp_path,
    monkeypatch,
):
    import agentlodge.dance.lodge as lodge_module
    from server.distributed.handlers import LodgeGenerateHandler

    shared = tmp_path / "shared"
    lodge_root = tmp_path / "LODGE"
    lodge_weights = tmp_path / "local.ckpt"
    global_weights = tmp_path / "global.ckpt"
    shared.mkdir()
    lodge_root.mkdir()
    lodge_weights.write_bytes(b"weights")
    global_weights.write_bytes(b"weights")
    calls = []

    def fake_generate(features, settings, work_dir, **kwargs):
        calls.append(
            {
                "features": np.asarray(features),
                "settings": settings,
                "work_dir": Path(work_dir),
                **kwargs,
            }
        )

    monkeypatch.setattr(
        lodge_module,
        "generate_lodge_dance",
        fake_generate,
    )
    handler = LodgeGenerateHandler(
        shared_root=shared,
        lodge_root=lodge_root,
        lodge_weights=lodge_weights,
        lodge_global_weights=global_weights,
    )

    handler.preload()

    assert len(calls) == 1
    assert calls[0]["features"].shape == (1, 35)
    assert calls[0]["preload_only"] is True
    assert calls[0]["settings"].lodge_code_path == lodge_root.resolve()


def test_jukebox_partitioning_is_contiguous_and_complete():
    from scripts.jukebox_extract_all import _partition_contiguous

    slices = [Path(f"slice-{index}") for index in range(10)]
    partitions = _partition_contiguous(slices, 3)

    assert [len(partition) for partition in partitions] == [4, 3, 3]
    assert [item for partition in partitions for item in partition] == slices


def test_packaged_generator_can_route_through_capability_worker(tmp_path, monkeypatch):
    import scripts.make_song_bestofk as generator
    from server.distributed.registry import WorkerSpec
    from server.distributed.worker import FileTaskWorker

    spec = WorkerSpec(
        worker_id="lodge-0",
        capabilities=("lodge.generate",),
        task_dir=tmp_path / "lodge-0",
    )

    def generate(payload):
        output = Path(payload["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.full((8, 139), 3.0, dtype=np.float32))
        return {"summary": "warm LODGE", "frames": 8}

    worker = FileTaskWorker(
        spec,
        {"lodge.generate": generate},
        poll_interval=0.01,
        heartbeat_interval=0.05,
    )
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    _wait_for(spec.is_healthy)
    monkeypatch.setenv(
        "AGENTLODGE_WORKERS_JSON",
        json.dumps(
            [
                {
                    "id": spec.worker_id,
                    "capabilities": list(spec.capabilities),
                    "task_dir": str(spec.task_dir),
                }
            ]
        ),
    )
    features = tmp_path / "features.npy"
    np.save(features, np.zeros((8, 35), dtype=np.float32))
    output = tmp_path / "work" / "lodge_motion.npy"

    result = generator._distributed_generation_job(
        "lodge",
        features,
        output,
        output.parent,
        seed=0,
    )

    assert result["error"] is None
    assert result["summary"] == "warm LODGE"
    assert result["worker_id"] == "lodge-0"
    np.testing.assert_array_equal(
        result["motion"],
        np.full((8, 139), 3.0, dtype=np.float32),
    )
    worker.stop()
    thread.join(timeout=2)


def test_render_handler_enforces_quality_and_frame_completeness(tmp_path, monkeypatch):
    from server import warm_render
    from server.distributed.handlers import RenderFramesHandler

    shared = tmp_path / "shared"
    shared.mkdir()
    poses = shared / "poses.npz"
    poses.write_bytes(b"poses")
    frames = shared / "frames"
    monkeypatch.setattr(warm_render, "on_pod", lambda: True)
    monkeypatch.setattr(warm_render, "ensure_pool", lambda **_kwargs: 1)

    def fake_render(_poses, frames_dir, **kwargs):
        target = Path(frames_dir)
        target.mkdir(parents=True, exist_ok=True)
        for frame in range(kwargs["frame_start"], kwargs["frame_end"]):
            (target / f"frame_{frame:04d}.tga").write_bytes(b"tga")
        return True

    monkeypatch.setattr(warm_render, "warm_render", fake_render)
    handler = RenderFramesHandler(shared_root=shared)
    handler.preload()
    payload = {
        "poses": str(poses),
        "frames_dir": str(frames),
        "frame_start": 3,
        "frame_end": 7,
        "width": 1080,
        "height": 1080,
        "samples": 96,
        "engine": "eevee",
        "denoise": 1,
        "frame_format": "tga",
        "timeout": 60,
    }

    result = handler(payload)
    assert result["frames"] == 4
    assert result["frame_start"] == 3
    assert result["frame_end"] == 7

    with pytest.raises(ValueError, match="quality mismatch"):
        handler({**payload, "samples": 8})


def test_render_handler_packages_local_frames_as_lossless_shard(
    tmp_path,
    monkeypatch,
):
    from server import warm_render
    from server.distributed import handlers

    shared = tmp_path / "shared"
    worker_tmp = tmp_path / "worker-tmp"
    shared.mkdir()
    poses = shared / "poses.npz"
    poses.write_bytes(b"poses")
    shard = shared / "shards" / "part-0.mkv"
    monkeypatch.setattr(warm_render, "on_pod", lambda: True)
    monkeypatch.setattr(warm_render, "ensure_pool", lambda **_kwargs: 1)
    monkeypatch.setattr(handlers, "_ffmpeg_executable", lambda: "ffmpeg")

    def fake_render(_poses, frames_dir, **kwargs):
        target = Path(frames_dir)
        target.mkdir(parents=True, exist_ok=True)
        for frame in range(kwargs["frame_start"], kwargs["frame_end"]):
            (target / f"frame_{frame:04d}.tga").write_bytes(
                f"frame-{frame}".encode()
            )
        return True

    packaged = {}

    def fake_package(frames_dir, output, **kwargs):
        packaged.update(frames_dir=Path(frames_dir), output=Path(output), **kwargs)
        Path(output).write_bytes(b"ffv1")

    monkeypatch.setattr(warm_render, "warm_render", fake_render)
    monkeypatch.setattr(handlers, "_package_ffv1", fake_package)
    handler = handlers.RenderFramesHandler(
        shared_root=shared,
        local_tmp=worker_tmp,
    )
    result = handler(
        {
            "poses": str(poses),
            "shard_output": str(shard),
            "frame_start": 0,
            "frame_end": 3,
            "width": 1080,
            "height": 1080,
            "samples": 96,
            "engine": "eevee",
            "denoise": 1,
            "frame_format": "tga",
            "fps": 30,
            "timeout": 60,
        }
    )

    assert shard.read_bytes() == b"ffv1"
    assert result["transport"] == "ffv1"
    assert result["frames"] == 3
    assert len(result["source_frames_sha256"]) == 64
    assert len(result["shard_sha256"]) == 64
    assert packaged["frame_start"] == 0
    assert packaged["frame_end"] == 3
    assert not packaged["frames_dir"].exists()


def test_full_render_dispatches_lossless_ranges_to_remote_workers(
    tmp_path,
    monkeypatch,
):
    import server.fk as fk
    import server.rendering as rendering
    import server.warm_render as warm_render
    from server.distributed.registry import WorkerSpec
    from server.distributed.worker import FileTaskWorker

    calls = []
    specs = [
        WorkerSpec(
            worker_id=f"render-{index}",
            capabilities=("render.frames",),
            task_dir=tmp_path / f"render-{index}",
        )
        for index in range(2)
    ]
    workers = []
    threads = []
    for spec in specs:
        def render(payload, worker_id=spec.worker_id):
            calls.append((worker_id, dict(payload)))
            shard = Path(payload["shard_output"])
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_bytes(b"ffv1")
            return {
                "frame_start": payload["frame_start"],
                "frame_end": payload["frame_end"],
                "frames": payload["frame_end"] - payload["frame_start"],
                "source_frames_sha256": "a" * 64,
                "shard_output": str(shard),
                "shard_sha256": "b" * 64,
                "transport": "ffv1",
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

    monkeypatch.setenv("AGENTLODGE_DISTRIBUTED", "1")
    monkeypatch.setenv("AGENTLODGE_FULL_RENDER_WORKERS", "2")
    monkeypatch.setenv("AGENTLODGE_SHARED_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENTLODGE_DISTRIBUTED_TMP", str(tmp_path / "tmp"))
    monkeypatch.setenv(
        "AGENTLODGE_WORKERS_JSON",
        json.dumps(
            [
                {
                    "id": spec.worker_id,
                    "capabilities": list(spec.capabilities),
                    "task_dir": str(spec.task_dir),
                }
                for spec in specs
            ]
        ),
    )
    monkeypatch.setattr(warm_render, "on_pod", lambda: False)
    monkeypatch.setattr(
        fk,
        "save_poses_npz",
        lambda _motion, path: Path(path).write_bytes(b"poses"),
    )

    encoded = {}

    def fake_encode(shards, output, **kwargs):
        encoded["shards"] = list(shards)
        encoded.update(kwargs)
        Path(output).write_bytes(b"video")
        return True

    monkeypatch.setattr(rendering, "_ffmpeg_shards", fake_encode)

    assert rendering._render_warm_local(
        "distributed",
        np.zeros((10, 139), dtype=np.float32),
        tmp_path / "media",
        "full",
    )
    assert sorted(
        (
            payload["frame_start"],
            payload["frame_end"],
            payload["width"],
            payload["height"],
            payload["samples"],
            payload["frame_format"],
            payload["fps"],
        )
        for _worker_id, payload in calls
    ) == [
        (0, 5, 1080, 1080, 96, "tga", 30),
        (5, 10, 1080, 1080, 96, "tga", 30),
    ]
    assert [path.name for path in encoded["shards"]] == [
        "shard_000000_000005.mkv",
        "shard_000005_000010.mkv",
    ]
    assert encoded["frame_count"] == 10
    assert (tmp_path / "media" / "edited.mp4").read_bytes() == b"video"
    assert rendering._RJOBS["distributed"]["rendered_frames"] == 10

    for worker in workers:
        worker.stop()
    for thread in threads:
        thread.join(timeout=2)


def test_ffmpeg_shards_preserves_final_codec_contract(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import server.rendering as rendering

    shards = [tmp_path / "a.mkv", tmp_path / "b.mkv"]
    for shard in shards:
        shard.write_bytes(b"ffv1")
    output = tmp_path / "result.mp4"
    calls = []
    concat_text = {}

    def fake_run(command, **_kwargs):
        calls.append(command)
        concat_path = Path(command[command.index("-i") + 1])
        concat_text["value"] = concat_path.read_text()
        Path(command[-1]).write_bytes(b"video")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert rendering._ffmpeg_shards(
        shards,
        output,
        frame_count=10,
        fps=30,
    )
    command = calls[0]
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-preset") + 1] == "veryfast"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-frames:v") + 1] == "10"
    assert concat_text["value"].index("a.mkv") < concat_text["value"].index(
        "b.mkv"
    )
