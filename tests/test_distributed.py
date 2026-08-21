import hashlib
import json
import os
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _render_provenance():
    from server.distributed.render_contract import RENDER_CONTRACT_VERSION

    return {
        "render_contract_version": RENDER_CONTRACT_VERSION,
        "daemon_protocol_version": 6,
        "scene": {
            "blend_sha256": "1" * 64,
            "ybot_sha256": "2" * 64,
        },
        "renderer": {
            "blender_version": "4.2.3 LTS",
            "blender_daemon_sha256": "3" * 64,
            "blender_render_ybot_sha256": "4" * 64,
            "blender_studio_sha256": "5" * 64,
            "render_root_motion_sha256": "6" * 64,
        },
        "selector": None,
    }


def _daemon_attestation(payload, provenance, gpu_index=0):
    return {
        "schema_version": 1,
        "pid": 1234,
        **provenance,
        "quality": {
            key: payload[key]
            for key in (
                "width",
                "height",
                "samples",
                "engine",
                "denoise",
                "frame_format",
            )
        },
        "gpu": {
            "cuda_index": gpu_index,
            "uuid": f"GPU-test-{gpu_index}",
            "pci_bus_id": f"00000000:{gpu_index + 1:02X}:00.0",
            "selection_mode": "single-visible-gpu",
        },
    }


def _render_identity(provenance, payload):
    from server.distributed.render_contract import render_identity_digest

    return render_identity_digest(
        provenance,
        {
            key: payload[key]
            for key in (
                "width",
                "height",
                "samples",
                "engine",
                "denoise",
                "frame_format",
                "fps",
            )
        },
    )


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
    gpu_guard = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "render_worker_env.sh"
    ).read_text(encoding="utf-8")
    launcher_surface = launcher + gpu_guard

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

    assert "AGENTLODGE_GPU_INDEX" in launcher_surface
    assert "export CUDA_VISIBLE_DEVICES" in launcher_surface
    assert "AGENTLODGE_EGL_SELECTOR_SHIM" in launcher_surface
    assert "AGENTLODGE_RENDER_DAEMON_ROOT" in launcher_surface
    assert "agentlodge_configure_gpu" in launcher
    assert "AGENTLODGE_DISTRIBUTED_TRANSPORT" in launcher
    assert "AGENTLODGE_HTTP_COORDINATOR_URL" in launcher
    assert "AGENTLODGE_HTTP_TOKEN_FILE" in launcher
    assert "AGENTLODGE_HTTP_WORKER_SCRATCH" in launcher


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


def test_jukebox_preload_syncs_the_runtime_models(tmp_path):
    from server.distributed.handlers import JukeboxExtractHandler

    edge = tmp_path / "edge"
    shared = tmp_path / "shared"
    edge.mkdir()
    shared.mkdir()
    models = (object(), object())
    runtime = SimpleNamespace(
        VQVAE=None,
        TOP_PRIOR=None,
        setup_models=lambda: models,
    )
    package = SimpleNamespace(VQVAE=None, TOP_PRIOR=None)
    handler = JukeboxExtractHandler(
        edge_root=edge,
        shared_root=shared,
        preload_audio_seconds=0,
    )
    handler._extractor = lambda _path: pytest.fail("no warm-up was requested")
    handler._jukebox_lib = runtime
    handler._jukebox_package = package

    handler.preload()

    assert (runtime.VQVAE, runtime.TOP_PRIOR) == models
    assert (package.VQVAE, package.TOP_PRIOR) == models


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
    with pytest.raises(ValueError, match="different request"):
        coordinator.submit(
            "echo.task",
            {"value": 7.0},
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


def test_filesystem_reassignment_stays_within_eligible_cohort(tmp_path):
    from server.distributed.coordinator import FileTaskCoordinator
    from server.distributed.registry import WorkerRegistry, WorkerSpec
    from server.distributed.tasks import PROTOCOL_VERSION

    workers = [
        WorkerSpec(
            worker_id=worker_id,
            capabilities=("echo.task",),
            task_dir=tmp_path / worker_id,
        )
        for worker_id in ("selected", "eligible-new", "incompatible-old")
    ]
    for worker in workers:
        worker.task_dir.mkdir()
        worker.heartbeat_path.write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_id": worker.worker_id,
                    "capabilities": ["echo.task"],
                    "status": "ready",
                    "updated_at": time.time(),
                }
            )
        )
    selected, eligible_new, incompatible_old = workers
    coordinator = FileTaskCoordinator(
        WorkerRegistry(workers),
        poll_interval=0.01,
        heartbeat_max_age=1,
    )
    handle = coordinator.submit(
        "echo.task",
        {"value": 23},
        worker=selected,
        eligible_worker_ids=("selected", "eligible-new"),
    )
    heartbeat = json.loads(selected.heartbeat_path.read_text())
    heartbeat["updated_at"] = time.time() - 30
    selected.heartbeat_path.write_text(json.dumps(heartbeat))

    reassigned = coordinator._safe_reassign(handle)

    assert reassigned is not None
    assert reassigned.worker == eligible_new
    assert not (
        incompatible_old.task_dir
        / "requests"
        / f"{handle.request.task_id}.json"
    ).exists()


def test_render_worker_selection_rejects_mixed_provenance_cohorts():
    import server.rendering as rendering

    current = _render_provenance()
    upgraded = {
        **current,
        "scene": {
            **current["scene"],
            "blend_sha256": "f" * 64,
        },
    }
    workers = [
        SimpleNamespace(
            worker_id="old",
            metadata={"render_provenance": current},
        ),
        SimpleNamespace(
            worker_id="new-1",
            metadata={"render_provenance": upgraded},
        ),
        SimpleNamespace(
            worker_id="new-2",
            metadata={"render_provenance": upgraded},
        ),
    ]
    quality = rendering._render_quality_contract(
        width=1080,
        height=1080,
        samples=96,
        engine="eevee",
        denoise=1,
        frame_format="tga",
        fps=30,
    )

    selected, eligible, provenance, _digest = (
        rendering._select_render_worker_cohort(
            workers,
            requested_workers=2,
            quality=quality,
        )
    )
    assert [worker.worker_id for worker in selected] == ["new-1", "new-2"]
    assert eligible == ("new-1", "new-2")
    assert provenance == upgraded
    with pytest.raises(RuntimeError, match="mixed provenance/quality"):
        rendering._select_render_worker_cohort(
            workers,
            requested_workers=3,
            quality=quality,
        )


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


def test_jukebox_handler_disables_repeated_cuda_cache_flushes(tmp_path):
    from server.distributed.handlers import JukeboxExtractHandler

    shared = tmp_path / "shared"
    edge = tmp_path / "edge"
    shared.mkdir()
    edge.mkdir()
    wav = shared / "song_slice0.wav"
    wav.write_bytes(b"wav")
    output = shared / "cache" / "song_slice0.npy"
    calls = []
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)

    runtime = SimpleNamespace(
        load_audio=lambda path: calls.append(("load", path)) or np.ones(8),
        extract=lambda **kwargs: calls.append(("extract", kwargs)) or {66: expected},
    )
    handler = JukeboxExtractHandler(
        edge_root=edge,
        shared_root=shared,
    )
    handler._extractor = lambda _path: pytest.fail("wrapper extraction was used")
    handler._jukebox_lib = runtime
    handler._jukebox_package = SimpleNamespace()
    payload = {"items": [{"wav": str(wav), "output": str(output)}]}

    result = handler(payload)

    np.testing.assert_array_equal(np.load(output), expected)
    assert result["cached"] == 0
    assert calls[0] == ("load", str(wav.resolve()))
    assert calls[1][0] == "extract"
    assert calls[1][1]["layers"] == [66]
    assert calls[1][1]["downsample_target_rate"] == 30
    assert calls[1][1]["force_empty_cache"] is False


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

    with pytest.raises(ValueError, match="outside the allowed roots"):
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


def test_jukebox_handler_accepts_configured_local_scratch(tmp_path):
    from server.distributed.handlers import JukeboxExtractHandler

    shared = tmp_path / "shared"
    scratch = tmp_path / "scratch"
    edge = tmp_path / "edge"
    shared.mkdir()
    scratch.mkdir()
    edge.mkdir()
    wav = scratch / "song_slice0.wav"
    wav.write_bytes(b"wav")
    output = scratch / "cache" / "song_slice0.npy"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    handler = JukeboxExtractHandler(
        edge_root=edge,
        shared_root=shared,
        scratch_root=scratch,
        extractor=lambda _path: (expected, ""),
    )

    result = handler(
        {"items": [{"wav": str(wav), "output": str(output)}]}
    )

    assert result["cached"] == 0
    np.testing.assert_array_equal(np.load(output), expected)


def test_audio_preprocess_handler_saves_exact_lodge_features(tmp_path, monkeypatch):
    from agentlodge.audio import preprocess
    from server.distributed.handlers import AudioPreprocessHandler

    shared = tmp_path / "shared"
    lodge = shared / "LODGE"
    shared.mkdir()
    lodge.mkdir()
    wav = shared / "song.wav"
    wav.write_bytes(b"wav")
    output = shared / "features.npy"
    expected = np.arange(70, dtype=np.float32).reshape(2, 35)
    monkeypatch.setattr(
        preprocess,
        "extract_lodge_features",
        lambda path, root: (
            expected
            if path == wav.resolve() and root == lodge.resolve()
            else pytest.fail("unexpected audio preprocessing input")
        ),
    )
    handler = AudioPreprocessHandler(
        mode="lodge",
        shared_root=shared,
        lodge_root=lodge,
    )

    result = handler(
        {
            "wav": str(wav),
            "output": str(output),
            "work_dir": str(shared / "work"),
        }
    )

    assert result["shape"] == [2, 35]
    assert result["dtype"] == "float32"
    assert np.array_equal(np.load(output), expected)


def test_audio_edge_preprocess_uses_and_cleans_local_scratch(
    tmp_path, monkeypatch
):
    from agentlodge.audio import preprocess
    from server.distributed.handlers import AudioPreprocessHandler

    shared = tmp_path / "shared"
    scratch = tmp_path / "scratch"
    edge = shared / "EDGE"
    shared.mkdir()
    scratch.mkdir()
    edge.mkdir()
    wav = shared / "song.wav"
    wav.write_bytes(b"wav")
    output = shared / "features.npy"
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    observed = {}

    def fake_extract(path, root, work_dir):
        observed["path"] = path
        observed["root"] = root
        observed["work_dir"] = work_dir
        assert work_dir.parent == scratch.resolve()
        return list(expected)

    monkeypatch.setattr(preprocess, "extract_edge_slices", fake_extract)
    handler = AudioPreprocessHandler(
        mode="edge",
        shared_root=shared,
        edge_root=edge,
        scratch_root=scratch,
    )

    result = handler(
        {
            "wav": str(wav),
            "output": str(output),
            "work_dir": str(shared / "network-work"),
        }
    )

    assert result["shape"] == [2, 3, 4]
    assert observed["path"] == wav.resolve()
    assert observed["root"] == edge.resolve()
    assert not observed["work_dir"].exists()
    np.testing.assert_array_equal(np.load(output), expected)


def test_dance_generation_handler_preserves_timing_context(tmp_path, monkeypatch):
    from server.distributed.handlers import DanceGenerationHandler

    shared = tmp_path / "shared"
    sid = "song_123"
    wav = shared / "LODGE/data/finedance/music_wav" / f"{sid}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    np.save(shared / f"lodge_fd_{sid}_feats.npy", np.zeros((2, 35)))
    np.save(shared / f"edge{sid}_slices.npy", np.zeros((2, 3)))
    timing = shared / "upload" / "timings.tsv"
    handler = DanceGenerationHandler(shared_root=shared)
    observed = {}
    sequence = []

    def fake_generate(value):
        sequence.append("generate")
        observed["sid"] = value
        observed["timing"] = os.environ.get("MAESTRO_TIMING_FILE")
        np.save(
            shared / f"fd_{sid}_STORY_bestofk.npy",
            np.zeros((4, 139), dtype=np.float32),
        )
        return {"frames": 4, "best_of_k": 1, "generation_workers": {}}

    def fake_build(value, bank_k, **kwargs):
        sequence.append("bank")
        observed["bank"] = {
            "sid": value,
            "bank_k": bank_k,
            "timing": os.environ.get("MAESTRO_TIMING_FILE"),
            **kwargs,
        }
        return {"sid": value, "bank_k": bank_k, "files": ["one", "two"]}

    handler._generate_song = fake_generate
    handler._build_bank = fake_build
    monkeypatch.setenv("MAESTRO_TIMING_FILE", "original")

    result = handler({"sid": sid, "timing_file": str(timing)})

    assert result["frames"] == 4
    assert sequence == ["generate", "bank"]
    assert observed == {
        "sid": sid,
        "timing": str(timing.resolve()),
        "bank": {
            "sid": sid,
            "bank_k": 1,
            "timing": str(timing.resolve()),
            "workspace": shared.resolve(),
            "distributed": True,
        },
    }
    assert os.environ["MAESTRO_TIMING_FILE"] == "original"


def test_dance_generation_handler_builds_bank_with_resident_workers(tmp_path):
    from server.distributed.handlers import DanceGenerationHandler

    shared = tmp_path / "shared"
    shared.mkdir()
    handler = DanceGenerationHandler(shared_root=shared)
    observed = {}

    def fake_build(sid, bank_k, **kwargs):
        observed.update({"sid": sid, "bank_k": bank_k, **kwargs})
        return {"sid": sid, "bank_k": bank_k, "files": ["one", "two"]}

    handler._build_bank = fake_build

    result = handler(
        {
            "operation": "build_bank",
            "sid": "song_123",
            "bank_k": 4,
        }
    )

    assert result["files"] == ["one", "two"]
    assert observed == {
        "sid": "song_123",
        "bank_k": 4,
        "workspace": shared.resolve(),
        "distributed": True,
    }


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


def test_distributed_jukebox_client_skips_local_model_import(tmp_path, monkeypatch):
    import scripts.jukebox_extract_all as client

    edge_root = tmp_path / "edge"
    module_root = edge_root / "data" / "audio_extraction"
    module_root.mkdir(parents=True)
    (module_root / "jukebox_features.py").write_text(
        "raise RuntimeError('local Jukebox model imported')\n",
        encoding="utf-8",
    )
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()
    (slice_dir / "song_slice0.wav").write_bytes(b"wav")
    cache_dir = tmp_path / "cache"

    def fake_extract(wav_slices, output_dir):
        for wav_slice in wav_slices:
            np.save(
                output_dir / f"{wav_slice.stem}.npy",
                np.zeros((1, 1), dtype=np.float32),
            )

    monkeypatch.setattr(client, "capability_enabled", lambda _capability: True)
    monkeypatch.setattr(client, "_extract_distributed", fake_extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jukebox_extract_all.py",
            "--edge-root",
            str(edge_root),
            "--slice-dir",
            str(slice_dir),
            "--cache-dir",
            str(cache_dir),
        ],
    )
    previous_directory = Path.cwd()
    previous_path = list(sys.path)
    try:
        assert client.main() == 0
    finally:
        os.chdir(previous_directory)
        sys.path[:] = previous_path


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
    provenance = _render_provenance()
    monkeypatch.setattr(
        warm_render,
        "render_provenance",
        lambda: provenance,
    )
    monkeypatch.setattr(
        warm_render,
        "daemon_attestation",
        lambda _daemon, **_kwargs: _daemon_attestation(payload, provenance),
    )
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
        "fps": 30,
        "timeout": 60,
        "render_contract_version": provenance["render_contract_version"],
        "render_provenance": provenance,
    }
    payload["render_identity_digest"] = _render_identity(provenance, payload)

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
    provenance = _render_provenance()
    decoded_rgb_hash = "c" * 64
    monkeypatch.setattr(warm_render, "render_provenance", lambda: provenance)
    source_digest_calls = []

    def fake_source_digest(*_args, **kwargs):
        source_digest_calls.append(dict(kwargs))
        return decoded_rgb_hash

    monkeypatch.setattr(
        handlers,
        "source_sequence_rgb_sha256",
        fake_source_digest,
    )
    probes = []

    def fake_probe(*_args, **kwargs):
        probes.append(dict(kwargs))
        return {
            "codec": "ffv1",
            "width": kwargs["width"],
            "height": kwargs["height"],
            "fps": kwargs["fps"],
            "frames": kwargs["frame_end"] - kwargs["frame_start"],
        }

    monkeypatch.setattr(handlers, "probe_ffv1_shard", fake_probe)
    handler = handlers.RenderFramesHandler(
        shared_root=shared,
        local_tmp=worker_tmp,
    )
    payload = {
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
        "render_contract_version": provenance["render_contract_version"],
        "render_provenance": provenance,
    }
    payload["render_identity_digest"] = _render_identity(provenance, payload)
    monkeypatch.setattr(
        warm_render,
        "daemon_attestation",
        lambda _daemon, **_kwargs: _daemon_attestation(payload, provenance),
    )
    result = handler(payload)

    assert shard.read_bytes() == b"ffv1"
    assert result["transport"] == "ffv1"
    assert result["frames"] == 3
    assert len(result["source_frames_sha256"]) == 64
    assert len(result["shard_sha256"]) == 64
    assert result["source_decoded_rgb_sha256"] == decoded_rgb_hash
    assert result["shard_decoded_rgb_sha256"] == decoded_rgb_hash
    assert result["shard_validation"]["decoded_rgb_sha256"] == decoded_rgb_hash
    assert result["shard_validation"]["worker_shard_full_decode"] is False
    assert len(source_digest_calls) == 1
    assert probes == [
        {
            "frame_start": 0,
            "frame_end": 3,
            "width": 1080,
            "height": 1080,
            "fps": 30,
        }
    ]
    assert packaged["frame_start"] == 0
    assert packaged["frame_end"] == 3
    assert not packaged["frames_dir"].exists()


def test_render_handler_falls_back_from_insufficient_shm(
    tmp_path,
    monkeypatch,
):
    from server.distributed import handlers

    shm_root = tmp_path / "shm"
    local_tmp = shm_root / "render-worker"
    fallback_root = tmp_path / "fallback"
    fallback = fallback_root / "agentlodge-render-render-worker"
    reservation = shm_root / ".agentlodge-reservations" / "worker.reservation"
    reservation.parent.mkdir(parents=True)
    reservation.write_text(f"{os.getpid()} 1 worker\n")
    monkeypatch.setenv("AGENTLODGE_SHM_ROOT", str(shm_root))
    monkeypatch.setenv("AGENTLODGE_SHM_RESERVATION_FILE", str(reservation))
    monkeypatch.setenv(
        "AGENTLODGE_RENDER_FALLBACK_ROOT",
        str(fallback_root),
    )
    monkeypatch.setattr(
        handlers.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=(
                10 * 1024**3
                if Path(path).resolve().is_relative_to(fallback.resolve())
                else 1024
            )
        ),
    )
    handler = handlers.RenderFramesHandler(
        shared_root=tmp_path,
        local_tmp=local_tmp,
        worker_id="render-worker",
    )

    selected, used_fallback = handler._scratch_parent(512 * 1024**2)

    assert used_fallback
    assert selected == fallback.resolve()


def test_render_handler_rejects_enospc_before_starting_blender(
    tmp_path,
    monkeypatch,
):
    from server import warm_render
    from server.distributed import handlers

    shared = tmp_path / "shared"
    shared.mkdir()
    poses = shared / "poses.npz"
    poses.write_bytes(b"poses")
    shm_root = tmp_path / "shm"
    local_tmp = shm_root / "render-worker"
    fallback_root = tmp_path / "fallback"
    fallback = fallback_root / "agentlodge-render-render-worker"
    reservation = shm_root / ".agentlodge-reservations" / "worker.reservation"
    reservation.parent.mkdir(parents=True)
    reservation.write_text(f"{os.getpid()} {10 * 1024**3} worker\n")
    monkeypatch.setenv("AGENTLODGE_SHM_ROOT", str(shm_root))
    monkeypatch.setenv("AGENTLODGE_SHM_RESERVATION_FILE", str(reservation))
    monkeypatch.setenv(
        "AGENTLODGE_RENDER_FALLBACK_ROOT",
        str(fallback_root),
    )
    monkeypatch.setattr(
        handlers.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )
    provenance = _render_provenance()
    monkeypatch.setattr(warm_render, "render_provenance", lambda: provenance)
    blender_calls = []
    monkeypatch.setattr(
        warm_render,
        "ensure_pool",
        lambda **_kwargs: blender_calls.append("ensure") or 1,
    )
    monkeypatch.setattr(
        warm_render,
        "warm_render",
        lambda *_args, **_kwargs: blender_calls.append("render") or True,
    )
    handler = handlers.RenderFramesHandler(
        shared_root=shared,
        local_tmp=local_tmp,
        worker_id="render-worker",
    )

    with pytest.raises(RuntimeError, match="both shared memory"):
        payload = {
            "poses": str(poses),
            "shard_output": str(shared / "shard.mkv"),
            "frame_start": 0,
            "frame_end": 10,
            "width": 1080,
            "height": 1080,
            "samples": 96,
            "engine": "eevee",
            "denoise": 1,
            "frame_format": "tga",
            "fps": 30,
            "render_contract_version": provenance[
                "render_contract_version"
            ],
            "render_provenance": provenance,
        }
        payload["render_identity_digest"] = _render_identity(
            provenance,
            payload,
        )
        handler(payload)
    assert blender_calls == []


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
    provenance = _render_provenance()
    decoded_rgb_hash = "c" * 64
    specs = [
        WorkerSpec(
            worker_id=f"render-{index}",
            capabilities=("render.frames",),
            task_dir=tmp_path / f"render-{index}",
            metadata={"render_provenance": provenance},
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
                "shard_sha256": hashlib.sha256(b"ffv1").hexdigest(),
                "transport": "ffv1",
                "render_contract_version": provenance[
                    "render_contract_version"
                ],
                "render_provenance": provenance,
                "render_identity_digest": payload["render_identity_digest"],
                "daemon_attestation": _daemon_attestation(
                    payload,
                    provenance,
                ),
                "source_decoded_rgb_sha256": decoded_rgb_hash,
                "shard_decoded_rgb_sha256": decoded_rgb_hash,
                "decoded_rgb_digest_version": "rgb24-global-frame-v1",
                "shard_validation": {
                    "codec": "ffv1",
                    "width": payload["width"],
                    "height": payload["height"],
                    "fps": payload["fps"],
                    "frames": (
                        payload["frame_end"] - payload["frame_start"]
                    ),
                    "decoded_rgb_digest_version": "rgb24-global-frame-v1",
                    "decoded_rgb_sha256": decoded_rgb_hash,
                    "worker_validation_version": (
                        "source-rgb-digest+ffprobe-v1"
                    ),
                    "worker_shard_full_decode": False,
                },
                **{
                    key: payload[key]
                    for key in (
                        "width",
                        "height",
                        "samples",
                        "engine",
                        "denoise",
                        "frame_format",
                        "fps",
                    )
                },
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
                    "metadata": {"render_provenance": provenance},
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
    coordinator_decodes = []

    def fake_inspect(path, **kwargs):
        coordinator_decodes.append((Path(path), dict(kwargs)))
        return {"decoded_rgb_sha256": decoded_rgb_hash}

    monkeypatch.setattr(rendering, "inspect_ffv1_shard", fake_inspect)

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
    assert [
        (details["frame_start"], details["frame_end"])
        for _path, details in coordinator_decodes
    ] == [(0, 5), (5, 10)]

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
