import hashlib
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest


def _render_provenance(version=None):
    from server.distributed.render_contract import RENDER_CONTRACT_VERSION

    return {
        "render_contract_version": version or RENDER_CONTRACT_VERSION,
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
            "render_root_motion_sha256": "8" * 64,
        },
        "selector": {
            "version": 2,
            "build_id": "sha256:" + "6" * 64,
            "binary_sha256": "7" * 64,
        },
    }


def _daemon_attestation(payload, provenance, *, gpu_index=0):
    attestation = {
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
            "selection_mode": "egl-cuda-device-nv",
        },
    }
    if provenance["selector"] is not None:
        attestation["selector"] = {
            **provenance["selector"],
            "requested_cuda_index": gpu_index,
            "selected_cuda_index": gpu_index,
            "egl_device_index": 1 - gpu_index,
        }
    return attestation


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return
        except Exception:
            pass
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


@contextmanager
def _running_coordinator(tmp_path, *, lease_seconds=0.5):
    from server.distributed import HttpCoordinatorStore, create_http_server

    token = "test-http-transport-secret"
    store = HttpCoordinatorStore(
        tmp_path / "state",
        tmp_path / "artifacts",
        default_lease_seconds=lease_seconds,
        minimum_lease_seconds=0.05,
        worker_max_age=5,
    )
    server = create_http_server(
        "127.0.0.1",
        0,
        token=token,
        store=store,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", token, store
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _heartbeat(client, worker_id, capability="echo.task"):
    client.heartbeat(
        worker_id,
        (capability,),
        status="ready",
        active_task="",
        metadata={"test": True},
    )


def test_http_transport_rejects_bad_auth_and_protocol(tmp_path):
    from server.distributed import (
        HttpTransportClient,
        HttpTransportError,
        PROTOCOL_VERSION,
    )

    with _running_coordinator(tmp_path) as (url, token, _store):
        wrong = HttpTransportClient(
            url,
            "wrong-secret",
            scratch_root=tmp_path / "wrong",
        )
        with pytest.raises(HttpTransportError) as rejected:
            wrong.list_workers("echo.task", max_age_seconds=5)
        assert rejected.value.status == 401
        assert rejected.value.code == "unauthorized"

        request = urllib.request.Request(
            f"{url}/v1/workers?capability=echo.task",
            headers={
                "Authorization": f"Bearer {token}",
                "X-AgentLODGE-Protocol": str(PROTOCOL_VERSION + 1),
            },
        )
        with pytest.raises(urllib.error.HTTPError) as mismatch:
            urllib.request.urlopen(request, timeout=2)
        assert mismatch.value.code == 426


def test_http_artifacts_verify_upload_download_and_tampering(tmp_path):
    from server.distributed import (
        ArtifactRef,
        HttpTaskCoordinator,
        HttpTransportError,
    )

    with _running_coordinator(tmp_path) as (url, token, store):
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator-scratch",
            poll_interval=0.01,
        )
        source = coordinator.scratch_root / "source.bin"
        source.write_bytes(b"verified artifact")
        reference = coordinator.upload_input(
            source,
            artifact_key="test-input:verified",
        )
        assert reference.sha256 == hashlib.sha256(
            b"verified artifact"
        ).hexdigest()
        assert reference.size == len(b"verified artifact")
        with pytest.raises(ValueError, match="scratch root"):
            coordinator.client.download_artifact(
                reference,
                tmp_path / "outside.bin",
            )
        with pytest.raises(ValueError, match="artifact id"):
            ArtifactRef.from_dict(
                {"artifact_id": "https://untrusted.example/file"}
            )

        bad_reference = coordinator.client.mint_artifact(
            artifact_key="test-input:wrong-hash",
            purpose="input",
            expected_sha256="0" * 64,
            expected_size=source.stat().st_size,
        )
        with pytest.raises(HttpTransportError) as local_mismatch:
            coordinator.client.upload_artifact(bad_reference, source)
        assert local_mismatch.value.code == "hash_mismatch"

        store.artifact_path(reference.artifact_id).write_bytes(b"tampered")
        with pytest.raises(HttpTransportError) as tampered:
            coordinator.client.download_artifact(
                reference,
                coordinator.scratch_root / "download.bin",
            )
        assert tampered.value.code == "artifact_tampered"


def test_http_tasks_dedupe_and_reject_task_id_collisions(tmp_path):
    from server.distributed import (
        HttpTaskCoordinator,
        HttpTaskWorker,
        HttpTransportError,
    )

    with _running_coordinator(tmp_path) as (url, token, _store):
        calls = []

        def echo(payload):
            calls.append(payload["value"])
            return {"value": payload["value"]}

        worker = HttpTaskWorker(
            "echo-0",
            ("echo.task",),
            {"echo.task": echo},
            base_url=url,
            token=token,
            scratch_root=tmp_path / "worker",
            poll_interval=0.01,
            heartbeat_interval=0.05,
            lease_seconds=0.5,
        )
        thread = threading.Thread(target=worker.run_forever, daemon=True)
        thread.start()
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator",
            poll_interval=0.01,
        )
        _wait_for(lambda: bool(coordinator.require_workers("echo.task")))

        first = coordinator.submit("echo.task", {"value": 7})
        repeated = coordinator.submit("echo.task", {"value": 7})
        assert first.request.task_id == repeated.request.task_id
        assert coordinator.wait(first, timeout=3).output == {"value": 7}
        assert coordinator.wait(repeated, timeout=3).output == {"value": 7}
        assert calls == [7]

        with pytest.raises(HttpTransportError) as collision:
            coordinator.submit(
                "echo.task",
                {"value": 8},
                task_id=first.request.task_id,
            )
        assert collision.value.code == "task_collision"
        worker.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_http_failed_task_requires_explicit_identical_retry(tmp_path):
    from server.distributed import (
        HttpTaskCoordinator,
        HttpTaskWorker,
        TaskExecutionError,
    )

    with _running_coordinator(tmp_path) as (url, token, _store):
        calls = []

        def flaky(payload):
            calls.append(payload["value"])
            if len(calls) == 1:
                raise RuntimeError("transient render failure")
            return {"value": payload["value"]}

        worker = HttpTaskWorker(
            "retry-0",
            ("echo.task",),
            {"echo.task": flaky},
            base_url=url,
            token=token,
            scratch_root=tmp_path / "worker",
            poll_interval=0.01,
            heartbeat_interval=0.05,
            lease_seconds=0.5,
        )
        thread = threading.Thread(target=worker.run_forever, daemon=True)
        thread.start()
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator",
            poll_interval=0.01,
        )
        _wait_for(lambda: bool(coordinator.require_workers("echo.task")))

        first = coordinator.submit("echo.task", {"value": 9})
        with pytest.raises(TaskExecutionError, match="transient render failure"):
            coordinator.wait(first, timeout=3)
        terminal = coordinator.submit("echo.task", {"value": 9})
        with pytest.raises(TaskExecutionError, match="transient render failure"):
            coordinator.wait(terminal, timeout=1)
        assert calls == [9]

        retried = coordinator.submit(
            "echo.task",
            {"value": 9},
            retry_failed=True,
        )
        result = coordinator.wait(retried, timeout=3)
        assert result.output == {"value": 9}
        assert calls == [9, 9]
        assert coordinator.client.task_status(retried.request.task_id)[
            "attempts"
        ] == 2
        worker.stop()
        thread.join(timeout=2)


def test_render_contract_version_changes_task_identity():
    from server.distributed.render_contract import RENDER_CONTRACT_VERSION
    from server.distributed.tasks import deterministic_task_id

    payload = {
        "frame_start": 0,
        "frame_end": 10,
        "scene": "scene-sha256",
        "renderer": "renderer-sha256",
        "task_protocol_version": 1,
        "artifact_transport": "http-v1",
    }
    previous = deterministic_task_id(
        "render.frames",
        {**payload, "render_contract_version": "render.frames-ffv1-v2"},
    )
    current = deterministic_task_id(
        "render.frames",
        {**payload, "render_contract_version": RENDER_CONTRACT_VERSION},
    )

    assert RENDER_CONTRACT_VERSION == "render.frames-ffv1-v3"
    assert previous != current
    assert current != deterministic_task_id(
        "render.frames",
        {
            **payload,
            "task_protocol_version": 2,
            "render_contract_version": RENDER_CONTRACT_VERSION,
        },
    )
    assert current != deterministic_task_id(
        "render.frames",
        {
            **payload,
            "artifact_transport": "http-v2",
            "render_contract_version": RENDER_CONTRACT_VERSION,
        },
    )


def test_http_lease_expiry_reassigns_to_another_worker(tmp_path):
    from server.distributed import (
        HttpTaskCoordinator,
        HttpTransportClient,
        HttpTransportError,
    )

    with _running_coordinator(
        tmp_path,
        lease_seconds=0.12,
    ) as (url, token, _store):
        first_client = HttpTransportClient(
            url,
            token,
            scratch_root=tmp_path / "first",
        )
        second_client = HttpTransportClient(
            url,
            token,
            scratch_root=tmp_path / "second",
        )
        _heartbeat(first_client, "worker-1")
        _heartbeat(second_client, "worker-2")
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator",
            poll_interval=0.01,
        )
        handle = coordinator.submit("echo.task", {"value": 3})

        first_claim = first_client.claim("worker-1", lease_seconds=0.12)
        assert first_claim is not None
        first_lease = first_claim["lease"]
        time.sleep(0.18)
        second_claim = second_client.claim("worker-2", lease_seconds=0.5)
        assert second_claim is not None
        assert second_claim["request"]["task_id"] == handle.request.task_id
        assert second_claim["lease"]["worker_id"] == "worker-2"
        assert second_claim["lease"]["attempt"] == 2

        from server.distributed.http_transport import TaskLease

        with pytest.raises(HttpTransportError) as lost:
            first_client.renew(
                TaskLease(
                    task_id=first_lease["task_id"],
                    worker_id=first_lease["worker_id"],
                    token=first_lease["token"],
                    expires_at=first_lease["expires_at"],
                    attempt=first_lease["attempt"],
                ),
                lease_seconds=0.5,
            )
        assert lost.value.code == "lease_lost"


def test_http_claims_and_reassignment_stay_within_eligible_cohort(tmp_path):
    from server.distributed import HttpTaskCoordinator, HttpTransportClient

    with _running_coordinator(
        tmp_path,
        lease_seconds=0.12,
    ) as (url, token, _store):
        clients = {
            worker_id: HttpTransportClient(
                url,
                token,
                scratch_root=tmp_path / worker_id,
            )
            for worker_id in ("old-worker", "new-worker-1", "new-worker-2")
        }
        for worker_id, client in clients.items():
            client.heartbeat(
                worker_id,
                ("render.frames",),
                status="ready",
                active_task="",
                metadata={
                    "render_identity_digest": (
                        "old-provenance"
                        if worker_id == "old-worker"
                        else "new-provenance"
                    )
                },
            )
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator",
            poll_interval=0.01,
        )
        workers = {
            worker.worker_id: worker
            for worker in coordinator.require_workers("render.frames")
        }
        handle = coordinator.submit(
            "render.frames",
            {"render_identity_digest": "new-provenance"},
            worker=workers["new-worker-1"],
            eligible_worker_ids=("new-worker-1", "new-worker-2"),
        )

        assert clients["old-worker"].claim(
            "old-worker",
            lease_seconds=0.12,
        ) is None
        first = clients["new-worker-1"].claim(
            "new-worker-1",
            lease_seconds=0.12,
        )
        assert first is not None
        assert first["request"]["task_id"] == handle.request.task_id
        time.sleep(0.18)
        assert clients["old-worker"].claim(
            "old-worker",
            lease_seconds=0.12,
        ) is None
        reassigned = clients["new-worker-2"].claim(
            "new-worker-2",
            lease_seconds=0.5,
        )
        assert reassigned is not None
        assert reassigned["request"]["task_id"] == handle.request.task_id
        assert reassigned["lease"]["attempt"] == 2


def test_http_retry_classification_and_bounded_status_polling(tmp_path):
    from server.distributed import HttpTaskCoordinator, HttpTransportError
    from server.distributed.http_transport import (
        HttpTaskHandle,
        _retryable_http_error,
    )
    from server.distributed.tasks import TaskRequest, TaskResult

    assert _retryable_http_error(HttpTransportError("network"))
    assert _retryable_http_error(
        HttpTransportError("unavailable", status=503, code="internal_error")
    )
    assert not _retryable_http_error(
        HttpTransportError("invalid", status=400, code="invalid_request")
    )
    assert not _retryable_http_error(
        HttpTransportError("mismatch", code="artifact_mismatch")
    )

    coordinator = HttpTaskCoordinator(
        "http://127.0.0.1:1",
        "token",
        tmp_path / "coordinator",
        poll_interval=0.001,
        request_timeout=1,
    )
    request = TaskRequest.create("echo.task", {"value": 1})
    handle = HttpTaskHandle(request)
    result = TaskResult(
        task_id=request.task_id,
        kind=request.kind,
        worker_id="worker-0",
        status="succeeded",
        started_at=1,
        finished_at=2,
        output={"value": 1},
    )
    calls = []

    status_timeouts = []

    def transient_status(_task_id, *, request_timeout=None):
        calls.append("status")
        status_timeouts.append(request_timeout)
        if len(calls) == 1:
            raise HttpTransportError(
                "temporary",
                status=503,
                code="internal_error",
            )
        return {"status": "succeeded", "result": result.to_dict()}

    coordinator.client.task_status = transient_status
    progress_states = []
    assert coordinator.wait(
        handle,
        timeout=1,
        on_poll=lambda: progress_states.append(
            coordinator.is_complete(handle)
        ),
    ).output == {"value": 1}
    assert calls == ["status", "status"]
    assert progress_states == [False]
    assert all(
        timeout is not None and 0 < timeout <= 1
        for timeout in status_timeouts
    )

    calls.clear()

    def terminal_status(_task_id, *, request_timeout=None):
        calls.append("status")
        raise HttpTransportError(
            "invalid",
            status=400,
            code="invalid_task",
        )

    coordinator.client.task_status = terminal_status
    with pytest.raises(HttpTransportError) as terminal:
        coordinator.wait(handle, timeout=1)
    assert terminal.value.code == "invalid_task"
    assert calls == ["status"]


def test_render_artifact_retries_survive_resets_without_rerender(
    tmp_path,
    monkeypatch,
):
    from server.distributed import (
        ARTIFACT_TRANSPORT,
        ArtifactRef,
        HttpTaskCoordinator,
        HttpTaskWorker,
        HttpTransportError,
    )

    with _running_coordinator(
        tmp_path,
        lease_seconds=2.0,
    ) as (url, token, _store):
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator",
            poll_interval=0.01,
        )
        source = tmp_path / "poses.npz"
        source.write_bytes(b"poses")
        input_ref = coordinator.upload_input(
            source,
            artifact_key="retry-input",
        )
        task_id = "render-transfer-retry"
        output_ref = coordinator.reserve_output(
            artifact_key="retry-output",
            task_id=task_id,
        )
        renders = []

        def render(payload):
            renders.append(dict(payload))
            shard = Path(payload["shard_output"])
            shard.write_bytes(b"rendered-once")
            return {
                "shard_output": str(shard),
                "shard_sha256": hashlib.sha256(
                    b"rendered-once"
                ).hexdigest(),
            }

        worker = HttpTaskWorker(
            "render-retry-worker",
            ("render.frames",),
            {"render.frames": render},
            base_url=url,
            token=token,
            scratch_root=tmp_path / "worker",
            lease_seconds=2.0,
        )
        worker.set_status("ready")
        spec = coordinator.require_workers("render.frames")[0]
        handle = coordinator.submit(
            "render.frames",
            {
                "artifact_transport": ARTIFACT_TRANSPORT,
                "poses_artifact": input_ref.to_dict(),
                "shard_artifact": output_ref.to_dict(),
            },
            worker=spec,
            task_id=task_id,
            eligible_worker_ids=(spec.worker_id,),
        )

        real_download = worker.client.download_artifact
        real_upload = worker.client.upload_artifact
        attempts = {"download": 0, "upload": 0}

        def flaky_download(*args, **kwargs):
            attempts["download"] += 1
            if attempts["download"] == 1:
                raise HttpTransportError("simulated TCP reset")
            return real_download(*args, **kwargs)

        def response_lost_upload(*args, **kwargs):
            attempts["upload"] += 1
            uploaded = real_upload(*args, **kwargs)
            if attempts["upload"] == 1:
                raise HttpTransportError("simulated lost upload response")
            return uploaded

        monkeypatch.setattr(
            worker.client,
            "download_artifact",
            flaky_download,
        )
        monkeypatch.setattr(
            worker.client,
            "upload_artifact",
            response_lost_upload,
        )

        assert worker.run_once()
        result = coordinator.wait(handle, timeout=3)
        artifact = ArtifactRef.from_dict(
            result.output["shard_artifact"],
            require_complete=True,
        )
        destination = tmp_path / "downloaded.mkv"
        coordinator.download_output(artifact, destination)

        assert destination.read_bytes() == b"rendered-once"
        assert len(renders) == 1
        assert attempts == {"download": 2, "upload": 2}
        assert not (worker.scratch_root / "tasks" / task_id).exists()


def test_render_artifact_permanent_4xx_stops_without_retry(
    tmp_path,
    monkeypatch,
):
    from server.distributed import (
        ARTIFACT_TRANSPORT,
        HttpTaskCoordinator,
        HttpTaskWorker,
        HttpTransportError,
        TaskExecutionError,
    )

    with _running_coordinator(
        tmp_path,
        lease_seconds=2.0,
    ) as (url, token, _store):
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator",
            poll_interval=0.01,
        )
        source = tmp_path / "poses.npz"
        source.write_bytes(b"poses")
        input_ref = coordinator.upload_input(
            source,
            artifact_key="terminal-input",
        )
        task_id = "render-terminal-upload"
        output_ref = coordinator.reserve_output(
            artifact_key="terminal-output",
            task_id=task_id,
        )
        rendered_paths = []

        def render(payload):
            shard = Path(payload["shard_output"])
            shard.write_bytes(b"rendered")
            rendered_paths.append(shard)
            return {
                "shard_output": str(shard),
                "shard_sha256": hashlib.sha256(b"rendered").hexdigest(),
            }

        worker = HttpTaskWorker(
            "render-terminal-worker",
            ("render.frames",),
            {"render.frames": render},
            base_url=url,
            token=token,
            scratch_root=tmp_path / "worker",
            lease_seconds=2.0,
        )
        worker.set_status("ready")
        spec = coordinator.require_workers("render.frames")[0]
        handle = coordinator.submit(
            "render.frames",
            {
                "artifact_transport": ARTIFACT_TRANSPORT,
                "poses_artifact": input_ref.to_dict(),
                "shard_artifact": output_ref.to_dict(),
            },
            worker=spec,
            task_id=task_id,
            eligible_worker_ids=(spec.worker_id,),
        )
        attempts = {"upload": 0}

        def reject_upload(*_args, **_kwargs):
            attempts["upload"] += 1
            assert rendered_paths[-1].is_file()
            raise HttpTransportError(
                "artifact rejected",
                status=409,
                code="artifact_mismatch",
            )

        monkeypatch.setattr(
            worker.client,
            "upload_artifact",
            reject_upload,
        )

        assert worker.run_once()
        with pytest.raises(TaskExecutionError, match="artifact rejected"):
            coordinator.wait(handle, timeout=3)
        assert attempts["upload"] == 1
        assert len(rendered_paths) == 1
        assert not rendered_paths[0].exists()


def test_worker_renewal_and_completion_retry_only_transient_errors(tmp_path):
    from server.distributed import HttpTaskWorker, HttpTransportError
    from server.distributed.http_transport import (
        TaskLease,
        _LeaseState,
    )
    from server.distributed.tasks import TaskResult

    worker = HttpTaskWorker(
        "worker-0",
        ("echo.task",),
        {"echo.task": lambda payload: payload},
        base_url="http://127.0.0.1:1",
        token="token",
        scratch_root=tmp_path / "worker",
        lease_seconds=1,
    )
    lease = TaskLease(
        task_id="echo-task-1234",
        worker_id="worker-0",
        token="lease-token",
        expires_at=time.time() + 2,
        attempt=1,
    )
    renew_stop = threading.Event()
    lost = threading.Event()
    renew_calls = []
    renew_timeouts = []

    def transient_renew(
        _lease,
        *,
        lease_seconds,
        request_timeout=None,
    ):
        renew_calls.append(lease_seconds)
        renew_timeouts.append(request_timeout)
        if len(renew_calls) == 1:
            raise HttpTransportError("network")
        renew_stop.set()
        return time.time() + 2

    worker.client.renew = transient_renew
    worker._renew_loop(
        lease,
        renew_stop,
        lost,
        _LeaseState(lease.expires_at),
    )
    assert renew_calls == [1, 1]
    assert all(
        timeout is not None and 0 < timeout <= 2
        for timeout in renew_timeouts
    )
    assert not lost.is_set()

    result = TaskResult(
        task_id=lease.task_id,
        kind="echo.task",
        worker_id=lease.worker_id,
        status="succeeded",
        started_at=1,
        finished_at=2,
        output={"ok": True},
    )
    completion_calls = []
    completion_timeouts = []

    def transient_complete(
        _lease,
        _result,
        *,
        request_timeout=None,
    ):
        completion_calls.append("complete")
        completion_timeouts.append(request_timeout)
        if len(completion_calls) == 1:
            raise HttpTransportError(
                "temporary",
                status=503,
                code="internal_error",
            )
        return {}

    worker.client.complete = transient_complete
    assert worker._complete_with_retry(
        lease,
        result,
        threading.Event(),
        _LeaseState(time.time() + 2),
    )
    assert completion_calls == ["complete", "complete"]
    assert all(
        timeout is not None and 0 < timeout <= 2
        for timeout in completion_timeouts
    )

    completion_calls.clear()

    def terminal_complete(
        _lease,
        _result,
        *,
        request_timeout=None,
    ):
        completion_calls.append("complete")
        raise HttpTransportError(
            "artifact mismatch",
            status=409,
            code="artifact_mismatch",
        )

    worker.client.complete = terminal_complete
    assert not worker._complete_with_retry(
        lease,
        result,
        threading.Event(),
        _LeaseState(time.time() + 2),
    )
    assert completion_calls == ["complete"]


def test_transient_renewal_failure_does_not_duplicate_execution(tmp_path):
    from server.distributed import (
        HttpTaskCoordinator,
        HttpTaskWorker,
        HttpTransportError,
    )

    with _running_coordinator(
        tmp_path,
        lease_seconds=0.6,
    ) as (url, token, _store):
        calls = []

        def slow(payload):
            calls.append(payload["value"])
            time.sleep(0.35)
            return {"value": payload["value"]}

        worker = HttpTaskWorker(
            "slow-0",
            ("echo.task",),
            {"echo.task": slow},
            base_url=url,
            token=token,
            scratch_root=tmp_path / "worker",
            poll_interval=0.01,
            heartbeat_interval=0.05,
            lease_seconds=0.6,
        )
        real_renew = worker.client.renew
        renew_calls = []

        def flaky_renew(
            lease,
            *,
            lease_seconds,
            request_timeout=None,
        ):
            renew_calls.append(lease.task_id)
            if len(renew_calls) == 1:
                raise HttpTransportError("temporary network loss")
            return real_renew(
                lease,
                lease_seconds=lease_seconds,
                request_timeout=request_timeout,
            )

        worker.client.renew = flaky_renew
        thread = threading.Thread(target=worker.run_forever, daemon=True)
        thread.start()
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator",
            poll_interval=0.01,
        )
        _wait_for(lambda: bool(coordinator.require_workers("echo.task")))
        handle = coordinator.submit("echo.task", {"value": 5})
        assert coordinator.wait(handle, timeout=3).output == {"value": 5}
        assert calls == [5]
        assert len(renew_calls) >= 2
        assert coordinator.client.task_status(handle.request.task_id)[
            "attempts"
        ] == 1
        worker.stop()
        thread.join(timeout=2)


def test_worker_heartbeat_status_updates_cannot_arrive_out_of_order(tmp_path):
    from server.distributed import HttpTaskWorker

    worker = HttpTaskWorker(
        "ordered-heartbeat-0",
        ("echo.task",),
        {"echo.task": lambda payload: payload},
        base_url="http://127.0.0.1:1",
        token="token",
        scratch_root=tmp_path / "worker",
    )
    busy_started = threading.Event()
    release_busy = threading.Event()
    calls = []

    def heartbeat(
        _worker_id,
        _capabilities,
        *,
        status,
        active_task,
        metadata,
    ):
        del active_task, metadata
        if status == "busy":
            busy_started.set()
            assert release_busy.wait(2)
        calls.append(status)
        return {}

    worker.client.heartbeat = heartbeat
    with worker._state_lock:
        worker._status = "busy"
        worker._active_task = "echo-task-1234"
    first = threading.Thread(target=worker._heartbeat)
    first.start()
    assert busy_started.wait(2)
    ready = threading.Thread(target=worker.set_status, args=("ready",))
    ready.start()
    time.sleep(0.05)
    release_busy.set()
    first.join(timeout=2)
    ready.join(timeout=2)

    assert not first.is_alive()
    assert not ready.is_alive()
    assert calls == ["busy", "ready"]


def test_http_completion_is_idempotent_and_provenance_checked(tmp_path):
    from server.distributed import HttpTaskCoordinator, HttpTransportClient
    from server.distributed.http_transport import TaskLease
    from server.distributed.tasks import TaskResult
    from server.distributed import HttpTransportError

    with _running_coordinator(tmp_path) as (url, token, _store):
        client = HttpTransportClient(
            url,
            token,
            scratch_root=tmp_path / "worker",
        )
        _heartbeat(client, "worker-1")
        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator",
        )
        handle = coordinator.submit("echo.task", {"value": 4})
        claim = client.claim("worker-1", lease_seconds=1)
        assert claim is not None
        raw_lease = claim["lease"]
        lease = TaskLease(
            task_id=raw_lease["task_id"],
            worker_id=raw_lease["worker_id"],
            token=raw_lease["token"],
            expires_at=raw_lease["expires_at"],
            attempt=raw_lease["attempt"],
        )
        result = TaskResult(
            task_id=handle.request.task_id,
            kind="echo.task",
            worker_id="worker-1",
            status="succeeded",
            started_at=10.0,
            finished_at=11.0,
            output={"value": 4},
        )
        first = client.complete(lease, result)
        second = client.complete(lease, result)
        assert first["result"] == second["result"]

        conflicting = TaskResult(
            task_id=handle.request.task_id,
            kind="echo.task",
            worker_id="worker-1",
            status="succeeded",
            started_at=10.0,
            finished_at=11.0,
            output={"value": 5},
        )
        with pytest.raises(HttpTransportError) as collision:
            client.complete(lease, conflicting)
        assert collision.value.code == "completion_collision"


def test_http_render_uses_exact_ranges_and_verified_artifacts(
    tmp_path,
    monkeypatch,
):
    import server.fk as fk
    import server.rendering as rendering
    import server.warm_render as warm_render
    from server.distributed import HttpTaskCoordinator, HttpTaskWorker

    with _running_coordinator(tmp_path) as (url, token, _store):
        calls = []
        workers = []
        threads = []
        provenance = _render_provenance()
        decoded_rgb_hash = "c" * 64
        for index in range(2):
            worker_root = tmp_path / f"worker-{index}"

            def render(
                payload,
                worker_id=f"render-{index}",
                root=worker_root,
                gpu_index=index,
            ):
                poses = Path(payload["poses"]).resolve()
                shard = Path(payload["shard_output"]).resolve()
                assert poses.is_relative_to(root.resolve())
                assert shard.is_relative_to(root.resolve())
                assert poses.read_bytes() == b"poses"
                content = (
                    f"{payload['frame_start']}:{payload['frame_end']}"
                ).encode()
                shard.parent.mkdir(parents=True, exist_ok=True)
                shard.write_bytes(content)
                calls.append(
                    (
                        worker_id,
                        payload["frame_start"],
                        payload["frame_end"],
                    )
                )
                return {
                    "frame_start": payload["frame_start"],
                    "frame_end": payload["frame_end"],
                    "frames": payload["frame_end"] - payload["frame_start"],
                    "source_frames_sha256": hashlib.sha256(
                        b"source-" + content
                    ).hexdigest(),
                    "shard_output": str(shard),
                    "shard_sha256": hashlib.sha256(content).hexdigest(),
                    "transport": "ffv1",
                    "render_contract_version": provenance[
                        "render_contract_version"
                    ],
                    "render_provenance": provenance,
                    "render_identity_digest": payload[
                        "render_identity_digest"
                    ],
                    "daemon_attestation": _daemon_attestation(
                        payload,
                        provenance,
                        gpu_index=gpu_index,
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
                        "decoded_rgb_digest_version": (
                            "rgb24-global-frame-v1"
                        ),
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

            worker = HttpTaskWorker(
                f"render-{index}",
                ("render.frames",),
                {"render.frames": render},
                base_url=url,
                token=token,
                scratch_root=worker_root,
                poll_interval=0.01,
                heartbeat_interval=0.05,
                lease_seconds=2.0,
                metadata={"render_provenance": provenance},
            )
            thread = threading.Thread(target=worker.run_forever, daemon=True)
            thread.start()
            workers.append(worker)
            threads.append(thread)

        coordinator = HttpTaskCoordinator(
            url,
            token,
            tmp_path / "coordinator-scratch",
            poll_interval=0.01,
        )
        _wait_for(
            lambda: len(coordinator.require_workers("render.frames")) == 2
        )
        monkeypatch.setenv("AGENTLODGE_DISTRIBUTED", "1")
        monkeypatch.setenv(
            "AGENTLODGE_DISTRIBUTED_CAPABILITIES",
            "render.frames",
        )
        monkeypatch.setenv("AGENTLODGE_DISTRIBUTED_TRANSPORT", "http")
        monkeypatch.setenv("AGENTLODGE_HTTP_COORDINATOR_URL", url)
        monkeypatch.setenv("AGENTLODGE_HTTP_TOKEN", token)
        monkeypatch.setenv(
            "AGENTLODGE_HTTP_COORDINATOR_SCRATCH",
            str(coordinator.scratch_root),
        )
        monkeypatch.setenv("AGENTLODGE_FULL_RENDER_WORKERS", "2")
        monkeypatch.setenv("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "5")
        monkeypatch.setattr(warm_render, "on_pod", lambda: False)
        monkeypatch.setattr(
            fk,
            "save_poses_npz",
            lambda _motion, path: Path(path).write_bytes(b"poses"),
        )
        encoded = {}

        def fake_encode(shards, output, **kwargs):
            encoded["contents"] = [Path(path).read_bytes() for path in shards]
            encoded.update(kwargs)
            Path(output).write_bytes(b"video")
            return True

        monkeypatch.setattr(rendering, "_ffmpeg_shards", fake_encode)
        monkeypatch.setattr(
            rendering,
            "inspect_ffv1_shard",
            lambda *_args, **_kwargs: {
                "decoded_rgb_sha256": decoded_rgb_hash
            },
        )
        assert rendering._render_warm_local(
            "http-render",
            np.zeros((10, 139), dtype=np.float32),
            tmp_path / "media",
            "full",
        )
        assert sorted((start, end) for _worker, start, end in calls) == [
            (0, 5),
            (5, 10),
        ]
        assert encoded["contents"] == [b"0:5", b"5:10"]
        assert encoded["frame_count"] == 10
        job = rendering._RJOBS["http-render"]
        assert len(job["render_source_frame_hashes"]) == 2
        assert len(job["render_shard_sha256"]) == 2
        assert (tmp_path / "media" / "edited.mp4").read_bytes() == b"video"

        for cached in (tmp_path / "media" / ".render_cache").glob("*.mp4"):
            cached.unlink()
        (tmp_path / "media" / "edited.mp4").unlink()
        assert rendering._render_warm_local(
            "http-render-retry",
            np.zeros((10, 139), dtype=np.float32),
            tmp_path / "media",
            "full",
        )
        assert len(calls) == 2
        assert (tmp_path / "media" / "edited.mp4").read_bytes() == b"video"

        for cached in (tmp_path / "media" / ".render_cache").glob("*.mp4"):
            cached.unlink()
        (tmp_path / "media" / "edited.mp4").unlink()
        monkeypatch.setattr(
            rendering,
            "inspect_ffv1_shard",
            lambda *_args, **_kwargs: {
                "decoded_rgb_sha256": "d" * 64
            },
        )
        with pytest.raises(RuntimeError, match="decoded RGB hash mismatch"):
            rendering._render_warm_local(
                "http-render-tampered-decode",
                np.zeros((10, 139), dtype=np.float32),
                tmp_path / "media",
                "full",
            )

        for worker in workers:
            worker.stop()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()


def test_http_only_render_does_not_require_legacy_pod_host(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import server.rendering as rendering

    monkeypatch.setenv("AGENTLODGE_DISTRIBUTED", "1")
    monkeypatch.setenv(
        "AGENTLODGE_DISTRIBUTED_CAPABILITIES",
        "render.frames",
    )
    monkeypatch.setenv("AGENTLODGE_DISTRIBUTED_TRANSPORT", "http")
    monkeypatch.delenv("AGENTLODGE_POD_HOST", raising=False)
    monkeypatch.setattr(
        rendering,
        "pod_config",
        lambda: SimpleNamespace(host="", port=22, ws="/workspace"),
    )
    calls = []

    def remote_render(sid, motion, media_dir, scope, *, audio_wav=None):
        calls.append((sid, len(motion), Path(media_dir), scope, audio_wav))
        Path(media_dir).mkdir(parents=True, exist_ok=True)
        (Path(media_dir) / "edited.mp4").write_bytes(b"video")
        return True

    monkeypatch.setattr(rendering, "_render_warm_local", remote_render)
    rendering._RJOBS["http-only"] = {"started": time.time()}
    rendering._render(
        "http-only",
        np.zeros((4, 139), dtype=np.float32),
        tmp_path / "media",
        "full",
        None,
        None,
    )

    assert calls and calls[0][3] == "full"
    assert rendering._RJOBS["http-only"]["status"] == "done"


def test_render_validation_rejects_gaps_hashes_and_transport_regressions(
    monkeypatch,
):
    from server.distributed.runtime import distributed_transport
    from server.rendering import _validate_render_output, _validate_render_ranges

    monkeypatch.delenv("AGENTLODGE_DISTRIBUTED_TRANSPORT", raising=False)
    assert distributed_transport("render.frames") == "filesystem"
    with pytest.raises(RuntimeError, match="not contiguous"):
        _validate_render_ranges([(0, 2), (3, 5)], 5)

    provenance = _render_provenance()
    decoded_rgb_hash = "c" * 64
    valid = {
        "frame_start": 0,
        "frame_end": 2,
        "frames": 2,
        "width": 1080,
        "height": 1080,
        "samples": 96,
        "engine": "eevee",
        "denoise": 1,
        "frame_format": "tga",
        "fps": 30,
        "transport": "ffv1",
        "render_contract_version": provenance["render_contract_version"],
        "render_provenance": provenance,
        "source_frames_sha256": "a" * 64,
        "shard_sha256": "b" * 64,
        "source_decoded_rgb_sha256": decoded_rgb_hash,
        "shard_decoded_rgb_sha256": decoded_rgb_hash,
        "decoded_rgb_digest_version": "rgb24-global-frame-v1",
        "shard_validation": {
            "codec": "ffv1",
            "width": 1080,
            "height": 1080,
            "fps": 30,
            "frames": 2,
            "decoded_rgb_digest_version": "rgb24-global-frame-v1",
            "decoded_rgb_sha256": decoded_rgb_hash,
            "worker_validation_version": (
                "source-rgb-digest+ffprobe-v1"
            ),
            "worker_shard_full_decode": False,
        },
    }
    from server.distributed.render_contract import render_identity_digest

    valid["render_identity_digest"] = render_identity_digest(
        provenance,
        {
            "width": 1080,
            "height": 1080,
            "samples": 96,
            "engine": "eevee",
            "denoise": 1,
            "frame_format": "tga",
            "fps": 30,
        },
    )
    valid["daemon_attestation"] = _daemon_attestation(valid, provenance)
    _validate_render_output(
        valid,
        start=0,
        end=2,
        width=1080,
        height=1080,
        samples=96,
        engine="eevee",
        denoise=1,
        frame_format="tga",
        fps=30,
        render_provenance=provenance,
    )
    with pytest.raises(RuntimeError, match="hashes are invalid"):
        _validate_render_output(
            {**valid, "shard_sha256": "not-a-hash"},
            start=0,
            end=2,
            width=1080,
            height=1080,
            samples=96,
            engine="eevee",
            denoise=1,
            frame_format="tga",
            fps=30,
            render_provenance=provenance,
        )
