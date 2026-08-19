"""Idempotent filesystem task submission over a shared RunPod volume."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from server.distributed.registry import WorkerRegistry, WorkerSpec
from server.distributed.tasks import TaskRequest, TaskResult


class TaskExecutionError(RuntimeError):
    """Raised when a worker returns a failed task result."""


class TaskTimeoutError(TimeoutError):
    """Raised when distributed work does not finish within its deadline."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class TaskHandle:
    request: TaskRequest
    worker: WorkerSpec

    @property
    def result_path(self) -> Path:
        return self.worker.task_dir / "results" / f"{self.request.task_id}.json"


class FileTaskCoordinator:
    def __init__(
        self,
        registry: WorkerRegistry | None = None,
        *,
        poll_interval: float = 0.2,
        heartbeat_max_age: float = 15.0,
    ):
        self.registry = registry or WorkerRegistry.from_env()
        self.poll_interval = max(0.02, float(poll_interval))
        self.heartbeat_max_age = max(1.0, float(heartbeat_max_age))
        self._round_robin: dict[str, int] = {}

    def _choose_worker(self, capability: str) -> WorkerSpec:
        workers = self.registry.require(
            capability,
            max_age_seconds=self.heartbeat_max_age,
        )
        index = self._round_robin.get(capability, 0) % len(workers)
        self._round_robin[capability] = index + 1
        return workers[index]

    def submit(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        worker: WorkerSpec | None = None,
        task_id: str | None = None,
    ) -> TaskHandle:
        selected = worker or self._choose_worker(kind)
        if kind not in selected.capabilities:
            raise ValueError(
                f"worker {selected.worker_id} does not advertise {kind!r}"
            )
        if not selected.is_healthy(max_age_seconds=self.heartbeat_max_age):
            raise RuntimeError(f"worker {selected.worker_id} is not healthy")

        request = TaskRequest.create(kind, payload, task_id=task_id)
        handle = TaskHandle(request=request, worker=selected)
        if handle.result_path.exists():
            return handle

        request_path = (
            selected.task_dir / "requests" / f"{request.task_id}.json"
        )
        claimed_path = (
            selected.task_dir / "claimed" / f"{request.task_id}.json"
        )
        if not request_path.exists() and not claimed_path.exists():
            _atomic_json(request_path, request.to_dict())
        return handle

    def wait(
        self,
        handle: TaskHandle,
        *,
        timeout: float,
        on_poll: Callable[[], None] | None = None,
    ) -> TaskResult:
        return self.wait_many(
            [handle],
            timeout=timeout,
            on_poll=on_poll,
        )[0]

    def wait_many(
        self,
        handles: list[TaskHandle],
        *,
        timeout: float,
        on_poll: Callable[[], None] | None = None,
        max_reassignments: int = 1,
    ) -> list[TaskResult]:
        if not handles:
            return []
        deadline = time.monotonic() + max(0.1, float(timeout))
        pending = {handle.request.task_id: handle for handle in handles}
        completed: dict[str, TaskResult] = {}
        reassignments = {handle.request.task_id: 0 for handle in handles}
        while pending and time.monotonic() < deadline:
            for task_id, handle in list(pending.items()):
                if not handle.result_path.exists():
                    continue
                try:
                    raw = json.loads(handle.result_path.read_text(encoding="utf-8"))
                    result = TaskResult.from_dict(raw)
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    continue
                if result.task_id != task_id or result.kind != handle.request.kind:
                    raise TaskExecutionError(
                        f"worker {handle.worker.worker_id} returned a mismatched result "
                        f"for {task_id}"
                    )
                if result.worker_id != handle.worker.worker_id:
                    raise TaskExecutionError(
                        f"task {task_id} was assigned to {handle.worker.worker_id} "
                        f"but result provenance names {result.worker_id}"
                    )
                if result.status == "failed":
                    raise TaskExecutionError(
                        f"{result.kind} failed on {result.worker_id}: {result.error}"
                    )
                completed[task_id] = result
                del pending[task_id]
            for task_id, handle in list(pending.items()):
                if handle.worker.is_healthy(
                    max_age_seconds=self.heartbeat_max_age
                ):
                    continue
                if reassignments[task_id] >= max(0, int(max_reassignments)):
                    continue
                alternatives = [
                    worker
                    for worker in self.registry.for_capability(
                        handle.request.kind,
                        require_healthy=True,
                        max_age_seconds=self.heartbeat_max_age,
                    )
                    if worker.worker_id != handle.worker.worker_id
                ]
                if not alternatives:
                    continue
                replacement = alternatives[
                    reassignments[task_id] % len(alternatives)
                ]
                for old_path in (
                    handle.worker.task_dir / "requests" / f"{task_id}.json",
                    handle.worker.task_dir / "claimed" / f"{task_id}.json",
                ):
                    old_path.unlink(missing_ok=True)
                pending[task_id] = self.submit(
                    handle.request.kind,
                    handle.request.payload,
                    worker=replacement,
                    task_id=handle.request.task_id,
                )
                reassignments[task_id] += 1
            if not pending:
                break
            if on_poll is not None:
                on_poll()
            time.sleep(self.poll_interval)
        if pending:
            details = ", ".join(
                f"{task_id}@{handle.worker.worker_id}"
                for task_id, handle in pending.items()
            )
            raise TaskTimeoutError(f"timed out waiting for distributed tasks: {details}")
        return [completed[handle.request.task_id] for handle in handles]
