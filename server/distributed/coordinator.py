"""Idempotent filesystem task submission over a shared RunPod volume."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from server.distributed.registry import WorkerRegistry, WorkerSpec
from server.distributed.tasks import TaskRequest, TaskResult, canonical_json


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
    eligible_worker_ids: tuple[str, ...] = ()

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
        configured_state = os.environ.get(
            "AGENTLODGE_DISTRIBUTED_STATE",
            "",
        ).strip()
        if configured_state:
            self.state_dir = Path(configured_state).resolve()
        else:
            parents = {
                worker.task_dir.resolve().parent
                for worker in self.registry.workers
            }
            if not parents:
                self.state_dir = Path(
                    os.environ.get(
                        "AGENTLODGE_SHARED_ROOT",
                        Path.cwd(),
                    )
                ).resolve() / "maestro-workers" / "_coordinator"
            elif len(parents) == 1:
                self.state_dir = next(iter(parents)) / "_coordinator"
            else:
                common = Path(
                    os.path.commonpath([str(path) for path in parents])
                )
                worker_paths = sorted(
                    str(worker.task_dir.resolve())
                    for worker in self.registry.workers
                )
                fingerprint = hashlib.sha256(
                    "\n".join(worker_paths).encode("utf-8")
                ).hexdigest()[:12]
                self.state_dir = common / f".maestro-coordinator-{fingerprint}"
        for name in ("locks", "tasks", "cancelled"):
            (self.state_dir / name).mkdir(parents=True, exist_ok=True)

    def _worker_load(self, worker: WorkerSpec) -> int:
        return sum(
            len(list((worker.task_dir / name).glob("*.json")))
            for name in ("requests", "claimed")
        )

    def _choose_worker(
        self,
        capability: str,
        task_id: str,
        *,
        exclude: set[str] | None = None,
        eligible_worker_ids: tuple[str, ...] = (),
    ) -> WorkerSpec:
        excluded = exclude or set()
        eligible = set(eligible_worker_ids)
        workers = [
            worker
            for worker in self.registry.require(
                capability,
                max_age_seconds=self.heartbeat_max_age,
            )
            if worker.worker_id not in excluded
            and (not eligible or worker.worker_id in eligible)
        ]
        if not workers:
            raise RuntimeError(
                f"no healthy replacement workers advertise {capability!r}"
            )
        return min(
            workers,
            key=lambda worker: (
                self._worker_load(worker),
                hashlib.sha256(
                    f"{task_id}\n{worker.worker_id}".encode("utf-8")
                ).hexdigest(),
            ),
        )

    @contextmanager
    def _task_lock(self, task_id: str):
        lock_path = self.state_dir / "locks" / f"{task_id}.lock"
        deadline = time.monotonic() + 10.0
        descriptor = None
        while descriptor is None:
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > 60.0:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring task lock for {task_id}"
                    )
                time.sleep(0.02)
        try:
            os.write(
                descriptor,
                f"{os.getpid()} {time.time()}\n".encode("ascii"),
            )
            os.close(descriptor)
            descriptor = None
            yield
        finally:
            if descriptor is not None:
                os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    def _record_task(self, request: TaskRequest) -> None:
        record_path = self.state_dir / "tasks" / f"{request.task_id}.json"
        if record_path.exists():
            try:
                existing = TaskRequest.from_dict(
                    json.loads(record_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"task record is invalid for {request.task_id}"
                ) from exc
            if (
                existing.kind != request.kind
                or canonical_json(existing.payload)
                != canonical_json(request.payload)
            ):
                raise ValueError(
                    f"task id {request.task_id} already names a different request"
                )
            return
        _atomic_json(record_path, request.to_dict())

    def _existing_handle(
        self,
        request: TaskRequest,
        eligible_worker_ids: tuple[str, ...] = (),
    ) -> TaskHandle | None:
        matches: list[TaskHandle] = []
        eligible = set(eligible_worker_ids)
        for worker in self.registry.workers:
            paths = (
                worker.task_dir / "results" / f"{request.task_id}.json",
                worker.task_dir / "claimed" / f"{request.task_id}.json",
                worker.task_dir / "requests" / f"{request.task_id}.json",
            )
            for path in paths:
                if not path.exists():
                    continue
                if path.parent.name == "results":
                    try:
                        result = TaskResult.from_dict(
                            json.loads(path.read_text(encoding="utf-8"))
                        )
                    except (
                        OSError,
                        json.JSONDecodeError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise RuntimeError(
                            f"task result is invalid for {request.task_id}"
                        ) from exc
                    if (
                        result.task_id != request.task_id
                        or result.kind != request.kind
                        or result.worker_id != worker.worker_id
                    ):
                        raise RuntimeError(
                            f"task result provenance mismatch for {request.task_id}"
                        )
                else:
                    try:
                        queued = TaskRequest.from_dict(
                            json.loads(path.read_text(encoding="utf-8"))
                        )
                    except (
                        OSError,
                        json.JSONDecodeError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise RuntimeError(
                            f"queued task is invalid for {request.task_id}"
                        ) from exc
                    if (
                        queued.kind != request.kind
                        or canonical_json(queued.payload)
                        != canonical_json(request.payload)
                    ):
                        raise ValueError(
                            f"task id {request.task_id} has conflicting payloads"
                        )
                    if eligible and worker.worker_id not in eligible:
                        raise RuntimeError(
                            f"queued task {request.task_id} is assigned outside "
                            "its eligible worker cohort"
                        )
                matches.append(
                    TaskHandle(
                        request=request,
                        worker=worker,
                        eligible_worker_ids=eligible_worker_ids,
                    )
                )
                break
        unique = {
            handle.worker.worker_id: handle
            for handle in matches
        }
        if len(unique) > 1:
            raise RuntimeError(
                f"task {request.task_id} exists on multiple workers: "
                f"{', '.join(sorted(unique))}"
            )
        return next(iter(unique.values()), None)

    def _queue_request(
        self,
        request: TaskRequest,
        worker: WorkerSpec,
        eligible_worker_ids: tuple[str, ...] = (),
    ) -> TaskHandle:
        request_path = (
            worker.task_dir / "requests" / f"{request.task_id}.json"
        )
        _atomic_json(request_path, request.to_dict())
        return TaskHandle(
            request=request,
            worker=worker,
            eligible_worker_ids=eligible_worker_ids,
        )

    def _safe_reassign(self, handle: TaskHandle) -> TaskHandle | None:
        task_id = handle.request.task_id
        with self._task_lock(task_id):
            result_path = handle.result_path
            if result_path.exists():
                return handle
            claimed_path = (
                handle.worker.task_dir / "claimed" / f"{task_id}.json"
            )
            if claimed_path.exists():
                return None
            existing = self._existing_handle(
                handle.request,
                handle.eligible_worker_ids,
            )
            if existing is not None and existing.worker != handle.worker:
                return existing
            request_path = (
                handle.worker.task_dir / "requests" / f"{task_id}.json"
            )
            cancelled_path = (
                self.state_dir
                / "cancelled"
                / f"{task_id}.{uuid.uuid4().hex}.json"
            )
            moved = False
            if request_path.exists():
                try:
                    os.replace(request_path, cancelled_path)
                    moved = True
                except FileNotFoundError:
                    if claimed_path.exists():
                        return None
            alternatives = [
                worker
                for worker in self.registry.for_capability(
                    handle.request.kind,
                    require_healthy=True,
                    max_age_seconds=self.heartbeat_max_age,
                )
                if worker.worker_id != handle.worker.worker_id
                and (
                    not handle.eligible_worker_ids
                    or worker.worker_id in handle.eligible_worker_ids
                )
            ]
            if not alternatives:
                if moved and cancelled_path.exists():
                    os.replace(cancelled_path, request_path)
                return None
            try:
                replacement = self._choose_worker(
                    handle.request.kind,
                    task_id,
                    exclude={handle.worker.worker_id},
                    eligible_worker_ids=handle.eligible_worker_ids,
                )
                reassigned = self._queue_request(
                    handle.request,
                    replacement,
                    handle.eligible_worker_ids,
                )
            except Exception:
                if moved and cancelled_path.exists():
                    os.replace(cancelled_path, request_path)
                raise
            cancelled_path.unlink(missing_ok=True)
            return reassigned

    def _validate_worker(
        self,
        kind: str,
        selected: WorkerSpec,
        eligible_worker_ids: tuple[str, ...] = (),
    ) -> None:
        if kind not in selected.capabilities:
            raise ValueError(
                f"worker {selected.worker_id} does not advertise {kind!r}"
            )
        if not selected.is_healthy(max_age_seconds=self.heartbeat_max_age):
            raise RuntimeError(f"worker {selected.worker_id} is not healthy")
        if (
            eligible_worker_ids
            and selected.worker_id not in eligible_worker_ids
        ):
            raise ValueError(
                f"worker {selected.worker_id} is outside the eligible cohort"
            )

    def submit(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        worker: WorkerSpec | None = None,
        task_id: str | None = None,
        eligible_worker_ids: tuple[str, ...] | list[str] = (),
    ) -> TaskHandle:
        eligible = tuple(sorted({str(value) for value in eligible_worker_ids}))
        known_worker_ids = {item.worker_id for item in self.registry.workers}
        if any(value not in known_worker_ids for value in eligible):
            raise ValueError("eligible worker cohort contains an unknown worker")
        request = TaskRequest.create(kind, payload, task_id=task_id)
        with self._task_lock(request.task_id):
            self._record_task(request)
            existing = self._existing_handle(request, eligible)
            if existing is not None:
                return existing
            selected = worker or self._choose_worker(
                kind,
                request.task_id,
                eligible_worker_ids=eligible,
            )
            self._validate_worker(kind, selected, eligible)
            return self._queue_request(request, selected, eligible)

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

    def is_complete(self, handle: TaskHandle) -> bool:
        return handle.result_path.is_file()

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
                replacement = self._safe_reassign(handle)
                if replacement is None:
                    continue
                pending[task_id] = replacement
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
