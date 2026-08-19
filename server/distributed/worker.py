"""Single-process worker loop for capability-specific GPU handlers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from server.distributed.coordinator import _atomic_json
from server.distributed.registry import WorkerSpec
from server.distributed.tasks import PROTOCOL_VERSION, TaskRequest, TaskResult

TaskHandler = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
logger = logging.getLogger(__name__)


class FileTaskWorker:
    def __init__(
        self,
        spec: WorkerSpec,
        handlers: Mapping[str, TaskHandler],
        *,
        poll_interval: float = 0.1,
        heartbeat_interval: float = 2.0,
    ):
        if spec.max_concurrency != 1:
            raise ValueError(
                "FileTaskWorker serves exactly one task at a time; "
                "run multiple worker processes for additional concurrency"
            )
        missing = set(spec.capabilities) - set(handlers)
        if missing:
            raise ValueError(
                f"worker {spec.worker_id} is missing handlers for {sorted(missing)}"
            )
        self.spec = spec
        self.handlers = dict(handlers)
        self.poll_interval = max(0.02, float(poll_interval))
        self.heartbeat_interval = max(0.2, float(heartbeat_interval))
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._heartbeat_write_lock = threading.Lock()
        self._status = "starting"
        self._active_task = ""
        self._heartbeat_thread: threading.Thread | None = None
        self._pending_results: dict[Path, TaskResult] = {}
        for name in ("requests", "claimed", "results"):
            (self.spec.task_dir / name).mkdir(parents=True, exist_ok=True)
        self._recover_claimed()

    def _recover_claimed(self) -> None:
        claimed_dir = self.spec.task_dir / "claimed"
        request_dir = self.spec.task_dir / "requests"
        result_dir = self.spec.task_dir / "results"
        for claimed_path in claimed_dir.glob("*.json"):
            if (result_dir / claimed_path.name).exists():
                claimed_path.unlink(missing_ok=True)
                continue
            request_path = request_dir / claimed_path.name
            if request_path.exists():
                claimed_path.unlink(missing_ok=True)
                continue
            try:
                os.replace(claimed_path, request_path)
            except OSError as exc:
                logger.warning(
                    "could not recover claimed task %s for %s: %s",
                    claimed_path.stem,
                    self.spec.worker_id,
                    exc,
                )

    def set_status(self, status: str, active_task: str = "") -> None:
        with self._state_lock:
            self._status = status
            self._active_task = active_task
        self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        with self._state_lock:
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "worker_id": self.spec.worker_id,
                "capabilities": list(self.spec.capabilities),
                "status": self._status,
                "active_task": self._active_task,
                "pid": os.getpid(),
                "updated_at": time.time(),
                "metadata": self.spec.metadata,
            }
        with self._heartbeat_write_lock:
            for attempt in range(3):
                try:
                    _atomic_json(self.spec.heartbeat_path, payload)
                    return
                except OSError as exc:
                    if attempt == 2:
                        logger.warning(
                            "could not update heartbeat for %s: %s",
                            self.spec.worker_id,
                            exc,
                        )
                        return
                    time.sleep(0.01)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            self._write_heartbeat()

    def start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._write_heartbeat()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"heartbeat-{self.spec.worker_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.set_status("stopping")

    def _claim_next(self) -> Path | None:
        for claimed_path in sorted(self._pending_results):
            if claimed_path.exists():
                return claimed_path
            self._pending_results.pop(claimed_path, None)
        request_dir = self.spec.task_dir / "requests"
        claimed_dir = self.spec.task_dir / "claimed"
        for request_path in sorted(request_dir.glob("*.json")):
            claimed_path = claimed_dir / request_path.name
            try:
                os.replace(request_path, claimed_path)
                return claimed_path
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return None

    def _process(self, claimed_path: Path) -> None:
        result = self._pending_results.get(claimed_path)
        if result is None:
            started_at = time.time()
            request: TaskRequest | None = None
            try:
                request = TaskRequest.from_dict(
                    json.loads(claimed_path.read_text(encoding="utf-8"))
                )
                result_path = (
                    self.spec.task_dir / "results" / f"{request.task_id}.json"
                )
                if result_path.exists():
                    self.set_status("ready")
                    return
                self.set_status("busy", request.task_id)
                handler = self.handlers.get(request.kind)
                if handler is None:
                    raise RuntimeError(
                        f"worker {self.spec.worker_id} cannot execute {request.kind!r}"
                    )
                output = dict(handler(request.payload) or {})
                result = TaskResult(
                    task_id=request.task_id,
                    kind=request.kind,
                    worker_id=self.spec.worker_id,
                    status="succeeded",
                    started_at=started_at,
                    finished_at=time.time(),
                    output=output,
                )
                result.validate()
            except Exception as exc:  # noqa: BLE001 - returned to the coordinator
                task_id = (
                    request.task_id
                    if request
                    else "invalid-"
                    + hashlib.sha256(
                        claimed_path.name.encode("utf-8")
                    ).hexdigest()[:32]
                )
                kind = request.kind if request else "invalid.task"
                result = TaskResult(
                    task_id=task_id,
                    kind=kind,
                    worker_id=self.spec.worker_id,
                    status="failed",
                    started_at=started_at,
                    finished_at=time.time(),
                    error=(
                        f"{type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()[-4000:]}"
                    ),
                )
            self._pending_results[claimed_path] = result
        result_path = (
            self.spec.task_dir / "results" / f"{result.task_id}.json"
        )
        last_error: OSError | None = None
        for attempt in range(5):
            try:
                _atomic_json(result_path, result.to_dict())
                self._pending_results.pop(claimed_path, None)
                self.set_status("ready")
                return
            except OSError as exc:
                last_error = exc
                time.sleep(min(1.0, 0.05 * (2**attempt)))
        self.set_status("degraded", result.task_id)
        raise OSError(
            f"could not persist result for {result.task_id}"
        ) from last_error

    def run_once(self) -> bool:
        claimed = self._claim_next()
        if claimed is None:
            return False
        try:
            self._process(claimed)
        except Exception:
            logger.exception(
                "worker %s retained task %s after processing failure",
                self.spec.worker_id,
                claimed.stem,
            )
            self.set_status("degraded", claimed.stem)
            return False
        else:
            claimed.unlink(missing_ok=True)
            return True

    def run_forever(self) -> None:
        self.set_status("ready")
        self.start_heartbeat()
        while not self._stop.is_set():
            if not self.run_once():
                self._stop.wait(self.poll_interval)
