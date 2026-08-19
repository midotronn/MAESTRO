"""Worker discovery and heartbeat validation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from server.distributed.tasks import PROTOCOL_VERSION


class WorkerRegistryError(RuntimeError):
    """Raised when distributed-worker configuration is missing or invalid."""


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    capabilities: tuple[str, ...]
    task_dir: Path
    max_concurrency: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WorkerSpec":
        worker_id = str(raw.get("id") or raw.get("worker_id") or "").strip()
        capabilities = tuple(
            str(value).strip()
            for value in raw.get("capabilities", [])
            if str(value).strip()
        )
        task_dir_value = str(raw.get("task_dir") or "").strip()
        try:
            max_concurrency = max(1, int(raw.get("max_concurrency") or 1))
        except (TypeError, ValueError) as exc:
            raise WorkerRegistryError(
                f"worker {worker_id or '<unknown>'} has invalid max_concurrency"
            ) from exc
        if not worker_id:
            raise WorkerRegistryError("worker is missing id")
        if not capabilities:
            raise WorkerRegistryError(f"worker {worker_id} has no capabilities")
        if not task_dir_value:
            raise WorkerRegistryError(f"worker {worker_id} is missing task_dir")
        return cls(
            worker_id=worker_id,
            capabilities=capabilities,
            task_dir=Path(task_dir_value),
            max_concurrency=max_concurrency,
            metadata=dict(raw.get("metadata") or {}),
        )

    @property
    def heartbeat_path(self) -> Path:
        return self.task_dir / "heartbeat.json"

    def heartbeat(self) -> dict[str, Any]:
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                return json.loads(
                    self.heartbeat_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                last_error = exc
                time.sleep(0.01)
        if isinstance(last_error, FileNotFoundError):
            raise WorkerRegistryError(
                f"worker {self.worker_id} has no heartbeat at {self.heartbeat_path}"
            ) from last_error
        raise WorkerRegistryError(
            f"worker {self.worker_id} heartbeat is unreadable"
        ) from last_error

    def is_healthy(self, *, max_age_seconds: float = 15.0) -> bool:
        try:
            heartbeat = self.heartbeat()
            updated_at = float(heartbeat.get("updated_at") or 0.0)
            protocol = int(heartbeat.get("protocol_version") or 0)
            status = str(heartbeat.get("status") or "")
            advertised = {
                str(value) for value in heartbeat.get("capabilities", [])
            }
        except (WorkerRegistryError, TypeError, ValueError):
            return False
        return (
            heartbeat.get("worker_id") == self.worker_id
            and protocol == PROTOCOL_VERSION
            and status in {"ready", "busy"}
            and time.time() - updated_at <= max_age_seconds
            and set(self.capabilities) <= advertised
        )


class WorkerRegistry:
    def __init__(self, workers: list[WorkerSpec]):
        ids = [worker.worker_id for worker in workers]
        if len(ids) != len(set(ids)):
            raise WorkerRegistryError("worker ids must be unique")
        self.workers = tuple(workers)

    @classmethod
    def from_dict(cls, raw: Any) -> "WorkerRegistry":
        entries = raw.get("workers", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise WorkerRegistryError("worker registry must be a list or {workers: [...]}")
        return cls([WorkerSpec.from_dict(entry) for entry in entries])

    @classmethod
    def from_env(cls) -> "WorkerRegistry":
        registry_path = os.environ.get("AGENTLODGE_WORKER_REGISTRY", "").strip()
        inline = os.environ.get("AGENTLODGE_WORKERS_JSON", "").strip()
        if registry_path:
            try:
                raw = json.loads(Path(registry_path).read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise WorkerRegistryError(
                    f"worker registry does not exist: {registry_path}"
                ) from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise WorkerRegistryError(
                    f"worker registry is invalid: {registry_path}"
                ) from exc
            return cls.from_dict(raw)
        if inline:
            try:
                return cls.from_dict(json.loads(inline))
            except json.JSONDecodeError as exc:
                raise WorkerRegistryError("AGENTLODGE_WORKERS_JSON is invalid") from exc
        return cls([])

    def for_capability(
        self,
        capability: str,
        *,
        require_healthy: bool = True,
        max_age_seconds: float = 15.0,
    ) -> list[WorkerSpec]:
        workers = [
            worker
            for worker in self.workers
            if capability in worker.capabilities
            and (
                not require_healthy
                or worker.is_healthy(max_age_seconds=max_age_seconds)
            )
        ]
        return sorted(workers, key=lambda worker: worker.worker_id)

    def require(
        self,
        capability: str,
        *,
        max_age_seconds: float = 15.0,
    ) -> list[WorkerSpec]:
        workers = self.for_capability(
            capability,
            require_healthy=True,
            max_age_seconds=max_age_seconds,
        )
        if not workers:
            raise WorkerRegistryError(
                f"no healthy workers advertise capability {capability!r}"
            )
        return workers
