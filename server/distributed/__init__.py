"""Filesystem-backed coordination for warm, capability-scoped GPU workers."""

from server.distributed.coordinator import (
    FileTaskCoordinator,
    TaskExecutionError,
    TaskHandle,
    TaskTimeoutError,
)
from server.distributed.registry import WorkerRegistry, WorkerRegistryError, WorkerSpec
from server.distributed.runtime import capability_enabled
from server.distributed.tasks import PROTOCOL_VERSION, TaskRequest, TaskResult

__all__ = [
    "FileTaskCoordinator",
    "PROTOCOL_VERSION",
    "TaskExecutionError",
    "TaskHandle",
    "TaskRequest",
    "TaskResult",
    "TaskTimeoutError",
    "WorkerRegistry",
    "WorkerRegistryError",
    "WorkerSpec",
    "capability_enabled",
]
