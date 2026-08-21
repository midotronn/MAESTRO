"""Versioned coordination for warm, capability-scoped GPU workers."""

from server.distributed.coordinator import (
    FileTaskCoordinator,
    TaskExecutionError,
    TaskHandle,
    TaskTimeoutError,
)
from server.distributed.http_transport import (
    ARTIFACT_TRANSPORT,
    ArtifactRef,
    HttpCoordinatorStore,
    HttpTaskCoordinator,
    HttpTaskHandle,
    HttpTaskWorker,
    HttpTransportClient,
    HttpTransportError,
    HttpWorkerSpec,
    create_http_server,
    sha256_file,
)
from server.distributed.registry import WorkerRegistry, WorkerRegistryError, WorkerSpec
from server.distributed.runtime import capability_enabled, distributed_transport
from server.distributed.tasks import (
    PROTOCOL_VERSION,
    TaskRequest,
    TaskResult,
    deterministic_task_id,
)

__all__ = [
    "FileTaskCoordinator",
    "ARTIFACT_TRANSPORT",
    "ArtifactRef",
    "HttpCoordinatorStore",
    "HttpTaskCoordinator",
    "HttpTaskHandle",
    "HttpTaskWorker",
    "HttpTransportClient",
    "HttpTransportError",
    "HttpWorkerSpec",
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
    "create_http_server",
    "distributed_transport",
    "deterministic_task_id",
    "sha256_file",
]
