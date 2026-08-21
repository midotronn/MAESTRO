"""Authenticated HTTP task and artifact transport for isolated GPU workers."""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import ssl
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

from server.distributed.coordinator import TaskExecutionError, TaskTimeoutError
from server.distributed.registry import WorkerRegistryError
from server.distributed.tasks import (
    PROTOCOL_VERSION,
    TaskRequest,
    TaskResult,
    canonical_json,
)

logger = logging.getLogger(__name__)

ARTIFACT_TRANSPORT = "http-v1"
_ARTIFACT_ID_RE = re.compile(r"^artifact-[a-f0-9]{32}$")
_WORKER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_MAX_JSON_BYTES = 1024 * 1024


class HttpTransportError(RuntimeError):
    """Raised for authenticated transport, protocol, or coordinator failures."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str = "transport_error",
    ):
        super().__init__(message)
        self.status = status
        self.code = code


def _retryable_http_error(error: HttpTransportError) -> bool:
    if error.status is None:
        return error.code == "transport_error"
    return 500 <= error.status <= 599


class _StoreError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_sha256(value: str, *, label: str = "sha256") -> str:
    normalized = str(value).lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"invalid {label}")
    return normalized


def _validate_worker_id(value: str) -> str:
    worker_id = str(value).strip()
    if not _WORKER_ID_RE.fullmatch(worker_id):
        raise ValueError(f"invalid worker id: {worker_id!r}")
    return worker_id


def _validate_task_id(value: str) -> str:
    task_id = str(value).strip()
    if not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError(f"invalid task id: {task_id!r}")
    return task_id


def _normalize_eligible_worker_ids(values: Any) -> tuple[str, ...]:
    if values is None or values == "":
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("eligible_worker_ids must be a list")
    return tuple(
        sorted({_validate_worker_id(str(value)) for value in values})
    )


def _confined(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    confined_root = Path(root).resolve()
    if not resolved.is_relative_to(confined_root):
        raise ValueError(f"path is outside the configured scratch root: {resolved}")
    return resolved


def _load_token(
    token: str | None = None,
    token_file: str | Path | None = None,
) -> str:
    resolved = str(token or "").strip()
    configured_file = str(token_file or "").strip()
    if not resolved and configured_file:
        try:
            resolved = Path(configured_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise HttpTransportError(
                f"could not read HTTP transport token file: {configured_file}"
            ) from exc
    if not resolved:
        raise HttpTransportError("HTTP transport authentication token is required")
    return resolved


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    sha256: str = ""
    size: int | None = None

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        require_complete: bool = False,
    ) -> "ArtifactRef":
        artifact_id = str(raw.get("artifact_id") or "")
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ValueError("invalid coordinator artifact id")
        sha256 = str(raw.get("sha256") or "").lower()
        size_value = raw.get("size")
        size = None if size_value in {None, ""} else int(size_value)
        if sha256:
            _validate_sha256(sha256)
        if size is not None and size < 0:
            raise ValueError("artifact size must not be negative")
        if require_complete and (not sha256 or size is None):
            raise ValueError("complete artifact metadata is required")
        return cls(artifact_id=artifact_id, sha256=sha256, size=size)

    def to_dict(self) -> dict[str, Any]:
        if not _ARTIFACT_ID_RE.fullmatch(self.artifact_id):
            raise ValueError("invalid coordinator artifact id")
        payload: dict[str, Any] = {"artifact_id": self.artifact_id}
        if self.sha256:
            payload["sha256"] = _validate_sha256(self.sha256)
        if self.size is not None:
            if self.size < 0:
                raise ValueError("artifact size must not be negative")
            payload["size"] = int(self.size)
        return payload


@dataclass(frozen=True)
class HttpWorkerSpec:
    worker_id: str
    capabilities: tuple[str, ...]
    status: str
    updated_at: float
    metadata: dict[str, Any]
    max_concurrency: int = 1


@dataclass(frozen=True)
class HttpTaskHandle:
    request: TaskRequest
    worker: HttpWorkerSpec | None = None


@dataclass(frozen=True)
class TaskLease:
    task_id: str
    worker_id: str
    token: str
    expires_at: float
    attempt: int


class _LeaseState:
    def __init__(self, expires_at: float):
        self._expires_at = float(expires_at)
        self._lock = threading.Lock()
        self.operation_lock = threading.Lock()

    def update(self, expires_at: float) -> None:
        with self._lock:
            self._expires_at = max(self._expires_at, float(expires_at))

    def deadline(self, safety_margin: float) -> float:
        with self._lock:
            return self._expires_at - max(0.01, float(safety_margin))


class HttpCoordinatorStore:
    """Persistent coordinator state backed by SQLite and confined artifact files."""

    def __init__(
        self,
        state_root: Path,
        artifact_root: Path | None = None,
        *,
        default_lease_seconds: float = 30.0,
        minimum_lease_seconds: float = 0.1,
        maximum_lease_seconds: float = 3600.0,
        worker_max_age: float = 60.0,
        max_artifact_bytes: int = 16 * 1024 * 1024 * 1024,
    ):
        self.state_root = Path(state_root).resolve()
        self.artifact_root = Path(
            artifact_root or self.state_root / "artifacts"
        ).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.files_root = self.artifact_root / "files"
        self.staged_root = self.artifact_root / "staged"
        self.incoming_root = self.artifact_root / "incoming"
        for path in (self.files_root, self.staged_root, self.incoming_root):
            path.mkdir(parents=True, exist_ok=True)
        self.database_path = self.state_root / "coordinator.sqlite3"
        self.default_lease_seconds = max(
            float(minimum_lease_seconds),
            float(default_lease_seconds),
        )
        self.minimum_lease_seconds = max(0.05, float(minimum_lease_seconds))
        self.maximum_lease_seconds = max(
            self.minimum_lease_seconds,
            float(maximum_lease_seconds),
        )
        self.worker_max_age = max(1.0, float(worker_max_age))
        self.max_artifact_bytes = max(1, int(max_artifact_bytes))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    capabilities_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active_task TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    protocol_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    preferred_worker_id TEXT,
                    eligible_worker_ids_json TEXT NOT NULL DEFAULT '{"values":[]}',
                    lease_owner TEXT,
                    lease_token TEXT,
                    lease_expires_at REAL,
                    last_worker_id TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT
                );
                CREATE INDEX IF NOT EXISTS tasks_status_created
                    ON tasks(status, created_at);
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_key TEXT UNIQUE NOT NULL,
                    purpose TEXT NOT NULL,
                    task_id TEXT,
                    expected_sha256 TEXT,
                    expected_size INTEGER,
                    status TEXT NOT NULL,
                    sha256 TEXT,
                    size INTEGER,
                    staged_lease_token TEXT,
                    staged_worker_id TEXT,
                    staged_sha256 TEXT,
                    staged_size INTEGER,
                    staged_path TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS artifacts_task
                    ON artifacts(task_id);
                """
            )
            task_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(tasks)")
            }
            if "eligible_worker_ids_json" not in task_columns:
                connection.execute(
                    """
                    ALTER TABLE tasks ADD COLUMN eligible_worker_ids_json
                    TEXT NOT NULL DEFAULT '{"values":[]}'
                    """
                )

    def _artifact_path(self, artifact_id: str) -> Path:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise _StoreError(400, "invalid_artifact", "invalid artifact id")
        return self.files_root / artifact_id[:11] / f"{artifact_id}.bin"

    def artifact_path(self, artifact_id: str) -> Path:
        """Return the confined final path for diagnostics and local tests."""
        return self._artifact_path(artifact_id)

    def _lease_seconds(self, requested: Any) -> float:
        try:
            value = float(requested)
        except (TypeError, ValueError):
            value = self.default_lease_seconds
        return min(
            self.maximum_lease_seconds,
            max(self.minimum_lease_seconds, value),
        )

    @staticmethod
    def _artifact_payload(row: sqlite3.Row) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "artifact_id": row["artifact_id"],
            "purpose": row["purpose"],
            "status": row["status"],
        }
        sha256 = row["sha256"] or row["staged_sha256"]
        size = row["size"]
        if size is None:
            size = row["staged_size"]
        if sha256:
            payload["sha256"] = sha256
        if size is not None:
            payload["size"] = int(size)
        if row["task_id"]:
            payload["task_id"] = row["task_id"]
        return payload

    @staticmethod
    def _task_payload(row: sqlite3.Row) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "task_id": row["task_id"],
            "kind": row["kind"],
            "status": row["status"],
            "attempts": int(row["attempts"]),
        }
        if row["lease_owner"]:
            payload["lease_owner"] = row["lease_owner"]
            payload["lease_expires_at"] = float(row["lease_expires_at"] or 0.0)
        if row["result_json"]:
            payload["result"] = json.loads(row["result_json"])
        payload["eligible_worker_ids"] = list(
            json.loads(
                row["eligible_worker_ids_json"] or '{"values":[]}'
            ).get("values", [])
        )
        return payload

    def heartbeat(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        if int(raw.get("protocol_version") or 0) != PROTOCOL_VERSION:
            raise _StoreError(
                426,
                "protocol_mismatch",
                f"expected protocol {PROTOCOL_VERSION}",
            )
        try:
            worker_id = _validate_worker_id(str(raw.get("worker_id") or ""))
        except ValueError as exc:
            raise _StoreError(400, "invalid_worker", str(exc)) from exc
        capabilities_value = raw.get("capabilities")
        if not isinstance(capabilities_value, list) or not capabilities_value:
            raise _StoreError(
                400,
                "invalid_worker",
                "worker capabilities must be a non-empty list",
            )
        capabilities = tuple(
            sorted({str(value).strip() for value in capabilities_value})
        )
        try:
            for capability in capabilities:
                TaskRequest.create(capability, {})
        except ValueError as exc:
            raise _StoreError(400, "invalid_worker", str(exc)) from exc
        status = str(raw.get("status") or "")
        if status not in {
            "starting",
            "warming",
            "ready",
            "busy",
            "degraded",
            "stopping",
        }:
            raise _StoreError(400, "invalid_worker", "invalid worker status")
        active_task = str(raw.get("active_task") or "")
        if active_task:
            try:
                _validate_task_id(active_task)
            except ValueError as exc:
                raise _StoreError(400, "invalid_worker", str(exc)) from exc
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise _StoreError(400, "invalid_worker", "worker metadata must be an object")
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workers (
                    worker_id, capabilities_json, metadata_json, status,
                    active_task, updated_at, protocol_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    capabilities_json=excluded.capabilities_json,
                    metadata_json=excluded.metadata_json,
                    status=excluded.status,
                    active_task=excluded.active_task,
                    updated_at=excluded.updated_at,
                    protocol_version=excluded.protocol_version
                """,
                (
                    worker_id,
                    canonical_json({"values": capabilities}),
                    canonical_json(dict(metadata)),
                    status,
                    active_task,
                    now,
                    PROTOCOL_VERSION,
                ),
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "worker_id": worker_id,
            "updated_at": now,
        }

    def list_workers(
        self,
        capability: str,
        *,
        max_age_seconds: float,
    ) -> list[dict[str, Any]]:
        try:
            TaskRequest.create(capability, {})
        except ValueError as exc:
            raise _StoreError(400, "invalid_capability", str(exc)) from exc
        cutoff = time.time() - max(0.1, float(max_age_seconds))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM workers
                WHERE protocol_version = ?
                  AND status IN ('ready', 'busy')
                  AND updated_at >= ?
                ORDER BY worker_id
                """,
                (PROTOCOL_VERSION, cutoff),
            ).fetchall()
        workers = []
        for row in rows:
            capabilities = tuple(
                json.loads(row["capabilities_json"]).get("values", [])
            )
            if capability not in capabilities:
                continue
            workers.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_id": row["worker_id"],
                    "capabilities": list(capabilities),
                    "status": row["status"],
                    "updated_at": float(row["updated_at"]),
                    "metadata": json.loads(row["metadata_json"]),
                    "max_concurrency": 1,
                }
            )
        return workers

    def submit(
        self,
        request: TaskRequest,
        *,
        preferred_worker_id: str = "",
        retry_failed: bool = False,
        eligible_worker_ids: tuple[str, ...] | list[str] = (),
    ) -> dict[str, Any]:
        request.validate()
        preferred = ""
        if preferred_worker_id:
            try:
                preferred = _validate_worker_id(preferred_worker_id)
            except ValueError as exc:
                raise _StoreError(400, "invalid_worker", str(exc)) from exc
        try:
            eligible = _normalize_eligible_worker_ids(eligible_worker_ids)
        except ValueError as exc:
            raise _StoreError(400, "invalid_worker", str(exc)) from exc
        if preferred and eligible and preferred not in eligible:
            raise _StoreError(
                400,
                "invalid_worker",
                "preferred worker is outside the eligible cohort",
            )
        eligible_json = canonical_json({"values": eligible})
        payload_json = canonical_json(request.payload)
        request_json = canonical_json(request.to_dict())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (request.task_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["kind"] != request.kind
                    or existing["payload_json"] != payload_json
                ):
                    raise _StoreError(
                        409,
                        "task_collision",
                        f"task id {request.task_id} already names a different request",
                    )
                if existing["status"] == "failed" and retry_failed:
                    connection.execute(
                        """
                        UPDATE tasks SET
                            status = 'queued',
                            result_json = NULL,
                            preferred_worker_id = ?,
                            eligible_worker_ids_json = ?,
                            lease_owner = NULL,
                            lease_token = NULL,
                            lease_expires_at = NULL
                        WHERE task_id = ? AND status = 'failed'
                        """,
                        (preferred or None, eligible_json, request.task_id),
                    )
                    existing = connection.execute(
                        "SELECT * FROM tasks WHERE task_id = ?",
                        (request.task_id,),
                    ).fetchone()
                elif (
                    existing["status"] in {"queued", "leased"}
                    and existing["eligible_worker_ids_json"] != eligible_json
                ):
                    raise _StoreError(
                        409,
                        "task_policy_collision",
                        "task already has a different eligible worker cohort",
                    )
                connection.commit()
                assert existing is not None
                return self._task_payload(existing)
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, kind, payload_json, request_json, created_at,
                    status, preferred_worker_id, eligible_worker_ids_json
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    request.task_id,
                    request.kind,
                    payload_json,
                    request_json,
                    request.created_at,
                    preferred or None,
                    eligible_json,
                ),
            )
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (request.task_id,),
            ).fetchone()
            connection.commit()
            assert row is not None
            return self._task_payload(row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _expire_locked(
        self,
        connection: sqlite3.Connection,
        now: float,
    ) -> list[Path]:
        expired = connection.execute(
            """
            SELECT task_id, lease_owner, lease_token
            FROM tasks
            WHERE status = 'leased' AND lease_expires_at <= ?
            """,
            (now,),
        ).fetchall()
        stale_paths: list[Path] = []
        for row in expired:
            staged = connection.execute(
                """
                SELECT staged_path FROM artifacts
                WHERE task_id = ? AND staged_lease_token = ?
                """,
                (row["task_id"], row["lease_token"]),
            ).fetchall()
            stale_paths.extend(
                Path(value["staged_path"])
                for value in staged
                if value["staged_path"]
            )
            connection.execute(
                """
                UPDATE artifacts SET
                    staged_lease_token = NULL,
                    staged_worker_id = NULL,
                    staged_sha256 = NULL,
                    staged_size = NULL,
                    staged_path = NULL,
                    updated_at = ?
                WHERE task_id = ? AND staged_lease_token = ?
                """,
                (now, row["task_id"], row["lease_token"]),
            )
            connection.execute(
                """
                UPDATE tasks SET
                    status = 'queued',
                    last_worker_id = lease_owner,
                    lease_owner = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL
                WHERE task_id = ? AND status = 'leased'
                """,
                (row["task_id"],),
            )
        return stale_paths

    @staticmethod
    def _unlink_paths(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove stale staged artifact %s", path)

    def task_status(self, task_id: str) -> dict[str, Any]:
        try:
            task_id = _validate_task_id(task_id)
        except ValueError as exc:
            raise _StoreError(400, "invalid_task", str(exc)) from exc
        connection = self._connect()
        stale_paths: list[Path] = []
        try:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise _StoreError(404, "task_not_found", "task does not exist")
            now = time.time()
            if (
                row["status"] == "leased"
                and float(row["lease_expires_at"] or 0.0) <= now
            ):
                connection.execute("BEGIN IMMEDIATE")
                stale_paths = self._expire_locked(connection, now)
                row = connection.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                connection.commit()
                assert row is not None
            return self._task_payload(row)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
            self._unlink_paths(stale_paths)

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: Any,
    ) -> dict[str, Any] | None:
        try:
            worker_id = _validate_worker_id(worker_id)
        except ValueError as exc:
            raise _StoreError(400, "invalid_worker", str(exc)) from exc
        now = time.time()
        duration = self._lease_seconds(lease_seconds)
        connection = self._connect()
        stale_paths: list[Path] = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            stale_paths = self._expire_locked(connection, now)
            worker = connection.execute(
                "SELECT * FROM workers WHERE worker_id = ?",
                (worker_id,),
            ).fetchone()
            if worker is None:
                raise _StoreError(
                    409,
                    "worker_not_registered",
                    "worker must heartbeat before claiming tasks",
                )
            if (
                int(worker["protocol_version"]) != PROTOCOL_VERSION
                or now - float(worker["updated_at"]) > self.worker_max_age
                or worker["status"] != "ready"
            ):
                raise _StoreError(409, "worker_stale", "worker heartbeat is stale")
            capabilities = tuple(
                json.loads(worker["capabilities_json"]).get("values", [])
            )
            if not capabilities:
                raise _StoreError(409, "worker_invalid", "worker has no capabilities")
            placeholders = ",".join("?" for _ in capabilities)
            rows = connection.execute(
                f"""
                SELECT * FROM tasks
                WHERE status = 'queued' AND kind IN ({placeholders})
                ORDER BY
                    CASE
                        WHEN preferred_worker_id = ? THEN 0
                        WHEN last_worker_id IS NULL OR last_worker_id != ? THEN 1
                        ELSE 2
                    END,
                    created_at,
                    task_id
                """,
                (*capabilities, worker_id, worker_id),
            ).fetchall()
            row = None
            for candidate in rows:
                eligible = tuple(
                    json.loads(
                        candidate["eligible_worker_ids_json"]
                        or '{"values":[]}'
                    ).get("values", [])
                )
                if not eligible or worker_id in eligible:
                    row = candidate
                    break
            if row is None:
                connection.commit()
                return None
            lease_token = secrets.token_urlsafe(32)
            expires_at = now + duration
            attempt = int(row["attempts"]) + 1
            connection.execute(
                """
                UPDATE tasks SET
                    status = 'leased',
                    lease_owner = ?,
                    lease_token = ?,
                    lease_expires_at = ?,
                    attempts = ?
                WHERE task_id = ? AND status = 'queued'
                """,
                (
                    worker_id,
                    lease_token,
                    expires_at,
                    attempt,
                    row["task_id"],
                ),
            )
            connection.execute(
                """
                UPDATE workers SET status = 'busy', active_task = ?, updated_at = ?
                WHERE worker_id = ?
                """,
                (row["task_id"], now, worker_id),
            )
            connection.commit()
            return {
                "protocol_version": PROTOCOL_VERSION,
                "request": json.loads(row["request_json"]),
                "lease": {
                    "task_id": row["task_id"],
                    "worker_id": worker_id,
                    "token": lease_token,
                    "expires_at": expires_at,
                    "attempt": attempt,
                },
            }
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._unlink_paths(stale_paths)

    def renew(
        self,
        task_id: str,
        worker_id: str,
        lease_token: str,
        *,
        lease_seconds: Any,
    ) -> dict[str, Any]:
        try:
            task_id = _validate_task_id(task_id)
            worker_id = _validate_worker_id(worker_id)
        except ValueError as exc:
            raise _StoreError(400, "invalid_lease", str(exc)) from exc
        now = time.time()
        expires_at = now + self._lease_seconds(lease_seconds)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "leased"
                or row["lease_owner"] != worker_id
                or not hmac.compare_digest(
                    str(row["lease_token"] or ""),
                    str(lease_token),
                )
                or float(row["lease_expires_at"] or 0.0) <= now
            ):
                raise _StoreError(409, "lease_lost", "task lease is not active")
            connection.execute(
                "UPDATE tasks SET lease_expires_at = ? WHERE task_id = ?",
                (expires_at, task_id),
            )
            connection.execute(
                """
                UPDATE workers SET status = 'busy', active_task = ?, updated_at = ?
                WHERE worker_id = ?
                """,
                (task_id, now, worker_id),
            )
            connection.commit()
            return {
                "protocol_version": PROTOCOL_VERSION,
                "task_id": task_id,
                "expires_at": expires_at,
            }
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mint_artifact(
        self,
        *,
        artifact_key: str,
        purpose: str,
        task_id: str = "",
        expected_sha256: str = "",
        expected_size: int | None = None,
    ) -> dict[str, Any]:
        key = str(artifact_key)
        if not key or len(key) > 512:
            raise _StoreError(
                400,
                "invalid_artifact",
                "artifact key must contain 1-512 characters",
            )
        if purpose not in {"input", "output"}:
            raise _StoreError(400, "invalid_artifact", "invalid artifact purpose")
        bound_task = ""
        if task_id:
            try:
                bound_task = _validate_task_id(task_id)
            except ValueError as exc:
                raise _StoreError(400, "invalid_artifact", str(exc)) from exc
        if purpose == "output" and not bound_task:
            raise _StoreError(
                400,
                "invalid_artifact",
                "output artifacts must be bound to a task",
            )
        sha256 = ""
        size: int | None = None
        if expected_sha256:
            try:
                sha256 = _validate_sha256(expected_sha256)
            except ValueError as exc:
                raise _StoreError(400, "invalid_artifact", str(exc)) from exc
        if expected_size is not None:
            size = int(expected_size)
            if size < 0 or size > self.max_artifact_bytes:
                raise _StoreError(400, "invalid_artifact", "invalid artifact size")
        if purpose == "input" and (not sha256 or size is None):
            raise _StoreError(
                400,
                "invalid_artifact",
                "input artifacts require SHA-256 and size",
            )
        now = time.time()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_key = ?",
                (key,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["purpose"] != purpose
                    or str(existing["task_id"] or "") != bound_task
                    or str(existing["expected_sha256"] or "") != sha256
                    or (
                        existing["expected_size"]
                        if existing["expected_size"] is not None
                        else None
                    )
                    != size
                ):
                    raise _StoreError(
                        409,
                        "artifact_collision",
                        "artifact key already names different metadata",
                    )
                connection.commit()
                return self._artifact_payload(existing)
            artifact_id = f"artifact-{secrets.token_hex(16)}"
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, artifact_key, purpose, task_id,
                    expected_sha256, expected_size, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    artifact_id,
                    key,
                    purpose,
                    bound_task or None,
                    sha256 or None,
                    size,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            connection.commit()
            assert row is not None
            return self._artifact_payload(row)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def authorize_upload(
        self,
        artifact_id: str,
        *,
        worker_id: str = "",
        task_id: str = "",
        lease_token: str = "",
    ) -> dict[str, Any]:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise _StoreError(400, "invalid_artifact", "invalid artifact id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise _StoreError(404, "artifact_not_found", "artifact does not exist")
            if row["purpose"] == "output":
                try:
                    worker_id = _validate_worker_id(worker_id)
                    task_id = _validate_task_id(task_id)
                except ValueError as exc:
                    raise _StoreError(400, "invalid_lease", str(exc)) from exc
                task = connection.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if (
                    row["task_id"] != task_id
                    or task is None
                    or task["status"] != "leased"
                    or task["lease_owner"] != worker_id
                    or not hmac.compare_digest(
                        str(task["lease_token"] or ""),
                        str(lease_token),
                    )
                    or float(task["lease_expires_at"] or 0.0) <= time.time()
                ):
                    raise _StoreError(409, "lease_lost", "output upload lease is not active")
            return dict(row)

    def finish_upload(
        self,
        artifact_id: str,
        temporary_path: Path,
        *,
        sha256: str,
        size: int,
        worker_id: str = "",
        task_id: str = "",
        lease_token: str = "",
    ) -> dict[str, Any]:
        connection = self._connect()
        previous_stage: Path | None = None
        staged_path: Path | None = None
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise _StoreError(404, "artifact_not_found", "artifact does not exist")
            if row["expected_sha256"] and row["expected_sha256"] != sha256:
                raise _StoreError(409, "hash_mismatch", "artifact SHA-256 does not match")
            if (
                row["expected_size"] is not None
                and int(row["expected_size"]) != int(size)
            ):
                raise _StoreError(409, "size_mismatch", "artifact size does not match")
            now = time.time()
            if row["purpose"] == "input":
                if row["status"] == "complete":
                    if row["sha256"] != sha256 or int(row["size"]) != int(size):
                        raise _StoreError(
                            409,
                            "artifact_collision",
                            "completed artifact has different content",
                        )
                    final_path = self._artifact_path(artifact_id)
                    if not final_path.is_file():
                        raise _StoreError(
                            409,
                            "artifact_missing",
                            "completed artifact content is missing",
                        )
                    stored_sha256, stored_size = sha256_file(final_path)
                    if stored_sha256 != sha256 or stored_size != size:
                        raise _StoreError(
                            409,
                            "artifact_tampered",
                            "completed artifact content failed verification",
                        )
                    connection.commit()
                    temporary_path.unlink(missing_ok=True)
                    return self._artifact_payload(row)
                final_path = self._artifact_path(artifact_id)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_path, final_path)
                connection.execute(
                    """
                    UPDATE artifacts SET
                        status = 'complete', sha256 = ?, size = ?, updated_at = ?
                    WHERE artifact_id = ?
                    """,
                    (sha256, size, now, artifact_id),
                )
            else:
                try:
                    worker_id = _validate_worker_id(worker_id)
                    task_id = _validate_task_id(task_id)
                except ValueError as exc:
                    raise _StoreError(400, "invalid_lease", str(exc)) from exc
                task = connection.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if (
                    row["task_id"] != task_id
                    or task is None
                    or task["status"] != "leased"
                    or task["lease_owner"] != worker_id
                    or not hmac.compare_digest(
                        str(task["lease_token"] or ""),
                        str(lease_token),
                    )
                    or float(task["lease_expires_at"] or 0.0) <= now
                ):
                    raise _StoreError(409, "lease_lost", "output upload lease is not active")
                if row["status"] == "complete":
                    if row["sha256"] != sha256 or int(row["size"]) != int(size):
                        raise _StoreError(
                            409,
                            "artifact_collision",
                            "completed artifact has different content",
                        )
                    connection.commit()
                    temporary_path.unlink(missing_ok=True)
                    return self._artifact_payload(row)
                previous_stage = (
                    Path(row["staged_path"]) if row["staged_path"] else None
                )
                token_digest = hashlib.sha256(
                    lease_token.encode("utf-8")
                ).hexdigest()[:16]
                staged_path = self.staged_root / (
                    f"{artifact_id}.{token_digest}.bin"
                )
                os.replace(temporary_path, staged_path)
                connection.execute(
                    """
                    UPDATE artifacts SET
                        staged_lease_token = ?,
                        staged_worker_id = ?,
                        staged_sha256 = ?,
                        staged_size = ?,
                        staged_path = ?,
                        updated_at = ?
                    WHERE artifact_id = ?
                    """,
                    (
                        lease_token,
                        worker_id,
                        sha256,
                        size,
                        str(staged_path),
                        now,
                        artifact_id,
                    ),
                )
            updated = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            connection.commit()
            assert updated is not None
            return self._artifact_payload(updated)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
            temporary_path.unlink(missing_ok=True)
            if previous_stage is not None:
                try:
                    if previous_stage != staged_path:
                        previous_stage.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "could not remove superseded artifact stage %s",
                        previous_stage,
                    )

    def artifact_for_download(self, artifact_id: str) -> tuple[Path, ArtifactRef]:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise _StoreError(400, "invalid_artifact", "invalid artifact id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise _StoreError(404, "artifact_not_found", "artifact does not exist")
        if row["status"] != "complete":
            raise _StoreError(409, "artifact_incomplete", "artifact is not complete")
        path = self._artifact_path(artifact_id)
        if not path.is_file():
            raise _StoreError(409, "artifact_missing", "artifact content is missing")
        actual_sha256, actual_size = sha256_file(path)
        if (
            actual_sha256 != row["sha256"]
            or actual_size != int(row["size"])
        ):
            raise _StoreError(
                409,
                "artifact_tampered",
                "artifact content failed coordinator verification",
            )
        return path, ArtifactRef(artifact_id, actual_sha256, actual_size)

    def complete(
        self,
        task_id: str,
        worker_id: str,
        lease_token: str,
        result: TaskResult,
    ) -> dict[str, Any]:
        try:
            task_id = _validate_task_id(task_id)
            worker_id = _validate_worker_id(worker_id)
        except ValueError as exc:
            raise _StoreError(400, "invalid_completion", str(exc)) from exc
        result.validate()
        incoming_result = canonical_json(result.to_dict())
        connection = self._connect()
        cleanup: list[Path] = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            task = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise _StoreError(404, "task_not_found", "task does not exist")
            if task["status"] in {"succeeded", "failed"}:
                if str(task["result_json"] or "") != incoming_result:
                    raise _StoreError(
                        409,
                        "completion_collision",
                        "task already has a different completion",
                    )
                connection.commit()
                return self._task_payload(task)
            now = time.time()
            if (
                task["status"] != "leased"
                or task["lease_owner"] != worker_id
                or not hmac.compare_digest(
                    str(task["lease_token"] or ""),
                    str(lease_token),
                )
                or float(task["lease_expires_at"] or 0.0) <= now
            ):
                raise _StoreError(409, "lease_lost", "task lease is not active")
            if (
                result.task_id != task_id
                or result.kind != task["kind"]
                or result.worker_id != worker_id
            ):
                raise _StoreError(
                    409,
                    "provenance_mismatch",
                    "task completion provenance does not match its lease",
                )
            request_payload = json.loads(task["payload_json"])
            expected_artifact = request_payload.get("shard_artifact")
            if result.status == "succeeded" and expected_artifact is not None:
                if (
                    request_payload.get("artifact_transport")
                    != ARTIFACT_TRANSPORT
                    or result.output.get("artifact_transport")
                    != ARTIFACT_TRANSPORT
                ):
                    raise _StoreError(
                        409,
                        "protocol_mismatch",
                        "render artifact transport version does not match",
                    )
                if not isinstance(expected_artifact, Mapping):
                    raise _StoreError(
                        409,
                        "artifact_mismatch",
                        "task output artifact metadata is invalid",
                    )
                try:
                    expected_ref = ArtifactRef.from_dict(expected_artifact)
                    completed_ref = ArtifactRef.from_dict(
                        result.output.get("shard_artifact") or {},
                        require_complete=True,
                    )
                except (TypeError, ValueError) as exc:
                    raise _StoreError(
                        409,
                        "artifact_mismatch",
                        str(exc),
                    ) from exc
                if expected_ref.artifact_id != completed_ref.artifact_id:
                    raise _StoreError(
                        409,
                        "artifact_mismatch",
                        "worker completed a different output artifact",
                    )
                artifact = connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id = ?",
                    (expected_ref.artifact_id,),
                ).fetchone()
                if (
                    artifact is None
                    or artifact["purpose"] != "output"
                    or artifact["task_id"] != task_id
                    or artifact["staged_worker_id"] != worker_id
                    or not hmac.compare_digest(
                        str(artifact["staged_lease_token"] or ""),
                        str(lease_token),
                    )
                    or artifact["staged_sha256"] != completed_ref.sha256
                    or int(artifact["staged_size"] or -1)
                    != int(completed_ref.size)
                    or result.output.get("shard_sha256") != completed_ref.sha256
                ):
                    raise _StoreError(
                        409,
                        "artifact_mismatch",
                        "staged output artifact does not match completion metadata",
                    )
                staged_path = Path(str(artifact["staged_path"] or ""))
                final_path = self._artifact_path(expected_ref.artifact_id)
                candidate = staged_path if staged_path.is_file() else final_path
                if not candidate.is_file():
                    raise _StoreError(
                        409,
                        "artifact_missing",
                        "staged output artifact is missing",
                    )
                actual_sha256, actual_size = sha256_file(candidate)
                if (
                    actual_sha256 != completed_ref.sha256
                    or actual_size != completed_ref.size
                ):
                    raise _StoreError(
                        409,
                        "artifact_tampered",
                        "staged output artifact failed coordinator verification",
                    )
                final_path.parent.mkdir(parents=True, exist_ok=True)
                if candidate != final_path:
                    os.replace(candidate, final_path)
                connection.execute(
                    """
                    UPDATE artifacts SET
                        status = 'complete',
                        sha256 = ?,
                        size = ?,
                        staged_lease_token = NULL,
                        staged_worker_id = NULL,
                        staged_sha256 = NULL,
                        staged_size = NULL,
                        staged_path = NULL,
                        updated_at = ?
                    WHERE artifact_id = ?
                    """,
                    (
                        completed_ref.sha256,
                        completed_ref.size,
                        now,
                        expected_ref.artifact_id,
                    ),
                )
            elif result.status == "failed":
                stages = connection.execute(
                    """
                    SELECT staged_path FROM artifacts
                    WHERE task_id = ? AND staged_lease_token = ?
                    """,
                    (task_id, lease_token),
                ).fetchall()
                cleanup.extend(
                    Path(row["staged_path"])
                    for row in stages
                    if row["staged_path"]
                )
                connection.execute(
                    """
                    UPDATE artifacts SET
                        staged_lease_token = NULL,
                        staged_worker_id = NULL,
                        staged_sha256 = NULL,
                        staged_size = NULL,
                        staged_path = NULL,
                        updated_at = ?
                    WHERE task_id = ? AND staged_lease_token = ?
                    """,
                    (now, task_id, lease_token),
                )
            connection.execute(
                """
                UPDATE tasks SET
                    status = ?,
                    result_json = ?,
                    last_worker_id = ?,
                    lease_owner = NULL,
                    lease_token = NULL,
                    lease_expires_at = NULL
                WHERE task_id = ?
                """,
                (
                    result.status,
                    incoming_result,
                    worker_id,
                    task_id,
                ),
            )
            connection.execute(
                """
                UPDATE workers SET status = 'ready', active_task = '', updated_at = ?
                WHERE worker_id = ?
                """,
                (now, worker_id),
            )
            completed = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            connection.commit()
            assert completed is not None
            return self._task_payload(completed)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
            self._unlink_paths(cleanup)


class _CoordinatorHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        token: str,
        store: HttpCoordinatorStore,
    ):
        self.auth_token = token
        self.store = store
        super().__init__(server_address, _CoordinatorHandler)


class _CoordinatorHandler(BaseHTTPRequestHandler):
    server: _CoordinatorHttpServer
    server_version = "AgentLODGECoordinator/1"

    def log_message(self, format_string: str, *args: Any) -> None:
        logger.debug("HTTP coordinator: " + format_string, *args)

    def _json_response(self, status: int, payload: Mapping[str, Any]) -> None:
        body = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-AgentLODGE-Protocol", str(PROTOCOL_VERSION))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json_response(
            status,
            {
                "protocol_version": PROTOCOL_VERSION,
                "error": code,
                "message": message,
            },
        )

    def _authorize(self) -> bool:
        expected = f"Bearer {self.server.auth_token}"
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, expected):
            self._error(401, "unauthorized", "authentication required")
            return False
        protocol = self.headers.get("X-AgentLODGE-Protocol", "")
        if protocol != str(PROTOCOL_VERSION):
            self._error(
                426,
                "protocol_mismatch",
                f"expected protocol {PROTOCOL_VERSION}",
            )
            return False
        return True

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise _StoreError(400, "invalid_json", "invalid content length") from exc
        if length < 0 or length > _MAX_JSON_BYTES:
            raise _StoreError(413, "request_too_large", "JSON request is too large")
        try:
            raw = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _StoreError(400, "invalid_json", "invalid JSON request") from exc
        if not isinstance(raw, dict):
            raise _StoreError(400, "invalid_json", "JSON request must be an object")
        if int(raw.get("protocol_version") or 0) != PROTOCOL_VERSION:
            raise _StoreError(
                426,
                "protocol_mismatch",
                f"expected protocol {PROTOCOL_VERSION}",
            )
        return raw

    def _dispatch(self, method: str) -> None:
        if not self._authorize():
            return
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if method == "POST" and path == "/v1/workers/heartbeat":
                self._json_response(
                    200,
                    self.server.store.heartbeat(self._read_json()),
                )
                return
            if method == "GET" and path == "/v1/workers":
                query = urllib.parse.parse_qs(parsed.query)
                capability = str((query.get("capability") or [""])[0])
                max_age = float(
                    (query.get("max_age_seconds") or ["15"])[0]
                )
                self._json_response(
                    200,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "workers": self.server.store.list_workers(
                            capability,
                            max_age_seconds=max_age,
                        ),
                    },
                )
                return
            if method == "POST" and path == "/v1/tasks":
                raw = self._read_json()
                request = TaskRequest.from_dict(raw.get("request") or {})
                retry_failed = raw.get("retry_failed", False)
                if not isinstance(retry_failed, bool):
                    raise _StoreError(
                        400,
                        "invalid_request",
                        "retry_failed must be a boolean",
                    )
                self._json_response(
                    200,
                    self.server.store.submit(
                        request,
                        preferred_worker_id=str(
                            raw.get("preferred_worker_id") or ""
                        ),
                        retry_failed=retry_failed,
                        eligible_worker_ids=raw.get(
                            "eligible_worker_ids",
                            [],
                        ),
                    ),
                )
                return
            if method == "POST" and path == "/v1/tasks/claim":
                raw = self._read_json()
                claim = self.server.store.claim(
                    str(raw.get("worker_id") or ""),
                    lease_seconds=raw.get("lease_seconds"),
                )
                if claim is None:
                    self.send_response(204)
                    self.send_header(
                        "X-AgentLODGE-Protocol",
                        str(PROTOCOL_VERSION),
                    )
                    self.end_headers()
                else:
                    self._json_response(200, claim)
                return
            if method == "POST" and path == "/v1/tasks/renew":
                raw = self._read_json()
                self._json_response(
                    200,
                    self.server.store.renew(
                        str(raw.get("task_id") or ""),
                        str(raw.get("worker_id") or ""),
                        str(raw.get("lease_token") or ""),
                        lease_seconds=raw.get("lease_seconds"),
                    ),
                )
                return
            if method == "POST" and path == "/v1/tasks/complete":
                raw = self._read_json()
                result = TaskResult.from_dict(raw.get("result") or {})
                self._json_response(
                    200,
                    self.server.store.complete(
                        str(raw.get("task_id") or ""),
                        str(raw.get("worker_id") or ""),
                        str(raw.get("lease_token") or ""),
                        result,
                    ),
                )
                return
            if method == "GET" and path.startswith("/v1/tasks/"):
                task_id = urllib.parse.unquote(path.removeprefix("/v1/tasks/"))
                self._json_response(
                    200,
                    self.server.store.task_status(task_id),
                )
                return
            if method == "POST" and path == "/v1/artifacts":
                raw = self._read_json()
                self._json_response(
                    200,
                    self.server.store.mint_artifact(
                        artifact_key=str(raw.get("artifact_key") or ""),
                        purpose=str(raw.get("purpose") or ""),
                        task_id=str(raw.get("task_id") or ""),
                        expected_sha256=str(raw.get("expected_sha256") or ""),
                        expected_size=raw.get("expected_size"),
                    ),
                )
                return
            if method == "PUT" and path.startswith("/v1/artifacts/"):
                artifact_id = urllib.parse.unquote(
                    path.removeprefix("/v1/artifacts/")
                )
                self._receive_artifact(artifact_id)
                return
            if method == "GET" and path.startswith("/v1/artifacts/"):
                artifact_id = urllib.parse.unquote(
                    path.removeprefix("/v1/artifacts/")
                )
                self._send_artifact(artifact_id)
                return
            self._error(404, "not_found", "endpoint does not exist")
        except _StoreError as exc:
            self._error(exc.status, exc.code, str(exc))
        except (TypeError, ValueError) as exc:
            self._error(400, "invalid_request", str(exc))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:
            logger.exception("HTTP coordinator request failed")
            self._error(500, "internal_error", "coordinator request failed")

    def _receive_artifact(self, artifact_id: str) -> None:
        try:
            length = int(self.headers.get("Content-Length", ""))
            declared_size = int(self.headers.get("X-Artifact-Size", ""))
        except ValueError as exc:
            raise _StoreError(
                400,
                "invalid_artifact",
                "artifact size headers are invalid",
            ) from exc
        if length != declared_size or length < 0:
            raise _StoreError(
                409,
                "size_mismatch",
                "artifact content length does not match declared size",
            )
        if length > self.server.store.max_artifact_bytes:
            raise _StoreError(413, "artifact_too_large", "artifact is too large")
        try:
            declared_sha256 = _validate_sha256(
                self.headers.get("X-Artifact-SHA256", "")
            )
        except ValueError as exc:
            raise _StoreError(400, "invalid_artifact", str(exc)) from exc
        worker_id = self.headers.get("X-AgentLODGE-Worker-ID", "")
        task_id = self.headers.get("X-AgentLODGE-Task-ID", "")
        lease_token = self.headers.get("X-AgentLODGE-Lease-Token", "")
        self.server.store.authorize_upload(
            artifact_id,
            worker_id=worker_id,
            task_id=task_id,
            lease_token=lease_token,
        )
        temporary = self.server.store.incoming_root / (
            f"{artifact_id}.{uuid.uuid4().hex}.tmp"
        )
        digest = hashlib.sha256()
        remaining = length
        try:
            with temporary.open("xb") as handle:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise _StoreError(
                            400,
                            "incomplete_upload",
                            "artifact upload ended before its declared size",
                        )
                    handle.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != declared_sha256:
                raise _StoreError(
                    409,
                    "hash_mismatch",
                    "artifact upload failed SHA-256 verification",
                )
            result = self.server.store.finish_upload(
                artifact_id,
                temporary,
                sha256=actual_sha256,
                size=length,
                worker_id=worker_id,
                task_id=task_id,
                lease_token=lease_token,
            )
            self._json_response(200, result)
        finally:
            temporary.unlink(missing_ok=True)

    def _send_artifact(self, artifact_id: str) -> None:
        path, reference = self.server.store.artifact_for_download(artifact_id)
        assert reference.size is not None
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(reference.size))
        self.send_header("X-Artifact-ID", reference.artifact_id)
        self.send_header("X-Artifact-SHA256", reference.sha256)
        self.send_header("X-Artifact-Size", str(reference.size))
        self.send_header("X-AgentLODGE-Protocol", str(PROTOCOL_VERSION))
        self.end_headers()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                self.wfile.write(chunk)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")


def create_http_server(
    host: str,
    port: int,
    *,
    token: str,
    store: HttpCoordinatorStore,
) -> ThreadingHTTPServer:
    return _CoordinatorHttpServer(
        (str(host), int(port)),
        _load_token(token),
        store,
    )


class HttpTransportClient:
    """Authenticated protocol client used by coordinators and workers."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        scratch_root: Path | None = None,
        timeout: float = 30.0,
    ):
        parsed = urllib.parse.urlsplit(str(base_url).rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise HttpTransportError("HTTP coordinator URL must use http or https")
        if parsed.query or parsed.fragment:
            raise HttpTransportError("HTTP coordinator URL must not contain query data")
        self.base_url = str(base_url).rstrip("/")
        self.parsed_url = parsed
        self.base_path = parsed.path.rstrip("/")
        self.token = _load_token(token)
        self.timeout = max(1.0, float(timeout))
        self.scratch_root = (
            Path(scratch_root).resolve() if scratch_root is not None else None
        )
        if self.scratch_root is not None:
            self.scratch_root.mkdir(parents=True, exist_ok=True)

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "X-AgentLODGE-Protocol": str(PROTOCOL_VERSION),
        }

    @staticmethod
    def _decode_error(status: int, body: bytes) -> HttpTransportError:
        code = "http_error"
        message = f"HTTP coordinator returned {status}"
        try:
            payload = json.loads(body.decode("utf-8"))
            code = str(payload.get("error") or code)
            message = str(payload.get("message") or message)
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
        return HttpTransportError(message, status=status, code=code)

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        expected: tuple[int, ...] = (200,),
        request_timeout: float | None = None,
    ) -> dict[str, Any] | None:
        body = None
        headers = self._headers()
        if payload is not None:
            normalized = dict(payload)
            normalized["protocol_version"] = PROTOCOL_VERSION
            body = canonical_json(normalized).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._endpoint(path),
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=(
                    self.timeout
                    if request_timeout is None
                    else max(
                        0.01,
                        min(self.timeout, float(request_timeout)),
                    )
                ),
            ) as response:
                status = int(response.status)
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise self._decode_error(exc.code, exc.read(65536)) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise HttpTransportError(
                f"HTTP coordinator request failed: {exc}"
            ) from exc
        if status not in expected:
            raise self._decode_error(status, raw)
        if status == 204 or not raw:
            return None
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpTransportError(
                "HTTP coordinator returned invalid JSON",
                status=status,
                code="invalid_response",
            ) from exc
        if not isinstance(decoded, dict):
            raise HttpTransportError(
                "HTTP coordinator returned a non-object response",
                status=status,
                code="invalid_response",
            )
        if int(decoded.get("protocol_version") or 0) != PROTOCOL_VERSION:
            raise HttpTransportError(
                "HTTP coordinator protocol mismatch",
                status=status,
                code="protocol_mismatch",
            )
        return decoded

    def heartbeat(
        self,
        worker_id: str,
        capabilities: tuple[str, ...],
        *,
        status: str,
        active_task: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        response = self.request_json(
            "POST",
            "/v1/workers/heartbeat",
            {
                "worker_id": worker_id,
                "capabilities": list(capabilities),
                "status": status,
                "active_task": active_task,
                "metadata": dict(metadata),
            },
        )
        assert response is not None
        return response

    def list_workers(
        self,
        capability: str,
        *,
        max_age_seconds: float,
    ) -> list[HttpWorkerSpec]:
        query = urllib.parse.urlencode(
            {
                "capability": capability,
                "max_age_seconds": float(max_age_seconds),
            }
        )
        response = self.request_json("GET", f"/v1/workers?{query}")
        assert response is not None
        workers = response.get("workers")
        if not isinstance(workers, list):
            raise HttpTransportError(
                "HTTP coordinator returned invalid workers",
                code="invalid_response",
            )
        return [
            HttpWorkerSpec(
                worker_id=_validate_worker_id(str(raw.get("worker_id") or "")),
                capabilities=tuple(str(value) for value in raw["capabilities"]),
                status=str(raw["status"]),
                updated_at=float(raw["updated_at"]),
                metadata=dict(raw.get("metadata") or {}),
                max_concurrency=int(raw.get("max_concurrency") or 1),
            )
            for raw in workers
        ]

    def submit(
        self,
        request: TaskRequest,
        *,
        preferred_worker_id: str = "",
        retry_failed: bool = False,
        eligible_worker_ids: tuple[str, ...] | list[str] = (),
    ) -> dict[str, Any]:
        response = self.request_json(
            "POST",
            "/v1/tasks",
            {
                "request": request.to_dict(),
                "preferred_worker_id": preferred_worker_id,
                "retry_failed": bool(retry_failed),
                "eligible_worker_ids": list(eligible_worker_ids),
            },
        )
        assert response is not None
        return response

    def task_status(
        self,
        task_id: str,
        *,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        response = self.request_json(
            "GET",
            f"/v1/tasks/{urllib.parse.quote(_validate_task_id(task_id), safe='')}",
            request_timeout=request_timeout,
        )
        assert response is not None
        return response

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
    ) -> dict[str, Any] | None:
        return self.request_json(
            "POST",
            "/v1/tasks/claim",
            {
                "worker_id": worker_id,
                "lease_seconds": float(lease_seconds),
            },
            expected=(200, 204),
        )

    def renew(
        self,
        lease: TaskLease,
        *,
        lease_seconds: float,
        request_timeout: float | None = None,
    ) -> float:
        response = self.request_json(
            "POST",
            "/v1/tasks/renew",
            {
                "task_id": lease.task_id,
                "worker_id": lease.worker_id,
                "lease_token": lease.token,
                "lease_seconds": float(lease_seconds),
            },
            request_timeout=request_timeout,
        )
        assert response is not None
        return float(response["expires_at"])

    def complete(
        self,
        lease: TaskLease,
        result: TaskResult,
        *,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        response = self.request_json(
            "POST",
            "/v1/tasks/complete",
            {
                "task_id": lease.task_id,
                "worker_id": lease.worker_id,
                "lease_token": lease.token,
                "result": result.to_dict(),
            },
            request_timeout=request_timeout,
        )
        assert response is not None
        return response

    def mint_artifact(
        self,
        *,
        artifact_key: str,
        purpose: str,
        task_id: str = "",
        expected_sha256: str = "",
        expected_size: int | None = None,
    ) -> ArtifactRef:
        response = self.request_json(
            "POST",
            "/v1/artifacts",
            {
                "artifact_key": artifact_key,
                "purpose": purpose,
                "task_id": task_id,
                "expected_sha256": expected_sha256,
                "expected_size": expected_size,
            },
        )
        assert response is not None
        return ArtifactRef.from_dict(response)

    def _connection(
        self,
        request_timeout: float | None = None,
    ) -> http.client.HTTPConnection:
        port = self.parsed_url.port
        timeout = (
            self.timeout
            if request_timeout is None
            else max(0.01, min(self.timeout, float(request_timeout)))
        )
        if self.parsed_url.scheme == "https":
            return http.client.HTTPSConnection(
                self.parsed_url.hostname,
                port or 443,
                timeout=timeout,
                context=ssl.create_default_context(),
            )
        return http.client.HTTPConnection(
            self.parsed_url.hostname,
            port or 80,
            timeout=timeout,
        )

    def _binary_path(self, artifact_id: str) -> str:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ValueError("invalid coordinator artifact id")
        return (
            f"{self.base_path}/v1/artifacts/"
            f"{urllib.parse.quote(artifact_id, safe='')}"
        )

    def upload_artifact(
        self,
        reference: ArtifactRef,
        source: Path,
        *,
        worker_id: str = "",
        task_id: str = "",
        lease_token: str = "",
        request_timeout: float | None = None,
    ) -> ArtifactRef:
        source_path = Path(source).resolve()
        if self.scratch_root is not None:
            source_path = _confined(source_path, self.scratch_root)
        actual_sha256, actual_size = sha256_file(source_path)
        if reference.sha256 and reference.sha256 != actual_sha256:
            raise HttpTransportError(
                "local artifact does not match its expected SHA-256",
                code="hash_mismatch",
            )
        if reference.size is not None and reference.size != actual_size:
            raise HttpTransportError(
                "local artifact does not match its expected size",
                code="size_mismatch",
            )
        connection = self._connection(request_timeout)
        try:
            connection.putrequest(
                "PUT",
                self._binary_path(reference.artifact_id),
            )
            headers = self._headers()
            headers.update(
                {
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(actual_size),
                    "X-Artifact-SHA256": actual_sha256,
                    "X-Artifact-Size": str(actual_size),
                }
            )
            if worker_id:
                headers["X-AgentLODGE-Worker-ID"] = worker_id
                headers["X-AgentLODGE-Task-ID"] = task_id
                headers["X-AgentLODGE-Lease-Token"] = lease_token
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            with source_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    connection.send(chunk)
            response = connection.getresponse()
            body = response.read(65536)
            if response.status != 200:
                raise self._decode_error(response.status, body)
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HttpTransportError(
                    "HTTP coordinator returned invalid artifact metadata",
                    code="invalid_response",
                ) from exc
            if int(payload.get("protocol_version") or 0) != PROTOCOL_VERSION:
                raise HttpTransportError(
                    "HTTP coordinator artifact protocol mismatch",
                    code="protocol_mismatch",
                )
            uploaded = ArtifactRef.from_dict(payload, require_complete=True)
            if (
                uploaded.artifact_id != reference.artifact_id
                or uploaded.sha256 != actual_sha256
                or uploaded.size != actual_size
            ):
                raise HttpTransportError(
                    "HTTP coordinator returned mismatched artifact metadata",
                    code="invalid_response",
                )
            return uploaded
        except (OSError, http.client.HTTPException) as exc:
            if isinstance(exc, HttpTransportError):
                raise
            raise HttpTransportError(f"artifact upload failed: {exc}") from exc
        finally:
            connection.close()

    def download_artifact(
        self,
        reference: ArtifactRef,
        destination: Path,
        *,
        request_timeout: float | None = None,
    ) -> ArtifactRef:
        if self.scratch_root is None:
            raise HttpTransportError("artifact downloads require a scratch root")
        destination_path = _confined(destination, self.scratch_root)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_path.with_name(
            f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
        )
        connection = self._connection(request_timeout)
        try:
            headers = self._headers()
            connection.request(
                "GET",
                self._binary_path(reference.artifact_id),
                headers=headers,
            )
            response = connection.getresponse()
            if response.status != 200:
                raise self._decode_error(response.status, response.read(65536))
            if (
                response.getheader("X-AgentLODGE-Protocol", "")
                != str(PROTOCOL_VERSION)
            ):
                raise HttpTransportError(
                    "HTTP coordinator artifact protocol mismatch",
                    code="protocol_mismatch",
                )
            response_id = response.getheader("X-Artifact-ID", "")
            response_sha256 = _validate_sha256(
                response.getheader("X-Artifact-SHA256", "")
            )
            response_size = int(response.getheader("X-Artifact-Size", ""))
            content_length = int(response.getheader("Content-Length", ""))
            if (
                response_id != reference.artifact_id
                or content_length != response_size
                or (reference.sha256 and reference.sha256 != response_sha256)
                or (
                    reference.size is not None
                    and reference.size != response_size
                )
            ):
                raise HttpTransportError(
                    "artifact response metadata does not match its reference",
                    code="artifact_mismatch",
                )
            digest = hashlib.sha256()
            size = 0
            with temporary.open("xb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != response_size:
                raise HttpTransportError(
                    "artifact download ended before its declared size",
                    code="size_mismatch",
                )
            if digest.hexdigest() != response_sha256:
                raise HttpTransportError(
                    "artifact download failed SHA-256 verification",
                    code="hash_mismatch",
                )
            os.replace(temporary, destination_path)
            return ArtifactRef(reference.artifact_id, response_sha256, size)
        except ValueError as exc:
            raise HttpTransportError(
                f"artifact response metadata is invalid: {exc}",
                code="artifact_mismatch",
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            if isinstance(exc, HttpTransportError):
                raise
            raise HttpTransportError(f"artifact download failed: {exc}") from exc
        finally:
            connection.close()
            temporary.unlink(missing_ok=True)


class HttpTaskCoordinator:
    """Coordinator facade matching the filesystem transport's submit/wait API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        scratch_root: Path,
        *,
        poll_interval: float = 0.2,
        heartbeat_max_age: float = 15.0,
        request_timeout: float = 30.0,
    ):
        self.scratch_root = Path(scratch_root).resolve()
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.poll_interval = max(0.02, float(poll_interval))
        self.heartbeat_max_age = max(1.0, float(heartbeat_max_age))
        self.client = HttpTransportClient(
            base_url,
            token,
            scratch_root=self.scratch_root,
            timeout=request_timeout,
        )
        self._poll_snapshot: dict[str, bool] | None = None

    @classmethod
    def from_env(cls) -> "HttpTaskCoordinator":
        base_url = os.environ.get(
            "AGENTLODGE_HTTP_COORDINATOR_URL",
            "",
        ).strip()
        if not base_url:
            raise HttpTransportError(
                "AGENTLODGE_HTTP_COORDINATOR_URL is required for HTTP transport"
            )
        token = _load_token(
            os.environ.get("AGENTLODGE_HTTP_TOKEN"),
            os.environ.get("AGENTLODGE_HTTP_TOKEN_FILE"),
        )
        scratch = os.environ.get(
            "AGENTLODGE_HTTP_COORDINATOR_SCRATCH",
            "",
        ).strip()
        if not scratch:
            raise HttpTransportError(
                "AGENTLODGE_HTTP_COORDINATOR_SCRATCH is required for HTTP transport"
            )
        return cls(
            base_url,
            token,
            Path(scratch),
            poll_interval=float(
                os.environ.get("AGENTLODGE_WORKER_POLL_INTERVAL", "0.1")
            ),
            heartbeat_max_age=float(
                os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
            ),
            request_timeout=float(
                os.environ.get("AGENTLODGE_HTTP_REQUEST_TIMEOUT", "30")
            ),
        )

    def require_workers(
        self,
        capability: str,
        *,
        max_age_seconds: float | None = None,
    ) -> list[HttpWorkerSpec]:
        workers = self.client.list_workers(
            capability,
            max_age_seconds=(
                self.heartbeat_max_age
                if max_age_seconds is None
                else max_age_seconds
            ),
        )
        if not workers:
            raise WorkerRegistryError(
                f"no healthy workers advertise capability {capability!r}"
            )
        return workers

    def submit(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        worker: HttpWorkerSpec | None = None,
        task_id: str | None = None,
        retry_failed: bool = False,
        eligible_worker_ids: tuple[str, ...] | list[str] = (),
    ) -> HttpTaskHandle:
        request = TaskRequest.create(kind, payload, task_id=task_id)
        self.client.submit(
            request,
            preferred_worker_id=worker.worker_id if worker else "",
            retry_failed=retry_failed,
            eligible_worker_ids=eligible_worker_ids,
        )
        return HttpTaskHandle(request=request, worker=worker)

    def is_complete(self, handle: HttpTaskHandle) -> bool:
        if (
            self._poll_snapshot is not None
            and handle.request.task_id in self._poll_snapshot
        ):
            return self._poll_snapshot[handle.request.task_id]
        return self.client.task_status(handle.request.task_id)["status"] in {
            "succeeded",
            "failed",
        }

    def wait(
        self,
        handle: HttpTaskHandle,
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
        handles: list[HttpTaskHandle],
        *,
        timeout: float,
        on_poll: Callable[[], None] | None = None,
        max_reassignments: int = 1,
    ) -> list[TaskResult]:
        del max_reassignments
        if not handles:
            return []
        deadline = time.monotonic() + max(0.1, float(timeout))
        pending = {handle.request.task_id: handle for handle in handles}
        completed: dict[str, TaskResult] = {}
        while pending and time.monotonic() < deadline:
            poll_snapshot = {task_id: True for task_id in completed}
            deadline_exhausted = False
            for task_id, handle in list(pending.items()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    deadline_exhausted = True
                    break
                try:
                    status = self.client.task_status(
                        task_id,
                        request_timeout=min(
                            self.client.timeout,
                            5.0,
                            max(0.01, remaining / 2.0),
                        ),
                    )
                except HttpTransportError as exc:
                    if not _retryable_http_error(exc):
                        raise
                    logger.warning(
                        "transient HTTP status poll failure for %s: %s",
                        task_id,
                        exc,
                    )
                    poll_snapshot[task_id] = False
                    continue
                terminal = status["status"] in {"succeeded", "failed"}
                poll_snapshot[task_id] = terminal
                if not terminal:
                    continue
                result = TaskResult.from_dict(status.get("result") or {})
                if (
                    result.task_id != task_id
                    or result.kind != handle.request.kind
                ):
                    raise TaskExecutionError(
                        f"HTTP worker returned a mismatched result for {task_id}"
                    )
                if result.status == "failed":
                    raise TaskExecutionError(
                        f"{result.kind} failed on {result.worker_id}: {result.error}"
                    )
                completed[task_id] = result
                del pending[task_id]
            if deadline_exhausted:
                break
            if not pending:
                break
            if on_poll is not None:
                self._poll_snapshot = poll_snapshot
                try:
                    on_poll()
                finally:
                    self._poll_snapshot = None
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(self.poll_interval, remaining))
        if pending:
            details = ", ".join(sorted(pending))
            raise TaskTimeoutError(
                f"timed out waiting for HTTP distributed tasks: {details}"
            )
        return [completed[handle.request.task_id] for handle in handles]

    def create_scratch_dir(self, *, prefix: str) -> Path:
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]", "-", prefix)[:20] or "task"
        path = self.scratch_root / f"{safe_prefix}-{uuid.uuid4().hex[:16]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def upload_input(
        self,
        source: Path,
        *,
        artifact_key: str,
    ) -> ArtifactRef:
        source_path = Path(source).resolve()
        sha256, size = sha256_file(source_path)
        upload_dir = self.scratch_root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        staged_source = upload_dir / f"{sha256}.bin"
        if source_path != staged_source:
            temporary = staged_source.with_name(
                f".{staged_source.name}.{uuid.uuid4().hex}.tmp"
            )
            shutil.copyfile(source_path, temporary)
            copied_sha256, copied_size = sha256_file(temporary)
            if copied_sha256 != sha256 or copied_size != size:
                temporary.unlink(missing_ok=True)
                raise HttpTransportError(
                    "coordinator scratch copy failed verification",
                    code="hash_mismatch",
                )
            os.replace(temporary, staged_source)
        reference = self.client.mint_artifact(
            artifact_key=artifact_key,
            purpose="input",
            expected_sha256=sha256,
            expected_size=size,
        )
        return self.client.upload_artifact(reference, staged_source)

    def reserve_output(
        self,
        *,
        artifact_key: str,
        task_id: str,
    ) -> ArtifactRef:
        reference = self.client.mint_artifact(
            artifact_key=artifact_key,
            purpose="output",
            task_id=task_id,
        )
        return ArtifactRef(reference.artifact_id)

    def download_output(
        self,
        reference: ArtifactRef,
        destination: Path,
    ) -> ArtifactRef:
        download_dir = self.scratch_root / "downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        scratch_path = download_dir / (
            f"{reference.artifact_id}.{uuid.uuid4().hex}.bin"
        )
        downloaded = self.client.download_artifact(reference, scratch_path)
        destination_path = Path(destination).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_path.with_name(
            f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(scratch_path, temporary)
            copied_sha256, copied_size = sha256_file(temporary)
            if (
                copied_sha256 != downloaded.sha256
                or copied_size != downloaded.size
            ):
                raise HttpTransportError(
                    "coordinator output copy failed verification",
                    code="hash_mismatch",
                )
            os.replace(temporary, destination_path)
        finally:
            temporary.unlink(missing_ok=True)
            scratch_path.unlink(missing_ok=True)
        return downloaded


class HttpTaskWorker:
    """Lease-based worker loop for authenticated HTTP tasks."""

    def __init__(
        self,
        worker_id: str,
        capabilities: tuple[str, ...],
        handlers: Mapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any] | None]],
        *,
        base_url: str,
        token: str,
        scratch_root: Path,
        metadata: Mapping[str, Any] | None = None,
        poll_interval: float = 0.1,
        heartbeat_interval: float = 2.0,
        lease_seconds: float = 30.0,
        request_timeout: float = 30.0,
    ):
        self.worker_id = _validate_worker_id(worker_id)
        self.capabilities = tuple(capabilities)
        missing = set(self.capabilities) - set(handlers)
        if missing:
            raise ValueError(
                f"worker {worker_id} is missing handlers for {sorted(missing)}"
            )
        self.handlers = dict(handlers)
        self.scratch_root = Path(scratch_root).resolve()
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        self.poll_interval = max(0.02, float(poll_interval))
        self.heartbeat_interval = max(0.2, float(heartbeat_interval))
        self.lease_seconds = max(0.5, float(lease_seconds))
        self.client = HttpTransportClient(
            base_url,
            token,
            scratch_root=self.scratch_root,
            timeout=request_timeout,
        )
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._heartbeat_send_lock = threading.Lock()
        self._status = "starting"
        self._active_task = ""
        self._heartbeat_thread: threading.Thread | None = None

    def _heartbeat(self) -> None:
        with self._heartbeat_send_lock:
            with self._state_lock:
                status = self._status
                active_task = self._active_task
            self.client.heartbeat(
                self.worker_id,
                self.capabilities,
                status=status,
                active_task=active_task,
                metadata=self.metadata,
            )

    def set_status(self, status: str, active_task: str = "") -> None:
        with self._state_lock:
            self._status = status
            self._active_task = active_task
        self._heartbeat()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            try:
                self._heartbeat()
            except HttpTransportError as exc:
                logger.exception(
                    "HTTP worker %s heartbeat failed",
                    self.worker_id,
                )
                if not _retryable_http_error(exc):
                    self._stop.set()
                    return

    def start_heartbeat(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"http-heartbeat-{self.worker_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.set_status("stopping")
        except HttpTransportError:
            logger.warning("could not send stopping heartbeat for %s", self.worker_id)

    @staticmethod
    def _lease_from_claim(raw: Mapping[str, Any]) -> TaskLease:
        lease = raw.get("lease")
        if not isinstance(lease, Mapping):
            raise HttpTransportError(
                "HTTP coordinator returned an invalid lease",
                code="invalid_response",
            )
        return TaskLease(
            task_id=_validate_task_id(str(lease.get("task_id") or "")),
            worker_id=_validate_worker_id(str(lease.get("worker_id") or "")),
            token=str(lease.get("token") or ""),
            expires_at=float(lease.get("expires_at") or 0.0),
            attempt=int(lease.get("attempt") or 0),
        )

    def _prepare_render_payload(
        self,
        request: TaskRequest,
        lease: TaskLease,
        lease_lost: threading.Event,
        lease_state: _LeaseState,
    ) -> tuple[dict[str, Any], Path, ArtifactRef]:
        payload = dict(request.payload)
        if payload.get("artifact_transport") != ARTIFACT_TRANSPORT:
            raise ValueError(
                "HTTP render task is missing the coordinator artifact transport"
            )
        input_ref = ArtifactRef.from_dict(
            payload.pop("poses_artifact", {}),
            require_complete=True,
        )
        output_ref = ArtifactRef.from_dict(payload.pop("shard_artifact", {}))
        payload.pop("artifact_transport", None)
        task_root = self.scratch_root / "tasks" / request.task_id
        task_root = _confined(task_root, self.scratch_root)
        shutil.rmtree(task_root, ignore_errors=True)
        task_root.mkdir(parents=True, exist_ok=False)
        poses_path = task_root / "poses.npz"
        shard_path = task_root / "shard.mkv"
        try:
            self._artifact_with_retry(
                lambda request_timeout: self.client.download_artifact(
                    input_ref,
                    poses_path,
                    request_timeout=request_timeout,
                ),
                operation="input artifact download",
                lease_lost=lease_lost,
                lease_state=lease_state,
            )
        except Exception:
            shutil.rmtree(task_root, ignore_errors=True)
            raise
        payload["poses"] = str(poses_path)
        payload["shard_output"] = str(shard_path)
        payload["_http_task_id"] = request.task_id
        payload["_http_lease_attempt"] = lease.attempt
        return payload, task_root, output_ref

    def _artifact_with_retry(
        self,
        callback: Callable[[float], ArtifactRef],
        *,
        operation: str,
        lease_lost: threading.Event,
        lease_state: _LeaseState,
    ) -> ArtifactRef:
        delay = max(0.02, min(0.1, self.lease_seconds / 10.0))
        safety_margin = max(0.02, min(1.0, self.lease_seconds / 10.0))
        while not self._stop.is_set() and not lease_lost.is_set():
            remaining = lease_state.deadline(safety_margin) - time.time()
            if remaining <= 0:
                lease_lost.set()
                raise HttpTransportError(
                    f"{operation} exceeded the confirmed task lease",
                    status=409,
                    code="lease_lost",
                )
            try:
                result = callback(
                    min(
                        self.client.timeout,
                        max(0.05, remaining - 0.01),
                    )
                )
                if (
                    lease_lost.is_set()
                    or time.time() >= lease_state.deadline(safety_margin)
                ):
                    lease_lost.set()
                    raise HttpTransportError(
                        f"{operation} finished after the confirmed task lease",
                        status=409,
                        code="lease_lost",
                    )
                return result
            except HttpTransportError as exc:
                if not _retryable_http_error(exc):
                    raise
                if time.time() + delay >= lease_state.deadline(safety_margin):
                    lease_lost.set()
                    raise HttpTransportError(
                        f"{operation} could not finish before lease expiry",
                        status=409,
                        code="lease_lost",
                    ) from exc
                logger.warning(
                    "HTTP worker %s transient %s failure for %s: %s",
                    self.worker_id,
                    operation,
                    self._active_task,
                    exc,
                )
                if self._stop.wait(delay):
                    break
                delay = min(1.0, delay * 2.0)
        raise HttpTransportError(
            f"{operation} stopped before completion",
            status=409,
            code="lease_lost",
        )

    def _execute(
        self,
        request: TaskRequest,
        lease: TaskLease,
        lease_lost: threading.Event,
        lease_state: _LeaseState,
    ) -> dict[str, Any]:
        handler = self.handlers.get(request.kind)
        if handler is None:
            raise RuntimeError(
                f"worker {self.worker_id} cannot execute {request.kind!r}"
            )
        task_root: Path | None = None
        output_ref: ArtifactRef | None = None
        payload: Mapping[str, Any] = request.payload
        try:
            if request.kind == "render.frames":
                payload, task_root, output_ref = self._prepare_render_payload(
                    request,
                    lease,
                    lease_lost,
                    lease_state,
                )
            output = dict(handler(payload) or {})
            if lease_lost.is_set():
                raise HttpTransportError(
                    "task lease expired during execution",
                    status=409,
                    code="lease_lost",
                )
            if request.kind == "render.frames":
                assert task_root is not None and output_ref is not None
                local_shard = task_root / "shard.mkv"
                uploaded = self._artifact_with_retry(
                    lambda request_timeout: self.client.upload_artifact(
                        output_ref,
                        local_shard,
                        worker_id=self.worker_id,
                        task_id=request.task_id,
                        lease_token=lease.token,
                        request_timeout=request_timeout,
                    ),
                    operation="output artifact upload",
                    lease_lost=lease_lost,
                    lease_state=lease_state,
                )
                if output.get("shard_sha256") != uploaded.sha256:
                    raise ValueError(
                        "render handler shard hash does not match uploaded content"
                    )
                output.pop("shard_output", None)
                output["shard_artifact"] = uploaded.to_dict()
                output["artifact_transport"] = ARTIFACT_TRANSPORT
            return output
        finally:
            if task_root is not None:
                shutil.rmtree(task_root, ignore_errors=True)

    def _renew_loop(
        self,
        lease: TaskLease,
        stop: threading.Event,
        lost: threading.Event,
        lease_state: _LeaseState,
    ) -> None:
        interval = max(0.02, min(1.0, self.lease_seconds / 3.0))
        retry_delay = max(0.02, min(0.25, self.lease_seconds / 10.0))
        safety_margin = max(0.02, min(1.0, self.lease_seconds / 10.0))
        while not stop.wait(interval):
            while not stop.is_set():
                try:
                    with lease_state.operation_lock:
                        if stop.is_set():
                            return
                        remaining = (
                            lease_state.deadline(safety_margin)
                            - time.time()
                        )
                        if remaining <= 0:
                            lost.set()
                            return
                        expires_at = self.client.renew(
                            lease,
                            lease_seconds=self.lease_seconds,
                            request_timeout=min(
                                self.client.timeout,
                                5.0,
                                max(0.01, remaining / 2.0),
                            ),
                        )
                    lease_state.update(expires_at)
                    break
                except HttpTransportError as exc:
                    if not _retryable_http_error(exc):
                        lost.set()
                        logger.error(
                            "HTTP worker %s terminal lease renewal failure "
                            "for %s: %s (%s)",
                            self.worker_id,
                            lease.task_id,
                            exc,
                            exc.code,
                        )
                        return
                    if (
                        time.time() + retry_delay
                        >= lease_state.deadline(safety_margin)
                    ):
                        lost.set()
                        logger.error(
                            "HTTP worker %s could not renew %s before its "
                            "last confirmed lease expiry",
                            self.worker_id,
                            lease.task_id,
                        )
                        return
                    logger.warning(
                        "HTTP worker %s transient renewal failure for %s: %s",
                        self.worker_id,
                        lease.task_id,
                        exc,
                    )
                    if stop.wait(retry_delay):
                        return

    def _complete_with_retry(
        self,
        lease: TaskLease,
        result: TaskResult,
        lease_lost: threading.Event,
        lease_state: _LeaseState,
    ) -> bool:
        delay = max(0.02, min(0.1, self.lease_seconds / 10.0))
        safety_margin = max(0.02, min(1.0, self.lease_seconds / 10.0))
        while not self._stop.is_set() and not lease_lost.is_set():
            try:
                with lease_state.operation_lock:
                    remaining = (
                        lease_state.deadline(safety_margin)
                        - time.time()
                    )
                    if remaining <= 0:
                        lease_lost.set()
                        return False
                    self.client.complete(
                        lease,
                        result,
                        request_timeout=min(
                            self.client.timeout,
                            5.0,
                            max(0.01, remaining / 2.0),
                        ),
                    )
                return True
            except HttpTransportError as exc:
                if not _retryable_http_error(exc):
                    lease_lost.set()
                    logger.error(
                        "HTTP worker %s terminal completion failure for %s: "
                        "%s (%s)",
                        self.worker_id,
                        lease.task_id,
                        exc,
                        exc.code,
                    )
                    return False
                if time.time() + delay >= lease_state.deadline(safety_margin):
                    lease_lost.set()
                    logger.error(
                        "HTTP worker %s could not complete %s before its "
                        "last confirmed lease expiry",
                        self.worker_id,
                        lease.task_id,
                    )
                    return False
                self._stop.wait(delay)
                delay = min(0.5, delay * 2.0)
        return False

    def run_once(self) -> bool:
        claim = self.client.claim(
            self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if claim is None:
            return False
        request = TaskRequest.from_dict(claim.get("request") or {})
        lease = self._lease_from_claim(claim)
        if lease.worker_id != self.worker_id or lease.task_id != request.task_id:
            raise HttpTransportError(
                "HTTP coordinator returned mismatched lease provenance",
                code="invalid_response",
            )
        self.set_status("busy", request.task_id)
        started_at = time.time()
        renew_stop = threading.Event()
        lease_lost = threading.Event()
        lease_state = _LeaseState(lease.expires_at)
        renew_thread = threading.Thread(
            target=self._renew_loop,
            args=(lease, renew_stop, lease_lost, lease_state),
            name=f"http-lease-{request.task_id}",
            daemon=True,
        )
        renew_thread.start()
        try:
            try:
                output = self._execute(
                    request,
                    lease,
                    lease_lost,
                    lease_state,
                )
                result = TaskResult(
                    task_id=request.task_id,
                    kind=request.kind,
                    worker_id=self.worker_id,
                    status="succeeded",
                    started_at=started_at,
                    finished_at=time.time(),
                    output=output,
                )
            except Exception as exc:  # noqa: BLE001 - sent to coordinator
                if lease_lost.is_set():
                    return True
                result = TaskResult(
                    task_id=request.task_id,
                    kind=request.kind,
                    worker_id=self.worker_id,
                    status="failed",
                    started_at=started_at,
                    finished_at=time.time(),
                    error=(
                        f"{type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()[-4000:]}"
                    ),
                )
            result.validate()
            self._complete_with_retry(
                lease,
                result,
                lease_lost,
                lease_state,
            )
            return True
        finally:
            renew_stop.set()
            renew_thread.join(timeout=2.0)
            if not self._stop.is_set():
                try:
                    self.set_status("ready")
                except HttpTransportError:
                    logger.exception(
                        "HTTP worker %s could not report ready",
                        self.worker_id,
                    )

    def run_forever(self) -> None:
        self.set_status("ready")
        self.start_heartbeat()
        while not self._stop.is_set():
            try:
                completed = self.run_once()
            except HttpTransportError as exc:
                logger.exception("HTTP worker %s loop failed", self.worker_id)
                if not _retryable_http_error(exc):
                    self._stop.set()
                    return
                completed = False
            except Exception:
                logger.exception("HTTP worker %s loop failed", self.worker_id)
                completed = False
            if not completed:
                self._stop.wait(self.poll_interval)
