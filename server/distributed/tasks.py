"""Versioned task and result records shared by coordinators and GPU workers."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

PROTOCOL_VERSION = 1
_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("task payload must be JSON serializable") from exc


def deterministic_task_id(kind: str, payload: Mapping[str, Any]) -> str:
    if not _KIND_RE.fullmatch(kind):
        raise ValueError(f"invalid task kind: {kind!r}")
    digest = hashlib.sha256(
        f"{kind}\n{_canonical_json(payload)}".encode("utf-8")
    ).hexdigest()[:32]
    prefix = kind.replace(".", "-")[:40]
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class TaskRequest:
    task_id: str
    kind: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    protocol_version: int = PROTOCOL_VERSION

    @classmethod
    def create(
        cls,
        kind: str,
        payload: Mapping[str, Any],
        *,
        task_id: str | None = None,
    ) -> "TaskRequest":
        normalized = dict(payload)
        resolved_id = task_id or deterministic_task_id(kind, normalized)
        request = cls(task_id=resolved_id, kind=kind, payload=normalized)
        request.validate()
        return request

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskRequest":
        request = cls(
            task_id=str(raw.get("task_id") or ""),
            kind=str(raw.get("kind") or ""),
            payload=dict(raw.get("payload") or {}),
            created_at=float(raw.get("created_at") or 0.0),
            protocol_version=int(raw.get("protocol_version") or 0),
        )
        request.validate()
        return request

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported task protocol {self.protocol_version}; "
                f"expected {PROTOCOL_VERSION}"
            )
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError(f"invalid task id: {self.task_id!r}")
        if not _KIND_RE.fullmatch(self.kind):
            raise ValueError(f"invalid task kind: {self.kind!r}")
        _canonical_json(self.payload)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": self.payload,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    kind: str
    worker_id: str
    status: str
    started_at: float
    finished_at: float
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    protocol_version: int = PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TaskResult":
        result = cls(
            task_id=str(raw.get("task_id") or ""),
            kind=str(raw.get("kind") or ""),
            worker_id=str(raw.get("worker_id") or ""),
            status=str(raw.get("status") or ""),
            started_at=float(raw.get("started_at") or 0.0),
            finished_at=float(raw.get("finished_at") or 0.0),
            output=dict(raw.get("output") or {}),
            error=str(raw.get("error") or ""),
            protocol_version=int(raw.get("protocol_version") or 0),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported result protocol {self.protocol_version}; "
                f"expected {PROTOCOL_VERSION}"
            )
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError(f"invalid task id: {self.task_id!r}")
        if not _KIND_RE.fullmatch(self.kind):
            raise ValueError(f"invalid task kind: {self.kind!r}")
        if self.status not in {"succeeded", "failed"}:
            raise ValueError(f"invalid task result status: {self.status!r}")
        if not self.worker_id:
            raise ValueError("task result is missing worker_id")
        if self.finished_at < self.started_at:
            raise ValueError("task result finished before it started")
        _canonical_json(self.output)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "task_id": self.task_id,
            "kind": self.kind,
            "worker_id": self.worker_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.finished_at - self.started_at, 3),
            "output": self.output,
            "error": self.error,
        }
