#!/usr/bin/env python3
"""Start one exact backbone generation early and publish a validated reuse marker."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed import FileTaskCoordinator, WorkerRegistry

EARLY_LODGE_CONTRACT_VERSION = "lodge-early-generation-v1"


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[2] != "lodge":
        print(
            "usage: dispatch_backbone_generation.py <sid> lodge",
            file=sys.stderr,
        )
        return 2
    sid = sys.argv[1]
    if not sid or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in sid
    ):
        raise ValueError(f"invalid song id: {sid!r}")

    workspace = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
    features = workspace / f"lodge_fd_{sid}_feats.npy"
    if not features.is_file():
        raise FileNotFoundError(features)
    work_dir = workspace / f"gen{sid}_work/lodge/early"
    output = workspace / f"lodge_early_{sid}.npy"
    marker = workspace / f"lodge_early_{sid}.json"
    pending = workspace / f"lodge_early_{sid}.pending"
    work_dir.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    pending.write_text(
        EARLY_LODGE_CONTRACT_VERSION + "\n",
        encoding="utf-8",
    )

    try:
        registry = WorkerRegistry.from_env()
        workers = registry.require(
            "lodge.generate",
            max_age_seconds=float(
                os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
            ),
        )
        coordinator = FileTaskCoordinator(
            registry,
            heartbeat_max_age=float(
                os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
            ),
        )
        handle = coordinator.submit(
            "lodge.generate",
            {
                "features": str(features),
                "output": str(output),
                "work_dir": str(work_dir),
                "seed": None,
                "source": _fingerprint(features),
            },
            worker=workers[0],
        )
        result = coordinator.wait(
            handle,
            timeout=float(
                os.environ.get("AGENTLODGE_GENERATION_TIMEOUT", "1200")
            ),
        )
        motion = np.load(output).astype(np.float32)
        if motion.ndim != 2 or motion.shape[0] < 1 or not np.isfinite(motion).all():
            raise RuntimeError(
                f"early LODGE generation produced invalid shape {motion.shape}"
            )
        _atomic_json(
            marker,
            {
                "contract_version": EARLY_LODGE_CONTRACT_VERSION,
                "sid": sid,
                "seed": None,
                "source": _fingerprint(features),
                "output": _fingerprint(output),
                "shape": [int(dimension) for dimension in motion.shape],
                "dtype": str(motion.dtype),
                "summary": str(result.output.get("summary") or ""),
                "worker_id": result.worker_id,
            },
        )
        print(
            f"EARLY_LODGE_{sid}_DONE frames={motion.shape[0]} "
            f"worker={result.worker_id}",
            flush=True,
        )
        return 0
    finally:
        pending.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
