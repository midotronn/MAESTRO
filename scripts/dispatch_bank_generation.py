#!/usr/bin/env python3
"""Dispatch editing-bank generation to the resident orchestration worker."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed import FileTaskCoordinator, WorkerRegistry

BANK_CONTRACT_VERSION = "dance-bank-v1-resident-workers"


def _source(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: dispatch_bank_generation.py <sid>", file=sys.stderr)
        return 2
    sid = sys.argv[1]
    bank_k = max(1, int(os.environ.get("AGENTLODGE_BANK_K", "1")))
    workspace = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
    inputs = (
        workspace / f"lodge_fd_{sid}_full.npy",
        workspace / f"edge_fd_{sid}_full.npy",
        workspace / f"lodge_fd_{sid}_feats.npy",
        workspace / f"edge{sid}_slices.npy",
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = WorkerRegistry.from_env()
    workers = registry.require(
        "dance.generate",
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
        "dance.generate",
        {
            "operation": "build_bank",
            "sid": sid,
            "bank_k": bank_k,
            "contract_version": BANK_CONTRACT_VERSION,
            "sources": {
                path.name: _source(path)
                for path in inputs
            },
        },
        worker=workers[0],
    )
    result = coordinator.wait(
        handle,
        timeout=float(os.environ.get("AGENTLODGE_BANK_TIMEOUT", "1200")),
    )
    print(
        f"BUILD_BANK_{sid}_DONE "
        f"{len(result.output.get('files') or [])} files",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
