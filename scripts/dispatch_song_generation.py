#!/usr/bin/env python3
"""Dispatch full-song generation to the resident orchestration worker."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed import FileTaskCoordinator, WorkerRegistry

GENERATION_CONTRACT_VERSION = (
    "dance-generation-v4-explicit-storyboard-search"
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _source(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: dispatch_song_generation.py <sid>", file=sys.stderr)
        return 2
    sid = sys.argv[1]
    workspace = Path(os.environ.get("WORKSPACE", "/workspace")).resolve()
    inputs = (
        workspace / "LODGE/data/finedance/music_wav" / f"{sid}.wav",
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
            "sid": sid,
            "contract_version": GENERATION_CONTRACT_VERSION,
            "penetration_cleanup": True,
            "best_of_k": max(
                1,
                int(os.environ.get("AGENTLODGE_BEST_OF_K", "1")),
            ),
            "require_full_best_of_k": _env_flag(
                "AGENTLODGE_REQUIRE_FULL_BEST_OF_K"
            ),
            "require_llm_storyboard": _env_flag(
                "AGENTLODGE_REQUIRE_LLM_STORYBOARD"
            ),
            "storyboard_model": os.environ.get(
                "OPENAI_CHAT_MODEL",
                "",
            ).strip() or "gpt-4o-mini",
            "timing_file": os.environ.get("MAESTRO_TIMING_FILE", ""),
            "sources": {
                path.name: _source(path)
                for path in inputs
            },
        },
        worker=workers[0],
    )
    result = coordinator.wait(
        handle,
        timeout=float(
            os.environ.get("AGENTLODGE_GENERATION_TIMEOUT", "1200")
        ),
    )
    print(
        f"MAKE_SONG_{sid}_DONE "
        f"{int(result.output.get('frames') or 0)} frames "
        f"best-of-{int(result.output.get('best_of_k') or 1)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
