#!/usr/bin/env python3
"""Extract Jukebox embeddings for EDGE slices in an isolated process."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.distributed.runtime import capability_enabled  # noqa: E402


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _partition_contiguous(items: list[Path], count: int) -> list[list[Path]]:
    if not items:
        return []
    groups = min(max(1, int(count)), len(items))
    base, remainder = divmod(len(items), groups)
    partitions: list[list[Path]] = []
    start = 0
    for index in range(groups):
        end = start + base + (1 if index < remainder else 0)
        partitions.append(items[start:end])
        start = end
    return partitions


def _extract_distributed(
    wav_slices: list[Path],
    cache_dir: Path,
) -> None:
    from server.distributed import FileTaskCoordinator, WorkerRegistry

    registry = WorkerRegistry.from_env()
    workers = registry.require(
        "jukebox.extract",
        max_age_seconds=float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
        ),
    )
    partitions = _partition_contiguous(wav_slices, len(workers))
    coordinator = FileTaskCoordinator(
        registry,
        heartbeat_max_age=float(
            os.environ.get("AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE", "30")
        ),
    )
    handles = []
    for worker, partition in zip(workers, partitions):
        items = []
        for wav_slice in partition:
            output_path = cache_dir / f"{wav_slice.stem}.npy"
            items.append(
                {
                    "wav": str(wav_slice.resolve()),
                    "output": str(output_path.resolve()),
                    "source": _fingerprint(wav_slice),
                }
            )
        handle = coordinator.submit(
            "jukebox.extract",
            {"items": items},
            worker=worker,
        )
        handles.append(handle)
        print(
            f"dispatched {len(items)} slices to {worker.worker_id} "
            f"as {handle.request.task_id}",
            flush=True,
        )
    timeout = float(os.environ.get("AGENTLODGE_JUKEBOX_TIMEOUT", "1800"))
    results = coordinator.wait_many(handles, timeout=timeout)
    for result in results:
        print(
            f"{result.worker_id} completed "
            f"{result.output.get('items', 0)} Jukebox slices",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-root", required=True)
    parser.add_argument("--slice-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    edge_root = Path(args.edge_root).resolve()
    slice_dir = Path(args.slice_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(edge_root)
    sys.path.insert(0, str(edge_root))

    wav_slices = sorted(slice_dir.glob("*.wav"), key=lambda p: int(p.stem.split("slice")[-1]))
    if not wav_slices:
        raise SystemExit(f"No wav slices found in {slice_dir}")

    pending = [
        wav_slice
        for wav_slice in wav_slices
        if not (cache_dir / f"{wav_slice.stem}.npy").exists()
    ]
    distributed = capability_enabled("jukebox.extract")
    if pending and distributed:
        _extract_distributed(pending, cache_dir)
    if pending and not distributed:
        import gc

        import numpy as np
        from data.audio_extraction.jukebox_features import extract as juke_extract

    for index, wav_slice in enumerate(wav_slices):
        out_path = cache_dir / f"{wav_slice.stem}.npy"
        if out_path.exists():
            print(f"[{index + 1}/{len(wav_slices)}] cached {out_path.name}")
            continue
        if distributed:
            raise RuntimeError(
                f"distributed Jukebox worker did not produce {out_path}"
            )
        print(f"[{index + 1}/{len(wav_slices)}] extracting {wav_slice.name}")
        reps, _ = juke_extract(str(wav_slice))
        np.save(out_path, np.asarray(reps, dtype=np.float32))
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"done {len(wav_slices)} slices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
