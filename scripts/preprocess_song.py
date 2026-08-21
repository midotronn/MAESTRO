#!/usr/bin/env python3
"""Preprocess a song into the features both backbones need, for the candidate bank / live pod mode.

Produces (in ``$WORKSPACE``):
    lodge_fd_<sid>_feats.npy   -- LODGE music features (librosa-based; no Jukebox, fast ~2min)
    edge<sid>_slices.npy       -- EDGE Jukebox feature slices (needs the ~10GB 5B prior, one-time)

``scripts/gen_take.py`` (live pod mode) and ``scripts/build_window_bank.py`` both consume these to
run LODGE / EDGE at new seeds. The wav must already be at
``$WORKSPACE/LODGE/data/finedance/music_wav/<sid>.wav``.

Usage:
    python scripts/preprocess_song.py <sid>              # both backbones
    python scripts/preprocess_song.py <sid> --lodge-only # skip Jukebox (LODGE live gen only)
    python scripts/preprocess_song.py <sid> --edge-only  # only Jukebox slices
Env: WORKSPACE (default /workspace)
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
AGENTLODGE_ROOT = os.environ.get("AGENTLODGE_ROOT", f"{WORKSPACE}/AgentLODGE")
sys.path.insert(0, AGENTLODGE_ROOT)

from server.distributed.runtime import capability_enabled


def _fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _distributed_preprocess(
    capability: str,
    wav_path: Path,
    output_path: Path,
    work_dir: Path,
) -> dict[str, object]:
    from server.distributed import FileTaskCoordinator, WorkerRegistry

    registry = WorkerRegistry.from_env()
    workers = registry.require(
        capability,
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
        capability,
        {
            "wav": str(wav_path),
            "output": str(output_path),
            "work_dir": str(work_dir),
            "source": _fingerprint(wav_path),
        },
        worker=workers[0],
    )
    result = coordinator.wait(
        handle,
        timeout=float(
            os.environ.get("AGENTLODGE_AUDIO_PREPROCESS_TIMEOUT", "900")
        ),
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"{capability} worker did not produce {output_path}"
        )
    return result.output


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        sys.stderr.write("usage: preprocess_song.py <sid> [--lodge-only|--edge-only]\n")
        return 2
    sid = args[0]
    want_lodge = "--edge-only" not in flags
    want_edge = "--lodge-only" not in flags

    ws = Path(WORKSPACE)
    lodge_out = ws / f"lodge_fd_{sid}_feats.npy"
    edge_out = ws / f"edge{sid}_slices.npy"
    if (not want_lodge or lodge_out.exists()) and (not want_edge or edge_out.exists()):
        print(f"[{sid}] already preprocessed; skipping", flush=True)
        return 0

    wav_path = Path(
        f"{WORKSPACE}/LODGE/data/finedance/music_wav/{sid}.wav"
    ).resolve()
    if not wav_path.exists():
        sys.stderr.write(f"missing wav {wav_path}\n")
        return 2
    work_dir = Path(f"{WORKSPACE}/gen{sid}_work")
    distributed_lodge = want_lodge and capability_enabled("audio.lodge")
    distributed_edge = want_edge and capability_enabled("audio.edge")
    print(
        f"=== preprocessing {sid} "
        f"(lodge={want_lodge} edge={want_edge}) ===",
        flush=True,
    )
    if want_lodge:
        if distributed_lodge:
            output = _distributed_preprocess(
                "audio.lodge",
                wav_path,
                lodge_out,
                work_dir,
            )
            print(
                f"[{sid}] LODGE feats {tuple(output.get('shape', []))} "
                f"-> {lodge_out.name}",
                flush=True,
            )
        else:
            import numpy as np

            from agentlodge.audio.preprocess import extract_lodge_features

            lodge_features = extract_lodge_features(
                wav_path,
                Path(WORKSPACE) / "LODGE",
            )
            np.save(
                lodge_out,
                np.asarray(lodge_features, dtype=np.float32),
            )
            print(
                f"[{sid}] LODGE feats {np.asarray(lodge_features).shape} "
                f"-> {lodge_out.name}",
                flush=True,
            )
    if want_edge:
        if distributed_edge:
            output = _distributed_preprocess(
                "audio.edge",
                wav_path,
                edge_out,
                work_dir,
            )
            shape = output.get("shape", [])
            count = shape[0] if isinstance(shape, list) and shape else 0
            print(
                f"[{sid}] EDGE slices {count} -> {edge_out.name}",
                flush=True,
            )
        else:
            import numpy as np

            from agentlodge.audio.preprocess import extract_edge_slices

            edge_slices = extract_edge_slices(
                wav_path,
                Path(WORKSPACE) / "EDGE",
                work_dir,
                max_slices=None,
            )
            np.save(edge_out, np.array(edge_slices))
            print(
                f"[{sid}] EDGE slices {len(edge_slices)} "
                f"-> {edge_out.name}",
                flush=True,
            )
    print(f"PREPROCESS_{sid}_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
