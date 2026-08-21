"""Build a window-edit *candidate bank* on the pod: sample K seeded LODGE/EDGE full takes, convert
them into the assembled Z-up 139 space, and save bank_<sid>_<backbone>_seed<n>.npy.

The interactive editor (agentlodge.editor) selects window candidates from this bank at edit time
(GPU-free), so the real generation cost is paid ONCE here (K real LODGE + K real EDGE diffusion
runs), not per edit. Conversions mirror build_story_dance exactly, so slices align frame-for-frame
with fd_<sid>_STORY_bestofk.npy.

Seed 0 reuses the already-generated best takes (lodge_fd_<sid>_full.npy / edge_fd_<sid>_full.npy);
seeds 1..K-1 re-run the backbones from the cached features/slices (no jukebox needed).

Usage:   AGENTLODGE_BANK_K=4 python scripts/build_window_bank.py <sid>
Env:     WORKSPACE (default /workspace)
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
AGENTLODGE_ROOT = os.environ.get("AGENTLODGE_ROOT", f"{WORKSPACE}/AgentLODGE")
sys.path.insert(0, AGENTLODGE_ROOT)

from agentlodge.dance.format import ensure_lodge139, to_agentlodge139
from agentlodge.dance.transition import to_zup


def to_lodge_zup(raw):
    return to_zup(to_agentlodge139(ensure_lodge139(np.asarray(raw, dtype=np.float32))))


def to_edge_zup(raw):
    return to_agentlodge139(ensure_lodge139(np.asarray(raw, dtype=np.float32)))


def save(workspace: Path, sid: str, backbone: str, seed: int, motion):
    p = workspace / f"bank_{sid}_{backbone}_seed{seed}.npy"
    np.save(p, np.asarray(motion, dtype=np.float32))
    print(f"  saved {p.name}  {np.asarray(motion).shape}", flush=True)


def build_bank(
    sid: str,
    bank_k: int,
    *,
    workspace: Path | str = WORKSPACE,
    distributed: bool = False,
) -> dict:
    if not sid or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in sid
    ):
        raise ValueError(f"invalid song id: {sid!r}")
    k = max(1, int(bank_k))
    ws = Path(workspace).resolve()
    lodge_best = ws / f"lodge_fd_{sid}_full.npy"
    edge_best = ws / f"edge_fd_{sid}_full.npy"
    lodge_features_path = ws / f"lodge_fd_{sid}_feats.npy"
    edge_slices_path = ws / f"edge{sid}_slices.npy"
    for path in (lodge_best, edge_best, lodge_features_path, edge_slices_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    lodge_raw = np.load(lodge_best).astype(np.float32)
    edge_raw = np.load(edge_best).astype(np.float32)
    print(f"[{sid}] seed 0 from best takes", flush=True)
    save(ws, sid, "lodge", 0, to_lodge_zup(lodge_raw))
    save(ws, sid, "edge", 0, to_edge_zup(edge_raw))

    if k > 1 and distributed:
        from scripts.make_song_bestofk import _distributed_generation_job

        work = ws / f"gen{sid}_work" / "bank"
        for seed in range(1, k):
            print(
                f"[{sid}] generating seed {seed} "
                "(resident LODGE + EDGE workers)...",
                flush=True,
            )
            lodge_work = work / "lodge" / f"seed_{seed}"
            edge_work = work / "edge" / f"seed_{seed}"
            with ThreadPoolExecutor(max_workers=2) as executor:
                lodge_future = executor.submit(
                    _distributed_generation_job,
                    "lodge",
                    lodge_features_path,
                    lodge_work / "lodge_motion.npy",
                    lodge_work,
                    seed=seed,
                )
                edge_future = executor.submit(
                    _distributed_generation_job,
                    "edge",
                    edge_slices_path,
                    edge_work / "edge_motion.npy",
                    edge_work,
                    seed=seed,
                )
                lj = lodge_future.result()
                ej = edge_future.result()
            if lj.get("error") is not None:
                raise RuntimeError(f"LODGE seed {seed} failed: {lj['error'][:500]}")
            if ej.get("error") is not None:
                raise RuntimeError(f"EDGE seed {seed} failed: {ej['error'][:500]}")
            save(ws, sid, "lodge", seed, to_lodge_zup(lj["motion"]))
            save(ws, sid, "edge", seed, to_edge_zup(ej["motion"]))
    elif k > 1:
        from agentlodge.config import Settings
        from agentlodge.pipeline import _run_lodge_job, _run_edge_job, _settings_to_dict
        settings = Settings.from_dict({
            "lodge_code_path": f"{ws}/LODGE",
            "lodge_weights_path": f"{ws}/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt",
            "lodge_global_weights_path": f"{ws}/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt",
            "edge_code_path": f"{ws}/EDGE",
            "edge_weights_path": f"{ws}/EDGE/checkpoint.pt",
            "lodge_genre": "Hiphop", "max_edge_slices": None,
        })
        sd = _settings_to_dict(settings)
        lodge_feats = np.load(lodge_features_path).astype(np.float32)
        edge_slices = [
            np.asarray(item, dtype=np.float32)
            for item in np.load(edge_slices_path)
        ]
        wav = f"{ws}/LODGE/data/finedance/music_wav/{sid}.wav"
        work = str(ws / f"gen{sid}_work")
        for seed in range(1, k):
            print(f"[{sid}] generating seed {seed} (real LODGE + EDGE diffusion)...", flush=True)
            lj = _run_lodge_job(lodge_feats, sd, work, seed=seed)
            if lj.get("error") is None:
                save(ws, sid, "lodge", seed, to_lodge_zup(lj["motion"]))
            else:
                print(f"  LODGE seed {seed} failed: {lj['error'][:200]}", flush=True)
            ej = _run_edge_job(wav, edge_slices, sd, work, seed=seed)
            if ej.get("error") is None:
                save(ws, sid, "edge", seed, to_edge_zup(ej["motion"]))
            else:
                print(f"  EDGE seed {seed} failed: {ej['error'][:200]}", flush=True)

    files = sorted(ws.glob(f"bank_{sid}_*.npy"))
    expected = k * 2
    if len(files) < expected:
        raise RuntimeError(
            f"bank generation produced {len(files)} files; expected {expected}"
        )
    print(f"BUILD_BANK_{sid}_DONE", flush=True)
    return {
        "sid": sid,
        "bank_k": k,
        "files": [str(path) for path in files],
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_window_bank.py <sid>")
    build_bank(
        sys.argv[1],
        int(os.environ.get("AGENTLODGE_BANK_K", "1")),
    )


if __name__ == "__main__":
    main()
