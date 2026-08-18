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
from pathlib import Path

import numpy as np

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
AGENTLODGE_ROOT = os.environ.get("AGENTLODGE_ROOT", f"{WORKSPACE}/AgentLODGE")
sys.path.insert(0, AGENTLODGE_ROOT)

from agentlodge.dance.format import ensure_lodge139, to_agentlodge139
from agentlodge.dance.transition import to_zup

SID = sys.argv[1]
K = int(os.environ.get("AGENTLODGE_BANK_K", "1"))
WS = Path(WORKSPACE)


def to_lodge_zup(raw):
    return to_zup(to_agentlodge139(ensure_lodge139(np.asarray(raw, dtype=np.float32))))


def to_edge_zup(raw):
    return to_agentlodge139(ensure_lodge139(np.asarray(raw, dtype=np.float32)))


def save(backbone, seed, motion):
    p = WS / f"bank_{SID}_{backbone}_seed{seed}.npy"
    np.save(p, np.asarray(motion, dtype=np.float32))
    print(f"  saved {p.name}  {np.asarray(motion).shape}", flush=True)


def main():
    lodge_raw = np.load(WS / f"lodge_fd_{SID}_full.npy").astype(np.float32)
    edge_raw = np.load(WS / f"edge_fd_{SID}_full.npy").astype(np.float32)
    print(f"[{SID}] seed 0 from best takes", flush=True)
    save("lodge", 0, to_lodge_zup(lodge_raw))
    save("edge", 0, to_edge_zup(edge_raw))

    if K > 1:
        from agentlodge.config import Settings
        from agentlodge.pipeline import _run_lodge_job, _run_edge_job, _settings_to_dict
        settings = Settings.from_dict({
            "lodge_code_path": f"{WORKSPACE}/LODGE",
            "lodge_weights_path": f"{WORKSPACE}/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt",
            "lodge_global_weights_path": f"{WORKSPACE}/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt",
            "edge_code_path": f"{WORKSPACE}/EDGE",
            "edge_weights_path": f"{WORKSPACE}/EDGE/checkpoint.pt",
            "lodge_genre": "Hiphop", "max_edge_slices": None,
        })
        sd = _settings_to_dict(settings)
        lodge_feats = np.load(WS / f"lodge_fd_{SID}_feats.npy").astype(np.float32)
        edge_slices = [np.asarray(s, dtype=np.float32) for s in np.load(WS / f"edge{SID}_slices.npy")]
        wav = f"{WORKSPACE}/LODGE/data/finedance/music_wav/{SID}.wav"
        work = str(WS / f"gen{SID}_work")
        for seed in range(1, K):
            print(f"[{SID}] generating seed {seed} (real LODGE + EDGE diffusion)...", flush=True)
            lj = _run_lodge_job(lodge_feats, sd, work, seed=seed)
            if lj.get("error") is None:
                save("lodge", seed, to_lodge_zup(lj["motion"]))
            else:
                print(f"  LODGE seed {seed} failed: {lj['error'][:200]}", flush=True)
            ej = _run_edge_job(wav, edge_slices, sd, work, seed=seed)
            if ej.get("error") is None:
                save("edge", seed, to_edge_zup(ej["motion"]))
            else:
                print(f"  EDGE seed {seed} failed: {ej['error'][:200]}", flush=True)

    print(f"BUILD_BANK_{SID}_DONE", flush=True)


if __name__ == "__main__":
    main()
