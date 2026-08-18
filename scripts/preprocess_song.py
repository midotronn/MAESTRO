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

import os
import sys
from pathlib import Path

import numpy as np

# librosa >= 0.10 dropped scipy.signal.hann that older LODGE/EDGE feature code imports.
import scipy.signal as _sps
if not hasattr(_sps, "hann"):
    from scipy.signal.windows import hann as _hann
    _sps.hann = _hann

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
AGENTLODGE_ROOT = os.environ.get("AGENTLODGE_ROOT", f"{WORKSPACE}/AgentLODGE")
sys.path.insert(0, AGENTLODGE_ROOT)

from agentlodge.config import Settings
from agentlodge.audio.preprocess import preprocess_audio


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

    wav = f"{WORKSPACE}/LODGE/data/finedance/music_wav/{sid}.wav"
    if not Path(wav).exists():
        sys.stderr.write(f"missing wav {wav}\n")
        return 2
    settings = Settings.from_dict({
        "lodge_code_path": f"{WORKSPACE}/LODGE",
        "lodge_weights_path": f"{WORKSPACE}/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt",
        "lodge_global_weights_path": f"{WORKSPACE}/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt",
        "edge_code_path": f"{WORKSPACE}/EDGE",
        "edge_weights_path": f"{WORKSPACE}/EDGE/checkpoint.pt",
        "lodge_genre": "Hiphop", "max_edge_slices": None,
    })
    print(f"=== preprocessing {sid} (lodge={want_lodge} edge={want_edge}) ===", flush=True)
    pre = preprocess_audio(wav, settings, Path(f"{WORKSPACE}/gen{sid}_work"),
                           extract_lodge=want_lodge, extract_edge=want_edge)
    if want_lodge:
        np.save(lodge_out, np.asarray(pre.lodge_features, dtype=np.float32))
        print(f"[{sid}] LODGE feats {np.asarray(pre.lodge_features).shape} -> {lodge_out.name}", flush=True)
    if want_edge:
        np.save(edge_out, np.array(pre.edge_feature_slices))
        print(f"[{sid}] EDGE slices {len(pre.edge_feature_slices)} -> {edge_out.name}", flush=True)
    print(f"PREPROCESS_{sid}_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
