"""Warm (persistent) LODGE generation daemon for the editor's live regeneration.

A cold ``gen_take.py`` reloads the ~1.7GB LODGE checkpoints every call (~150s). This daemon loads the
models ONCE (``agentlodge.dance.lodge`` caches them in-process) and then serves windowed samples on
request, so every new seed after warm-up is just the diffusion (seconds). Same request/marker
protocol as ``scripts/blender_daemon.py``.

Run (in the LODGE/CUDA venv, on the pod):
    WORKSPACE=/workspace python scripts/gen_daemon.py --requests-dir /workspace/gen_daemon

Requests: ``<id>.req`` JSON ``{"id","sid","backbone":"lodge","seed":N,"a":A,"b":B}``.
Writes:   ``bank_<sid>_lodge_seed<N>_w<A>_<B>.npy`` in WORKSPACE + ``<id>.done`` ("<path> <frames> <sec>")
          or ``<id>.fail`` (traceback). ``daemon.ready`` once up; touches ``daemon.hb`` every poll.
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
sys.path.insert(0, f"{WORKSPACE}/AgentLODGE")
sys.path.insert(0, f"{WORKSPACE}/AgentLODGE/scripts")
WS = Path(WORKSPACE)

from gen_take import _to_lodge_zup, _trim_zup  # noqa: E402 - pure helpers (no subprocess)


def _settings():
    from agentlodge.config import Settings
    return Settings.from_dict({
        "lodge_code_path": f"{WORKSPACE}/LODGE",
        "lodge_weights_path": f"{WORKSPACE}/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt",
        "lodge_global_weights_path": f"{WORKSPACE}/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt",
        "edge_code_path": f"{WORKSPACE}/EDGE",
        "edge_weights_path": f"{WORKSPACE}/EDGE/checkpoint.pt",
        "lodge_genre": "Hiphop", "max_edge_slices": None,
    })


def _gen_lodge_window(sid: str, settings, seed: int, a: int, b: int) -> np.ndarray:
    """Generate JUST the [a, b) window (plus a 2048-frame LODGE context) with the WARM models."""
    from agentlodge.dance.lodge import generate_lodge_dance

    feats = np.load(WS / f"lodge_fd_{sid}_feats.npy").astype(np.float32)
    MIN, L = 2048, feats.shape[0]
    if L <= MIN:
        in_start, in_end = 0, L
    else:
        center = (a + b) // 2
        in_start = max(0, min(center - MIN // 2, L - MIN))
        in_end = in_start + MIN
    res = generate_lodge_dance(feats[in_start:in_end], settings, WS / "gen_daemon_work", seed=seed)
    zup = _to_lodge_zup(res.motion)
    return np.ascontiguousarray(_trim_zup(zup, in_start, in_end - in_start, a, b), dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--requests-dir", required=True)
    p.add_argument("--idle-exit", type=int, default=0, help="Exit after N idle seconds (0 = never).")
    args = p.parse_args()

    rdir = args.requests_dir
    os.makedirs(rdir, exist_ok=True)
    settings = _settings()

    _stop = threading.Event()

    def _beat():
        while not _stop.is_set():
            try:
                with open(os.path.join(rdir, "daemon.hb"), "w") as f:
                    f.write(str(int(time.time())))
            except Exception:  # noqa: BLE001
                pass
            _stop.wait(3)

    threading.Thread(target=_beat, daemon=True).start()
    open(os.path.join(rdir, "daemon.ready"), "w").close()
    print("GEN_DAEMON_READY", flush=True)

    last_activity = time.time()
    while True:
        for reqpath in sorted(glob.glob(os.path.join(rdir, "*.req"))):
            rid = os.path.splitext(os.path.basename(reqpath))[0]
            done = os.path.join(rdir, rid + ".done")
            fail = os.path.join(rdir, rid + ".fail")
            try:
                with open(reqpath) as f:
                    req = json.load(f)
                os.remove(reqpath)                          # claim the request
            except Exception:  # noqa: BLE001 - half-written file: retry next poll
                continue
            sid = str(req.get("sid", ""))
            bb = str(req.get("backbone", "lodge"))
            seed = int(req.get("seed", 0))
            a, b = int(req["a"]), int(req["b"])
            out = WS / f"bank_{sid}_{bb}_seed{seed}_w{a}_{b}.npy"
            t0 = time.time()
            try:
                if bb != "lodge":
                    raise RuntimeError("warm daemon serves lodge only")
                if out.exists():
                    motion = np.load(out).astype(np.float32)
                else:
                    motion = _gen_lodge_window(sid, settings, seed, a, b)
                    tmp = out.with_suffix(".tmp.npy")
                    np.save(tmp, motion)
                    os.replace(tmp, out)                    # atomic publish
                with open(done, "w") as f:
                    f.write(f"{out} {motion.shape[0]} {time.time() - t0:.1f}")
                print(f"GEN_DAEMON_DONE {rid} {bb} seed{seed} [{a},{b}) "
                      f"{motion.shape[0]}f {time.time() - t0:.1f}s", flush=True)
            except Exception:  # noqa: BLE001 - one bad request must not kill the daemon
                with open(fail, "w") as f:
                    f.write(traceback.format_exc()[-1500:])
                print(f"GEN_DAEMON_FAIL {rid}", flush=True)
            last_activity = time.time()
        if args.idle_exit and (time.time() - last_activity) > args.idle_exit:
            print("GEN_DAEMON_IDLE_EXIT", flush=True)
            _stop.set()
            return
        time.sleep(0.2)


if __name__ == "__main__":
    main()
