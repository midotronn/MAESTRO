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


_EDGE_MODEL = None  # cached EDGE model per process (keyed implicitly by the daemon's single checkpoint)


def _edge_model(edge_root: Path, checkpoint):
    """Load EDGE('jukebox', checkpoint) once and cache it (the ~1.2GB checkpoint load is the cold cost)."""
    global _EDGE_MODEL
    if _EDGE_MODEL is None:
        os.chdir(edge_root)
        if str(edge_root) not in sys.path:
            sys.path.insert(0, str(edge_root))
        from EDGE import EDGE  # noqa: N811 - EDGE repo module
        m = EDGE("jukebox", str(checkpoint))
        m.eval()
        _EDGE_MODEL = m
    return _EDGE_MODEL


def _pkl_to_edge151(pkl_path: Path) -> np.ndarray:
    import pickle

    import torch
    from dataset.quaternion import ax_to_6v  # EDGE repo

    data = pickle.load(open(pkl_path, "rb"))
    trans = data["smpl_trans"].astype(np.float32)
    poses_aa = data["smpl_poses"].reshape(-1, 24, 3)
    rot_6d = ax_to_6v(torch.from_numpy(poses_aa)).numpy().reshape(len(trans), 144)
    contact = np.zeros((len(trans), 4), dtype=np.float32)
    return np.concatenate([trans, rot_6d, contact], axis=1)


def _gen_edge_window(sid: str, settings, seed: int, a: int, b: int) -> np.ndarray:
    """Generate JUST the [a, b) window from EDGE (150-frame Jukebox slices) with the WARM model."""
    import glob as _glob
    import shutil

    import torch

    from gen_take import _to_edge_zup

    edge_root = Path(f"{WORKSPACE}/EDGE")
    slices = np.load(WS / f"edge{sid}_slices.npy", allow_pickle=True)
    i0 = max(0, a // 150)                                    # EDGE slices are 150-frame (5s) chunks
    i1 = max(min(len(slices), (b + 149) // 150), i0 + 1)
    sub = [np.asarray(s, dtype=np.float32) for s in slices[i0:i1]]
    in_start = i0 * 150

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed) % (2 ** 32 - 1))

    model = _edge_model(edge_root, settings.edge_weights_path)
    work = WS / "gen_daemon_edge_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    render_dir, motion_dir = work / "edge_renders", work / "edge_motions"
    render_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)
    cond = torch.from_numpy(np.array(sub))
    model.render_sample((None, cond, []), "test", str(render_dir), render_count=-1,
                        fk_out=str(motion_dir), render=False)
    pkls = sorted(_glob.glob(str(motion_dir / "test_*.pkl")))
    if not pkls:
        raise RuntimeError("no EDGE motion output")
    motion = _pkl_to_edge151(Path(pkls[0]))
    expected = int(5.0 * 30 + (len(sub) - 1) * 2.5 * 30)
    if motion.shape[0] > expected:
        motion = motion[:expected]
    zup = _to_edge_zup(motion)
    return np.ascontiguousarray(_trim_zup(zup, in_start, len(sub) * 150, a, b), dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--requests-dir", required=True)
    p.add_argument("--backbone", default="lodge", choices=["lodge", "edge"],
                   help="One backbone per daemon process (so LODGE and EDGE code paths never mix).")
    p.add_argument("--idle-exit", type=int, default=0, help="Exit after N idle seconds (0 = never).")
    args = p.parse_args()

    rdir = args.requests_dir
    daemon_bb = args.backbone
    os.makedirs(rdir, exist_ok=True)
    settings = _settings()
    _gen = _gen_lodge_window if daemon_bb == "lodge" else _gen_edge_window

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
            bb = str(req.get("backbone", daemon_bb))
            seed = int(req.get("seed", 0))
            a, b = int(req["a"]), int(req["b"])
            out = WS / f"bank_{sid}_{bb}_seed{seed}_w{a}_{b}.npy"
            t0 = time.time()
            try:
                if bb != daemon_bb:
                    raise RuntimeError(f"this daemon serves {daemon_bb}, not {bb}")
                if out.exists():
                    motion = np.load(out).astype(np.float32)
                else:
                    motion = _gen(sid, settings, seed, a, b)
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
