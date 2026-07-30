"""Generate ONE seeded LODGE/EDGE full take on the pod for live window editing.

The interactive editor's *live pod mode* (:class:`agentlodge.editor.remote_generator.LiveWindowGenerator`)
calls this once per unseen ``(backbone, seed)``: it runs the real diffusion backbone at ``seed``,
converts the take into the assembled Z-up 139 space (identical to build_story_dance /
build_window_bank so slices align frame-for-frame), and caches it as
``bank_<sid>_<backbone>_seed<n>.npy`` in ``WORKSPACE``. The client then scps that file back and
slices whatever window it needs. Because the file name matches the candidate-bank convention, a live
session also *grows the bank* for free.

Seed 0 reuses the already-generated best take (``lodge_fd_<sid>_full.npy`` / ``edge_fd_<sid>_full.npy``)
when present; other seeds re-run the backbone from the cached features/slices (no Jukebox needed).

Usage:   python scripts/gen_take.py <sid> <lodge|edge> <seed>
Env:     WORKSPACE (default /workspace)
Prints:  ``TAKE_CACHED <path>``  if already on disk,
         ``TAKE_DONE <path> <frames>``  on a fresh generation,
         ``TAKE_ERROR <reason>``  on failure (exit code 2) so the client can fall back cleanly.
"""
import os
import sys
from pathlib import Path

import numpy as np

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
sys.path.insert(0, f"{WORKSPACE}/AgentLODGE")

WS = Path(WORKSPACE)


def _fail(reason: str) -> "NoReturn":  # noqa: F821
    print(f"TAKE_ERROR {reason}", flush=True)
    raise SystemExit(2)


def _to_lodge_zup(raw):
    from agentlodge.dance.format import ensure_lodge139, to_agentlodge139
    from agentlodge.dance.transition import to_zup
    return to_zup(to_agentlodge139(ensure_lodge139(np.asarray(raw, dtype=np.float32))))


def _to_edge_zup(raw):
    from agentlodge.dance.format import ensure_lodge139, to_agentlodge139
    return to_agentlodge139(ensure_lodge139(np.asarray(raw, dtype=np.float32)))


def _generate(sid: str, backbone: str, seed: int) -> np.ndarray:
    """Produce a full Z-up 139 take for ``(backbone, seed)`` (reuses best take at seed 0)."""
    if seed == 0:
        best = WS / f"{backbone}_fd_{sid}_full.npy"
        if best.exists():
            raw = np.load(best).astype(np.float32)
            return _to_lodge_zup(raw) if backbone == "lodge" else _to_edge_zup(raw)

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
    work = str(WS / f"gen{sid}_work")

    if backbone == "lodge":
        feats_p = WS / f"lodge_fd_{sid}_feats.npy"
        if not feats_p.exists():
            _fail(f"missing {feats_p.name} (song not preprocessed for generation on this pod)")
        job = _run_lodge_job(np.load(feats_p).astype(np.float32), sd, work, seed=seed)
        if job.get("error"):
            _fail(f"LODGE seed {seed}: {str(job['error'])[:180]}")
        return _to_lodge_zup(job["motion"])

    slices_p = WS / f"edge{sid}_slices.npy"
    wav = f"{WORKSPACE}/LODGE/data/finedance/music_wav/{sid}.wav"
    if not slices_p.exists():
        _fail(f"missing {slices_p.name} (song not preprocessed for generation on this pod)")
    edge_slices = [np.asarray(s, dtype=np.float32) for s in np.load(slices_p)]
    job = _run_edge_job(wav, edge_slices, sd, work, seed=seed)
    if job.get("error"):
        _fail(f"EDGE seed {seed}: {str(job['error'])[:180]}")
    return _to_edge_zup(job["motion"])


def main() -> None:
    if len(sys.argv) != 4:
        _fail("usage: gen_take.py <sid> <lodge|edge> <seed>")
    sid, backbone, seed = sys.argv[1], sys.argv[2].lower(), int(sys.argv[3])
    if backbone not in ("lodge", "edge"):
        _fail(f"unknown backbone {backbone!r}")

    out = WS / f"bank_{sid}_{backbone}_seed{seed}.npy"
    if out.exists():
        print(f"TAKE_CACHED {out}", flush=True)
        return

    motion = np.asarray(_generate(sid, backbone, seed), dtype=np.float32)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, motion)
    os.replace(tmp, out)                       # atomic: client never scps a half-written file
    print(f"TAKE_DONE {out} {motion.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
