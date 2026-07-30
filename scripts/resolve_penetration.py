"""Reduce hand-through-body self-penetration in an assembled AgentLODGE 139 motion.

LODGE/EDGE are kinematic generators that do not enforce self-collision, so wrists sometimes pass
through the torso. This is a training-free cleanup: FK the SMPL joints (LODGE SMPLX_Skeleton),
model the torso core as a capsule along the pelvis->neck axis, and for frames where a wrist is
clearly INSIDE the capsule, greedily nudge that arm's shoulder rotation outward (guided only by FK
wrist-to-axis distance, so there are no rotation-convention pitfalls) until the wrist clears the
surface + a margin. Corrections are capped and temporally smoothed so the dance is preserved.

Usage:
    python resolve_penetration.py <in_motion.npy> <out_motion.npy> [--radius 0.11] [--margin 0.02]
                                  [--max-deg 30] [--report-only]
Env: WORKSPACE (default /workspace), LODGE at $WORKSPACE/LODGE.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
sys.path.insert(0, f"{WORKSPACE}/AgentLODGE")
from agentlodge.dance.format import to_native_finedance139
from agentlodge.env_paths import lodge_import_paths, use_code_paths

LODGE = f"{WORKSPACE}/LODGE"
JPATH = f"{LODGE}/data/smplx_neu_J_1.npy"

PELVIS, NECK = 0, 12
L_SH, R_SH, L_WR, R_WR = 16, 17, 20, 21
ARMS = [("L", L_SH, L_WR, 18), ("R", R_SH, R_WR, 19)]  # name, shoulder, wrist, elbow
_HAND = 0.11  # hand mesh reaches ~this far past the wrist joint (toward the fingertips)


def _fk_module():
    with use_code_paths(*lodge_import_paths(Path(LODGE))):
        from dld.data.render_joints.smplfk import SMPLX_Skeleton, ax_from_6v, ax_to_6v
    return SMPLX_Skeleton, ax_from_6v, ax_to_6v


def _joints(fk, ax, trans):
    """ax (N,22,3) axis-angle, trans (N,3) -> joints (N,22,3)."""
    poses = torch.from_numpy(ax.reshape(-1, 66)).float()
    tr = torch.from_numpy(trans).float()
    with torch.no_grad():
        j = fk.forward(poses, tr)
    return j[:, :22].detach().cpu().numpy()


def _dist_to_axis(w, a, b):
    """Perp distance of points w (N,3) to the segment a->b (N,3 each), + closest point."""
    ab = b - a
    t = np.clip(np.sum((w - a) * ab, axis=1) / (np.sum(ab * ab, axis=1) + 1e-9), 0, 1)
    c = a + t[:, None] * ab
    d = np.linalg.norm(w - c, axis=1)
    return d, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--radius", type=float, default=0.11)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--max-deg", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--step-deg", type=float, default=4.0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    motion = np.load(args.inp).astype(np.float32)          # AgentLODGE 139
    native = to_native_finedance139(motion)
    trans = native[:, 4:7].astype(np.float32)
    rot6d = native[:, 7:139].reshape(-1, 22, 6).astype(np.float32)
    n = motion.shape[0]

    SMPLX_Skeleton, ax_from_6v, ax_to_6v = _fk_module()
    fk = SMPLX_Skeleton(device="cpu", batch=1, Jpath=JPATH)
    ax = ax_from_6v(torch.from_numpy(rot6d).float()).reshape(-1, 22, 3).numpy().astype(np.float32)
    ax0 = ax.copy()

    j = _joints(fk, ax, trans)
    a, b = j[:, PELVIS], j[:, NECK]
    thr = args.radius
    target = thr + args.margin

    def tipdist(jj, wr, el, aa, bb):
        v = jj[:, wr] - jj[:, el]
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        d, _ = _dist_to_axis(jj[:, wr] + _HAND * v, aa, bb)
        return d

    for name, sh, wr, el in ARMS:
        dt = tipdist(j, wr, el, a, b)
        print(f"[penetration] frames={n} r={thr:.3f} hand-inside before {name}: "
              f"{(dt < thr).mean() * 100:.2f}%  (min {dt.min():.3f})", flush=True)
    if args.report_only:
        print("REPORT_ONLY_DONE", flush=True)
        return

    step = np.radians(args.step_deg)
    cap = np.radians(args.max_deg)
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32)

    for name, sh, wr, el in ARMS:
        d = tipdist(j, wr, el, a, b)
        P = np.where(d < thr)[0]
        if P.size == 0:
            continue
        cur = ax[P].copy()
        trP, aP, bP = trans[P], a[P], b[P]
        for _ in range(args.iters):
            dl = tipdist(_joints(fk, cur, trP), wr, el, aP, bP)
            need = dl < target
            if not need.any():
                break
            best_d = dl.copy()
            best = cur[:, sh, :].copy()
            for e in axes:
                trial = cur.copy()
                newv = trial[:, sh, :] + step * e
                over = np.linalg.norm(newv - ax0[P][:, sh, :], axis=1) > cap
                newv[over] = trial[over, sh, :]
                trial[:, sh, :] = newv
                wd = tipdist(_joints(fk, trial, trP), wr, el, aP, bP)
                take = need & (wd > best_d)
                best[take] = trial[take, sh, :]
                best_d[take] = wd[take]
            cur[:, sh, :] = best
        ax[P] = cur

    # temporal smoothing of the shoulder corrections (keep the rest of the pose intact)
    corr = ax - ax0
    k = np.ones(5, np.float32) / 5
    for jid in (L_SH, R_SH):
        for c in range(3):
            corr[:, jid, c] = np.convolve(np.pad(corr[:, jid, c], 2, mode="edge"), k, "valid")
    ax = ax0 + corr

    j2 = _joints(fk, ax, trans)
    a2, b2 = j2[:, PELVIS], j2[:, NECK]
    for name, sh, wr, el in ARMS:
        dt2 = tipdist(j2, wr, el, a2, b2)
        print(f"[penetration] hand-inside after {name}: {(dt2 < thr).mean() * 100:.2f}%  "
              f"(max shoulder correction {np.degrees(np.linalg.norm(ax[:, sh] - ax0[:, sh], axis=1).max()):.1f} deg)",
              flush=True)

    # write corrected shoulder rotations back into the AgentLODGE 139 (same joint order; rot at 3+j*6)
    new6d = ax_to_6v(torch.from_numpy(ax).float()).reshape(-1, 22, 6).numpy().astype(np.float32)
    out = motion.copy()
    for jid in (L_SH, R_SH):
        out[:, 3 + jid * 6: 3 + jid * 6 + 6] = new6d[:, jid, :]
    np.save(args.out, out)
    print(f"RESOLVE_PENETRATION_DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
