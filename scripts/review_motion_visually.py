"""Render a named motion large enough to judge its skeleton from multiple angles.

The contact sheet this sits beside packs the full bank onto one page, and at that size a
broken clap -- two wrists driven through each other to the same point -- is indistinguishable
from a good one. Hands near the chest look like hands meeting. So this renders ONE motion at a
time, big, and adds a top-down view where left and right wrists either sit side by side or occupy
the same point. SMPL FK stops at the wrist, so this cannot judge palm orientation; clap review
must also include a close Y-Bot render from the front and side.

    python scripts/review_motion_visually.py clap_single
    python scripts/review_motion_visually.py --all --outdir /tmp/review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.editor.motion_bank import default_motion_bank  # noqa: E402
from server.fk import BODY_PARENTS, compute_poses  # noqa: E402

L_WRIST, R_WRIST = 20, 21
COLS = 7


def _draw(ax, pose, h, v, *, lo, hi, title, gap):
    for joint, parent in enumerate(BODY_PARENTS):
        if parent < 0:
            continue
        ax.plot([pose[joint, h], pose[parent, h]], [pose[joint, v], pose[parent, v]],
                color="#6c4ce0", linewidth=2.0, solid_capstyle="round", zorder=2)
    ax.scatter([pose[L_WRIST, h]], [pose[L_WRIST, v]], s=110, color="#1f77ff",
               zorder=4, edgecolors="white", linewidths=1.4)
    ax.scatter([pose[R_WRIST, h]], [pose[R_WRIST, v]], s=110, color="#f0453a",
               zorder=4, edgecolors="white", linewidths=1.4)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, color="#333")
    if gap is not None:
        ax.text(0.5, 0.01, f"{gap:.3f} m", transform=ax.transAxes, ha="center",
                fontsize=10, color="#c02020" if gap < 0.06 else "#207020")


def review(motion_id: str, outdir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank = default_motion_bank()
    spec = next(s for s in bank.specs if s.id == motion_id)
    motion = bank.load_clip(spec)
    joints = compute_poses(motion)["fk_joints"]
    gaps = np.linalg.norm(joints[:, L_WRIST] - joints[:, R_WRIST], axis=-1)
    # Always show the closest approach. An evenly spaced sample steps straight over it -- the
    # frame that decides whether a clap is a clap is exactly the one most likely to be missed.
    closest = int(np.argmin(gaps))
    frames = np.linspace(0, motion.shape[0] - 1, COLS - 1).round().astype(int)
    frames = np.unique(np.append(frames, closest))

    fig, axes = plt.subplots(4, len(frames), figsize=(3.1 * len(frames), 13.5), facecolor="white")
    # Keep world height in the front row. Subtracting the root centres the skeleton but also
    # subtracts the jump out of a jump and the drop out of a crouch, which is the whole motion.
    floor = float(joints[..., 2].min())
    ceil_ = float(joints[..., 2].max())
    for col, f in enumerate(frames):
        pose = joints[f] - joints[f, 0]
        tall = joints[f].copy()
        tall[:, :2] -= joints[f, 0, :2]
        mark = " <- event" if abs(int(f) - int(spec.event_frame)) <= 1 else ""
        mark += " *closest*" if int(f) == closest else ""
        # Front: the way you would watch the dancer, standing on the floor.
        _draw(axes[0, col], tall, 0, 2, lo=(-1.0, floor - 0.1), hi=(1.0, ceil_ + 0.1),
              title=f"f{f}{mark}", gap=None)
        # Side: a body roll or a chest pop travels through the spine front-to-back and is simply
        # not visible head-on. Without this view there is no honest way to say whether it happened.
        _draw(axes[1, col], tall, 1, 2, lo=(-1.0, floor - 0.1), hi=(1.0, ceil_ + 0.1),
              title=None, gap=None)
        # Top down: the only view where two hands at one point cannot hide.
        _draw(axes[2, col], pose, 0, 1, lo=(-0.8, -0.8), hi=(0.8, 0.8),
              title=None, gap=float(gaps[f]))
        # Hands, close.
        mid = 0.5 * (pose[L_WRIST] + pose[R_WRIST])
        _draw(axes[3, col], pose, 0, 2, lo=(mid[0] - 0.22, mid[2] - 0.22),
              hi=(mid[0] + 0.22, mid[2] + 0.22), title=None, gap=None)

    axes[0, 0].set_ylabel("front", fontsize=12)
    # Every panel is drawn root-centred so the skeleton stays in frame, which hides exactly what a
    # step or a turn is for. State the movement numerically instead of pretending the picture shows it.
    from agentlodge.editor.motion_bank import _root_yaw_series
    yaw = _root_yaw_series(motion)
    travel = float(np.linalg.norm(joints[:, 0, :2] - joints[0, 0, :2], axis=-1).max())
    spin = float(np.rad2deg(np.abs(yaw - yaw[0]).max()))
    lift = float(joints[:, 0, 2].max() - joints[:, 0, 2].min())
    fig.suptitle(
        f"{motion_id}   {motion.shape[0]} frames ({motion.shape[0] / 30:.2f}s)   "
        f"min wrist gap {gaps.min():.4f} m   travel {travel:.2f} m   "
        f"turn {spin:.0f} deg   root rise/fall {lift:.2f} m   blue = left hand, red = right",
        fontsize=15, y=0.985,
    )
    fig.text(0.005, 0.87, "FRONT", rotation=90, fontsize=12, color="#555")
    fig.text(0.005, 0.63, "SIDE", rotation=90, fontsize=12, color="#555")
    fig.text(0.005, 0.38, "TOP DOWN", rotation=90, fontsize=12, color="#555")
    fig.text(0.005, 0.13, "HANDS", rotation=90, fontsize=12, color="#555")
    fig.tight_layout(rect=(0.012, 0, 1, 0.97))
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"review_{motion_id}.png"
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_id", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--outdir", type=Path, default=ROOT / "assets" / "motion_bank" / "review")
    args = ap.parse_args()

    ids = [s.id for s in default_motion_bank().specs] if args.all else args.motion_id
    if not ids:
        ap.error("give a motion id or --all")
    for mid in ids:
        print(review(mid, args.outdir))


if __name__ == "__main__":
    main()
