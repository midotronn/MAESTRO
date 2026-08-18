"""Render a compact visual review sheet for every canonical named motion."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.editor.motion_bank import default_motion_bank  # noqa: E402
from server.fk import BODY_PARENTS, compute_poses  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "assets" / "motion_bank" / "review_contact_sheet.png",
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bank = default_motion_bank()
    columns = 4
    rows = int(math.ceil(len(bank.specs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.4 * rows), facecolor="#f7f5fc")
    colors = ("#9b8aca", "#6c4ce0", "#f07845")
    flat_axes = list(np.asarray(axes).flat)
    for ax, spec in zip(flat_axes, bank.specs):
        motion = bank.load_clip(spec)
        joints = compute_poses(motion)["fk_joints"]
        event = int(spec.event_frame)
        frames = (
            max(0, event - max(4, spec.frames // 5)),
            event,
            min(spec.frames - 1, event + max(4, spec.frames // 5)),
        )
        for offset, frame, color in zip((-0.75, 0.0, 0.75), frames, colors):
            pose = joints[frame]
            # Centre each frame horizontally on its own root: travelling actions (steps, turns)
            # otherwise drift into the neighbouring frame and read as a broken skeleton. Vertical
            # position is kept so jumps, crouches, and rises still show their level change.
            pose = pose - np.array([joints[frame][0, 0], joints[frame][0, 1], 0.0])
            x = pose[:, 0] + 0.55 * pose[:, 1] + offset
            z = pose[:, 2]
            for child, parent in enumerate(BODY_PARENTS):
                if parent >= 0:
                    ax.plot([x[parent], x[child]], [z[parent], z[child]],
                            color=color, linewidth=1.6, alpha=0.9)
            ax.scatter(x, z, s=5, color=color)
        ax.set_title(spec.name, fontsize=10, fontweight="bold")
        ax.text(0.5, 0.01, spec.category, transform=ax.transAxes, ha="center",
                fontsize=8, color="#736f80")
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in flat_axes[len(bank.specs):]:
        ax.axis("off")
    fig.suptitle(
        "MAESTRO named motion bank\nbefore event  |  event  |  after event",
        fontsize=17, fontweight="bold", color="#25212d",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
