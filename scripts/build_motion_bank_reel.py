"""Stitch the canonical bank clips into one continuous reel for visual review.

Rendering twenty two-second clips separately gives twenty videos that are all
camera-framed differently and impossible to compare. This walks the manifest in
order, holds each action briefly so the eye can register where it lands, and
bridges between them with the editor's own seam blend, so what you watch is the
clip quality rather than the stitching.

Every action starts from the canonical facing, which keeps the camera framing
consistent — without it the two turns would leave the dancer facing away for the
rest of the reel. The cost is that a seam can span a large pose change, so the
bridge between actions is sized to how far the pose actually has to travel and
is placed *before* each action rather than blended into its opening frames. A
fixed short blend snapped 180 degrees of turn_half in eight frames, which reads
as a pop rather than a reset.

Writes the reel next to the clips plus a JSON sidecar mapping each action id to
its frame range, which the render wrapper turns into on-screen labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agentlodge.editor.motion_bank import _ROT, MotionBank, _ease_join, _sixd_to_matrix  # noqa: E402


def seam_travel(left: np.ndarray, right: np.ndarray) -> float:
    """Largest per-joint rotation, in radians, between two poses."""
    a = _sixd_to_matrix(left[None, _ROT].reshape(1, 22, 6))
    b = _sixd_to_matrix(right[None, _ROT].reshape(1, 22, 6))
    rel = b @ np.swapaxes(a, -1, -2)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    return float(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)).max())


def build_reel(
    bank: MotionBank, hold: int, min_bridge: int, max_bridge: int
) -> tuple[np.ndarray, list[dict]]:
    reel = np.zeros((0, 139), dtype=np.float32)
    spans: list[dict] = []
    for spec in bank.specs:
        clip = bank.load_clip(spec)
        if reel.shape[0]:
            # Ease to the action's opening pose first, then play the action untouched.
            travel = seam_travel(reel[-1], clip[0])
            frames = int(np.clip(round(min_bridge + travel * 12.0), min_bridge, max_bridge))
            bridge = np.repeat(clip[None, 0], frames, axis=0)
            reel = _ease_join(reel, bridge, frames)
            reel = _ease_join(reel, clip, 0)
        else:
            reel = clip.astype(np.float32)
        start = reel.shape[0] - clip.shape[0]
        spans.append(
            {
                "id": spec.id,
                "label": spec.name,
                "start": start,
                "end": reel.shape[0],
                "event": start + int(spec.event_frame),
            }
        )
        reel = np.concatenate([reel, np.repeat(reel[-1:], hold, axis=0)], axis=0)
    return reel, spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=12, help="frames to hold each action's final pose")
    ap.add_argument("--min-bridge", type=int, default=8, help="shortest reset between actions")
    ap.add_argument("--max-bridge", type=int, default=30, help="longest reset between actions")
    ap.add_argument("--out", type=Path, default=REPO / "assets/motion_bank/review_reel.npy")
    args = ap.parse_args()

    bank = MotionBank()
    reel, spans = build_reel(bank, args.hold, args.min_bridge, args.max_bridge)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, reel)
    sidecar = args.out.with_suffix(".labels.json")
    sidecar.write_text(json.dumps({"fps": 30, "frames": int(reel.shape[0]), "spans": spans}, indent=2))

    drift = np.abs(reel[-1, :2] - reel[0, :2]).max()
    print(f"reel {reel.shape} -> {args.out}")
    print(f"labels -> {sidecar}")
    print(f"{len(spans)} actions, {reel.shape[0] / 30:.1f}s, max planar drift {drift:.2f}m")
    for span in spans:
        print(f"  {span['start']:5d}-{span['end']:5d}  {span['id']}")


if __name__ == "__main__":
    main()
