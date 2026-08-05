"""Stitch the canonical bank clips into one continuous reel for visual review.

Rendering twenty two-second clips separately gives twenty videos that are all
camera-framed differently and impossible to compare. This walks the manifest in
order, holds each action briefly so the eye can register where it lands, and
eases between them through the same seam blend the editor uses, so what you
watch is the clip quality rather than the stitching.

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

from agentlodge.editor.motion_bank import MotionBank, _ease_join  # noqa: E402


def build_reel(bank: MotionBank, hold: int, blend: int) -> tuple[np.ndarray, list[dict]]:
    reel = np.zeros((0, 139), dtype=np.float32)
    spans: list[dict] = []
    for spec in bank.specs:
        clip = bank.load_clip(spec)
        held = np.concatenate([clip, np.repeat(clip[-1:], hold, axis=0)], axis=0)
        start = reel.shape[0]
        reel = _ease_join(reel, held, blend) if reel.shape[0] else held.astype(np.float32)
        spans.append(
            {
                "id": spec.id,
                "label": spec.name,
                "start": start,
                "end": reel.shape[0],
                "event": start + int(spec.event_frame),
            }
        )
    return reel, spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=12, help="frames to hold each action's final pose")
    ap.add_argument("--blend", type=int, default=8, help="frames of seam blend between actions")
    ap.add_argument("--out", type=Path, default=REPO / "assets/motion_bank/review_reel.npy")
    args = ap.parse_args()

    bank = MotionBank()
    reel, spans = build_reel(bank, args.hold, args.blend)

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
