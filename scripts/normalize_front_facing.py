"""Normalize generated motion files to one camera-facing root heading."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentlodge.dance.transition import root_facing_yaw, stabilize_root_facing


def save_atomic(path: Path, motion: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".front-facing.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, motion)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("related", nargs="*", type=Path)
    args = parser.parse_args()

    base = np.load(args.base)
    target_yaw = root_facing_yaw(base)
    save_atomic(args.base, stabilize_root_facing(base, target_yaw=target_yaw))
    for path in args.related:
        save_atomic(
            path,
            stabilize_root_facing(np.load(path), target_yaw=target_yaw),
        )
    print(f"FRONT_FACING_READY yaw={target_yaw:.8f} files={1 + len(args.related)}")


if __name__ == "__main__":
    main()
