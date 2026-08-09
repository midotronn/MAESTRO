"""Write the render-completion receipt required by the motion-audit finalizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.editor.motion_audit import record_audit_render_receipt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    receipt = record_audit_render_receipt(args.audit)
    print(
        f"recorded render receipt for {receipt['audit_id']}: "
        f"{len(receipt['artifacts'])} artifacts"
    )


if __name__ == "__main__":
    main()
