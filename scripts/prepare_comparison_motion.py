#!/usr/bin/env python3
"""Convert a retained baseline output into the common MAESTRO motion format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agentlodge.evaluation.adapters import (
    convert_agentlodge_motion,
    convert_bailando_motion,
    convert_finedance_motion,
    save_conversion,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=[
        "lodge",
        "edge",
        "maestro",
        "finedance",
        "bailando_plus_plus",
    ])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--source-up-axis", choices=["y", "z"])
    parser.add_argument("--translation-key", default="trans")
    parser.add_argument("--rotation-key", default="rot_matrices")
    parser.add_argument("--contacts-key", default="contacts")
    args = parser.parse_args()

    if args.method == "finedance":
        result = convert_finedance_motion(
            np.load(args.input),
            source_fps=args.source_fps or 30.0,
            source_up_axis=args.source_up_axis or "y",
        )
    elif args.method == "bailando_plus_plus":
        payload = np.load(args.input)
        contacts = payload[args.contacts_key] if args.contacts_key in payload else None
        result = convert_bailando_motion(
            payload[args.translation_key],
            payload[args.rotation_key],
            contacts=contacts,
            source_fps=args.source_fps or 60.0,
            source_up_axis=args.source_up_axis or "y",
        )
    else:
        result = convert_agentlodge_motion(
            np.load(args.input),
            method=args.method,
            source_fps=args.source_fps or 30.0,
        )
    save_conversion(
        result,
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
    )
    print(
        f"{args.method}: {result.report['frames']} frames at "
        f"{result.report['target_fps']} FPS -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
