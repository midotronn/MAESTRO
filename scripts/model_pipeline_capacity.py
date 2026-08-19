#!/usr/bin/env python3
"""Derive transparent throughput requirements from a measured MAESTRO trace."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "experiments" / "performance" / "historical_single_gpu.json"


def model_capacity(
    baseline: dict,
    *,
    target_frames: int = 5400,
    generation_budget_seconds: float = 20.0,
    render_budget_seconds: float = 23.0,
) -> dict:
    timings = baseline["timings_seconds"]
    workload = baseline["workload"]
    generation_seconds = float(timings["preprocessing_and_generation"])
    render_seconds = float(timings["full_quality_rendering_and_staging"])
    measured_frames = int(workload["output_frames"])
    render_fps = measured_frames / render_seconds
    required_render_fps = target_frames / render_budget_seconds
    render_gpu_equivalents = math.ceil(required_render_fps / render_fps)
    duration_scale = target_frames / measured_frames
    scaled_generation_seconds = generation_seconds * duration_scale
    generation_throughput_equivalents = math.ceil(
        scaled_generation_seconds / generation_budget_seconds
    )
    return {
        "baseline_record_id": baseline["record_id"],
        "target_frames": target_frames,
        "budgets_seconds": {
            "preprocessing_and_generation": generation_budget_seconds,
            "rendering_and_encoding": render_budget_seconds,
        },
        "historical_throughput": {
            "render_frames_per_second": round(render_fps, 3),
            "scaled_preprocessing_and_generation_seconds": round(
                scaled_generation_seconds, 1
            ),
        },
        "required_throughput": {
            "render_frames_per_second": round(required_render_fps, 3),
            "render_gpu_equivalents_under_ideal_linear_scaling": render_gpu_equivalents,
            "generation_throughput_equivalents_under_ideal_linear_scaling": (
                generation_throughput_equivalents
            ),
        },
        "warnings": [
            "These are throughput equivalents, not a recommended topology.",
            "Generation is not fully parallelizable, so its equivalent count is only a lower-bound diagnostic.",
            "The historical render timing includes staging and comes from one run.",
            "Replace this model with p50 and p95 trace data before provisioning GPUs."
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--target-frames", type=int, default=5400)
    parser.add_argument("--generation-budget", type=float, default=20.0)
    parser.add_argument("--render-budget", type=float, default=23.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.baseline.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    report = model_capacity(
        baseline,
        target_frames=args.target_frames,
        generation_budget_seconds=args.generation_budget,
        render_budget_seconds=args.render_budget,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
