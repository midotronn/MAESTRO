#!/usr/bin/env python3
"""Model MAESTRO capacity from controlled browser-to-final traces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_pipeline_traces import percentile


def _timeline_duration(trace: dict, stage_name: str) -> float:
    return sum(
        float(stage["duration_seconds"])
        for stage in trace.get("stage_timeline") or []
        if stage.get("stage") == stage_name
        and isinstance(stage.get("duration_seconds"), (int, float))
    )


def _remote_duration(trace: dict, stage_name: str) -> float | None:
    stage = (
        (trace.get("remote_pipeline_timings") or {})
        .get("stages", {})
        .get(stage_name, {})
    )
    duration = stage.get("duration_seconds")
    return float(duration) if isinstance(duration, (int, float)) else None


def _stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "stddev": round(statistics.pstdev(values), 3),
    }


def model_trace_capacity(
    traces: list[dict],
    *,
    target_frames: int = 5400,
    preprocess_p50_budget: float = 8.0,
    preprocess_p95_budget: float = 13.0,
    generation_p50_budget: float = 20.0,
    generation_p95_budget: float = 30.0,
    render_p50_budget: float = 23.0,
    render_p95_budget: float = 32.0,
) -> dict:
    runs = []
    for trace in traces:
        render_seconds = _timeline_duration(trace, "render")
        frames = trace.get("rendered_frames") or trace.get("render_frames")
        if (
            trace.get("status") != "done"
            or not isinstance(frames, int)
            or frames < 1
            or render_seconds <= 0
        ):
            continue
        runs.append(
            {
                "request_id": trace.get("request_id"),
                "benchmark_sid": trace.get("benchmark_sid"),
                "service_state": trace.get("service_state"),
                "browser_total_seconds": float(trace["browser_total_seconds"]),
                "frames": frames,
                "render_seconds": render_seconds,
                "render_frames_per_second": frames / render_seconds,
                "preprocess_edge_seconds": _remote_duration(
                    trace, "preprocess_edge"
                ),
                "generation_lodge_seconds": _remote_duration(
                    trace, "generation_lodge"
                ),
                "generation_edge_seconds": _remote_duration(
                    trace, "generation_edge"
                ),
            }
        )
    if not runs:
        raise ValueError("no complete trace has render frames and duration")

    render_rates = [run["render_frames_per_second"] for run in runs]
    p50_rate = percentile(render_rates, 0.50)
    p05_rate = percentile(render_rates, 0.05)
    required_p50_rate = target_frames / render_p50_budget
    required_p95_rate = target_frames / render_p95_budget
    p50_workers = math.ceil(required_p50_rate / p50_rate)
    p95_workers = math.ceil(required_p95_rate / p05_rate)
    ideal_workers = max(p50_workers, p95_workers)

    role_values = {}
    for key in (
        "preprocess_edge_seconds",
        "generation_lodge_seconds",
        "generation_edge_seconds",
    ):
        values = [run[key] for run in runs if run[key] is not None]
        if values:
            role_values[key] = values

    role_requirements = {}
    if "preprocess_edge_seconds" in role_values:
        values = role_values["preprocess_edge_seconds"]
        role_requirements["jukebox_preprocessing"] = {
            "measured_seconds": _stats(values),
            "required_speedup": {
                "p50_budget": round(
                    percentile(values, 0.50) / preprocess_p50_budget, 2
                ),
                "p95_budget": round(
                    percentile(values, 0.95) / preprocess_p95_budget, 2
                ),
            },
            "parallelism": "Independent audio slices can be distributed.",
        }
    for role, key in (
        ("lodge_generation", "generation_lodge_seconds"),
        ("edge_generation", "generation_edge_seconds"),
    ):
        if key not in role_values:
            continue
        values = role_values[key]
        role_requirements[role] = {
            "measured_seconds": _stats(values),
            "required_speedup": {
                "p50_budget": round(
                    percentile(values, 0.50) / generation_p50_budget, 2
                ),
                "p95_budget": round(
                    percentile(values, 0.95) / generation_p95_budget, 2
                ),
            },
            "parallelism": (
                "A single fixed-seed generation remains serial; role isolation "
                "removes contention but target-GPU latency must be calibrated."
            ),
        }

    return {
        "schema_version": 1,
        "accepted_runs": len(runs),
        "service_states": sorted(
            {str(run["service_state"] or "unknown") for run in runs}
        ),
        "target": {
            "frames": int(target_frames),
            "p50_total_seconds": 60,
            "p95_total_seconds": 90,
            "stage_budgets_seconds": {
                "preprocessing": {
                    "p50": preprocess_p50_budget,
                    "p95": preprocess_p95_budget,
                },
                "generation": {
                    "p50": generation_p50_budget,
                    "p95": generation_p95_budget,
                },
                "render": {
                    "p50": render_p50_budget,
                    "p95": render_p95_budget,
                },
            },
        },
        "browser_total_seconds": _stats(
            [run["browser_total_seconds"] for run in runs]
        ),
        "render": {
            "measured_frames_per_second": _stats(render_rates),
            "p05_frames_per_second": round(p05_rate, 3),
            "required_frames_per_second": {
                "p50_budget": round(required_p50_rate, 3),
                "p95_budget": round(required_p95_rate, 3),
            },
            "ideal_gpu_equivalents": {
                "p50_budget_at_p50_rate": p50_workers,
                "p95_budget_at_p05_rate": p95_workers,
                "minimum_satisfying_both": ideal_workers,
            },
            "planning_gpu_equivalents": {
                "at_90_percent_scaling_efficiency": math.ceil(
                    ideal_workers / 0.90
                ),
                "at_80_percent_scaling_efficiency": math.ceil(
                    ideal_workers / 0.80
                ),
            },
        },
        "roles": role_requirements,
        "runs": [
            {
                **run,
                "render_seconds": round(run["render_seconds"], 3),
                "render_frames_per_second": round(
                    run["render_frames_per_second"], 3
                ),
            }
            for run in runs
        ],
        "warnings": [
            "Render counts are ideal equivalents before measured scaling, coordination, and encoding overhead.",
            "The current render stage includes final encoding and is therefore a conservative frame-rate estimate.",
            "Do not provision from this report until target-GPU one-, two-, and four-worker calibration is complete.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--target-frames", type=int, default=5400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    traces = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.traces
    ]
    report = model_trace_capacity(traces, target_frames=args.target_frames)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
