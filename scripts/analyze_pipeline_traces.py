#!/usr/bin/env python3
"""Summarize browser-to-final MAESTRO performance traces."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stage_durations(trace: dict) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for stage in trace.get("stage_timeline") or []:
        duration = stage.get("duration_seconds")
        if isinstance(duration, (int, float)):
            totals[str(stage.get("stage") or "unknown")] += float(duration)
    remote = (trace.get("remote_pipeline_timings") or {}).get("stages") or {}
    for stage, summary in remote.items():
        duration = summary.get("duration_seconds")
        if isinstance(duration, (int, float)):
            totals[f"remote:{stage}"] += float(duration)
    return dict(totals)


def summarize_traces(traces: list[dict], *, warm_only: bool = True) -> dict:
    accepted = []
    rejected = []
    for trace in traces:
        if warm_only and trace.get("service_state") != "warm":
            rejected.append(
                {
                    "request_id": trace.get("request_id"),
                    "reason": "service_state is not warm",
                }
            )
            continue
        total = trace.get("browser_total_seconds")
        if not isinstance(total, (int, float)):
            rejected.append(
                {
                    "request_id": trace.get("request_id"),
                    "reason": "browser_total_seconds is missing",
                }
            )
            continue
        accepted.append(trace)
    if not accepted:
        raise ValueError("no complete traces matched the requested filters")

    totals = [float(trace["browser_total_seconds"]) for trace in accepted]
    uploads = [
        float(trace["browser_upload_seconds"])
        for trace in accepted
        if isinstance(trace.get("browser_upload_seconds"), (int, float))
    ]
    stage_values: dict[str, list[float]] = defaultdict(list)
    for trace in accepted:
        for stage, duration in _stage_durations(trace).items():
            stage_values[stage].append(duration)

    def stats(values: list[float]) -> dict:
        return {
            "n": len(values),
            "p50": round(percentile(values, 0.50), 3),
            "p95": round(percentile(values, 0.95), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "accepted_runs": len(accepted),
        "rejected_runs": rejected,
        "browser_total_seconds": stats(totals),
        "browser_upload_seconds": stats(uploads) if uploads else None,
        "stages": {
            stage: stats(values)
            for stage, values in sorted(stage_values.items())
        },
        "sla": {
            "p50_target_seconds": 60,
            "p95_target_seconds": 90,
            "p50_pass": percentile(totals, 0.50) <= 60,
            "p95_pass": percentile(totals, 0.95) <= 90,
        },
        "request_ids": [trace.get("request_id") for trace in accepted],
    }


def _load_paths(paths: list[Path]) -> list[dict]:
    traces = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            traces.append(json.load(handle))
    return traces


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--include-non-warm", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize_traces(
        _load_paths(args.traces),
        warm_only=not args.include_non_warm,
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
