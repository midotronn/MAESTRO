#!/usr/bin/env python3
"""Validate the frozen comparison and three-minute performance protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = ROOT / "experiments" / "comparisons"
PERFORMANCE_DIR = ROOT / "experiments" / "performance"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_protocols(root: Path = ROOT) -> list[str]:
    comparison_dir = root / "experiments" / "comparisons"
    performance_dir = root / "experiments" / "performance"
    protocol = _load(comparison_dir / "protocol.json")
    songs = _load(comparison_dir / "songs.json")
    baselines = _load(comparison_dir / "baselines.json")
    performance = _load(performance_dir / "three_minute_protocol.json")
    errors: list[str] = []

    mandatory = set(protocol["methods"]["mandatory"])
    if mandatory != {"lodge", "edge", "maestro"}:
        errors.append("comparison protocol must require LODGE, EDGE, and MAESTRO")
    if protocol["methods"]["external_target_count"] != 2:
        errors.append("comparison protocol must retain exactly two external methods")
    if len(songs["songs"]) != protocol["song_sets"]["long_form"]["target_count"]:
        errors.append("long-form song registry does not match the target count")
    standard = protocol["song_sets"]["standard_duration"]
    expected_excerpts = len(songs["songs"]) * songs["standard_excerpt_derivation"]["windows_per_song"]
    if standard["target_count"] != expected_excerpts:
        errors.append("standard-duration target does not match deterministic excerpt derivation")
    if len(baselines["external_candidates"]) < baselines["selection_target"]:
        errors.append("not enough external candidates to satisfy the selection target")
    if protocol["first_round"]["primary_condition"]["seed"] != 0:
        errors.append("primary first-round condition must use preregistered seed 0")
    if protocol["rendering"]["playback_speed"] != 1.0:
        errors.append("comparison playback speed must remain 1.0")
    if protocol["metrics"]["composite_score_allowed"]:
        errors.append("unvalidated metric composites are forbidden")

    workload = performance["workload"]
    quality = performance["quality_contract"]
    if workload["expected_frames"] != workload["song_duration_seconds"] * workload["fps"]:
        errors.append("performance frame count does not match duration and FPS")
    if (quality["width"], quality["height"], quality["samples"]) != (1080, 1080, 96):
        errors.append("performance protocol changed the established full-quality settings")
    if not quality["render_every_frame"]:
        errors.append("performance protocol must render every frame")
    if performance["sla_seconds"] != {"p50": 60, "p95": 90}:
        errors.append("performance SLA must remain p50 60s and p95 90s")
    if workload["minimum_validation_runs"] < 20:
        errors.append("SLA validation requires at least 20 runs")
    if "explicit user approval" not in performance["spend_gate"]:
        errors.append("performance protocol must retain the paid-resource approval gate")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = validate_protocols(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Focused comparison and performance protocols are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
