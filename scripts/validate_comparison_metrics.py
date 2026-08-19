#!/usr/bin/env python3
"""Run deterministic synthetic controls for comparison-ready metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.dance.beat_metrics import beat_alignment_score, beat_coverage
from agentlodge.dance.story_metrics import (
    arc_adherence,
    motif_recurrence,
    seam_jerk,
    sectional_contrast,
    section_repetition_correlation,
)

DEFAULT_OUTPUT = (
    ROOT / "experiments" / "comparisons" / "metric_validation_report.json"
)


def _motion(frames: int) -> np.ndarray:
    motion = np.zeros((frames, 139), dtype=np.float32)
    identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 3:135] = np.tile(identity, 22)
    motion[:, 135:139] = 1.0
    return motion


def validate_metric_controls() -> dict:
    beat_motion = _motion(120)
    music_beats = np.array([15, 45, 75, 105])
    aligned = beat_alignment_score(
        beat_motion,
        music_beats,
        motion_beats=music_beats,
    )
    shifted = beat_alignment_score(
        beat_motion,
        music_beats,
        motion_beats=music_beats + 10,
    )
    full_coverage = beat_coverage(
        beat_motion,
        music_beats,
        motion_beats=music_beats,
    )
    partial_coverage = beat_coverage(
        beat_motion,
        music_beats,
        motion_beats=music_beats[:1],
        tol_frames=2,
    )

    arc_motion = _motion(180)
    t = np.linspace(0.0, 1.0, len(arc_motion), dtype=np.float32)
    arc_motion[:, 0] = t ** 3
    rising = arc_adherence(arc_motion, t)
    falling = arc_adherence(arc_motion, t[::-1])

    sections = [
        SimpleNamespace(start_frame=0, end_frame=60, label="A"),
        SimpleNamespace(start_frame=60, end_frame=120, label="B"),
        SimpleNamespace(start_frame=120, end_frame=180, label="A"),
    ]
    structured = _motion(180)
    structured[60:120, 0] = 2.0
    identical_recurrence = motif_recurrence(structured, sections)
    identical_correlation = section_repetition_correlation(structured, sections)
    contrast = sectional_contrast(structured, sections)
    altered = structured.copy()
    altered[120:180, 0] = -2.0
    altered_recurrence = motif_recurrence(altered, sections)
    altered_correlation = section_repetition_correlation(altered, sections)

    smooth = _motion(180)
    smooth[:, 0] = np.linspace(0, 1, 180)
    discontinuous = smooth.copy()
    discontinuous[120:, 0] += 4.0
    smooth_peak, smooth_area = seam_jerk(smooth, sections)
    jump_peak, jump_area = seam_jerk(discontinuous, sections)

    checks = {
        "beat_alignment": {
            "pass": aligned > 0.999 and shifted < 0.01,
            "aligned": aligned,
            "shifted": shifted,
        },
        "beat_coverage": {
            "pass": full_coverage == 1.0 and partial_coverage == 0.25,
            "full": full_coverage,
            "partial": partial_coverage,
        },
        "energy_arc": {
            "pass": rising > 0.5 and falling < -0.5,
            "matching": rising,
            "reversed": falling,
        },
        "section_structure": {
            "pass": (
                identical_recurrence > altered_recurrence
                and identical_correlation > altered_correlation
                and contrast > 0
            ),
            "identical_recurrence": identical_recurrence,
            "altered_recurrence": altered_recurrence,
            "identical_correlation": identical_correlation,
            "altered_correlation": altered_correlation,
            "sectional_contrast": contrast,
        },
        "transition_jerk": {
            "pass": jump_peak > smooth_peak and jump_area > smooth_area,
            "smooth_peak": smooth_peak,
            "jump_peak": jump_peak,
            "smooth_area": smooth_area,
            "jump_area": jump_area,
        },
    }
    return {
        "schema_version": 1,
        "passed": all(check["pass"] for check in checks.values()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate_metric_controls()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
