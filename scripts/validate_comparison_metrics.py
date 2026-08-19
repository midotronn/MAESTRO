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

from agentlodge.dance.beat_metrics import (
    beat_alignment_score,
    beat_coverage,
    foot_contact_consistency,
)
from agentlodge.dance.transition import _matrix_to_sixd, mirror, retrograde
from agentlodge.dance.story_metrics import (
    arc_adherence,
    boundary_alignment,
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


def _motif(frames: int = 60) -> np.ndarray:
    motion = _motion(frames)
    phase = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    motion[:, 0] = 0.4 * np.sin(phase)
    motion[:, 1] = 0.2 * np.cos(phase * 2.0)
    motion[:, 2] = 0.1 * np.sin(phase * 3.0)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (frames, 22, 3, 3),
    ).copy()
    cosine = np.cos(0.4 * np.sin(phase))
    sine = np.sin(0.4 * np.sin(phase))
    rotations[:, 5, 0, 0] = cosine
    rotations[:, 5, 0, 1] = -sine
    rotations[:, 5, 1, 0] = sine
    rotations[:, 5, 1, 1] = cosine
    motion[:, 3:135] = _matrix_to_sixd(rotations).reshape(frames, 132)
    return motion


def _structure_score(variant: np.ndarray, base: np.ndarray, other: np.ndarray, sections: list) -> dict:
    motion = np.concatenate([base, other, variant], axis=0)
    return {
        "raw_repetition": section_repetition_correlation(motion, sections),
        "invariant_recurrence": motif_recurrence(motion, sections),
    }


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
    base = _motif()
    other = _motif()
    other[:, 0] = np.linspace(-0.5, 0.5, len(other))
    other[:, 1] = 0.3 * np.sin(np.linspace(0.0, 5.0 * np.pi, len(other)))
    structured = np.concatenate([base, other, base], axis=0)
    contrast = sectional_contrast(structured, sections)
    rng = np.random.default_rng(7)
    unrelated = _motif()
    unrelated[:, :12] += rng.normal(0.0, 0.5, size=(len(unrelated), 12))
    shifted_motif = np.roll(base, 6, axis=0)
    mirrored_motif = mirror(base)
    retrograded_motif = retrograde(base)
    frozen_motif = np.repeat(base[:1], len(base), axis=0)
    jittered_motif = base.copy()
    jittered_motif[:, :12] += rng.normal(
        0.0, 0.18, size=(len(jittered_motif), 12)
    )
    structure_controls = {
        "identical": _structure_score(base, base, other, sections),
        "unrelated": _structure_score(unrelated, base, other, sections),
        "shifted": _structure_score(shifted_motif, base, other, sections),
        "mirrored": _structure_score(mirrored_motif, base, other, sections),
        "retrograded": _structure_score(
            retrograded_motif, base, other, sections
        ),
        "frozen": _structure_score(frozen_motif, base, other, sections),
        "jittered": _structure_score(jittered_motif, base, other, sections),
    }

    smooth = _motion(180)
    smooth[:, 0] = np.linspace(0, 1, 180)
    discontinuous = smooth.copy()
    discontinuous[120:, 0] += 4.0
    smooth_peak, smooth_area = seam_jerk(smooth, sections)
    jump_peak, jump_area = seam_jerk(discontinuous, sections)
    jitter_motion = np.concatenate([base, other, jittered_motif], axis=0)
    jitter_peak, jitter_area = seam_jerk(jitter_motion, sections)

    aligned_boundaries = _motion(180)
    aligned_boundaries[60:, 0] += 2.0
    aligned_boundaries[120:, 0] -= 4.0
    misaligned_boundaries = _motion(180)
    misaligned_boundaries[30:, 0] += 2.0
    misaligned_boundaries[90:, 0] -= 4.0
    misaligned_boundaries[150:, 0] += 2.0
    aligned_boundary_score = boundary_alignment(
        aligned_boundaries,
        sections,
        tol_seconds=0.2,
    )
    misaligned_boundary_score = boundary_alignment(
        misaligned_boundaries,
        sections,
        tol_seconds=0.2,
    )

    planted = _motion(60)
    sliding = planted.copy()
    sliding[:, 0] = np.linspace(0.0, 2.0, len(sliding))
    planted_contact = foot_contact_consistency(planted, move_thresh=0.01)
    sliding_contact = foot_contact_consistency(sliding, move_thresh=0.01)

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
                structure_controls["identical"]["raw_repetition"] > 0.999
                and structure_controls["identical"]["invariant_recurrence"] > 0.999
                and structure_controls["unrelated"]["invariant_recurrence"] < 0.8
                and structure_controls["shifted"]["invariant_recurrence"] > 0.9
                and structure_controls["mirrored"]["invariant_recurrence"] > 0.9
                and structure_controls["retrograded"]["invariant_recurrence"] > 0.9
                and structure_controls["frozen"]["invariant_recurrence"] < 0.1
                and structure_controls["jittered"]["invariant_recurrence"]
                < structure_controls["identical"]["invariant_recurrence"]
                and contrast > 0
            ),
            "controls": structure_controls,
            "sectional_contrast": contrast,
        },
        "transition_jerk": {
            "pass": (
                jump_peak > smooth_peak
                and jump_area > smooth_area
                and jitter_peak > smooth_peak
                and jitter_area > smooth_area
            ),
            "smooth_peak": smooth_peak,
            "jump_peak": jump_peak,
            "smooth_area": smooth_area,
            "jump_area": jump_area,
            "jitter_peak": jitter_peak,
            "jitter_area": jitter_area,
        },
        "boundary_alignment": {
            "pass": (
                aligned_boundary_score > 0.9
                and misaligned_boundary_score < 0.1
            ),
            "aligned": aligned_boundary_score,
            "misaligned": misaligned_boundary_score,
        },
        "foot_contact": {
            "pass": planted_contact > 0.999 and sliding_contact < 0.1,
            "planted": planted_contact,
            "sliding": sliding_contact,
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
