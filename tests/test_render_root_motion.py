"""Regression tests for the pure root-motion part of the Blender Y-Bot renderer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_root_motion import (  # noqa: E402
    centered_follow_locations,
    prepare_root_motion,
    smooth_path,
)


def _fk() -> np.ndarray:
    fk = np.zeros((7, 22, 3), dtype=np.float64)
    fk[:, 0] = np.array([
        [4.0, -3.0, 1.00],
        [4.1, -3.0, 1.04],
        [4.3, -2.9, 1.16],
        [4.6, -2.7, 1.28],
        [5.0, -2.5, 1.18],
        [5.3, -2.4, 1.06],
        [5.4, -2.4, 1.01],
    ])
    for joint in (7, 8, 10, 11):
        fk[:, joint, :2] = fk[:, 0, :2]
        fk[:, joint, 2] = np.array([0.03, 0.02, 0.12, 0.24, 0.14, 0.01, 0.02])
    return fk


def test_prepare_root_motion_preserves_travel_and_height_without_mutating_fk():
    fk = _fk()
    original = fk.copy()
    plan = prepare_root_motion(fk, follow_window=3, calibration_count=3)

    assert np.array_equal(fk, original)
    assert np.allclose(plan.root_path[0, :2], 0.0)
    assert np.allclose(plan.root_path[:, :2], fk[:, 0, :2] - fk[0, 0, :2])
    assert np.allclose(plan.root_path[:, 2], fk[:, 0, 2] - 0.01)
    assert np.array_equal(plan.calibration_frames, np.array([5, 1, 6]))
    assert np.allclose(plan.follow_xy[0], 0.0)


def test_prepare_root_motion_rotates_travel_with_the_render_yaw():
    plan = prepare_root_motion(_fk(), yaw_degrees=90.0, follow_window=1)
    original_delta = _fk()[-1, 0, :2] - _fk()[0, 0, :2]
    assert np.allclose(plan.root_path[-1, :2], [-original_delta[1], original_delta[0]])


def test_smooth_path_reduces_camera_jitter_and_keeps_the_shape():
    frame_count = 9
    path = np.column_stack([
        np.linspace(0.0, 1.0, frame_count),
        np.array([0.0, 0.2, -0.1, 0.25, -0.2, 0.2, -0.1, 0.1, 0.0]),
    ])
    smoothed = smooth_path(path, window=5)
    assert smoothed.shape == (frame_count, 2)
    assert np.abs(np.diff(smoothed[:, 1], n=2)).sum() < np.abs(
        np.diff(path[:, 1], n=2)
    ).sum()


def test_centered_follow_locations_prevents_warm_daemon_state_leakage():
    camera = np.array([8.5, -1.0, 3.0])
    target = np.array([7.0, 4.0, 1.0])
    spot = np.array([7.0, 0.0, 6.0])
    home_camera, home_target, home_spot = centered_follow_locations(camera, target, spot)

    assert np.allclose(home_target, [0.0, 0.0, 1.0])
    assert np.allclose(home_camera - home_target, camera - target)
    assert home_spot is not None
    assert np.allclose(home_spot - home_target, spot - target)


@pytest.mark.parametrize(
    "fk",
    [
        np.zeros((0, 22, 3)),
        np.zeros((3, 10, 3)),
        np.zeros((3, 22, 2)),
        np.full((3, 22, 3), np.nan),
    ],
)
def test_prepare_root_motion_rejects_invalid_fk(fk):
    with pytest.raises(ValueError):
        prepare_root_motion(fk)
