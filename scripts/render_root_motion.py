"""Pure root-motion helpers shared by Blender renderers and unit tests."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


FOOT_JOINTS = (7, 8, 10, 11)


class RootMotionPlan(NamedTuple):
    root_path: np.ndarray
    follow_xy: np.ndarray
    calibration_frames: np.ndarray


def smooth_path(xy: np.ndarray, window: int = 11) -> np.ndarray:
    """Smooth an ``(frames, 2)`` path with edge padding instead of zero-padding drift."""
    path = np.asarray(xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"path must have shape (frames, 2), got {path.shape}")
    if not np.isfinite(path).all():
        raise ValueError("path contains non-finite values")
    if path.shape[0] < 3:
        return path.copy()

    width = max(1, int(window))
    if width % 2 == 0:
        width += 1
    largest_odd = path.shape[0] if path.shape[0] % 2 else path.shape[0] - 1
    width = min(width, largest_odd)
    if width <= 1:
        return path.copy()

    pad = width // 2
    padded = np.pad(path, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)],
        axis=1,
    )


def prepare_root_motion(
    fk_joints: np.ndarray,
    *,
    yaw_degrees: float = 0.0,
    follow_window: int = 11,
    calibration_count: int = 12,
    foot_joints: tuple[int, ...] = FOOT_JOINTS,
) -> RootMotionPlan:
    """Build a floor-relative root path and a stable camera-follow path from FK joints."""
    fk = np.asarray(fk_joints, dtype=np.float64)
    if fk.ndim != 3 or fk.shape[2] != 3 or fk.shape[0] == 0:
        raise ValueError(f"FK joints must have shape (frames, joints, 3), got {fk.shape}")
    if not foot_joints or fk.shape[1] <= max(foot_joints):
        raise ValueError("FK joints do not contain the requested feet")
    if not np.isfinite(fk).all():
        raise ValueError("FK joints contain non-finite values")

    root_path = fk[:, 0].copy()
    root_path[:, :2] -= root_path[:1, :2]
    floor_z = float(np.min(fk[:, foot_joints, 2]))
    root_path[:, 2] -= floor_z

    angle = np.deg2rad(float(yaw_degrees))
    if abs(angle) > 1e-12:
        c, s = float(np.cos(angle)), float(np.sin(angle))
        x, y = root_path[:, 0].copy(), root_path[:, 1].copy()
        root_path[:, 0] = c * x - s * y
        root_path[:, 1] = s * x + c * y

    follow_xy = smooth_path(root_path[:, :2], follow_window)
    follow_xy -= follow_xy[:1]

    foot_height = np.min(fk[:, foot_joints, 2], axis=1)
    count = min(max(1, int(calibration_count)), fk.shape[0])
    calibration_frames = np.argsort(foot_height, kind="stable")[:count]
    return RootMotionPlan(root_path, follow_xy, calibration_frames)


def centered_follow_locations(
    camera_location: np.ndarray,
    target_location: np.ndarray,
    spot_location: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reset a reused follow rig to the origin while preserving its relative offsets."""
    camera = np.asarray(camera_location, dtype=np.float64)
    target = np.asarray(target_location, dtype=np.float64)
    spot = None if spot_location is None else np.asarray(spot_location, dtype=np.float64)
    invalid_spot = spot is not None and spot.shape != (3,)
    if camera.shape != (3,) or target.shape != (3,) or invalid_spot:
        raise ValueError("follow-rig locations must be 3D vectors")
    if not np.isfinite(camera).all() or not np.isfinite(target).all() or (
        spot is not None and not np.isfinite(spot).all()
    ):
        raise ValueError("follow-rig locations contain non-finite values")

    home_target = target.copy()
    home_target[:2] = 0.0
    home_camera = home_target + (camera - target)
    home_spot = None if spot is None else home_target + (spot - target)
    return home_camera, home_target, home_spot
