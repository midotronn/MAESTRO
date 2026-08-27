import numpy as np
import pytest

from agentlodge.dance.transition import (
    _matrix_to_sixd,
    _sixd_to_matrix,
    root_facing_yaw,
    stabilize_root_facing,
)


def _motion_with_yaw(yaw: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(yaw), np.sin(yaw)
    roots = np.stack(
        [
            cos, -sin, np.zeros_like(cos),
            sin, cos, np.zeros_like(cos),
            np.zeros_like(cos), np.zeros_like(cos), np.ones_like(cos),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    motion = np.zeros((len(yaw), 139), dtype=np.float32)
    motion[:, 3:9] = _matrix_to_sixd(roots)
    motion[:, :3] = np.arange(len(yaw) * 3, dtype=np.float32).reshape(-1, 3)
    motion[:, 9:] = np.arange(130, dtype=np.float32)
    return motion


def _yaw(motion: np.ndarray) -> np.ndarray:
    roots = _sixd_to_matrix(motion[:, 3:9])
    return np.arctan2(roots[:, 1, 0], roots[:, 0, 0])


def test_stabilize_root_facing_uses_requested_heading():
    motion = _motion_with_yaw(np.linspace(-2.5, 2.5, 120, dtype=np.float32))

    stabilized = stabilize_root_facing(motion, target_yaw=0.7)

    np.testing.assert_allclose(_yaw(stabilized), 0.7, atol=1e-5)
    np.testing.assert_array_equal(stabilized[:, :3], motion[:, :3])
    np.testing.assert_array_equal(stabilized[:, 9:], motion[:, 9:])


def test_stabilize_root_facing_anchors_to_opening_circular_mean():
    opening = np.array([3.10, -3.12, 3.13, -3.11], dtype=np.float32)
    motion = _motion_with_yaw(np.concatenate([opening, np.linspace(-1.0, 1.0, 20)]))

    stabilized = stabilize_root_facing(motion, anchor_frames=4)

    expected = root_facing_yaw(motion, anchor_frames=4)
    np.testing.assert_allclose(
        np.arctan2(np.sin(_yaw(stabilized)), np.cos(_yaw(stabilized))),
        expected,
        atol=1e-5,
    )


def test_stabilize_root_facing_rejects_invalid_layout():
    with pytest.raises(ValueError, match="expected motion shape"):
        stabilize_root_facing(np.zeros((10, 138), dtype=np.float32))


def test_root_facing_yaw_rejects_empty_motion():
    with pytest.raises(ValueError, match="empty motion"):
        root_facing_yaw(np.zeros((0, 139), dtype=np.float32))
