from __future__ import annotations

import numpy as np

from agentlodge.dance.transition import root_facing_yaw
from scripts.build_song_comparisons import _prepare_motions
from scripts.rank_interview_windows import _pose_difference


def _identity_rotations(frames: int) -> np.ndarray:
    identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    return np.tile(identity, (frames, 22))


def test_prepare_motions_converts_edge_retimes_and_shares_front_facing_yaw(tmp_path):
    sid = "approved_song"
    frames = 8
    rotations = _identity_rotations(frames)

    lodge = np.zeros((frames, 139), dtype=np.float32)
    lodge[:, 7:139] = rotations
    edge = np.zeros((frames, 151), dtype=np.float32)
    edge[:, 3:135] = rotations
    maestro = np.zeros((frames, 139), dtype=np.float32)
    maestro[:, 3:135] = rotations

    np.save(tmp_path / f"lodge_fd_{sid}_full.npy", lodge)
    np.save(tmp_path / f"edge_fd_{sid}_full.npy", edge)
    np.save(tmp_path / f"fd_{sid}_STORY_bestofk.npy", maestro)

    paths, report = _prepare_motions(tmp_path, sid, 10, tmp_path / "output")

    assert report["source_shapes"]["edge"] == [frames, 151]
    prepared = {method: np.load(path) for method, path in paths.items()}
    assert all(motion.shape == (10, 139) for motion in prepared.values())
    target_yaw = root_facing_yaw(prepared["maestro"])
    assert all(
        all(
            np.isclose(root_facing_yaw(motion[index:index + 1]), target_yaw)
            for index in range(len(motion))
        )
        for motion in prepared.values()
    )


def test_pose_difference_rejects_identical_motion():
    motion = np.zeros((10, 139), dtype=np.float32)
    motion[:, 3:135] = _identity_rotations(10)

    identical = _pose_difference(motion, motion)

    assert identical["mean_rotation_degrees"] == 0
    assert identical["near_identical_rotation_ratio"] == 1
