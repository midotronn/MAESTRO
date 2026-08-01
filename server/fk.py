"""Server-side numpy replica of ``scripts.render_blender_dance.compute_smpl_poses``.

Produces the render's ``poses.npz`` (SMPL axis-angle poses + root translation + FK joint positions)
in pure numpy, so the GPU pod never pays the ~12-24s torch import for forward-kinematics -- the single
biggest source of render-time variance. The 22 body joints form a closed subtree of the SMPL-X tree
(every parent < 22) and the 6D->matrix convention matches pytorch3d's ``rotation_6d_to_matrix`` (what
LODGE's ``ax_from_6v`` uses), so the rotation is exactly ``_sixd_to_matrix(rot6d)``; the FK joints were
validated against the pod's ``SMPLX_Skeleton`` to max diff 3.4e-7. Blender re-derives each bone matrix
from the axis-angle, so any valid axis-angle for the same rotation renders identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentlodge.dance.format import to_native_finedance139
from agentlodge.dance.transition import _matrix_to_axis_angle, _sixd_to_matrix

# SMPL(-X) body kinematic tree for the 22 joints LODGE emits (== smplx_parents[:22]).
BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
_J_TEMPLATE: np.ndarray | None = None


def _template() -> np.ndarray:
    global _J_TEMPLATE
    if _J_TEMPLATE is None:
        p = Path(__file__).resolve().parent / "data" / "smplx_neu_J_1.npy"
        _J_TEMPLATE = np.load(p)[:22].astype(np.float64)
    return _J_TEMPLATE


def _fk_joints(R: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Batch rigid transform: R (L,22,3,3) + trans (L,3) -> world joint positions (L,22,3)."""
    L = int(R.shape[0])
    J = _template()
    parents = BODY_PARENTS
    rel = J.copy()
    for i in range(1, 22):
        rel[i] = J[i] - J[parents[i]]
    eye = np.broadcast_to(np.eye(4), (L, 4, 4))
    chain: list = [None] * 22
    T0 = eye.copy()
    T0[:, :3, :3] = R[:, 0]
    T0[:, :3, 3] = rel[0]
    chain[0] = T0
    for i in range(1, 22):
        Ti = eye.copy()
        Ti[:, :3, :3] = R[:, i]
        Ti[:, :3, 3] = rel[i]
        chain[i] = chain[parents[i]] @ Ti
    posed = np.stack([chain[i][:, :3, 3] for i in range(22)], axis=1)   # (L,22,3)
    posed += trans[:, None, :]
    return posed


def _orient_joints_zup(joints: np.ndarray) -> np.ndarray:
    """Reorient FK joints so the body's vertical axis maps to +Z (matches render_blender_dance)."""
    if joints.size == 0:
        return joints
    hf = (joints[:, 15, :] - joints[:, [7, 8], :].mean(axis=1)).mean(axis=0)
    up = int(np.argmax(np.abs(hf)))
    positive = hf[up] >= 0
    if up == 1:
        if positive:
            joints = np.stack([joints[..., 0], -joints[..., 2], joints[..., 1]], axis=-1)
        else:
            joints = np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)
    elif up == 0:
        if positive:
            joints = np.stack([-joints[..., 2], joints[..., 1], joints[..., 0]], axis=-1)
        else:
            joints = np.stack([joints[..., 2], joints[..., 1], -joints[..., 0]], axis=-1)
    elif not positive:
        joints = np.stack([joints[..., 0], -joints[..., 1], -joints[..., 2]], axis=-1)
    return joints.astype(np.float32)


def compute_poses(motion139: np.ndarray) -> dict:
    """(L, 139) AgentLODGE motion -> {poses (L,24,3), trans (L,3), fk_joints (L,22,3)} for the render."""
    native = to_native_finedance139(np.asarray(motion139, dtype=np.float32))
    L = int(native.shape[0])
    trans = native[:, 4:7].astype(np.float32)
    R = _sixd_to_matrix(native[:, 7:139].reshape(L, 22, 6)).astype(np.float64)   # (L,22,3,3)
    ax = _matrix_to_axis_angle(R).astype(np.float32)                             # (L,22,3)
    fkj = _orient_joints_zup(_fk_joints(R, trans.astype(np.float64)))
    poses = np.zeros((L, 24, 3), dtype=np.float32)
    poses[:, :22] = ax
    return {"poses": poses, "trans": trans, "fk_joints": fkj.astype(np.float32)}


def save_poses_npz(motion139: np.ndarray, path) -> None:
    d = compute_poses(motion139)
    np.savez(path, poses=d["poses"], trans=d["trans"], fk_joints=d["fk_joints"])
