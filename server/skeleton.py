"""Local numpy forward-kinematics: AgentLODGE 139 motion -> (L, 22, 3) joint positions.

Replicates LODGE's ``SMPLX_Skeleton.forward`` (the batch_rigid_transform) in pure numpy so the server
can produce a stick figure for the in-browser fast preview with NO torch, NO LODGE, and NO pod round
trip. The 22 body joints form a closed subtree of the SMPL-X kinematic tree (every parent index is
< 22), so only the 22-joint template + parents are needed; the 6D->matrix convention matches
pytorch3d's ``rotation_6d_to_matrix`` (what LODGE's ``ax_from_6v`` uses), so the rotation used in FK
is exactly ``_sixd_to_matrix(rot6d)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentlodge.dance.format import to_native_finedance139
from agentlodge.dance.transition import _sixd_to_matrix

# SMPL(-X) body kinematic tree for the 22 joints LODGE emits (== smplx_parents[:22]).
BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
# bone pairs (child, parent) for drawing, skipping the root's -1 parent
BONES = [(i, p) for i, p in enumerate(BODY_PARENTS) if p >= 0]

_J_TEMPLATE: np.ndarray | None = None


def _template() -> np.ndarray:
    global _J_TEMPLATE
    if _J_TEMPLATE is None:
        p = Path(__file__).resolve().parent / "data" / "smplx_neu_J_1.npy"
        _J_TEMPLATE = np.load(p)[:22].astype(np.float64)
    return _J_TEMPLATE


def fk_joints(motion139: np.ndarray) -> np.ndarray:
    """(L, 139) AgentLODGE motion -> (L, 22, 3) world joint positions (float32)."""
    native = to_native_finedance139(np.asarray(motion139, dtype=np.float32))
    L = int(native.shape[0])
    trans = native[:, 4:7].astype(np.float64)                       # root translation
    R = _sixd_to_matrix(native[:, 7:139].reshape(L, 22, 6)).astype(np.float64)   # (L,22,3,3)

    J = _template()                                                 # (22,3) rest joints
    parents = BODY_PARENTS
    rel = J.copy()
    for i in range(1, 22):                                          # bone offset relative to parent
        rel[i] = J[i] - J[parents[i]]

    eye = np.broadcast_to(np.eye(4), (L, 4, 4))
    chain: list[np.ndarray] = [None] * 22                          # type: ignore[list-item]
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
    return posed.astype(np.float32)
