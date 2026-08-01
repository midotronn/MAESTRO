"""Server-side FK: shape + rotation-fidelity checks. The axis-angle we emit must represent EXACTLY
the 6D rotation Blender poses the Y-Bot armature from (validated end-to-end against the pod torch FK
to max diff 3.8e-7). Skips when the licence-gated SMPL-X joint template isn't present locally."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "server", "data", "smplx_neu_J_1.npy")
pytestmark = pytest.mark.skipif(not os.path.exists(_TMPL),
                                reason="SMPL-X joint template not present (fetched from the pod)")


def _motion(n: int = 120) -> np.ndarray:
    from agentlodge.editor.window_edit import MockWindowGenerator
    return MockWindowGenerator().generate("edge", 0, n, 1, energy=0.6)


def test_server_fk_shapes_and_rotation_fidelity():
    from agentlodge.dance.format import to_native_finedance139
    from agentlodge.dance.transition import _axis_angle_to_matrix, _sixd_to_matrix
    from server.fk import compute_poses
    m = _motion(120)
    d = compute_poses(m)
    assert d["poses"].shape == (120, 24, 3)
    assert d["trans"].shape == (120, 3)
    assert d["fk_joints"].shape == (120, 22, 3)
    assert np.allclose(d["poses"][:, 22:], 0.0)                 # SMPL hand joints stay at rest
    # the emitted axis-angle must reconstruct the exact 6D rotation Blender poses from
    native = to_native_finedance139(m)
    R = _sixd_to_matrix(native[:, 7:139].reshape(120, 22, 6))
    R2 = _axis_angle_to_matrix(d["poses"][:, :22])
    assert np.abs(R - R2).max() < 1e-4
    # trans is exactly the native root translation
    assert np.allclose(d["trans"], native[:, 4:7], atol=1e-6)


def test_server_fk_deterministic():
    from server.fk import compute_poses
    m = _motion(80)
    a, b = compute_poses(m), compute_poses(m)
    for k in ("poses", "trans", "fk_joints"):
        assert np.array_equal(a[k], b[k])
