"""Unit test for the local numpy forward-kinematics (in-browser stick-figure preview)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor.window_edit import MockWindowGenerator
from server.skeleton import BODY_PARENTS, BONES, fk_joints


def test_fk_shape_finite_and_bones():
    m = MockWindowGenerator().generate("edge", 0, 60, 1, energy=0.5, beats=None)
    joints = fk_joints(m)
    assert joints.shape == (60, 22, 3)
    assert np.isfinite(joints).all()
    assert len(BODY_PARENTS) == 22 and BODY_PARENTS[0] == -1
    assert len(BONES) == 21 and all(0 <= p < 22 and 0 < c < 22 for c, p in BONES)


def test_fk_is_non_degenerate():
    # FK of a valid pose spreads the joints out (not collapsed to a point). Exact correctness vs
    # LODGE's SMPLX_Skeleton was validated separately (max diff 3.4e-7).
    m = MockWindowGenerator().generate("lodge", 0, 30, 0, energy=0.4, beats=None)
    j = fk_joints(m)[0]
    extent = j.max(0) - j.min(0)
    assert float(extent.max()) > 0.2 and float(np.linalg.norm(j.std(0))) > 0.05
