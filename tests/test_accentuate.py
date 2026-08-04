"""Property tests for `transition.accentuate` -- the deterministic, monotone energy lever.

These lock the invariants the edit loop relies on to GUARANTEE 'more/less energetic' without
regenerating: energy moves monotonically with the gain, the rotations stay on SO(3), the seams are
tapered, contacts are untouched, and gain==1 is a no-op. A break here is exactly the class of bug
(off-manifold / jittery / seam pops) that made the retired amplitude_scale unusable.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.dance import transition as T
from agentlodge.editor.window_edit import window_metrics


def _smooth_motion(n: int = 120, seed: int = 0, n_joints: int = 22) -> np.ndarray:
    """A smooth, structurally valid 139-dim motion: each joint sways sinusoidally about a random axis
    (low-frequency, so most of the signal is the mid-band the energy lever amplifies)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, n)
    axes = rng.standard_normal((n_joints, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True) + 1e-9
    amp = 0.15 + 0.25 * rng.random((n_joints, 1))
    freq = 1.2 + 1.6 * rng.random((n_joints, 1))             # ~0.6-1.4 Hz: dance-tempo mid-band
    phase = 2.0 * np.pi * rng.random((n_joints, 1))
    ang = amp * np.sin(freq * t[None, :] + phase)            # (J, n)
    aa = np.transpose(axes[:, None, :] * ang[:, :, None], (1, 0, 2))   # (n, J, 3)
    r6 = T._matrix_to_sixd(T._axis_angle_to_matrix(aa)).reshape(n, n_joints * 6)
    trans = 0.02 * np.cumsum(np.sin(t)[:, None] * np.ones((1, 3)), axis=0)
    contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
    return np.concatenate([trans, r6, contact], axis=1).astype(np.float32)


def _energy(clip: np.ndarray) -> float:
    return window_metrics(clip)["energy"]


def test_gain_one_is_identity():
    m = _smooth_motion(80, seed=1)
    assert np.array_equal(T.accentuate(m, 1.0), m)


def test_energy_is_monotonic_in_gain():
    m = _smooth_motion(150, seed=2)
    gains = [0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6, 1.9]
    energies = [_energy(T.accentuate(m, g)) for g in gains]
    # strictly increasing across the whole sweep (the property the loop dials against)
    for lo, hi in zip(energies, energies[1:]):
        assert hi > lo + 1e-6, f"energy not increasing: {list(zip(gains, energies))}"


def test_gain_up_raises_energy_gain_down_lowers_it():
    m = _smooth_motion(120, seed=3)
    base = _energy(m)
    assert _energy(T.accentuate(m, 1.6)) > base * 1.03        # clearly more energetic
    assert _energy(T.accentuate(m, 0.6)) < base * 0.97        # clearly calmer


def test_energy_gain_is_meaningful_across_seeds():
    # on dance-tempo motion the lever should move energy by a clear margin, not a token amount
    ratios = []
    for seed in range(6):
        m = _smooth_motion(150, seed=seed)
        ratios.append(_energy(T.accentuate(m, 1.7)) / max(_energy(m), 1e-9))
    assert min(ratios) > 1.05 and float(np.mean(ratios)) > 1.15, ratios


def test_rotations_stay_on_so3():
    m = _smooth_motion(60, seed=4)
    for g in (0.6, 1.5):
        out = T.accentuate(m, g)
        R = T._sixd_to_matrix(out[:, 3:135].reshape(60, 22, 6))
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-4)
        # orthonormal: R R^T == I
        rt = np.einsum("...ij,...kj->...ik", R, R)
        assert np.allclose(rt, np.eye(3)[None, None], atol=1e-4)


def test_contacts_are_untouched():
    m = _smooth_motion(70, seed=5)
    out = T.accentuate(m, 1.7)
    assert np.array_equal(out[:, 135:139], m[:, 135:139])


def test_taper_keeps_the_ends_closer_to_the_original_than_the_middle():
    m = _smooth_motion(120, seed=6)
    out = T.accentuate(m, 1.8, taper_frames=10)
    edge = np.linalg.norm(out[0, :135] - m[0, :135]) + np.linalg.norm(out[-1, :135] - m[-1, :135])
    mid = np.linalg.norm(out[60, :135] - m[60, :135])
    assert edge < mid                                        # the tapered seams move far less


def test_shape_and_dtype_preserved():
    m = _smooth_motion(50, seed=7)
    out = T.accentuate(m, 1.3)
    assert out.shape == m.shape and out.dtype == np.float32


def test_short_clip_is_safe():
    m = _smooth_motion(2, seed=8)
    out = T.accentuate(m, 1.5)                                # n < 3 -> returns a copy, no crash
    assert out.shape == m.shape
