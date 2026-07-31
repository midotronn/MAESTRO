"""Unit tests for the interactive windowed editor: parse, splice fidelity, regen loop, verify."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.dance import transition as T
from agentlodge.editor import window_edit as WE
from agentlodge.editor.window_edit import (
    EditGoal,
    MockWindowGenerator,
    apply_window_edit,
    goal_reward,
    parse_window_instruction,
    reward_weights_for,
    window_metrics,
)


def _base(n: int = 300, energy: float = 0.5, seed: int = 0) -> np.ndarray:
    """A mock-generated 'assembled dance' (same scale as edits) for realistic before/after deltas."""
    return MockWindowGenerator().generate("edge", 0, n, seed, energy=energy, beats=None)


def _beats(n: int = 300, step: int = 15) -> np.ndarray:
    return np.arange(0, n, step).astype(float)


# --------------------------------------------------------------------------- parsing
def test_parse_objectives():
    cases = {
        "make the drop more energetic": "more_energetic",
        "can you make this calmer and softer": "calmer",
        "tighten this to the beat": "more_on_beat",
        "make it sharper and more percussive": "sharper",
        "smoother and more flowing please": "smoother",
        "reverse this section": "reverse",
        "mirror this bit left to right": "mirror",
        "really exaggerate the movement here": "exaggerate",
    }
    for text, obj in cases.items():
        assert parse_window_instruction(text).objective == obj, text


def test_parse_default_is_on_beat():
    assert parse_window_instruction("do something nice here").objective == "more_on_beat"


def test_parse_backbone_hint_and_magnitude():
    g = parse_window_instruction("make this much more energetic")
    assert g.backbone == "edge" and g.magnitude > 0.5
    assert parse_window_instruction("smoother").backbone == "lodge"


def test_reward_weights_shape():
    w = reward_weights_for("more_energetic")
    assert w["w_energy"] > w["w_bas"] and w["target_intensity"] == 1.0
    assert reward_weights_for("more_on_beat")["w_bas"] >= 0.6


# --------------------------------------------------------------------------- splice fidelity
def test_splice_preserves_outside_frames_exactly():
    m = _base(240)
    gen = MockWindowGenerator()
    a, b = 60, 150
    win = gen.generate("edge", a, b, 2, energy=0.9, beats=None)
    out = T.splice_window(m, a, b, win, blend_frames=12)
    assert out.shape == m.shape
    assert np.array_equal(out[:a], m[:a])          # prefix byte-identical
    assert np.array_equal(out[b:], m[b:])          # suffix byte-identical
    assert not np.array_equal(out[a:b], m[a:b])    # window actually changed


def test_splice_retimes_mismatched_window():
    m = _base(200)
    gen = MockWindowGenerator()
    a, b = 50, 130
    win = gen.generate("edge", 0, 40, 1, energy=0.5)   # wrong length (40 != 80)
    out = T.splice_window(m, a, b, win, blend_frames=10)
    assert out.shape == m.shape
    assert np.array_equal(out[:a], m[:a]) and np.array_equal(out[b:], m[b:])


def test_splice_rejects_bad_window():
    m = _base(100)
    for a, b in [(-1, 10), (50, 50), (10, 5), (0, 101)]:
        try:
            T.splice_window(m, a, b, m[:10])
            assert False, f"expected ValueError for [{a},{b})"
        except ValueError:
            pass


def test_window_beats_slices_and_shifts():
    beats = np.array([0, 10, 20, 30, 40, 50], dtype=float)
    wb = WE._window_beats(beats, 15, 45)
    assert list(wb) == [5.0, 15.0, 25.0]           # 20,30,40 shifted by -15


# --------------------------------------------------------------------------- regen objectives
def test_more_energetic_increases_energy():
    m = _base(300, energy=0.5)
    r = apply_window_edit(m, 90, 180, "make this more energetic",
                          MockWindowGenerator(), beats=_beats(), k=6, max_cycles=3)
    assert r.ok
    assert r.metrics_after["energy"] > r.metrics_before["energy"]
    assert np.array_equal(r.motion[:90], m[:90]) and np.array_equal(r.motion[180:], m[180:])


def test_calmer_decreases_energy():
    m = _base(300, energy=0.6)
    r = apply_window_edit(m, 90, 180, "make this calmer",
                          MockWindowGenerator(), beats=_beats(), k=6, max_cycles=3)
    assert r.ok
    assert r.metrics_after["energy"] < r.metrics_before["energy"]


def test_on_beat_increases_bas():
    m = _base(300, energy=0.5)
    r = apply_window_edit(m, 90, 180, "tighten this to the beat",
                          MockWindowGenerator(), beats=_beats(), k=6, max_cycles=3)
    assert r.ok
    assert r.metrics_after["bas"] > r.metrics_before["bas"]


def test_smoother_decreases_jerk():
    m = _base(300, energy=0.6)
    r = apply_window_edit(m, 90, 180, "make this smoother and flowing",
                          MockWindowGenerator(), beats=_beats(), k=6, max_cycles=3)
    assert r.ok
    assert r.metrics_after["jerk"] < r.metrics_before["jerk"]


def test_reverse_is_deterministic_and_preserves_outside():
    m = _base(240)
    r = apply_window_edit(m, 60, 150, "reverse this section",
                          MockWindowGenerator(), beats=_beats(240))
    assert r.ok and r.goal.objective == "reverse" and r.backbone == "none"
    assert np.array_equal(r.motion[:60], m[:60]) and np.array_equal(r.motion[150:], m[150:])
    assert len(r.cycles) == 1


def test_mirror_is_deterministic():
    m = _base(240)
    r = apply_window_edit(m, 60, 150, "mirror this bit", MockWindowGenerator(), beats=_beats(240))
    assert r.ok and r.goal.objective == "mirror"
    assert np.array_equal(r.motion[:60], m[:60]) and np.array_equal(r.motion[150:], m[150:])


def test_on_beat_is_deterministic_warp_and_raises_bas():
    # 'more on beat' now snaps motion beats onto the music beats (deterministic), so it needs no
    # generator, preserves everything outside the window, and reliably raises BAS.
    m = _base(300, energy=0.5)
    r = apply_window_edit(m, 90, 210, "make this much more on beat", None, beats=_beats())
    assert r.ok and r.goal.objective == "more_on_beat" and r.backbone == "none"
    assert r.metrics_after["bas"] > r.metrics_before["bas"]
    assert np.array_equal(r.motion[:90], m[:90]) and np.array_equal(r.motion[210:], m[210:])


# --------------------------------------------------------------------------- refine / failure paths
class _WeakGenerator:
    """Ignores the energy request (always low energy) -> 'more energetic' can never be satisfied."""

    def generate(self, backbone, a, b, seed, *, energy=0.5, beats=None, context=None):
        return MockWindowGenerator().generate(backbone, a, b, seed, energy=0.15, beats=beats)


def test_unsatisfiable_edit_returns_best_attempt_not_error():
    m = _base(300, energy=0.9)                     # already high energy
    r = apply_window_edit(m, 90, 180, "make this way more energetic",
                          _WeakGenerator(), beats=_beats(), k=4, max_cycles=2)
    assert r.ok is False                           # weak generator can't raise energy
    assert r.motion.shape == m.shape               # still returns a valid best-effort motion
    assert np.array_equal(r.motion[:90], m[:90]) and np.array_equal(r.motion[180:], m[180:])
    assert len(r.cycles) >= 1                       # tried and reported


def test_regen_requires_generator():
    m = _base(120)
    try:
        apply_window_edit(m, 30, 60, "more energetic", None, beats=_beats(120))
        assert False, "expected ValueError without a generator"
    except ValueError:
        pass


def test_deterministic_edit_needs_no_generator():
    m = _base(120)
    r = apply_window_edit(m, 30, 60, "reverse this", None, beats=_beats(120))
    assert r.ok and r.goal.objective == "reverse"


def test_goal_reward_penalizes_frozen_window():
    frozen = np.tile(_base(1)[0], (60, 1))         # a single repeated pose = no motion
    r, mets = goal_reward(frozen, _beats(60), EditGoal("more_energetic"))
    assert mets["energy"] < WE._FROZEN_ENERGY and r < 0


def test_window_metrics_keys():
    m = _base(90)
    mk = window_metrics(m[10:70], _beats(90))
    assert set(mk) == {"energy", "bas", "jerk", "foot"}
