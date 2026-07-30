"""Unit tests for the on-demand :class:`LiveWindowGenerator` (pod-backed live search).

The pod I/O is fully abstracted behind a ``take_provider`` callable, so these tests exercise the
memoization, slicing, counting and fallback behaviour with an in-process fake -- no SSH, no GPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor.remote_generator import LiveWindowGenerator
from agentlodge.editor.window_edit import MockWindowGenerator, apply_window_edit


def _take(energy: float, n: int = 300, backbone: str = "edge", seed: int = 0) -> np.ndarray:
    return MockWindowGenerator().generate(backbone, 0, n, seed, energy=energy, beats=None)


class _FakeProvider:
    """Records every (backbone, seed) it is asked to generate; returns a deterministic take."""

    def __init__(self, n: int = 300, fail_for=None):
        self.n = n
        self.calls: list[tuple[str, int]] = []
        self.fail_for = set(fail_for or ())

    def __call__(self, backbone: str, seed: int):
        self.calls.append((backbone, int(seed)))
        if (backbone, int(seed)) in self.fail_for:
            return None
        energy = 0.9 if backbone == "edge" else 0.2
        return _take(energy, self.n, backbone, int(seed))


def test_live_returns_window_slice():
    p = _FakeProvider(240)
    g = LiveWindowGenerator(p)
    win = g.generate("edge", 60, 150, 0)
    assert win.shape == (90, 139)
    assert p.calls == [("edge", 0)]


def test_live_memoizes_one_generation_per_seed():
    p = _FakeProvider(240)
    g = LiveWindowGenerator(p)
    # many windows out of the same (backbone, seed) => exactly one pod generation
    for a in (0, 30, 60, 90):
        g.generate("edge", a, a + 30, 0)
    assert p.calls == [("edge", 0)]
    assert g.n_generated == 1
    # a new seed and a new backbone each trigger exactly one more generation
    g.generate("edge", 0, 30, 1)
    g.generate("lodge", 0, 30, 0)
    assert g.n_generated == 3
    assert p.calls == [("edge", 0), ("edge", 1), ("lodge", 0)]


def test_live_pads_when_take_too_short():
    g = LiveWindowGenerator(_FakeProvider(100))
    win = g.generate("edge", 80, 140, 0)       # 60 requested, only 20 available
    assert win.shape == (60, 139)
    assert np.array_equal(win[20:], np.repeat(win[19:20], 40, axis=0))


def test_live_falls_back_when_provider_returns_none():
    p = _FakeProvider(200, fail_for=[("lodge", 0)])
    g = LiveWindowGenerator(p, fallback=MockWindowGenerator())
    win = g.generate("lodge", 0, 30, 0)        # provider fails -> fallback fills in
    assert win is not None and win.shape == (30, 139)


def test_live_returns_none_without_fallback_on_failure():
    p = _FakeProvider(200, fail_for=[("edge", 0)])
    g = LiveWindowGenerator(p)                  # no fallback
    assert g.generate("edge", 0, 30, 0) is None


def test_live_provider_exception_is_swallowed():
    def boom(backbone, seed):
        raise RuntimeError("pod unreachable")
    g = LiveWindowGenerator(boom, fallback=MockWindowGenerator())
    win = g.generate("edge", 0, 30, 0)
    assert win is not None and win.shape == (30, 139)


def test_live_drives_a_full_edit():
    n = 300
    p = _FakeProvider(n)
    g = LiveWindowGenerator(p)
    base = _take(0.2, n, "lodge", 0)           # calm assembled dance
    beats = np.arange(0, n, 15).astype(float)
    r = apply_window_edit(base, 90, 180, "make this more energetic", g,
                          beats=beats, k=2, max_cycles=2)
    assert r.ok and r.backbone == "edge"
    assert r.metrics_after["energy"] > r.metrics_before["energy"]
    # frames outside the window are byte-identical
    assert np.array_equal(r.motion[:90], base[:90]) and np.array_equal(r.motion[180:], base[180:])
    # every distinct seed was generated at most once despite k*cycles candidate evaluations
    assert len(set(p.calls)) == len(p.calls)
