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
    """Records every (backbone, seed, a, b) window it is asked to generate; returns a window take."""

    def __init__(self, n: int = 300, fail_for=None):
        self.n = n
        self.calls: list[tuple] = []
        self.fail_for = set(fail_for or ())

    def __call__(self, backbone: str, seed: int, a: int, b: int):
        self.calls.append((backbone, int(seed), int(a), int(b)))
        if (backbone, int(seed)) in self.fail_for:
            return None
        energy = 0.9 if backbone == "edge" else 0.2
        length = min(self.n, int(b) - int(a))     # the pod returns just the window (may be shorter)
        return _take(energy, max(2, length), backbone, int(seed))


def test_live_returns_window():
    p = _FakeProvider(240)
    g = LiveWindowGenerator(p)
    win = g.generate("edge", 60, 150, 0)
    assert win.shape == (90, 139)
    assert p.calls == [("edge", 0, 60, 150)]      # asked the pod for exactly that window


def test_live_memoizes_per_window_and_seed():
    p = _FakeProvider(240)
    g = LiveWindowGenerator(p)
    # the SAME window+seed reused => exactly one pod generation
    for _ in range(4):
        g.generate("edge", 30, 90, 0)
    assert p.calls == [("edge", 0, 30, 90)]
    assert g.n_generated == 1
    # a different window, a new seed, and a new backbone each trigger one more generation
    g.generate("edge", 60, 120, 0)                # different window
    g.generate("edge", 30, 90, 1)                 # new seed
    g.generate("lodge", 30, 90, 0)                # new backbone
    assert g.n_generated == 4


def test_live_pads_when_take_too_short():
    g = LiveWindowGenerator(_FakeProvider(20))    # pod returns only 20 frames
    win = g.generate("edge", 80, 140, 0)          # 60 requested
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
    def boom(backbone, seed, a, b):
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
