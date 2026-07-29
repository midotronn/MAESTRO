"""Unit tests for the backbone-backed window generators (candidate bank + resilient fallback)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor.remote_generator import BankWindowGenerator, ResilientWindowGenerator
from agentlodge.editor.window_edit import MockWindowGenerator, apply_window_edit


def _take(energy: float, n: int = 300, backbone: str = "edge", seed: int = 0) -> np.ndarray:
    return MockWindowGenerator().generate(backbone, 0, n, seed, energy=energy, beats=None)


def _bank(n: int = 300):
    # a calm LODGE take and an energetic EDGE take, plus a 2nd edge seed
    return BankWindowGenerator({
        "lodge": [_take(0.2, n, "lodge", 0)],
        "edge": [_take(0.9, n, "edge", 0), _take(0.85, n, "edge", 1)],
    })


def test_bank_returns_window_slice():
    b = _bank(240)
    win = b.generate("edge", 60, 150, 0)
    assert win.shape == (90, 139)
    assert np.array_equal(win, b.bank["edge"][0][60:150])


def test_bank_seed_cycles_available_takes():
    b = _bank(240)
    assert b.n_takes("edge") == 2
    w0 = b.generate("edge", 0, 30, 0)
    w1 = b.generate("edge", 0, 30, 1)
    w2 = b.generate("edge", 0, 30, 2)          # wraps back to seed 0
    assert not np.array_equal(w0, w1)
    assert np.array_equal(w0, w2)


def test_bank_pads_when_take_too_short():
    b = BankWindowGenerator({"edge": [_take(0.5, 100, "edge", 0)]})
    win = b.generate("edge", 80, 140, 0)       # 60 frames requested, only 20 available
    assert win.shape == (60, 139)
    assert np.array_equal(win[20:], np.repeat(win[19:20], 40, axis=0))  # edge-padded tail


def test_bank_missing_backbone_uses_fallback():
    fb = MockWindowGenerator()
    b = BankWindowGenerator({"edge": [_take(0.5)]}, fallback=fb)
    win = b.generate("lodge", 0, 30, 0)        # no lodge take -> fallback
    assert win is not None and win.shape == (30, 139)


def test_energetic_edit_selects_edge_slice_from_bank():
    n = 300
    bank = _bank(n)
    base = bank.bank["lodge"][0]               # start from the calm take as the "assembled" dance
    beats = np.arange(0, n, 15).astype(float)
    r = apply_window_edit(base, 90, 180, "make this more energetic", bank,
                          beats=beats, k=2, max_cycles=2)
    assert r.ok and r.backbone == "edge"
    assert r.metrics_after["energy"] > r.metrics_before["energy"]
    assert np.array_equal(r.motion[:90], base[:90]) and np.array_equal(r.motion[180:], base[180:])


def test_calmer_edit_selects_lodge_slice_from_bank():
    n = 300
    bank = _bank(n)
    base = bank.bank["edge"][0]                # start from the energetic take
    beats = np.arange(0, n, 15).astype(float)
    r = apply_window_edit(base, 90, 180, "make this calmer", bank,
                          beats=beats, k=2, max_cycles=2)
    assert r.ok and r.backbone == "lodge"
    assert r.metrics_after["energy"] < r.metrics_before["energy"]


def test_resilient_falls_back_on_empty_primary():
    empty = BankWindowGenerator({})            # no takes at all
    res = ResilientWindowGenerator(empty, MockWindowGenerator())
    win = res.generate("edge", 0, 30, 0)
    assert win is not None and win.shape == (30, 139)
    assert res.used_fallback is True


def test_from_dir_loads_bank(tmp_path):
    n = 120
    np.save(tmp_path / "bank_trs_lodge_seed0.npy", _take(0.2, n, "lodge", 0))
    np.save(tmp_path / "bank_trs_edge_seed0.npy", _take(0.9, n, "edge", 0))
    np.save(tmp_path / "bank_trs_edge_seed1.npy", _take(0.85, n, "edge", 1))
    b = BankWindowGenerator.from_dir(tmp_path, "trs")
    assert b.n_takes("lodge") == 1 and b.n_takes("edge") == 2
    assert set(b.backbones) == {"lodge", "edge"}
