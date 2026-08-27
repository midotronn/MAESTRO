"""Unit tests for Phase-2 best-of-K seed selection (generator-agnostic; synthetic candidates)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.dance import best_of_k as BK

_IDENTITY_6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _pulsing_motion(length: int, period: int, phase: int = 0) -> np.ndarray:
    """Valid 139-dim motion whose kinematic speed dips to ~0 every `period` frames, offset by
    `phase` frames -> motion beats at {phase, phase+period, ...}."""
    t = np.arange(length)
    speed = np.abs(np.sin(np.pi * (t - phase) / period))
    trans = np.zeros((length, 3), dtype=np.float32)
    trans[:, 0] = np.cumsum(speed)
    rot = np.tile(_IDENTITY_6D, (length, 22)).reshape(length, 132).astype(np.float32)
    contact = np.ones((length, 4), dtype=np.float32)
    return np.concatenate([trans, rot, contact], axis=1).astype(np.float32)


def _music():
    return np.arange(0, 151, 15)


def test_aligned_candidate_scores_higher_bas():
    aligned = _pulsing_motion(150, 15, phase=0)
    off = _pulsing_motion(150, 15, phase=7)  # ~half a beat out of phase
    scores = BK.score_candidates([off, aligned], _music())
    assert scores[1]["bas"] > scores[0]["bas"]


def test_select_best_picks_the_beat_aligned_candidate():
    aligned = _pulsing_motion(150, 15, phase=0)
    off = _pulsing_motion(150, 15, phase=7)
    best, scores = BK.select_best([off, aligned], _music())
    assert best == 1
    assert len(scores) == 2


def test_best_of_k_selects_winning_seed():
    def generate_fn(seed):
        return _pulsing_motion(150, 15, phase=0) if seed == 42 else _pulsing_motion(150, 15, phase=7)

    motion, seed, report = BK.best_of_k(generate_fn, [1, 42, 3], _music())
    assert seed == 42
    assert report["winner_seed"] == 42
    assert report["k"] == 3
    assert report["scores"][report["winner_index"]]["bas"] == report["winner_bas"]


def test_best_of_k_skips_invalid_and_raises_when_all_fail():
    def bad(seed):
        return None if seed % 2 else np.zeros((1, 139))  # None or too-short

    with pytest.raises(ValueError):
        BK.best_of_k(bad, [1, 2, 3], _music())


def test_best_of_k_survives_a_throwing_seed():
    def flaky(seed):
        if seed == 2:
            raise RuntimeError("bad seed")
        return _pulsing_motion(150, 15, phase=0 if seed == 1 else 7)

    motion, seed, report = BK.best_of_k(flaky, [1, 2, 3], _music())
    assert 2 not in report["seeds"]          # the throwing seed was skipped
    assert seed == 1                          # aligned candidate wins


def test_target_intensity_affects_energy_match():
    calm = _pulsing_motion(150, 15, phase=0)
    calm[:, :3] *= 0.3                          # smaller movement -> lower energy
    lively = _pulsing_motion(150, 15, phase=0)
    lively[:, :3] *= 3.0                         # larger movement -> higher energy
    hi = BK.score_candidates([calm, lively], _music(), target_intensity=1.0)
    assert hi[1]["energy_match"] >= hi[0]["energy_match"]  # target=high favors the livelier one


# --------------------------------------------------------------------------- score_transform
def _marker(tag):
    return np.full((1, 1), tag, dtype=np.float32)


def _phase_transform(m):
    return _pulsing_motion(150, 15, phase=int(m[0, 0]))


def test_score_transform_applied_before_scoring():
    # raw candidates are tiny markers; the transform expands them to aligned/misaligned motions
    scores = BK.score_candidates([_marker(0), _marker(7)], _music(), score_transform=_phase_transform)
    assert scores[0]["bas"] > scores[1]["bas"]


def test_generate_best_of_k_selects_winning_seed():
    def gen(seed):
        return _pulsing_motion(150, 15, phase=0) if seed == 2 else _pulsing_motion(150, 15, phase=7)

    motion, seed, report = BK.generate_best_of_k(gen, 4, _music(), base_seed=0)
    assert seed == 2 and report["seeds"] == [0, 1, 2, 3]


# --------------------------------------------------------------------------- best_of_k_job
def test_best_of_k_job_single_run_when_k_is_one():
    calls = []

    def job(seed):
        calls.append(seed)
        return {"motion": _pulsing_motion(150, 15, 0), "error": None, "summary": "one"}

    r = BK.best_of_k_job(job, 1, _music())
    assert calls == [None] and r["summary"] == "one"
    assert r["best_of_k_requested"] == 1
    assert r["best_of_k_completed"] == 1


def test_best_of_k_job_records_failed_single_run_as_incomplete():
    r = BK.best_of_k_job(
        lambda _seed: {"motion": None, "error": "boom", "summary": ""},
        1,
        _music(),
    )

    assert r["best_of_k_requested"] == 1
    assert r["best_of_k_completed"] == 0


def test_best_of_k_job_selects_best_seed_and_annotates():
    def job(seed):
        return {"motion": _pulsing_motion(150, 15, 0 if seed == 1 else 7),
                "error": None, "summary": "gen"}

    r = BK.best_of_k_job(job, 3, _music())
    assert "best-of-3" in r["summary"] and "seed 1" in r["summary"]
    assert r["best_of_k_requested"] == 3
    assert r["best_of_k_completed"] == 3
    assert r["best_of_k_selected_seed"] == 1
    assert "best_of_k_fallback" not in r


def test_best_of_k_job_falls_back_when_all_fail():
    def job(seed):
        return {"motion": None, "error": "boom", "summary": ""}

    r = BK.best_of_k_job(job, 3, _music())
    assert r["error"] == "boom"   # fell back to a single run
    assert r["best_of_k_requested"] == 3
    assert r["best_of_k_completed"] == 0
    assert r["best_of_k_fallback"]
