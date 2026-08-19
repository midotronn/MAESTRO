"""Unit tests for Phase-1 choreography structure: spectral labels, repeat_of, recapitulation, SRC."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.audio import structure as S
from agentlodge.agent import storyboard as SB
from agentlodge.dance import story as ST
from agentlodge.dance import story_metrics as SM
from agentlodge.dance import transition as T


def _valid_motion(n: int, scale: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r6 = T._matrix_to_sixd(T._sixd_to_matrix(rng.standard_normal((n, 22, 6)))).reshape(n, 132)
    trans = np.cumsum(rng.standard_normal((n, 3)) * scale, axis=0)
    contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
    return np.concatenate([trans, r6, contact], axis=1).astype(np.float32)


def _distinct_structure(nsec: int = 3, seclen: int = 30) -> S.MusicStructure:
    """A structure with all-distinct section labels (no natural repetition)."""
    bounds = [i * seclen for i in range(nsec + 1)]
    total = nsec * seclen
    ec = np.interp(np.arange(total), [0, total // 2, total - 1], [0.0, 1.0, 0.2]).astype(np.float32)
    feats = np.eye(nsec, dtype=float)  # all distinct
    return S._build_sections(bounds, ec, feats, 120.0, total, used_fallback=False)


# --------------------------------------------------------------------------- spectral labels
def test_spectral_labels_group_identical_sections():
    feats = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    labels = S._spectral_labels(feats)
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


def test_spectral_labels_all_distinct_when_no_repeats():
    feats = np.eye(4, dtype=float)
    labels = S._spectral_labels(feats)
    assert len(set(labels.tolist())) == 4


# --------------------------------------------------------------------------- repeat_of
def test_repeat_of_from_labels():
    assert S._repeat_of_from_labels(np.array([0, 1, 2, 1, 2, 0])) == [None, None, None, 1, 2, 0]


def test_build_sections_sets_repeat_of():
    bounds = [0, 30, 60, 90]
    ec = np.linspace(0, 1, 90).astype(np.float32)
    feats = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)  # section 2 repeats 0
    ms = S._build_sections(bounds, ec, feats, 120.0, 90, used_fallback=False)
    assert ms.sections[0].repeat_of is None
    assert ms.sections[2].repeat_of == 0
    assert ms.sections[2].to_dict()["repeat_of"] == 0


# --------------------------------------------------------------------------- recapitulation
def test_recapitulation_injects_mirrored_retrograded_intro_at_end():
    ms = _distinct_structure(3, 30)
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    lodge = _valid_motion(90, seed=3)
    edge = _valid_motion(90, seed=4)

    plain = ST.select_sources(lodge, edge, ms, board, motif_reuse=True, recapitulate=False)
    recap = ST.select_sources(lodge, edge, ms, board, motif_reuse=True, recapitulate=True)

    assert not plain[-1]["source"].startswith("reuse")     # no recap without the flag
    assert recap[-1]["source"] == "reuse:0"                 # recap wins the final section
    assert "reuse:0" in recap[-1]["costs"]
    # the recap clip is the opening clip, retimed + mirrored + retrograded (not a raw slice)
    a, b = recap[-1]["a"], recap[-1]["b"]
    assert recap[-1]["clip"].shape[0] == b - a
    assert not np.allclose(recap[-1]["clip"], lodge[a:b])
    assert not np.allclose(recap[-1]["clip"], edge[a:b])


# --------------------------------------------------------------------------- SRC metric
def test_section_repetition_correlation_high_for_recurring_material():
    base = _valid_motion(30, seed=7)
    other = _valid_motion(30, seed=8)
    motion = np.concatenate([base, other, base], axis=0)  # sections 0 and 2 identical
    bounds = [0, 30, 60, 90]
    ec = np.linspace(0, 1, 90).astype(np.float32)
    feats = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
    ms = S._build_sections(bounds, ec, feats, 120.0, 90, used_fallback=False)
    src = SM.section_repetition_correlation(motion, ms.sections)
    assert src > 0.99
    assert "section_repetition_correlation" in SM.compute_story_metrics(motion, ms)
