"""Unit tests for the structure-aware ("story") choreography stage.

Covers the pure-numpy pieces that run without the heavy rotation/audio backends:
motion transforms (mirror/retime/amplitude_scale), music-structure helpers, the storyboard
agent's rule-based fallback + JSON parsing, and per-section source selection. The librosa
segmentation path and torch-based inertial assembly are validated separately (on-GPU).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.dance import transition as T
from agentlodge.audio import structure as S
from agentlodge.agent import storyboard as SB
from agentlodge.dance import story as ST


def _valid_motion(n: int, scale: float = 0.05, seed: int = 0) -> np.ndarray:
    """A random but structurally valid AgentLODGE 139-dim motion (orthonormal 6D rotations)."""
    rng = np.random.default_rng(seed)
    r6 = T._matrix_to_sixd(T._sixd_to_matrix(rng.standard_normal((n, 22, 6)))).reshape(n, 132)
    trans = np.cumsum(rng.standard_normal((n, 3)) * scale, axis=0)
    contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
    return np.concatenate([trans, r6, contact], axis=1).astype(np.float32)


def _structure(seed: int = 0) -> S.MusicStructure:
    bounds = [0, 30, 90, 150, 210, 270, 300]
    ec = np.interp(np.arange(300), [0, 150, 299], [0.0, 1.0, 0.0]).astype(np.float32)
    feats = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, .05], [0, 0, 1], [1, 0, .05]], float)
    return S._build_sections(bounds, ec, feats, 120.0, 300, used_fallback=False)


# --------------------------------------------------------------------------- motion transforms
def test_mirror_is_involution():
    m = _valid_motion(40, seed=1)
    assert np.allclose(T.mirror(T.mirror(m)), m, atol=1e-4)


def test_mirror_swaps_contacts_and_keeps_valid_rotations():
    m = _valid_motion(30, seed=2)
    mm = T.mirror(m)
    assert np.allclose(mm[:, 135:139], m[:, [137, 138, 135, 136]])
    R = T._sixd_to_matrix(mm[:, 3:135].reshape(30, 22, 6))
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-4)


def test_retime_changes_length_binary_contacts_and_is_identity_at_same_length():
    m = _valid_motion(40, seed=3)
    r = T.retime(m, 25)
    assert r.shape == (25, 139)
    assert set(np.unique(r[:, 135:139]).tolist()) <= {0.0, 1.0}
    assert np.array_equal(T.retime(m, 40), m)


def test_reuse_fit_preserves_tempo_by_selecting_a_centered_excerpt():
    source = _valid_motion(96, seed=31)
    fitted, fit = ST._fit_reuse_clip(source, 36, requested_time_scale=1.0)
    assert fitted.shape == (36, 139)
    assert fit["source_start"] == 30
    assert fit["source_end"] == 66
    assert fit["playback_speed"] == 1.0
    assert fit["cropped"]
    assert not fit["capped"]
    assert np.array_equal(fitted, source[30:66])


def test_reuse_fit_prevents_extreme_speedup():
    source = _valid_motion(96, seed=32)
    fitted, fit = ST._fit_reuse_clip(source, 36, requested_time_scale=0.375)
    assert fitted.shape == (36, 139)
    assert fit["selected_frames"] == 43
    assert fit["playback_speed"] == 1.1944
    assert fit["capped"]
    assert fit["playback_speed"] <= ST.MAX_REUSE_PLAYBACK_SPEED


def test_reuse_cues_are_filtered_and_remapped_after_cropping():
    cues = [
        SB.CommonMotionCue("wave", position=0.1),
        SB.CommonMotionCue("clap_single", position=0.6, direction="left"),
        SB.CommonMotionCue("point_side", position=0.9),
    ]
    _, fit = ST._fit_reuse_clip(_valid_motion(100, seed=33), 40, 1.0)
    remapped = ST._remap_reuse_cues(cues, fit)
    assert [cue.motion_id for cue in remapped] == ["clap_single"]
    assert abs(remapped[0].position - 0.75) < 0.02
    reversed_cues = ST._remap_reuse_cues(
        cues, fit, mirrored=True, retrograded=True
    )
    assert abs(reversed_cues[0].position - 0.25) < 0.02
    assert reversed_cues[0].direction == "right"
    assert reversed_cues[0].mirror


def test_amplitude_scale_preserves_valid_rotations_and_clamps():
    m = _valid_motion(40, seed=4)
    a = T.amplitude_scale(m, 5.0)  # clamped to 1.4
    R = T._sixd_to_matrix(a[:, 3:135].reshape(40, 22, 6))
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-4)
    assert np.array_equal(T.amplitude_scale(m, 1.0), m)


# --------------------------------------------------------------------------- structure helpers
def test_merge_short_and_snap():
    assert S._merge_short([0, 10, 12, 60, 120], 15) == [0, 60, 120]
    assert S._snap_to_downbeats([0, 50, 120], np.array([0, 48, 96, 144]), 5) == [0, 48, 120]


def test_label_sections_detects_repetition():
    feats = np.array([[1, 0, 0], [0, 1, 0], [1, 0, .05], [0, 1, .05], [0, 0, 1]], float)
    assert list(S._label_sections(feats)) == [0, 1, 0, 1, 2]


def test_build_sections_covers_full_span():
    ms = _structure()
    assert ms.boundaries() == [0, 30, 90, 150, 210, 270, 300]
    assert ms.sections[0].start_frame == 0 and ms.sections[-1].end_frame == 300
    assert 0 <= ms.climax_index < len(ms.sections)
    for s in ms.sections:
        assert 0.0 <= s.energy <= 1.0
        assert s.role in S._ROLES


# --------------------------------------------------------------------------- storyboard agent
def test_rule_based_storyboard_reuses_earlier_same_label():
    ms = _structure()
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    assert len(board.plans) == len(ms.sections)
    assert board.used_fallback
    assert board.plans[0].reuse_of is None
    for p in board.plans:
        if p.reuse_of is not None:
            assert p.reuse_of < p.section_index
            assert ms.sections[p.reuse_of].label == ms.sections[p.section_index].label


def test_storyboard_parse_validates_and_rejects_bad_reuse():
    ms = _structure()
    payload = {"arc": "x", "reasoning": "y", "plans": [
        {"section_index": i, "role": s.role, "target_intensity": float(s.energy),
         "vocabulary": "explosive_fast", "generator_bias": "edge",
         "reuse_of": None, "variation": {"mirror": False, "retime": 1.0, "amplitude": 1.1}}
        for i, s in enumerate(ms.sections)]}
    board = SB._parse_response(json.dumps(payload), ms)
    assert not board.used_fallback and len(board.plans) == len(ms.sections)
    # reuse of a section with a different label must be rejected
    bad = {"section_index": 1, "role": "verse", "target_intensity": 0.5,
           "vocabulary": "explosive_fast", "generator_bias": "edge", "reuse_of": 0}
    assert SB._coerce_plan(bad, 1, ms).reuse_of is None


def test_storyboard_parse_rejects_wrong_plan_count():
    ms = _structure()
    try:
        SB._parse_response(json.dumps({"arc": "x", "plans": []}), ms)
    except ValueError:
        return
    raise AssertionError("expected ValueError on plan-count mismatch")


# --------------------------------------------------------------------------- source selection
def test_select_sources_is_gap_free_and_covers_full_length():
    lodge = _valid_motion(300, scale=0.02, seed=5)
    edge = _valid_motion(300, scale=0.12, seed=6)
    ms = _structure()
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    dec = ST.select_sources(lodge, edge, ms, board, motif_reuse=True)
    assert dec[0]["a"] == 0
    assert dec[-1]["b"] == 300
    assert all(dec[i]["b"] == dec[i + 1]["a"] for i in range(len(dec) - 1))
    for d in dec:
        assert d["source"] in {"lodge", "edge"} or d["source"].startswith("reuse:")


def test_select_sources_generates_motif_candidate_when_planned():
    lodge = _valid_motion(300, scale=0.05, seed=7)
    edge = _valid_motion(300, scale=0.05, seed=8)
    ms = _structure()
    board = SB._rule_based_storyboard(ms, motif_reuse=True)
    dec = ST.select_sources(lodge, edge, ms, board, motif_reuse=True)
    # sections 3 and 4 repeat sections 1 and 2 -> a reuse candidate must be scored
    assert any(k.startswith("reuse") for k in dec[3]["costs"])
    assert any(k.startswith("reuse") for k in dec[4]["costs"])


def test_select_sources_reports_bounded_reuse_playback_speed():
    sections = [
        S.Section(0, 90, 0.0, 3.0, 0, "intro", 0.3),
        S.Section(90, 120, 3.0, 4.0, 0, "chorus", 0.3, repeat_of=0),
    ]
    structure = S.MusicStructure(
        sections=sections,
        energy_curve=np.full(120, 0.3, dtype=np.float32),
        recurrence=np.eye(2, dtype=np.float32),
        climax_index=0,
        tempo=120.0,
        total_frames=120,
    )
    board = SB.Storyboard(
        arc="repeat",
        plans=[
            SB.SectionPlan(0, "intro", 0.3, "grounded_minimal", "auto"),
            SB.SectionPlan(
                1,
                "chorus",
                0.3,
                "grounded_minimal",
                "auto",
                reuse_of=0,
                variation={"mirror": False, "retime": 0.25, "amplitude": 1.0},
            ),
        ],
    )
    motion = _valid_motion(120, scale=0.02, seed=34)
    decisions = ST.select_sources(motion, motion.copy(), structure, board, motif_reuse=True)
    assert decisions[1]["source"] == "reuse:0"
    assert decisions[1]["reuse_fit"]["playback_speed"] <= ST.MAX_REUSE_PLAYBACK_SPEED
    assert decisions[1]["reuse_fit"]["capped"]
