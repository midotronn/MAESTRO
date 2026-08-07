"""Exhaustive contracts for the manifest-driven named-motion bank."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor import agent_edit as AE
from agentlodge.editor.motion_bank import (
    MotionBank,
    _JOIN_MAX_FRAMES,
    _root_yaw_series,
    default_motion_bank,
    validate_semantics,
    verify_applied_motion,
)
from agentlodge.editor.window_edit import MockWindowGenerator, window_metrics


def _base(n=240):
    return MockWindowGenerator().generate("edge", 0, n, 4, energy=0.5, beats=None)


_TMPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "server", "data", "smplx_neu_J_1.npy")
_LODGE_SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "lodge_sample_dance.npy")
# The posture checks below need forward kinematics, which needs the licence-gated
# SMPL-X joint template. It is fetched from the pod, so it is absent on a clean clone.
needs_fk = pytest.mark.skipif(not os.path.exists(_TMPL),
                              reason="SMPL-X joint template not present (fetched from the pod)")

_CLAP_HAND_CONTRACTS = ("clap_single", "clap_repeat", "clap_overhead")
_HOST_WRIST_CONTRACTS = ("jump_two_foot",)
_FOREARM_HAND_CONTRACTS = (
    "jump_arms_up",
    "wave",
    "point_side",
    "celebrate_hands_up",
    "arm_punch",
    "rise_reach",
)
_WRIST_STEP_LIMITS = {
    "clap_single": 0.70,
    "clap_repeat": 0.95,
    "clap_overhead": 0.80,
    "jump_two_foot": 0.22,
    "jump_arms_up": 0.25,
    "wave": 0.60,
    "point_side": 0.25,
    "celebrate_hands_up": 0.28,
    "arm_punch": 0.25,
    "rise_reach": 0.38,
}


def _body_forward(joints, frame=0):
    """The direction the dancer faces, read off the feet rather than a world convention.

    The reference frame matters. A window is mostly the song's own choreography, so frame 0 is
    whatever the dancer happened to be doing then -- mid-stride, feet splayed -- and its toes can
    point 70 degrees away from where they face when the spliced action actually happens. Read the
    facing at the frame you are measuring travel from.
    """
    toe = (joints[frame, 10] - joints[frame, 7]) + (joints[frame, 11] - joints[frame, 8])
    forward = np.array([toe[0], toe[1], 0.0])
    return forward / np.linalg.norm(forward)


def _local_joint_rotations(motion):
    from agentlodge.dance.transition import _sixd_to_matrix

    return _sixd_to_matrix(np.asarray(motion)[:, 3:135].reshape(-1, 22, 6))


def _global_joint_rotations(motion):
    from server.fk import BODY_PARENTS

    local = _local_joint_rotations(motion)
    global_r = np.empty_like(local)
    global_r[:, 0] = local[:, 0]
    for joint in range(1, 22):
        global_r[:, joint] = global_r[:, BODY_PARENTS[joint]] @ local[:, joint]
    return global_r


def _hand_forearm_alignment(motion, wrist):
    """Finger/forearm alignment in one local frame, independent of torso and world heading."""
    elbow = 18 if wrist == 20 else 19
    finger = np.array([1.0 if wrist == 20 else -1.0, 0.0, 0.0], dtype=np.float32)
    template = np.load(_TMPL)[:22]
    forearm = template[wrist] - template[elbow]
    forearm /= np.linalg.norm(forearm)
    return (_local_joint_rotations(motion)[:, wrist] @ finger) @ forearm


def _clap_hand_alignment(motion, frame):
    global_r = _global_joint_rotations(motion)
    palm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    left_fingers = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right_fingers = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    left_palm = global_r[frame, 20] @ palm
    right_palm = global_r[frame, 21] @ palm
    left_fingers = global_r[frame, 20] @ left_fingers
    right_fingers = global_r[frame, 21] @ right_fingers
    return float(left_palm @ right_palm), float(left_fingers @ right_fingers)


def test_manifest_has_twenty_valid_redistributable_motions():
    bank = default_motion_bank()
    assert len(bank.specs) == 20
    assert len({s.id for s in bank.specs}) == 20
    for spec in bank.specs:
        clip = bank.load_clip(spec)
        assert clip.shape == (spec.frames, 139)
        assert spec.source and spec.license and spec.attribution
        assert validate_semantics(clip, spec)["ok"]


def test_every_wrist_owning_motion_has_an_explicit_hand_contract():
    wrist_owners = {
        spec.id
        for spec in default_motion_bank().specs
        if (set(spec.absolute_joints) | set(spec.additive_joints)).intersection({20, 21})
    }
    assert wrist_owners == (
        set(_CLAP_HAND_CONTRACTS)
        | set(_HOST_WRIST_CONTRACTS)
        | set(_FOREARM_HAND_CONTRACTS)
    )


@pytest.mark.parametrize(
    "motion_id",
    [
        spec.id
        for spec in default_motion_bank().specs
        if not (set(spec.absolute_joints) | set(spec.additive_joints)).intersection({20, 21})
    ],
)
def test_motions_without_wrist_ownership_preserve_both_host_hands(motion_id):
    base = _base(180)
    out, _ = default_motion_bank().apply(
        base, motion_id, beats=np.arange(0, 180, 15), anchor="beat",
    )
    for wrist in (20, 21):
        channels = slice(3 + 6 * wrist, 3 + 6 * (wrist + 1))
        assert np.array_equal(out[:, channels], base[:, channels]), (motion_id, wrist)


def test_stationary_clips_render_in_agentlodge_layout_despite_zero_translation():
    from agentlodge.dance.format import to_agentlodge139, to_native_finedance139
    clip = default_motion_bank().load_clip("clap_single")
    native = to_native_finedance139(clip)
    restored = to_agentlodge139(native)
    assert np.allclose(restored, clip)
    # Native FineDance layout begins with contacts and keeps the complete rotation block.
    assert np.array_equal(native[:, :4], clip[:, 135:139])
    assert np.allclose(native[:, 7:139], clip[:, 3:135])


def test_every_name_and_alias_routes_through_the_single_generic_tool():
    bank = default_motion_bank()
    for spec in bank.specs:
        for phrase in (spec.name, *spec.aliases):
            plan = AE.plan_edit(f"add {phrase} here", {}, 1.0, 6.0)
            assert [s.tool for s in plan.steps] == ["motion_bank"], (spec.id, phrase, plan.steps)
            assert plan.steps[0].params["motion_id"] == spec.id
            assert plan.steps[0].params["mode"] == "replace"


def test_motion_defaults_distinguish_beat_hits_from_phrase_motions():
    beat_hits = {
        "clap_single", "clap_repeat", "clap_overhead",
        "jump_two_foot", "jump_arms_up",
        "point_side", "celebrate_hands_up", "chest_pop", "arm_punch",
        "crouch_drop", "rise_reach",
    }
    bank = default_motion_bank()
    assert {spec.id for spec in bank.specs if spec.default_anchor == "beat"} == beat_hits
    assert all(spec.default_anchor in {"beat", "center"} for spec in bank.specs)


def test_named_motions_default_to_a_slight_exaggeration():
    bank = default_motion_bank()
    base = _base(180)
    for spec in bank.specs:
        _out, report = bank.apply(base, spec.id, beats=np.arange(0, 180, 15))
        assert report["intensity"] == pytest.approx(0.65), spec.id


@pytest.mark.parametrize(
    "motion_id",
    [spec.id for spec in default_motion_bank().specs if spec.default_anchor == "beat"],
)
def test_beat_hit_motion_defaults_land_on_a_musical_beat(motion_id):
    base = _base(180)
    _out, report = default_motion_bank().apply(
        base, motion_id, beats=np.arange(0, 180, 15),
    )
    assert report["anchor"] == "beat"
    assert report["beat_error_frames"] <= 0.5


def test_beat_anchor_prefers_the_strongest_feasible_beat():
    base = _base(180)
    beats = np.array([30.0, 75.0, 120.0, 165.0])
    strengths = np.array([1.0, 0.3, 0.7, 0.1])

    _out, strongest = default_motion_bank().apply(
        base, "clap_single", beats=beats, beat_strengths=strengths,
    )
    _out, nearest = default_motion_bank().apply(
        base, "clap_single", beats=beats,
    )

    assert strongest["event_frame"] == 30
    assert nearest["event_frame"] == 75


def test_agent_threads_beat_strengths_to_named_motion_placement():
    base = _base(180)
    beats = np.array([30.0, 75.0, 120.0, 165.0])
    strengths = np.array([1.0, 0.3, 0.7, 0.1])
    result = AE.run_agent_edit(
        base, 0, 180, "add a clap here",
        beats=beats, beat_strengths=strengths, max_refine=0,
    )
    report = next(step["motion_bank"] for step in result.log if "motion_bank" in step)
    assert report["anchor"] == "beat"
    assert report["event_frame"] == 30
    assert result.ok


@pytest.mark.parametrize(
    "motion_id",
    [spec.id for spec in default_motion_bank().specs if spec.id not in _CLAP_HAND_CONTRACTS],
)
def test_default_exaggeration_strengthens_owned_pose_without_moving_the_beat(motion_id):
    from agentlodge.dance.transition import _matrix_to_axis_angle, _sixd_to_matrix

    base = _base(180)
    beats = np.arange(0, 180, 15)
    neutral, neutral_report = default_motion_bank().apply(
        base, motion_id, beats=beats, intensity=0.5,
    )
    boosted, boosted_report = default_motion_bank().apply(
        base, motion_id, beats=beats,
    )
    spec = default_motion_bank().resolve(motion_id)
    start, end = neutral_report["action_range"]
    joints = sorted(
        (set(spec.absolute_joints) | set(spec.additive_joints)) - {0, 20, 21}
    )
    base_rot = _sixd_to_matrix(base[start:end, 3:135].reshape(-1, 22, 6))
    neutral_rot = _sixd_to_matrix(neutral[start:end, 3:135].reshape(-1, 22, 6))
    boosted_rot = _sixd_to_matrix(boosted[start:end, 3:135].reshape(-1, 22, 6))
    neutral_delta = np.linalg.norm(_matrix_to_axis_angle(
        neutral_rot[:, joints] @ np.swapaxes(base_rot[:, joints], -1, -2)
    ), axis=-1)
    boosted_delta = np.linalg.norm(_matrix_to_axis_angle(
        boosted_rot[:, joints] @ np.swapaxes(base_rot[:, joints], -1, -2)
    ), axis=-1)
    neutral_rms = float(np.sqrt(np.mean(neutral_delta ** 2)))
    boosted_rms = float(np.sqrt(np.mean(boosted_delta ** 2)))
    assert boosted_rms > 1.03 * neutral_rms
    assert boosted_report["event_frame"] == neutral_report["event_frame"]
    assert boosted_report["action_range"] == neutral_report["action_range"]


@pytest.mark.parametrize("motion_id", _CLAP_HAND_CONTRACTS)
def test_default_clap_exaggeration_preserves_safe_contact(motion_id):
    from server.fk import compute_poses

    base = _base(180)
    beats = np.arange(0, 180, 15)
    neutral, neutral_report = default_motion_bank().apply(
        base, motion_id, beats=beats, intensity=0.5,
    )
    boosted, boosted_report = default_motion_bank().apply(
        base, motion_id, beats=beats,
    )
    event = boosted_report["event_frame"]
    neutral_joints = compute_poses(neutral)["fk_joints"]
    boosted_joints = compute_poses(boosted)["fk_joints"]
    a, b = boosted_report["action_range"]
    neutral_gaps = np.linalg.norm(
        neutral_joints[:, 20] - neutral_joints[:, 21], axis=-1,
    )
    boosted_gaps = np.linalg.norm(
        boosted_joints[:, 20] - boosted_joints[:, 21], axis=-1,
    )
    contact_frames = [
        frame
        for frame in range(a + 1, b - 1)
        if neutral_gaps[frame] < 0.08
        and neutral_gaps[frame] <= neutral_gaps[frame - 1]
        and neutral_gaps[frame] <= neutral_gaps[frame + 1]
    ]
    assert contact_frames
    assert np.allclose(
        boosted_gaps[contact_frames], neutral_gaps[contact_frames], atol=1e-5,
    )
    for frame in contact_frames:
        palm_dot, finger_dot = _clap_hand_alignment(boosted, frame)
        assert palm_dot < -0.95
        assert finger_dot > 0.95

    gap = float(boosted_gaps[event])
    neutral_travel = float(np.linalg.norm(
        np.diff(neutral_joints[a:b, [20, 21]], axis=0), axis=-1
    ).sum())
    boosted_travel = float(np.linalg.norm(
        np.diff(boosted_joints[a:b, [20, 21]], axis=0), axis=-1
    ).sum())
    assert 0.005 < gap < 0.12
    assert boosted_travel > neutral_travel
    assert boosted_report["event_frame"] == neutral_report["event_frame"]


def test_beat_strengths_must_align_one_to_one_with_beats():
    with pytest.raises(ValueError, match="one value for every beat"):
        default_motion_bank().apply(
            _base(180),
            "clap_single",
            beats=np.array([30.0, 75.0]),
            beat_strengths=np.array([1.0]),
        )


@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_every_motion_fits_replacement_and_preserves_contract(motion_id):
    bank = default_motion_bank()
    base = _base(180)
    out, report = bank.apply(base, motion_id, beats=np.arange(0, 180, 15))
    assert out.shape == base.shape
    assert report["id"] == motion_id and report["mode"] == "replace"
    assert report["validation"]["ok"]
    assert np.isfinite(out).all()


@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_every_motion_supports_fixed_duration_insertion(motion_id):
    bank = default_motion_bank()
    base = _base(210)
    out, report = bank.apply(
        base, motion_id, mode="insert", anchor="beat", beats=np.arange(0, 210, 15),
    )
    assert out.shape == base.shape
    assert report["mode"] == "insert" and report["validation"]["ok"]
    a, b = report["action_range"]
    assert 0 < a < b < base.shape[0]


def test_insertion_time_warps_the_host_instead_of_aliasing_replacement():
    """Insert keeps the full timeline by making room around the action; replace leaves it alone."""
    bank = default_motion_bank()
    base = _base(180)
    beats = np.arange(0, 180, 15)
    replaced, replace_report = bank.apply(
        base, "wave", mode="replace", anchor="early", beats=beats,
    )
    inserted, insert_report = bank.apply(
        base, "wave", mode="insert", anchor="early", beats=beats,
    )

    assert replace_report["action_range"] == insert_report["action_range"]
    assert replace_report["event_frame"] == insert_report["event_frame"]
    assert not np.array_equal(inserted, replaced)
    a, b = insert_report["action_range"]
    assert np.array_equal(replaced[:a], base[:a])
    assert np.array_equal(replaced[b:], base[b:])
    assert not np.array_equal(inserted[:a], base[:a])
    assert np.allclose(inserted[[0, -1]], base[[0, -1]], atol=1e-6)


@pytest.mark.parametrize(
    "motion_id,mirrored_chain,original_only",
    [
        ("wave", (13, 16, 18, 20), (14, 17, 19, 21)),
        ("point_side", (13, 16, 18, 20), (14, 17, 19, 21)),
        ("arm_punch", (13, 16, 18, 20), (14, 19, 21)),
    ],
)
def test_mirroring_moves_composition_ownership_to_the_other_side(
    motion_id, mirrored_chain, original_only,
):
    """Mirroring the clip without mirroring its ownership silently discarded the left action."""
    base = _base(180)
    out, _ = default_motion_bank().apply(
        base, motion_id, mirror=True, beats=np.arange(0, 180, 15), anchor="beat",
    )
    for joint in mirrored_chain:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        assert not np.array_equal(out[:, channels], base[:, channels]), (motion_id, joint)
    for joint in original_only:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        assert np.array_equal(out[:, channels], base[:, channels]), (motion_id, joint)


def test_an_unreachable_beat_reports_the_event_that_was_actually_placed():
    """A beat beyond the last feasible action slot must not be reported as a zero-error hit."""
    base = _base(180)
    beats = np.array([179])
    _out, report = default_motion_bank().apply(
        base, "clap_single", beats=beats, anchor="beat",
    )
    a, b = report["action_range"]
    event = report["event_frame"]
    assert a < event < b - 1
    assert report["beat_error_frames"] == pytest.approx(abs(179 - event))
    assert report["beat_error_frames"] > 0.5

    result = AE.run_agent_edit(
        base, 0, len(base), "add a clap on the beat here", beats=beats, max_refine=0,
    )
    semantic = next(
        check for check in result.trace["final"]["checks"] if check["metric"] == "semantic"
    )
    assert not semantic["met"]


@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_every_motion_changes_only_the_channels_declared_in_its_composition(motion_id):
    """The named action owns a layer, not the whole dancer.

    This is the architectural guard against the original clap failure: an arm gesture may not
    rewrite legs, root travel, or contacts, while a jump or step may touch those channels only
    because its manifest explicitly says so.
    """
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    base = _base(180)
    out, report = bank.apply(base, motion_id, beats=np.arange(0, 180, 15), anchor="beat")
    a, b = report["action_range"]

    owned = set(spec.absolute_joints) | set(spec.additive_joints)
    for joint in set(range(22)) - owned:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        assert np.array_equal(out[:, channels], base[:, channels]), (motion_id, joint)

    if not spec.translation_axes:
        assert np.array_equal(out[:, :3], base[:, :3]), motion_id
    else:
        if 2 not in spec.translation_axes:
            assert np.array_equal(out[:, 2], base[:, 2]), motion_id
        if not any(axis < 2 for axis in spec.translation_axes):
            assert np.array_equal(out[:, :2], base[:, :2]), motion_id

    if spec.replace_contacts:
        assert np.array_equal(out[:a, 135:139], base[:a, 135:139]), motion_id
        assert np.array_equal(out[b:, 135:139], base[b:, 135:139]), motion_id
    else:
        assert np.array_equal(out[:, 135:139], base[:, 135:139]), motion_id

    assert report["beat_error_frames"] <= 0.5, motion_id
    assert report["action_frames"] == round(spec.recommended_beats * 15), motion_id


def test_editor_does_not_send_a_composed_named_motion_through_a_second_splice(monkeypatch):
    """The bank already edits the selected host window; only the outer editor crossfade remains."""
    def forbidden_splice(*_args, **_kwargs):
        raise AssertionError("named motion was treated as foreign and spliced twice")

    monkeypatch.setattr(AE, "splice_window", forbidden_splice)
    motion = _base(300)
    result = AE.run_agent_edit(
        motion, 60, 240, "add a clap here", beats=np.arange(0, 300, 15),
        max_refine=0,
    )
    assert result.log[0]["motion_bank"]["id"] == "clap_single"
    assert result.trace["final"]["checks"][0]["met"]


@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_named_motion_verdict_surfaces_artifact_guards_without_erasing_the_edit():
    """A semantic match is kept, but quality regressions remain explicit in the trace."""
    from agentlodge.dance.format import to_editor139

    motion = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:300]
    result = AE.run_agent_edit(
        motion, 60, 240, "add a clap here", beats=np.arange(0, 300, 15),
        max_refine=0,
    )
    guards = [check for check in result.trace["final"]["checks"] if check.get("guard")]
    assert result.ok
    assert any(not check["met"] for check in guards)
    assert not result.trace["attempts"][0]["verify"]["ok"]


@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_raw_lodge_input_is_normalized_before_motion_composition():
    from agentlodge.dance.format import to_editor139

    raw = np.load(_LODGE_SAMPLE).astype(np.float32)[:180]
    normalized = to_editor139(raw)
    assert not np.array_equal(raw, normalized)

    bank = default_motion_bank()
    kwargs = {"beats": np.arange(0, 180, 15), "anchor": "beat"}
    from_raw, raw_report = bank.apply(raw, "clap_single", **kwargs)
    from_normalized, normalized_report = bank.apply(normalized, "clap_single", **kwargs)
    assert np.array_equal(from_raw, from_normalized)
    assert raw_report == normalized_report


@needs_fk
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_real_output_clap_preserves_the_host_rhythm_and_reads_as_a_clap():
    from agentlodge.dance.format import to_editor139
    from server.fk import compute_poses

    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:120]
    out, report = default_motion_bank().apply(
        base, "clap_single", beats=np.arange(0, 120, 15), anchor="beat",
    )

    assert np.array_equal(out[:, :3], base[:, :3])
    assert np.array_equal(out[:, 135:139], base[:, 135:139])
    for joint in range(13):
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        assert np.array_equal(out[:, channels], base[:, channels]), joint

    before = compute_poses(base)["fk_joints"]
    after = compute_poses(out)["fk_joints"]
    assert np.allclose(after[:, :16], before[:, :16], atol=1e-6)

    event = int(report["event_frame"])
    gap = float(np.linalg.norm(after[event, 20] - after[event, 21]))
    shoulders = 0.5 * (after[event, 16] + after[event, 17])
    upper = shoulders - after[event, 9]
    upper /= np.linalg.norm(upper)
    hands = 0.5 * (after[event, 20] + after[event, 21]) - shoulders
    assert 0.005 < gap < 0.08
    palm_dot, finger_dot = _clap_hand_alignment(out, event)
    assert palm_dot < -0.95
    assert finger_dot > 0.95
    assert -0.32 < float(hands @ upper) < -0.10

    beats = np.arange(0, 120, 15)
    before_metrics = window_metrics(base, beats)
    after_metrics = window_metrics(out, beats)
    # Correct palm contact adds a fast, localized wrist rotation. It may sharpen the aggregate
    # jerk score, but remains bounded to roughly twice the host rather than restoring the old
    # whole-arm wind-up that made the gesture slow and pasted-on.
    assert after_metrics["jerk"] < 2.2 * before_metrics["jerk"]


@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
@pytest.mark.parametrize(
    "motion_id,max_jerk_ratio",
    [
        ("clap_overhead", 3.0),
        ("jump_arms_up", 2.0),
        ("point_side", 1.6),
        ("celebrate_hands_up", 2.0),
        ("arm_punch", 1.7),
        ("rise_reach", 3.2),
    ],
)
def test_beat_compressed_pose_accents_take_the_shortest_path(
    motion_id, max_jerk_ratio,
):
    """Retiming a long ready-stance clip into one beat must not preserve its whole wind-up.

    The semantic pose belongs on the beat, but making the host visit every authored in-between
    pose first creates the same acceleration spike that made the original clap feel pasted onto
    the dance. Event-pose joints travel directly from the host to the accent and back; joints
    carrying real internal motion (the jump, rise, or torso response) keep their clip trajectory.
    """
    from agentlodge.dance.format import to_editor139

    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:120]
    beats = np.arange(0, 120, 15)
    before = window_metrics(base, beats)
    out, report = default_motion_bank().apply(
        base, motion_id, beats=beats, anchor="beat",
    )
    after = window_metrics(out, beats)
    assert report["validation"]["ok"]
    assert after["jerk"] < max_jerk_ratio * before["jerk"]


@needs_fk
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_outer_crossfade_never_overwrites_a_named_event_in_a_short_window():
    """The editor seam blend may use only frames outside the bank's semantic action."""
    from agentlodge.dance.format import to_editor139
    from server.fk import compute_poses

    motion = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))
    beats = np.arange(5, len(motion), 15)
    result = AE.run_agent_edit(
        motion, 0, 30, "add a clap on the beat here", beats=beats, max_refine=0,
    )
    report = next(step["motion_bank"] for step in result.log if "motion_bank" in step)
    event = int(report["event_frame"])
    joints = compute_poses(result.motion)["fk_joints"]
    gap = float(np.linalg.norm(joints[event, 20] - joints[event, 21]))

    assert report["action_range"] == [12, 27]
    assert event == 20
    assert result.ok
    assert 0.005 < gap < 0.12
    palm_dot, finger_dot = _clap_hand_alignment(result.motion, event)
    assert palm_dot < -0.95
    assert finger_dot > 0.95


def test_short_selections_keep_global_beat_cadence_and_fail_when_no_hit_fits():
    """One in-window beat still has a period; zero feasible beats must not report alignment."""
    motion = _base(90)
    beats = np.arange(0, len(motion), 15)

    fitted = AE.run_agent_edit(
        motion, 5, 25, "add a clap on the beat here", beats=beats, max_refine=0,
    )
    fitted_report = next(
        step["motion_bank"] for step in fitted.log if "motion_bank" in step
    )
    assert fitted_report["action_frames"] == 15
    assert fitted_report["event_frame"] == 10
    assert fitted_report["beat_error_frames"] == pytest.approx(0.0)
    assert fitted.ok

    impossible = AE.run_agent_edit(
        motion, 2, 14, "add a clap on the beat here", beats=beats, max_refine=0,
    )
    impossible_report = next(
        step["motion_bank"] for step in impossible.log if "motion_bank" in step
    )
    assert impossible_report["beat_error_frames"] > 0.5
    assert not impossible.ok

    beatless = AE.run_agent_edit(
        motion, 5, 25, "add a clap here", beats=np.array([]), max_refine=0,
    )
    beatless_report = next(
        step["motion_bank"] for step in beatless.log if "motion_bank" in step
    )
    assert beatless_report["beat_error_frames"] is None
    assert beatless.ok


def test_explicit_whole_window_beat_goal_survives_a_named_beat_action():
    motion = _base(180)
    beats = np.arange(0, len(motion), 15)
    anchored = AE.run_agent_edit(
        motion, 0, len(motion), "add a clap on the beat here",
        beats=beats, max_refine=0,
    )
    assert not any(goal["metric"] == "bas" for goal in anchored.trace["goals"])

    instructions = (
        "add a clap, then make the rest of the window more on beat",
        "add a clap and put everything on the beat",
        "add a clap and make it tighter to the beat",
        "add a clap and lock the window to the beat",
    )
    for instruction in instructions:
        result = AE.run_agent_edit(
            motion, 0, len(motion), instruction, beats=beats, max_refine=0,
        )

        assert any(goal["metric"] == "bas" for goal in result.trace["goals"]), instruction
        assert any(step["tool"] == "beat_align" for step in result.log), instruction
        before = result.trace["final"]["metrics_before"]["bas"]
        after = result.trace["final"]["metrics_after"]["bas"]
        assert after >= before, instruction
        assert result.ok, instruction


@needs_fk
def test_motion_bank_runs_after_temporal_tools_so_its_report_is_not_stale(monkeypatch):
    from server.fk import compute_poses

    plan = AE.AgentPlan(
        "reverse and add a clap",
        [
            AE.PlanStep(
                "motion_bank",
                {"motion_id": "clap_single", "mode": "replace", "anchor": "early"},
                "place clap",
            ),
            AE.PlanStep("reverse", {}, "reverse the host"),
        ],
        goals=[],
    )
    monkeypatch.setattr(
        AE,
        "plan_edit",
        lambda *args, **kwargs: AE.AgentPlan(
            plan.summary,
            [AE.PlanStep(step.tool, dict(step.params), step.why) for step in plan.steps],
            goals=[],
        ),
    )

    motion = _base(180)
    result = AE.run_agent_edit(
        motion, 0, len(motion), "reverse and add a clap",
        beats=np.arange(0, len(motion), 15), max_refine=0,
    )
    assert [step["tool"] for step in result.log] == ["reverse", "motion_bank"]

    report = result.log[-1]["motion_bank"]
    event = int(report["event_frame"])
    joints = compute_poses(result.motion)["fk_joints"]
    gap = float(np.linalg.norm(joints[event, 20] - joints[event, 21]))
    semantic = next(
        check for check in result.trace["final"]["checks"] if check["metric"] == "semantic"
    )
    assert report["beat_error_frames"] == pytest.approx(0.0)
    assert 0.005 < gap < 0.12
    palm_dot, finger_dot = _clap_hand_alignment(result.motion, event)
    assert palm_dot < -0.95
    assert finger_dot > 0.95
    assert semantic["met"] and result.ok


def test_composed_joint_activity_can_be_valid_relative_to_the_host():
    from agentlodge.dance.transition import _axis_angle_to_matrix, _matrix_to_sixd

    spec = default_motion_bank().resolve("point_side")
    identity = np.eye(3, dtype=np.float32)
    rotations = np.repeat(identity[None, None], 2 * 22, axis=0).reshape(2, 22, 3, 3)
    reference = rotations.copy()
    rotations[1, 17] = _axis_angle_to_matrix(
        np.array([0.0, 0.0, 0.70], dtype=np.float32)
    )
    reference[1, 17] = _axis_angle_to_matrix(
        np.array([0.0, 0.0, -0.70], dtype=np.float32)
    )

    clip = np.zeros((2, 139), dtype=np.float32)
    host = clip.copy()
    clip[:, 3:135] = _matrix_to_sixd(rotations).reshape(2, 132)
    host[:, 3:135] = _matrix_to_sixd(reference).reshape(2, 132)

    assert not validate_semantics(clip, spec)["ok"]
    assert validate_semantics(clip, spec, reference=host)["ok"]


def test_agent_replacement_preserves_every_frame_outside_selection():
    motion = _base(300)
    result = AE.run_agent_edit(
        motion, 60, 240, "add a clap motion here", beats=np.arange(0, 300, 15),
    )
    assert result.log[0]["tool"] == "motion_bank"
    assert result.log[0]["motion_bank"]["id"] == "clap_single"
    assert result.trace["final"]["checks"][0]["met"]
    assert np.array_equal(result.motion[:60], motion[:60])
    assert np.array_equal(result.motion[240:], motion[240:])
    assert not np.array_equal(result.motion[60:240], motion[60:240])


def test_agent_insert_preserves_total_duration_and_uses_relational_policy():
    motion = _base(300)
    result = AE.run_agent_edit(
        motion, 60, 240, "insert a wave before the next move", beats=np.arange(0, 300, 15),
    )
    meta = result.log[0]["motion_bank"]
    assert result.motion.shape == motion.shape
    assert meta["id"] == "wave" and meta["mode"] == "insert" and meta["anchor"] == "early"
    assert result.trace["final"]["checks"][0]["met"]
    assert np.array_equal(result.motion[:60], motion[:60])
    assert np.array_equal(result.motion[240:], motion[240:])


def test_direction_repeat_and_intensity_variants_are_data_driven():
    plan = AE.plan_edit("add a big point left twice here", {}, 0.0, 6.0)
    step = plan.steps[0]
    assert step.tool == "motion_bank"
    assert step.params == {
        "motion_id": "point_side", "mode": "replace", "anchor": "beat",
        "mirror": True, "intensity": 0.8, "repeats": 2,
    }
    # Point is intentionally non-repeatable. The request still lands as a single point with the
    # dropped repetition stated, rather than failing and leaving the window untouched.
    motion = _base(240)
    result = AE.run_agent_edit(motion, 30, 210, "add a big point left twice here")
    assert result.log[0]["status"] != "failed"
    assert result.trace["final"]["checks"][0]["met"]
    assert "does not support repetition" in result.log[0]["note"]


def test_unknown_motion_never_silently_resolves():
    bank = default_motion_bank()
    with pytest.raises(KeyError, match="unknown named motion"):
        bank.resolve("moonwalk backflip")
    with pytest.raises(KeyError, match="unknown named motion"):
        bank.apply(_base(120), "moonwalk_backflip")


def test_insertion_rejects_too_short_window_instead_of_corrupting_timing():
    with pytest.raises(ValueError, match="too short"):
        default_motion_bank().apply(_base(20), "clap_single", mode="insert")


def test_manifest_rejects_duplicate_aliases(tmp_path):
    original = json.loads((default_motion_bank().root / "manifest.json").read_text(encoding="utf-8"))
    original["motions"] = original["motions"][:2]
    original["motions"][1]["aliases"].append(original["motions"][0]["aliases"][0])
    (tmp_path / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate motion alias"):
        MotionBank(tmp_path)


def test_motion_bank_is_one_tool_not_twenty_special_cases():
    assert "motion_bank" in AE.TOOLS
    bank_ids = {s.id for s in default_motion_bank().specs}
    assert not (bank_ids & set(AE.TOOLS))


@needs_fk
@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_planted_feet_never_sink_through_the_floor(motion_id):
    """Hand-authoring a root drop next to a knee bend used to push the feet underground."""
    from server.fk import compute_poses
    bank = default_motion_bank()
    clip = bank.load_clip(motion_id)
    joints = compute_poses(clip)["fk_joints"]
    lowest = np.min(joints[:, (7, 8, 10, 11), 2], axis=1)
    grounded = clip[:, 135:139].sum(axis=1) > 0
    assert grounded.any(), motion_id
    floor = float(np.median(lowest[grounded]))
    assert float(np.min(lowest[grounded])) > floor - 0.03, motion_id


@needs_fk
@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_every_motion_animates_a_meaningful_share_of_the_body(motion_id):
    """Guards against clips that read as a mannequin with a single moving limb."""
    from server.fk import compute_poses
    bank = default_motion_bank()
    joints = compute_poses(bank.load_clip(motion_id))["fk_joints"]
    relative = joints - joints[:, :1, :]
    excursion = np.linalg.norm(relative - relative[0], axis=-1).max(axis=0)
    assert int((excursion > 0.02).sum()) >= 10, motion_id


@needs_fk
def test_lateral_steps_spread_the_stance_instead_of_leaning_both_legs_one_way():
    """A side step is a weight transfer: the legs must open, not swing over together."""
    from server.fk import compute_poses
    bank = default_motion_bank()
    for motion_id in ("side_step", "step_touch"):
        joints = compute_poses(bank.load_clip(motion_id))["fk_joints"]
        left = joints[:, 7] - joints[:, 1]
        right = joints[:, 8] - joints[:, 2]
        left_angle = np.arctan2(left[:, 0], -left[:, 2])
        right_angle = np.arctan2(right[:, 0], -right[:, 2])
        same_side = float(np.mean(np.sign(left_angle) == np.sign(right_angle)))
        assert same_side < 0.75, (motion_id, same_side)
        separation = np.linalg.norm(joints[:, 7, :2] - joints[:, 8, :2], axis=-1)
        assert float(separation.max()) > 0.35, (motion_id, separation.max())


# The manifest validators are generic shape checks -- joint_activity only asks that the arms
# move, so an action can satisfy its contract while failing to depict what it is named after.
# These pin the meaning of the actions where the rendered review found the gap.

@needs_fk
@pytest.mark.parametrize("motion_id", ["clap_single", "clap_repeat", "clap_overhead"])
def test_claps_bring_the_hands_together_on_the_beat(motion_id):
    """The event frame is what the editor snaps to a beat, so that is where the hands must meet.

    Wrist distance alone is not enough: the original fix held the wrist centres apart but left
    the rendered palms in incompatible planes. Once the wrists explicitly orient the hands,
    their IK targets can sit close to the midline without the forearms crossing. Keep a small
    positive lower bound as a positional sanity check; the orientation contract below decides
    whether those nearby hands can actually make palm contact.
    """
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    joints = compute_poses(bank.load_clip(motion_id))["fk_joints"]
    gap = float(np.linalg.norm(joints[spec.event_frame, 20] - joints[spec.event_frame, 21]))
    assert 0.005 < gap < 0.08, (motion_id, gap)


@needs_fk
@pytest.mark.parametrize("motion_id", ["clap_single", "clap_repeat", "clap_overhead"])
def test_claps_align_the_palm_planes_and_fingers_on_contact(motion_id):
    """Hands at the same point still look broken when one palm is up and the other is sideways."""
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    palm_dot, finger_dot = _clap_hand_alignment(bank.load_clip(motion_id), spec.event_frame)
    assert palm_dot < -0.98, (motion_id, "palms are not facing each other", palm_dot)
    assert finger_dot > 0.98, (motion_id, "fingers are not parallel", finger_dot)


@needs_fk
@pytest.mark.parametrize("motion_id", _HOST_WRIST_CONTRACTS + _FOREARM_HAND_CONTRACTS)
def test_authored_non_contact_hands_continue_the_forearm(motion_id):
    """Open hands may twist around the arm, but may not bend into a detached hand plane."""
    motion = default_motion_bank().load_clip(motion_id)
    spec = default_motion_bank().resolve(motion_id)
    owned = set(spec.absolute_joints) | set(spec.additive_joints)
    for wrist in (20, 21):
        if wrist not in owned:
            continue
        alignment = _hand_forearm_alignment(motion, wrist)
        assert float(alignment.min()) > 0.99, (motion_id, wrist, float(alignment.min()))


@pytest.mark.parametrize("motion_id", _HOST_WRIST_CONTRACTS + _FOREARM_HAND_CONTRACTS)
def test_authored_non_contact_wrists_do_not_flip_or_hyperextend(motion_id):
    from agentlodge.dance.transition import _matrix_to_axis_angle, _sixd_to_matrix

    spec = default_motion_bank().resolve(motion_id)
    clip = default_motion_bank().load_clip(spec)
    local = _sixd_to_matrix(clip[:, 3:135].reshape(-1, 22, 6))
    owned = (set(spec.absolute_joints) | set(spec.additive_joints)).intersection({20, 21})
    for wrist in owned:
        angle = np.linalg.norm(_matrix_to_axis_angle(local[:, wrist]), axis=-1)
        limit = 0.70 if motion_id == "wave" else 0.05
        assert float(angle.max()) < limit, (motion_id, wrist, float(angle.max()))
        step = local[1:, wrist] @ np.swapaxes(local[:-1, wrist], -1, -2)
        step_angle = np.linalg.norm(_matrix_to_axis_angle(step), axis=-1)
        assert float(step_angle.max()) < 0.35, (
            motion_id, wrist, "abrupt wrist flip", float(step_angle.max()),
        )


@pytest.mark.parametrize("degrees", [0, 45, 90, 135, 180, 225, 270, 315])
def test_host_relative_hand_motion_does_not_repose_wrists(degrees):
    from agentlodge.editor.motion_bank import _yaw_rotate

    base = _base()
    if degrees:
        base = _yaw_rotate(base.copy(), np.deg2rad(degrees), base[0, :3].copy())
    out, _ = default_motion_bank().apply(base, "jump_two_foot", mode="replace")
    for wrist in (20, 21):
        channels = slice(3 + 6 * wrist, 3 + 6 * (wrist + 1))
        assert np.allclose(out[:, channels], base[:, channels], atol=1e-6), (
            degrees, wrist,
        )


@needs_fk
@pytest.mark.parametrize("motion_id", _FOREARM_HAND_CONTRACTS)
@pytest.mark.parametrize("mode", ["replace", "insert"])
def test_authored_hand_plane_survives_composition_at_every_host_heading(motion_id, mode):
    from agentlodge.editor.motion_bank import _yaw_rotate

    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    original_wrists = (set(spec.absolute_joints) | set(spec.additive_joints)).intersection({20, 21})
    mirrors = (False, True) if spec.mirrorable else (False,)
    for mirror in mirrors:
        wrists = {21 if wrist == 20 else 20 for wrist in original_wrists} if mirror else original_wrists
        for degrees in range(0, 360, 45):
            base = _base()
            if degrees:
                base = _yaw_rotate(base, np.deg2rad(degrees), base[0, :3].copy())
            out, report = bank.apply(
                base, motion_id, mode=mode, mirror=mirror, anchor="center",
            )
            event = int(report["event_frame"])
            for wrist in wrists:
                alignment = float(_hand_forearm_alignment(out, wrist)[event])
                assert alignment > 0.99, (
                    motion_id, mode, mirror, degrees, wrist, alignment,
                )


@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
@pytest.mark.parametrize("motion_id", tuple(_WRIST_STEP_LIMITS))
@pytest.mark.parametrize("mode", ["replace", "insert"])
def test_composed_wrists_do_not_flip_on_real_dance_phases(motion_id, mode):
    from agentlodge.dance.format import to_editor139
    from agentlodge.dance.transition import _matrix_to_axis_angle

    dance = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    wrists = sorted(
        (set(spec.absolute_joints) | set(spec.additive_joints)).intersection({20, 21})
    )
    for start in (0, 160, 320):
        base = dance[start:start + 120].copy()
        out, report = bank.apply(
            base, motion_id, mode=mode, anchor="center", beats=np.arange(0, 120, 16),
        )
        local = _local_joint_rotations(out)
        action_start, action_end = map(int, report["action_range"])
        for wrist in wrists:
            step = (
                local[action_start + 1:action_end, wrist]
                @ np.swapaxes(local[action_start:action_end - 1, wrist], -1, -2)
            )
            step_angle = np.linalg.norm(_matrix_to_axis_angle(step), axis=-1)
            maximum = float(step_angle.max())
            assert maximum < _WRIST_STEP_LIMITS[motion_id], (
                motion_id, mode, start, wrist, maximum,
            )


@needs_fk
@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_no_motion_drives_the_two_hands_through_each_other(motion_id):
    """Whatever a motion is doing, the dancer has two separate hands for the whole of it.

    Checked over every frame of every clip rather than only the claps, because nothing about
    the defect was clap-specific: any two-armed recipe that shares one IK target reproduces it.
    Non-clap motions keep distinct wrists. Claps additionally prove that the palm planes oppose
    and the fingers agree, because their rotated hand surfaces can meet with nearby wrist centres.
    """
    from server.fk import compute_poses
    motion = default_motion_bank().load_clip(motion_id)
    joints = compute_poses(motion)["fk_joints"]
    gap = float(np.linalg.norm(joints[:, 20] - joints[:, 21], axis=-1).min())
    if motion_id.startswith("clap_"):
        spec = default_motion_bank().resolve(motion_id)
        palm_dot, finger_dot = _clap_hand_alignment(motion, spec.event_frame)
        assert gap > 0.005, (motion_id, gap)
        assert palm_dot < -0.98, (motion_id, palm_dot)
        assert finger_dot > 0.98, (motion_id, finger_dot)
    else:
        assert gap > 0.06, (motion_id, gap)


@needs_fk
@pytest.mark.parametrize("motion_id", ["clap_single", "clap_repeat"])
def test_a_clap_lands_in_front_of_the_chest_not_under_the_chin(motion_id):
    """Hands meeting is not enough -- WHERE they meet is what makes it read as a clap.

    The target was authored at y=+0.04 in the template frame. The shoulders sit at +0.083 and
    the chest at -0.057, so that put the palms ABOVE the chest, level with the collarbones and
    tucked in close: rendered, it read as a bow or a prayer rather than a clap, and the user
    rejected it on sight. Every existing assertion passed, because all of them only ever asked
    whether the hands got close to each other. So pin the height and the reach as well.
    """
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    joints = compute_poses(bank.load_clip(motion_id))["fk_joints"]
    ev = spec.event_frame
    hands = 0.5 * (joints[ev, 20] + joints[ev, 21])
    shoulders = 0.5 * (joints[ev, 16] + joints[ev, 17])
    drop = float(hands[2] - shoulders[2])
    assert -0.32 < drop < -0.10, (motion_id, "clap height vs shoulders", drop)
    ahead = float((hands - shoulders) @ np.array([0.0, -1.0, 0.0]))
    assert ahead > 0.15, (motion_id, "clap is not out in front of the body", ahead)


@needs_fk
@pytest.mark.parametrize("motion_id", ["clap_single", "clap_repeat"])
def test_a_clap_keeps_its_elbows_down_rather_than_winged_out(motion_id):
    """Two arm poses reach the same hand target, and only one of them looks like clapping.

    ``_solve_arm`` picks between them with a single hint vector, which pointed straight out to
    the side for every motion. For a clap that lifts the elbows to near shoulder height and
    splays them wide, so the clip reads as flapping wings. Nobody clapping holds their elbows
    up; they hang down by the ribs. The winged build measured -0.08/-0.11 m below the shoulder
    line and 0.25-0.31 m out from the midline, so this would have failed on both counts.
    """
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    joints = compute_poses(bank.load_clip(motion_id))["fk_joints"]
    ev = spec.event_frame
    for elbow, shoulder in ((18, 16), (19, 17)):
        drop = float(joints[ev, elbow, 2] - joints[ev, shoulder, 2])
        assert drop < -0.18, (motion_id, "elbow is winged up to shoulder height", drop)
        out = abs(float(joints[ev, elbow, 0] - joints[ev, 0, 0]))
        assert out < 0.24, (motion_id, "elbow is splayed out wide", out)


@needs_fk
def test_the_repeated_clap_keeps_the_hands_up_between_claps():
    """A repeat clap parts the hands a few inches; it does not drop them back to the hips.

    Driving the arms straight off the hit pulses returned them to the rest stance after every
    clap, so the hands swung the full 0.84 m of the stance width three times over. Rendered,
    that is flapping, not clapping. Measured on the broken build, the hands re-opened to
    0.48-0.68 m between claps and fell to 0.29-0.44 m below the shoulders.

    The ends are deliberately excluded: the clip still has to start and finish on the shared
    neutral stance, because that is what the splice hands over to the surrounding dance.
    """
    from server.fk import compute_poses
    bank = default_motion_bank()
    joints = compute_poses(bank.load_clip("clap_repeat"))["fk_joints"]
    n = len(joints)
    held = slice(int(0.25 * n), int(0.85 * n))
    gap = np.linalg.norm(joints[held, 20] - joints[held, 21], axis=-1)
    assert gap.max() < 0.36, ("hands drop back to the sides between claps", float(gap.max()))

    hands = 0.5 * (joints[held, 20] + joints[held, 21])
    shoulders = 0.5 * (joints[held, 16] + joints[held, 17])
    assert float((hands[:, 2] - shoulders[:, 2]).min()) > -0.26, "hands fall away between claps"

    # ...and it is still three separate claps, not one long squeeze. Count dips the hands
    # recover from on both sides, since once they correctly stay up no fixed threshold on the
    # gap can separate the claps from the hold.
    full = np.linalg.norm(joints[:, 20] - joints[:, 21], axis=-1)
    closings = sum(
        1 for i in range(1, len(full) - 1)
        if full[i] <= full[i - 1] and full[i] < full[i + 1]
        and min(full[:i].max(initial=full[i]), full[i + 1:].max(initial=full[i])) - full[i] >= 0.03
    )
    assert closings == 3, ("expected three claps", closings)


@needs_fk
@pytest.mark.parametrize("motion_id", ["clap_single", "clap_repeat"])
@pytest.mark.parametrize("window", [(60, 240), (100, 196), (200, 460)])
def test_a_spliced_clap_still_reads_as_a_clap(motion_id, window):
    """The clip is not what anyone watches -- the spliced result is.

    Retiming to a beat-locked window and blending both ends into the surrounding choreography
    all happen after the clip is authored, so the posture has to be re-checked downstream of
    them rather than assumed to survive.

    Measured in the DANCER'S frame, not world Z. The host dance leans and twists the torso, and
    reading "hands above the shoulders" off the world vertical charges that lean to the clap:
    the same bit-identical arm pose scores anywhere from -0.14 to +0.26 depending only on what
    the song was doing. That is the same mistake as grading a clip's speed against the song's.
    """
    from server.fk import compute_poses
    a, b = window
    base = _base(600)
    spliced, report = default_motion_bank().apply(
        base[a:b], motion_id, beats=np.arange(0, b - a, 16),
    )
    out = base.copy()
    out[a:b] = spliced
    j = compute_poses(out)["fk_joints"]

    k = a + int(report["event_frame"])
    gap = float(np.linalg.norm(j[k, 20] - j[k, 21]))
    assert 0.005 < gap < 0.12, ("hands do not meet after splicing", gap)
    palm_dot, finger_dot = _clap_hand_alignment(out, k)
    assert palm_dot < -0.95, ("palms lose alignment after splicing", palm_dot)
    assert finger_dot > 0.95, ("fingers lose alignment after splicing", finger_dot)

    # The arms inherit the upper torso, not an imaginary straight line from pelvis to neck.
    # On a bent host pose that pelvis axis can point across the body and report a canonical,
    # bit-identical clap as being above the shoulders. Build the frame from the chest and
    # shoulder girdle, which is what the viewer reads as the dancer's local upright.
    shoulders = 0.5 * (j[k, 16] + j[k, 17])
    up = shoulders - j[k, 9]
    up = up / np.linalg.norm(up)
    hands = 0.5 * (j[k, 20] + j[k, 21]) - shoulders
    assert -0.34 < float(hands @ up) < -0.08, ("clap is not at chest height", float(hands @ up))
    for elbow, shoulder in ((18, 16), (19, 17)):
        assert float((j[k, elbow] - j[k, shoulder]) @ up) < -0.15, "elbow winged after splicing"


@needs_fk
def test_the_overhead_clap_actually_clears_the_head():
    """Height is the only thing separating it from clap_single, and nothing pinned it."""
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve("clap_overhead")
    joints = compute_poses(bank.load_clip("clap_overhead"))["fk_joints"]
    ev = spec.event_frame
    assert float(min(joints[ev, 20, 2], joints[ev, 21, 2]) - joints[ev, 15, 2]) > 0.12


@needs_fk
def test_pointing_to_the_side_reaches_sideways_rather_than_forward():
    """A target only slightly off the forward axis renders as a forward reach, not a side point."""
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve("point_side")
    joints = compute_poses(bank.load_clip("point_side"))["fk_joints"]
    offset = joints[spec.event_frame, 21] - joints[spec.event_frame, 17]
    assert abs(offset[0]) > 1.5 * abs(offset[1]), offset


@needs_fk
def test_a_chest_pop_moves_the_chest_and_not_the_head():
    """A chest pop is an isolation: the ribcage leads and the head stays where it is.

    Only joints BELOW the chest can move the chest. The recipe drove spine3 (+0.66) and the
    neck, which sit above it, so they swung the head instead -- the chest stayed at exactly its
    rest offset, +0.018 m, unchanged to the millimetre, while the head speared 0.272 m forward
    and dropped 0.134 m. Rendered, that is a pigeon peck, not a pop. Nothing caught it: the
    articulation validator only asks whether those joints moved at all, and "a meaningful share
    of the body animates" cannot tell a pop from a nod. This is the same failure as the clap
    that read as a namaste, so pin the thing the name actually promises.
    """
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve("chest_pop")
    joints = compute_poses(bank.load_clip("chest_pop"))["fk_joints"]
    fwd = np.array([0.0, -1.0, 0.0])

    def travel(joint):
        """Displacement relative to the pelvis, so a step or lean cannot fake a pop."""
        return ((joints[spec.event_frame, joint] - joints[spec.event_frame, 0])
                - (joints[0, joint] - joints[0, 0]))

    chest, head = travel(9), travel(15)
    chest_fwd, head_fwd = float(chest @ fwd), float(head @ fwd)
    assert chest_fwd > 0.04, ("the chest never actually pops forward", chest_fwd)
    assert abs(head_fwd) < 0.6 * chest_fwd, (
        "the head travels as far as the chest, so this reads as a nod", head_fwd, chest_fwd)
    assert float(head[2]) > -0.06, ("the head dives instead of staying level", float(head[2]))


@needs_fk
@pytest.mark.parametrize("motion_id", ["jump_two_foot", "jump_arms_up"])
def test_jumps_are_off_the_ground_at_their_accent(motion_id):
    """A jump whose accent lands while the feet are planted reads as a squat."""
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    clip = bank.load_clip(motion_id)
    joints = compute_poses(clip)["fk_joints"]
    assert int(clip[spec.event_frame, 135:139].sum()) == 0, motion_id
    lift = float(joints[spec.event_frame, (7, 8, 10, 11), 2].min()
                 - joints[0, (7, 8, 10, 11), 2].min())
    assert lift > 0.08, (motion_id, lift)


@needs_fk
def test_a_bounce_actually_travels_far_enough_to_see():
    """The bounce lives entirely in the knees, and a shallow one is invisible on a rendered body.

    ``_ground`` re-derives pelvis height from the leg pose on every grounded frame, so an authored
    root rise is silently cancelled and only leg flexion moves the dancer. That made an earlier
    version bob by 2cm -- numerically a bounce, visually a statue. Pin the travel a viewer sees.
    """
    from server.fk import compute_poses
    clip = default_motion_bank().load_clip("bounce_in_place")
    pelvis = compute_poses(clip)["fk_joints"][:, 0, 2]
    travel = float(pelvis.max() - pelvis.min())
    assert travel > 0.07, travel


@pytest.mark.parametrize("motion_id", ["side_step", "step_touch"])
def test_mirroring_a_step_reverses_it_sideways_not_front_to_back(motion_id):
    """Mirroring reflects across the sagittal plane, so it must not touch forward travel.

    The lateral and sagittal axes are adjacent in the stored layout, and the bank was once
    authored against the wrong one. If `mirror` ever negated the sagittal component instead,
    mirroring a step would quietly turn it into its opposite rather than its reflection.
    """
    from agentlodge.dance.transition import mirror
    clip = default_motion_bank().load_clip(motion_id)
    flipped = mirror(clip)
    lateral = float(clip[-1, 0] - clip[0, 0])
    assert abs(lateral) > 0.2, lateral
    assert float(flipped[-1, 0] - flipped[0, 0]) == pytest.approx(-lateral, abs=1e-5)
    sagittal = float(clip[-1, 1] - clip[0, 1])
    assert float(flipped[-1, 1] - flipped[0, 1]) == pytest.approx(sagittal, abs=1e-5)


@needs_fk
@pytest.mark.parametrize("motion_id,sign", [("step_forward", 1.0), ("step_backward", -1.0)])
def test_a_named_step_travels_and_leans_the_way_its_name_says(motion_id, sign):
    """`root_displacement` is unsigned, so it cannot tell a forward step from a moonwalk.

    Both step clips satisfied their contract while travelling in opposite directions to the
    ones they are named after, because the recipes were authored against the wrong template
    axis. Facing has to be measured off the skeleton for the check to mean anything.
    """
    from server.fk import compute_poses
    bank = default_motion_bank()
    clip = bank.load_clip(motion_id)
    joints = compute_poses(clip)["fk_joints"]
    forward = _body_forward(joints)
    travel = float((clip[-1, :3] - clip[0, :3]) @ forward)
    assert sign * travel > 0.25, (motion_id, travel)
    lean = (joints[:, 9] - joints[:, 0]) @ forward - float((joints[0, 9] - joints[0, 0]) @ forward)
    assert sign * float(lean[np.argmax(np.abs(lean))]) > 0, (motion_id, lean.min(), lean.max())


@needs_fk
@pytest.mark.parametrize("motion_id,wrist", [("clap_single", 20), ("clap_repeat", 20),
                                             ("clap_overhead", 20), ("arm_punch", 21)])
def test_the_hands_do_their_work_in_front_of_the_body(motion_id, wrist):
    """The same axis error clapped and punched behind the back while passing every validator."""
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    joints = compute_poses(bank.load_clip(motion_id))["fk_joints"]
    reach = float((joints[spec.event_frame, wrist] - joints[spec.event_frame, 9])
                  @ _body_forward(joints))
    assert reach > 0.15, (motion_id, reach)


@needs_fk
@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_no_motion_strands_a_raised_hand_behind_the_dancer(motion_id):
    """A raised hand belongs in front of the body or beside it, never behind the back.

    This is the general form of the axis bug: an arm target authored with the wrong sign puts
    the hand behind the torso, which every manifest validator accepts because `joint_activity`
    only asks that the joint moved. Rather than enumerate the motions that happen to reach, this
    checks the whole bank, so a new recipe cannot reintroduce the mistake unnoticed. Hands hanging
    in the rest stance are exempt -- the bound only applies once a hand is actually lifted.

    The bound is placed against measured evidence rather than guessed. Rebuilding the pre-fix
    recipes (3122133) trips it on five motions -- arm_punch -0.557, rise_reach -0.308,
    point_side -0.296, clap_repeat -0.236, clap_single -0.220 -- while the current bank's worst
    case is celebrate_hands_up at -0.083, whose hands genuinely travel a little behind the chest
    plane on the way into an overhead V. That leaves room on both sides: loose enough not to
    outlaw a natural celebration, tight enough to catch a flipped sign.
    """
    from server.fk import compute_poses
    joints = compute_poses(default_motion_bank().load_clip(motion_id))["fk_joints"]
    forward = _body_forward(joints)
    chest = joints[:, 9]
    for wrist in (20, 21):
        lifted = joints[:, wrist, 2] > chest[:, 2]
        if not lifted.any():
            continue
        behind = ((joints[:, wrist] - chest) @ forward)[lifted].min()
        assert behind > -0.2, (motion_id, wrist, float(behind))


@pytest.mark.parametrize("motion_id", ["step_forward", "step_backward", "side_step",
                                       "turn_half", "jump_two_foot", "clap_single"])
@pytest.mark.parametrize("degrees", [0, 45, 90, 135, 180, 225, 270, 315])
@pytest.mark.parametrize("mode", ["replace", "insert"])
def test_an_edit_is_legal_no_matter_which_way_the_dancer_is_facing(motion_id, degrees, mode):
    """Rotating the whole song must not change whether an edit is allowed.

    It did. `insert` yaw-aligns the action to the dancer's heading at the splice point, but the
    `root_displacement` contract was measured along a fixed world axis, so once the dancer turned,
    the travel rotated onto the other axis and the check read almost zero. Every travelling motion
    -- forward, backward, sideways -- failed at 45, 90, 225 and 270 degrees while passing at 0 and
    180. A real song points the dancer wherever the music takes them, so this surfaced as an edit
    that worked or failed depending on nothing the user could see or control.
    """
    from agentlodge.editor.motion_bank import _yaw_rotate
    base = _base()
    if degrees:
        base = _yaw_rotate(base.copy(), np.deg2rad(degrees), base[0, :3].copy())
    window, report = default_motion_bank().apply(base, motion_id, mode=mode, anchor="center")
    assert window.shape[1] == base.shape[1]
    assert report["validation"]["ok"]


@pytest.mark.parametrize("degrees", [0, 45, 90, 135, 180, 225, 270, 315])
@pytest.mark.parametrize("mode", ["replace", "insert"])
@pytest.mark.parametrize("motion_id", _CLAP_HAND_CONTRACTS)
def test_clap_palm_alignment_survives_every_host_heading(motion_id, degrees, mode):
    from agentlodge.editor.motion_bank import _yaw_rotate

    base = _base()
    if degrees:
        base = _yaw_rotate(base.copy(), np.deg2rad(degrees), base[0, :3].copy())
    window, report = default_motion_bank().apply(
        base, motion_id, mode=mode, anchor="center",
    )
    palm_dot, finger_dot = _clap_hand_alignment(window, int(report["event_frame"]))
    assert palm_dot < -0.95, (motion_id, degrees, mode, "palms", palm_dot)
    assert finger_dot > 0.95, (motion_id, degrees, mode, "fingers", finger_dot)


@needs_fk
@pytest.mark.parametrize("motion_id,sign", [("step_forward", 1.0), ("step_backward", -1.0)])
def test_a_spliced_step_still_travels_the_way_it_is_named(motion_id, sign):
    """The bank clip going the right way is not the same as the spliced result going the right way.

    Splicing pins the window's end to wherever the song expects the dancer, so start-to-end travel
    is the base's, not the motion's, and says nothing about the action. What has to survive is the
    excursion: a forward step must reach forward of where it began before the tail hands the root
    back. Nothing checked this, and `root_displacement` is unsigned, so a reversed step would have
    validated cleanly -- exactly how the original axis bug stayed hidden.

    The base is pinned in place first. The mock base wanders 0.79 backward on its own, which is
    more than either step contributes, so against it a backward step is indistinguishable from the
    base's own drift -- it "passed" even when built from the reversed pre-fix recipe. Dancing on
    the spot makes the travel attributable to the motion: the fixed bank reaches +0.410 and -0.410,
    the pre-fix bank 0.000 in both directions.
    """
    from server.fk import compute_poses
    base = _base()
    base[:, :2] = base[0, :2]
    window, _ = default_motion_bank().apply(base, motion_id, mode="replace", anchor="center")
    forward = _body_forward(compute_poses(window)["fk_joints"])
    travel = (window[:, :3] - window[0, :3]) @ forward
    peak = float(travel.max() if sign > 0 else travel.min())
    assert sign * peak > 0.25, (motion_id, peak, travel.min(), travel.max())


@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
@pytest.mark.parametrize("mode", ["replace", "insert"])
def test_every_motion_applies_to_real_backbone_output(mode):
    """Every check above this point runs against a mock base. This one uses the real thing.

    The fixture is 24 seconds of genuine LODGE diffusion output, generated from a click track on
    the pod: it turns through 136 degrees, travels a metre in each direction, and carries the pose
    statistics of the diffusion model rather than a hand-written loop. `MockWindowGenerator` is a
    fine stand-in for splice arithmetic but it cannot tell you that the bank survives contact with
    what the product actually generates -- which, until this test, nothing did.

    It is deliberately not the guard for the world-axis bug. Real output mostly faces one way, so
    the sampled windows sit near yaw 0 and the old validator passes here; reverting the fix leaves
    this test green. `test_an_edit_is_legal_no_matter_which_way_the_dancer_is_facing` rotates the
    song explicitly and is what pins that. The two cover different things and both are needed.
    """
    from agentlodge.dance.format import to_editor139
    from agentlodge.editor.motion_bank import _root_yaw_series
    dance = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))
    bank, n = default_motion_bank(), 120
    yaw = _root_yaw_series(dance)
    spread = sorted(range(0, len(dance) - n, 8), key=lambda s: yaw[s])
    starts = [spread[int(i * (len(spread) - 1) / 5)] for i in range(6)]
    failures = []
    for start in starts:
        for spec in bank.specs:
            # turn_half in replace mode legitimately loses its turn to the closing rotation when
            # the song is itself turning hard; see "Closing the window" in docs/research.
            if spec.id == "turn_half" and mode == "replace":
                continue
            try:
                window, report = bank.apply(dance[start:start + n].copy(), spec.id,
                                            mode=mode, anchor="center")
                assert np.isfinite(window).all() and report["validation"]["ok"]
            except Exception as exc:
                failures.append(f"{spec.id} at yaw {np.rad2deg(yaw[start]):+.0f}: {exc}")
    assert not failures, failures


@needs_fk
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
@pytest.mark.parametrize("motion_id,sign", [("step_forward", 1.0), ("step_backward", -1.0)])
def test_a_step_spliced_into_real_output_travels_the_way_it_is_named(motion_id, sign):
    """Direction has to survive contact with a dancer who is not facing down a world axis."""
    from agentlodge.dance.format import to_editor139
    from agentlodge.editor.motion_bank import _root_yaw_series
    from server.fk import compute_poses
    dance = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))
    bank, n = default_motion_bank(), 120
    yaw = _root_yaw_series(dance)
    for start in (int(np.argmin(yaw)), int(np.argmax(yaw[:len(dance) - n]))):
        start = min(start, len(dance) - n - 1)
        base = dance[start:start + n].copy()
        base[:, :2] = base[0, :2]
        window, report = bank.apply(base, motion_id, mode="replace", anchor="center")
        # Measure the step against the frame it starts from, facing the way the dancer faces
        # there. The frames around it are the song's own dance and travel on their own account.
        a, b = report["action_range"]
        forward = _body_forward(compute_poses(window)["fk_joints"], a)
        travel = (window[a:b, :3] - window[a, :3]) @ forward
        peak = float(travel.max() if sign > 0 else travel.min())
        assert sign * peak > 0.25, (motion_id, np.rad2deg(yaw[start]), peak)


@needs_fk
def test_rise_reach_leads_with_one_arm_up_and_forward_as_its_recipe_claims():
    """The recipe comment promises a reach "up and slightly forward" with one arm leading.

    Before this was pinned the right hand actually sat 0.098 behind the chest, so the recipe
    documented an intent the clip did not honour -- the same class of silent mismatch as the
    original axis bug, just smaller. Asserting the relationship rather than the raw numbers
    keeps the target free to be retuned as long as it still reads as a leading diagonal rise.
    """
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve("rise_reach")
    joints = compute_poses(bank.load_clip("rise_reach"))["fk_joints"]
    forward = _body_forward(joints)
    chest = joints[spec.event_frame, 9]
    left, right = joints[spec.event_frame, 20], joints[spec.event_frame, 21]
    reach = [float((h - chest) @ forward) for h in (left, right)]
    lift = [float(h[2] - chest[2]) for h in (left, right)]
    assert min(reach) > 0.0, reach
    assert reach[0] > reach[1] + 0.05, reach
    assert min(lift) > 0.4, lift
    assert min(lift) > max(reach), (lift, reach)


def _spliced(bank, motion_id, mode, n=240):
    """Apply a named motion and put it back in the song the way the editor does."""
    from agentlodge.dance.transition import crossfade_edit
    song = np.concatenate([_base(n)] * 3, axis=0)
    a, b = n, 2 * n
    window, report = bank.apply(song[a:b], motion_id, mode=mode, anchor="center")
    return song, window, crossfade_edit(song, a, b, window, blend_frames=8)[a:b], report


@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_a_named_motion_still_reads_as_itself_once_it_is_spliced(motion_id):
    """Validating the clip in isolation proves nothing: the splice pins the window's edges
    back to the song, which used to erase a turn or a step from the record entirely."""
    bank = default_motion_bank()
    for mode in ("replace", "insert"):
        _song, _window, spliced, report = _spliced(bank, motion_id, mode)
        check = verify_applied_motion(spliced, report)
        assert check["ok"], (motion_id, mode, check["detail"])


@pytest.mark.parametrize("motion_id", ["turn_half", "turn_quarter", "step_forward",
                                       "step_backward", "side_step", "crouch_drop", "rise_reach"])
def test_a_travelling_motion_hands_the_root_back_before_the_window_ends(motion_id):
    """Whatever the dancer does inside the window, the next window starts where the song
    left off. Anything still owed at the last frame gets ripped back by the crossfade."""
    bank = default_motion_bank()
    for mode in ("replace", "insert"):
        song, window, _spliced_win, _report = _spliced(bank, motion_id, mode)
        gap = float(np.linalg.norm(window[-1, :3] - song[2 * 240 - 1, :3]))
        yaw = _root_yaw_series(np.stack([window[-1], song[2 * 240 - 1]]))
        assert gap < 0.02, (motion_id, mode, gap)
        assert abs(float((yaw[1] - yaw[0] + np.pi) % (2 * np.pi) - np.pi)) < 0.05, (motion_id, mode)


@pytest.mark.parametrize("motion_id", ["turn_half", "turn_quarter"])
def test_a_spliced_turn_does_not_whip_round_at_the_seam(motion_id):
    """A half turn left owing 180 degrees at the seam used to be unwound across the eight
    blend frames -- 55 rad/s, six times anything in the song."""
    bank = default_motion_bank()
    for mode in ("replace", "insert"):
        _song, _window, spliced, _report = _spliced(bank, motion_id, mode)
        rate = float(np.abs(np.diff(_root_yaw_series(spliced))).max() * 30)
        assert rate < 10.0, (motion_id, mode, rate)


@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_a_motion_uses_its_declared_musical_duration_however_wide_the_selection(motion_id):
    """Authored frames define motion shape; recommended beats define playback duration.

    This is the defect a user reported as "extremely slow and off putting from the actual dance".
    A 45-frame source clap declares one beat, so at this tempo it must occupy 16 frames whether
    the user selected three seconds or sixteen. Stretching it toward its authored frame count
    would recreate the slow-motion mismatch the beat contract exists to prevent.
    """
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    beats = np.arange(0, 480, 16)                    # a steady 112.5 BPM grid
    expected = int(round(spec.recommended_beats * 16))
    for n in (max(96, expected * 3), 480):           # selections far wider than the action
        base = np.concatenate([_base(240)] * 2, axis=0)[:n]
        _window, report = bank.apply(base, motion_id, mode="replace", anchor="center",
                                     beats=beats[beats < n])
        a, b = report["action_range"]
        assert b - a < n, (motion_id, n, report["action_range"])
        assert b - a == expected, (motion_id, n, b - a, expected)


@pytest.mark.parametrize("motion_id", ["clap_single", "wave", "chest_pop", "turn_quarter"])
def test_the_dance_outside_a_spliced_gesture_is_left_alone(motion_id):
    """Only the frames the action occupies are the bank's to touch.

    Retiming the whole window replaced the song's own choreography either side of the gesture with
    a slowed-down copy of the clip, which is why the edit read as a tempo change rather than as a
    gesture. The pose either side has to still be the pose the dancer was already in.
    """
    bank = default_motion_bank()
    n = 480
    base = np.concatenate([_base(240)] * 2, axis=0)[:n]
    window, report = bank.apply(base, motion_id, mode="replace", anchor="center",
                                beats=np.arange(0, n, 16))
    a, b = report["action_range"]
    assert a > 8 and b < n - 8, (motion_id, report["action_range"])
    # Body joints only. Translation and root orientation (channels 0:9) are legitimately
    # re-anchored so the window still starts and ends where the song expects; the pose the
    # dancer is holding either side of the gesture is what must survive untouched.
    # The hand-over either side fades the clip's pose offset out over at most one beat, so the
    # dance is only its own from there on. That bound is the point: without it the blend grows
    # with the selection and a one-second gesture quietly rewrites seconds of choreography.
    margin = _JOIN_MAX_FRAMES + 4
    for lo, hi in ((0, a - margin), (b + margin, n)):
        drift = float(np.abs(window[lo:hi, 9:135] - base[lo:hi, 9:135]).max())
        assert drift < 1e-3, (motion_id, (lo, hi), drift)


@pytest.mark.parametrize("motion_id", sorted(s.id for s in default_motion_bank().specs))
def test_a_spliced_motion_never_stops_the_dancer_dead(motion_id):
    """No frame of a spliced window may be a still copy of the one before it.

    The seams used to hand over by landing the incoming clip exactly on the outgoing pose, so a
    frame of time passed with nobody moving and the dance then rushed to catch up. Measured joint
    speed was precisely 0.000 at every seam. A held frame is far more visible than a fast one --
    it reads as a hitch or dropped frame -- and the splice now creates two seams inside every
    window, so this has to hold for each of them.
    """
    from server.fk import compute_poses

    bank = default_motion_bank()
    n = 240
    base = _base(n)
    window, report = bank.apply(base, motion_id, mode="replace", anchor="center",
                                beats=np.arange(0, n, 16))
    speed = np.linalg.norm(np.diff(compute_poses(window)["fk_joints"], axis=0), axis=-1).mean(axis=-1)
    a, b = report["action_range"]
    # A held frame shows up as a local dip, not as a low absolute speed: a calm gesture is
    # legitimately slower than the dance around it, so comparing against the window as a whole
    # would flag a quiet wave. What must not happen is the hand-over frame sitting still while
    # the frames on both sides of it are moving.
    for seam in (a, b):
        if seam < 2 or seam >= len(speed):
            continue
        held, before, after = speed[seam - 1], speed[seam - 2], speed[seam]
        floor = 0.25 * float(min(before, after))
        assert float(held) > floor, (motion_id, seam, float(held), float(before), float(after))


@pytest.mark.parametrize("motion_id", sorted(s.id for s in default_motion_bank().specs))
def test_a_spliced_motion_leaves_the_dance_room_to_come_back(motion_id):
    """An action may never run to the last frame of the window.

    The hand-over needs frames of real dance to fade into. When the action finished two frames
    from the edge the entire pose difference was closed inside those two frames, and the dancer
    was flung at nearly six times any speed in the song. Reserving a tail costs at most one beat
    of the gesture and is what keeps the return to the choreography watchable.
    """
    bank = default_motion_bank()
    for n in (96, 120, 168, 240):
        base = _base(240)[:n]
        _, report = bank.apply(base, motion_id, mode="replace", anchor="center",
                               beats=np.arange(0, n, 16))
        a, b = report["action_range"]
        assert n - b >= min(8, n // 8), (motion_id, n, report["action_range"])
        assert a >= 0 and b <= n, (motion_id, n, report["action_range"])


def test_asking_for_repetition_a_motion_cannot_do_still_edits_the_window():
    """The bank refuses repetition it cannot perform. Letting that sink the whole edit hands
    the user back an unchanged window, which is worse than the action happening once."""
    bank = default_motion_bank()
    spec = next(s for s in bank.specs if not s.repeatable)
    clip = _base(120)
    ctx = {"wbeats": None}
    out, note = AE._tool_motion_bank(clip, ctx, motion_id=spec.id, repeats=3)
    assert out.shape == clip.shape
    assert ctx["_motion_bank_report"]["repeats"] == 1
    assert ctx["_motion_bank_report"]["dropped"] == ["repetition"]
    assert "does not support repetition" in note


def test_a_motion_that_can_repeat_still_repeats():
    bank = default_motion_bank()
    spec = next(s for s in bank.specs if s.repeatable)
    ctx = {"wbeats": None}
    AE._tool_motion_bank(_base(120), ctx, motion_id=spec.id, repeats=2)
    assert ctx["_motion_bank_report"]["repeats"] == 2
    assert ctx["_motion_bank_report"]["dropped"] == []


def test_the_planner_tells_the_model_which_motions_repeat_or_mirror():
    """The model cannot avoid asking for repetition it will not get unless it is told."""
    import inspect
    src = inspect.getsource(AE._llm_plan)
    assert "repeats>1" in src and "mirror=true" in src
    bank = default_motion_bank()
    assert any(not s.repeatable for s in bank.specs)
