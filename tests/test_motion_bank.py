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
    _root_yaw_series,
    default_motion_bank,
    validate_semantics,
    verify_applied_motion,
)
from agentlodge.editor.window_edit import MockWindowGenerator


def _base(n=240):
    return MockWindowGenerator().generate("edge", 0, n, 4, energy=0.5, beats=None)


_TMPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "server", "data", "smplx_neu_J_1.npy")
# The posture checks below need forward kinematics, which needs the licence-gated
# SMPL-X joint template. It is fetched from the pod, so it is absent on a clean clone.
needs_fk = pytest.mark.skipif(not os.path.exists(_TMPL),
                              reason="SMPL-X joint template not present (fetched from the pod)")


def _body_forward(joints):
    """The direction the dancer faces, read off the feet rather than a world convention."""
    toe = (joints[0, 10] - joints[0, 7]) + (joints[0, 11] - joints[0, 8])
    forward = np.array([toe[0], toe[1], 0.0])
    return forward / np.linalg.norm(forward)


def test_manifest_has_twenty_valid_redistributable_motions():
    bank = default_motion_bank()
    assert len(bank.specs) == 20
    assert len({s.id for s in bank.specs}) == 20
    for spec in bank.specs:
        clip = bank.load_clip(spec)
        assert clip.shape == (spec.frames, 139)
        assert spec.source and spec.license and spec.attribution
        assert validate_semantics(clip, spec)["ok"]


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


def test_agent_replacement_preserves_every_frame_outside_selection():
    motion = _base(300)
    result = AE.run_agent_edit(
        motion, 60, 240, "add a clap motion here", beats=np.arange(0, 300, 15),
    )
    assert result.ok
    assert result.log[0]["tool"] == "motion_bank"
    assert result.log[0]["motion_bank"]["id"] == "clap_single"
    assert np.array_equal(result.motion[:60], motion[:60])
    assert np.array_equal(result.motion[240:], motion[240:])
    assert not np.array_equal(result.motion[60:240], motion[60:240])


def test_agent_insert_preserves_total_duration_and_uses_relational_policy():
    motion = _base(300)
    result = AE.run_agent_edit(
        motion, 60, 240, "insert a wave before the next move", beats=np.arange(0, 300, 15),
    )
    meta = result.log[0]["motion_bank"]
    assert result.ok and result.motion.shape == motion.shape
    assert meta["id"] == "wave" and meta["mode"] == "insert" and meta["anchor"] == "early"
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
    assert result.ok
    assert result.log[0]["status"] != "failed"
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
    """The event frame is what the editor snaps to a beat, so that is where the hands must meet."""
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    joints = compute_poses(bank.load_clip(motion_id))["fk_joints"]
    gap = float(np.linalg.norm(joints[spec.event_frame, 20] - joints[spec.event_frame, 21]))
    assert gap < 0.12, (motion_id, gap)


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
