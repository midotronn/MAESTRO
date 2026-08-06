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
from agentlodge.editor.window_edit import MockWindowGenerator


def _base(n=240):
    return MockWindowGenerator().generate("edge", 0, n, 4, energy=0.5, beats=None)


_TMPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "server", "data", "smplx_neu_J_1.npy")
# The posture checks below need forward kinematics, which needs the licence-gated
# SMPL-X joint template. It is fetched from the pod, so it is absent on a clean clone.
needs_fk = pytest.mark.skipif(not os.path.exists(_TMPL),
                              reason="SMPL-X joint template not present (fetched from the pod)")


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
    """The event frame is what the editor snaps to a beat, so that is where the hands must meet.

    The bound is two-sided, and the lower half is the one that was missing. `_clap_arms` aimed
    both hands at x=0, so the solver drove the two wrists to the same point and the forearms
    passed through each other -- rendered, it read as folded, tangled arms rather than a clap.
    A one-sided "close enough" assertion cannot see that: the broken clips measured 0.016
    (clap_single), 0.012 (clap_repeat) and 0.046 (clap_overhead) at the event frame and passed
    cleanly. Palms in contact leave the wrists about a hand apart, so anything that collapses
    below 0.06 is bodies interpenetrating, not hands meeting.
    """
    from server.fk import compute_poses
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    joints = compute_poses(bank.load_clip(motion_id))["fk_joints"]
    gap = float(np.linalg.norm(joints[spec.event_frame, 20] - joints[spec.event_frame, 21]))
    assert 0.06 < gap < 0.12, (motion_id, gap)


@needs_fk
@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_no_motion_drives_the_two_hands_through_each_other(motion_id):
    """Whatever a motion is doing, the dancer has two separate hands for the whole of it.

    Checked over every frame of every clip rather than only the claps, because nothing about
    the defect was clap-specific: any two-armed recipe that shares one IK target reproduces it.
    The claps were the only offenders (0.0001 to 0.044); every other motion clears 0.35.
    """
    from server.fk import compute_poses
    joints = compute_poses(default_motion_bank().load_clip(motion_id))["fk_joints"]
    gap = float(np.linalg.norm(joints[:, 20] - joints[:, 21], axis=-1).min())
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
    spliced, _ = default_motion_bank().apply(base[a:b], motion_id,
                                             beats=np.arange(0, b - a, 16))
    out = base.copy()
    out[a:b] = spliced
    j = compute_poses(out)["fk_joints"]

    gap = np.linalg.norm(j[a:b, 20] - j[a:b, 21], axis=-1)
    k = int(np.argmin(gap)) + a
    assert 0.06 < float(gap.min()) < 0.16, ("hands do not meet after splicing", float(gap.min()))

    up = j[k, 12] - j[k, 0]
    up = up / np.linalg.norm(up)
    hands = 0.5 * (j[k, 20] + j[k, 21]) - 0.5 * (j[k, 16] + j[k, 17])
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


_LODGE_SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "lodge_sample_dance.npy")


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
    from agentlodge.editor.motion_bank import _root_yaw_series
    dance = np.load(_LODGE_SAMPLE).astype(np.float32)
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
    from server.fk import compute_poses
    from agentlodge.editor.motion_bank import _root_yaw_series
    dance = np.load(_LODGE_SAMPLE).astype(np.float32)
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
def test_a_motion_keeps_its_authored_speed_however_wide_the_selection(motion_id):
    """A gesture takes as long as it takes. It must not be stretched to fill the selection.

    This is the defect a user reported as "extremely slow and off putting from the actual dance".
    Duration used to come from the dragged window -- `_fit_event(raw, event, n, ...)` retimed the
    clip to the full width -- so a 45-frame clap dropped on a 4-second selection played at 0.375x
    and smeared one hand-meet across eight beats. Duration now comes from the music instead.
    """
    bank = default_motion_bank()
    authored = bank.load_clip(motion_id).shape[0]
    beats = np.arange(0, 480, 16)                    # a steady 112.5 BPM grid
    for n in (authored * 2, 480):                    # selections far wider than the action
        base = np.concatenate([_base(240)] * 2, axis=0)[:n]
        _window, report = bank.apply(base, motion_id, mode="replace", anchor="center",
                                     beats=beats[beats < n])
        a, b = report["action_range"]
        assert b - a < n, (motion_id, n, report["action_range"])
        assert 0.6 * authored <= b - a <= 1.6 * authored, (motion_id, n, b - a, authored)


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
