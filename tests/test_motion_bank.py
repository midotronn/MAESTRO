"""Exhaustive contracts for the manifest-driven named-motion bank."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor import agent_edit as AE
from agentlodge.editor.motion_bank import MotionBank, default_motion_bank, validate_semantics
from agentlodge.editor.window_edit import MockWindowGenerator


def _base(n=240):
    return MockWindowGenerator().generate("edge", 0, n, 4, energy=0.5, beats=None)


_TMPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "server", "data", "smplx_neu_J_1.npy")
# The posture checks below need forward kinematics, which needs the licence-gated
# SMPL-X joint template. It is fetched from the pod, so it is absent on a clean clone.
needs_fk = pytest.mark.skipif(not os.path.exists(_TMPL),
                              reason="SMPL-X joint template not present (fetched from the pod)")


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
    # Point is intentionally non-repeatable, so the generic capability contract rejects it visibly.
    motion = _base(240)
    result = AE.run_agent_edit(motion, 30, 210, "add a big point left twice here")
    assert not result.ok
    assert result.log[0]["status"] == "failed"
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
