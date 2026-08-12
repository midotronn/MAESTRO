"""Exhaustive contracts for the manifest-driven named-motion bank."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor import agent_edit as AE
from agentlodge.editor.motion_bank import (
    MotionBank,
    _JOIN_MAX_FRAMES,
    _fit_event_frame,
    _root_yaw_series,
    default_motion_bank,
    validate_semantics,
    verify_applied_motion,
)
from agentlodge.editor.window_edit import MockWindowGenerator, window_metrics


def _base(n=240):
    return MockWindowGenerator().generate("edge", 0, n, 4, energy=0.5, beats=None)


def _declared_action_frames(spec, beat_period):
    """Apply the readable-duration floor without changing the motion's musical recommendation."""
    beats_wide = float(spec.recommended_beats)
    frames = int(round(beats_wide * beat_period))
    while frames < spec.minimum_frames:
        beats_wide += 1.0
        frames = int(round(beats_wide * beat_period))
    return frames


_TMPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "server", "data", "smplx_neu_J_1.npy")
_LODGE_SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "lodge_sample_dance.npy")
_CLAP_REGRESSION = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "clap_regression_case.npz",
)
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
    "side_step",
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


@needs_fk
def test_committed_clips_match_the_procedural_authoring_source():
    """A blind audit must not certify stale .npy files after the recipe changes."""
    from scripts.build_motion_bank import build_motion

    bank = default_motion_bank()
    for spec in bank.specs:
        expected = build_motion(spec.id, spec.frames)
        assert np.allclose(bank.load_clip(spec), expected, rtol=1e-6, atol=1e-6), spec.id
        for direction, _path in spec.direction_clips:
            expected = build_motion(spec.id, spec.frames, direction=direction)
            actual = bank.load_clip(spec, direction=direction)
            assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6), (
                spec.id,
                direction,
            )


def test_every_wrist_owning_motion_has_an_explicit_hand_contract():
    wrist_owners = {
        spec.id
        for spec in default_motion_bank().specs
        if (set(spec.absolute_joints) | set(spec.additive_joints)).intersection({20, 21})
    }
    assert wrist_owners == (
        set(_CLAP_HAND_CONTRACTS)
        | set(_FOREARM_HAND_CONTRACTS)
    )


def test_every_motion_has_a_blind_visual_acceptance_contract():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "motion_bank", "visual_contracts.json",
    )
    payload = json.loads(open(path, encoding="utf-8").read())
    bank_ids = {spec.id for spec in default_motion_bank().specs}

    assert set(payload["motions"]) == bank_ids
    assert "unlabeled" in payload["acceptance_rule"].lower()
    assert "cannot override" in payload["acceptance_rule"].lower()
    for motion_id, contract in payload["motions"].items():
        assert contract["recognizable_as"].strip(), motion_id
        assert len(contract["required_phases"]) >= 4, motion_id
        assert all(str(phase).strip() for phase in contract["required_phases"]), motion_id


def test_blind_review_locks_every_guess_before_loading_the_separate_answer_key():
    """Revealing one card must not expose answers while other takes remain unlocked."""
    from scripts.build_motion_bank_audit import _review_html

    page = _review_html(
        [
            {"take": "take_01", "control": "control_01"},
            {"take": "take_02", "control": "control_02"},
        ],
        "recognize the unlabeled edit",
        audit_id="paired-test",
        normalized_facing=True,
        motion_fingerprint_value="fingerprint-test",
    )

    assert '"take_01": {"id":' not in page
    assert page.count('fetch("answer_key.json"') == 1
    assert 'const takeIds = ["take_01", "take_02"]' in page
    assert "takeIds.every" in page
    assert 'class="lock"' in page
    assert 'class="direction-guess"' in page
    assert 'id="reveal-all" type="button" disabled' in page
    assert 'class="play-pair"' in page
    assert 'class="review-link"' in page
    assert "Play both from the start" in page
    assert "dancer-left appears on screen right" in page
    assert "Supported motion vocabulary" in page
    assert "normalSpeedPlayback" in page
    assert "comparisonOpenedAt" in page
    assert "comparisonAcknowledgment" in page
    assert "validComparisonAcknowledgment" in page
    assert "event.origin === window.location.origin" in page
    assert "event.source === comparisonWindows.get(take)" in page
    assert 'acknowledgment.auditId === auditId' in page
    assert 'acknowledgment.motionFingerprint === motionFingerprint' in page
    assert 'acknowledgment.takeId === take' in page
    assert "pause_count" in page
    assert "maxSyncDrift > 0.12" in page
    assert (
        "if (video.currentTime <= 0.01) {\n"
        "          return Promise.resolve();"
    ) in page
    assert 'class="visual-verdict"' in page
    assert 'class="visual-evidence"' in page
    assert "phase-reviewed" in page
    assert "allVisualReviewsComplete" in page
    assert "direction_recognized" in page
    assert "verified_phases" in page
    assert "motion_fingerprint" in page
    assert 'class="reveal"' not in page
    assert "localStorage" in page
    assert (
        "phase_sheets/take_01_review.html?"
        "audit_id=paired-test&amp;motion_fingerprint=fingerprint-test&amp;take_id=take_01"
    ) in page
    assert (
        "videos/control_01.mp4?"
        "audit_id=paired-test&amp;motion_fingerprint=fingerprint-test&amp;take_id=take_01"
    ) in page
    assert (
        "videos/take_01.mp4?"
        "audit_id=paired-test&amp;motion_fingerprint=fingerprint-test&amp;take_id=take_01"
    ) in page
    assert "Source choreography" in page
    reveal_handler = page.split('revealAll.addEventListener("click"', 1)[1]
    assert reveal_handler.index("if (!allGuessesLocked())") < reveal_handler.index(
        "await loadAnswers()"
    )
    click_handler = page.split('reviewLink.addEventListener("click"', 1)[1].split(
        'lock.addEventListener("click"', 1
    )[0]
    assert "window.open" in click_handler
    assert "comparisonOpenedAt" not in click_handler
    message_handler = page.split('window.addEventListener("message"', 1)[1].split(
        'document.querySelectorAll(".take").forEach', 1
    )[0]
    assert "validComparisonAcknowledgment(event, take)" in message_handler
    assert message_handler.index("validComparisonAcknowledgment") < message_handler.index(
        "comparisonOpenedAt"
    )


def test_parent_rejects_mismatched_or_stale_comparison_acknowledgments():
    from scripts.build_motion_bank_audit import _review_html

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute generated comparison-proof JavaScript")
    page = _review_html(
        [{"take": "take_01", "control": "control_01"}],
        "recognize the unlabeled edit",
        audit_id="ack-audit",
        normalized_facing=True,
        motion_fingerprint_value="ack-fingerprint",
    )
    function_source = page.split(
        "    function validComparisonAcknowledgment", 1
    )[1].split("    function takeReadyToLock", 1)[0]
    function_source = "function validComparisonAcknowledgment" + function_source
    harness = f"""
      const auditId = "ack-audit";
      const motionFingerprint = "ack-fingerprint";
      const expectedChild = {{}};
      const staleChild = {{}};
      const comparisonWindows = new Map([["take_01", expectedChild]]);
      const window = {{location: {{origin: "https://audit.example"}}}};
      {function_source}
      const good = {{
        origin: window.location.origin,
        source: expectedChild,
        data: {{
          type: "maestro-motion-audit-comparison-ready",
          auditId,
          motionFingerprint,
          takeId: "take_01",
        }},
      }};
      const checks = [
        validComparisonAcknowledgment(good, "take_01"),
        !validComparisonAcknowledgment({{...good, origin: "https://stale.example"}}, "take_01"),
        !validComparisonAcknowledgment({{...good, source: staleChild}}, "take_01"),
        !validComparisonAcknowledgment({{
          ...good, data: {{...good.data, auditId: "stale-audit"}},
        }}, "take_01"),
        !validComparisonAcknowledgment({{
          ...good, data: {{...good.data, motionFingerprint: "stale-fingerprint"}},
        }}, "take_01"),
        !validComparisonAcknowledgment({{
          ...good, data: {{...good.data, takeId: "take_02"}},
        }}, "take_01"),
      ];
      if (!checks.every(Boolean)) process.exit(1);
    """
    subprocess.run(
        [node, "-e", harness],
        check=True,
        capture_output=True,
        text=True,
    )


def test_rebuilt_audit_cannot_restore_revealed_state_from_the_previous_build():
    from pathlib import Path

    from scripts.build_motion_bank_audit import _new_audit_id, _review_html

    first_id = _new_audit_id(Path("current"), 84593)
    second_id = _new_audit_id(Path("current"), 84593)
    assert first_id.startswith("current-seed-84593-")
    assert second_id.startswith("current-seed-84593-")
    assert first_id != second_id

    second_page = _review_html(
        [{"take": "take_01", "control": "control_01"}],
        "recognize the unlabeled edit",
        audit_id=second_id,
        normalized_facing=True,
    )
    assert json.dumps(second_id) in second_page
    assert first_id not in second_page


def test_rebuilt_audit_removes_only_stale_generated_evidence(tmp_path):
    from scripts.build_motion_bank_audit import _prepare_output

    output = tmp_path / "audit"
    (output / "videos").mkdir(parents=True)
    (output / "videos" / "take_01.mp4").write_bytes(b"stale")
    (output / "take_01_front_frames").mkdir()
    (output / "take_01.npy").write_bytes(b"stale")
    (output / "render_receipt.json").write_text("{}", encoding="utf-8")
    (output / "reviewer-notes.txt").write_text("keep", encoding="utf-8")

    _prepare_output(output)

    assert not (output / "videos").exists()
    assert not (output / "take_01_front_frames").exists()
    assert not (output / "take_01.npy").exists()
    assert not (output / "render_receipt.json").exists()
    assert (output / "reviewer-notes.txt").read_text(encoding="utf-8") == "keep"


def test_semantic_audit_views_normalize_facing_without_changing_the_edit_delta():
    """Forward/back scoring needs a known profile direction on a faceless audit rig."""
    from agentlodge.editor.motion_bank import _root_yaw_series, _yaw_rotate
    from scripts.build_motion_bank_audit import _paired_views

    source = _base(36)
    source = _yaw_rotate(source, 0.91, source[0, :3].copy())
    edited = source.copy()
    edited[12:, 0] += np.linspace(0.0, 0.4, len(edited) - 12, dtype=np.float32)
    expected_heading = float(_root_yaw_series(edited[12:13])[0])

    front, side, control, control_side, heading = _paired_views(
        edited,
        source,
        12,
        normalize_facing=True,
    )

    front_yaw = float(_root_yaw_series(front[12:13])[0])
    side_yaw = float(_root_yaw_series(side[12:13])[0])
    quarter_turn = (side_yaw - front_yaw + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(front_yaw) < 1e-5
    assert quarter_turn == pytest.approx(np.pi / 2.0, abs=1e-5)
    assert heading == pytest.approx(expected_heading, abs=1e-5)
    assert np.allclose(front[12, :3], control[12, :3], atol=1e-6)
    assert np.allclose(side[12, :3], control_side[12, :3], atol=1e-6)
    before = np.linalg.norm(edited[:, :2] - source[:, :2], axis=1)
    after = np.linalg.norm(front[:, :2] - control[:, :2], axis=1)
    assert np.allclose(after, before, atol=1e-6)


def test_audit_phase_sheets_keep_action_boundaries_and_event_visible():
    from scripts.build_motion_audit_sheets import _selected_frames

    item = {
        "frames": 54,
        "action_range": [12, 42],
        "event_frame": 29,
    }
    selected = _selected_frames(item, stride=4, context=2)

    assert selected == sorted(set(selected))
    assert 10 == selected[0]
    assert 43 == selected[-1]
    assert 12 in selected
    assert 29 in selected
    assert 41 in selected


def test_audit_phase_sheets_build_synchronized_dual_view(tmp_path):
    from PIL import Image

    from scripts.build_motion_audit_sheets import build_sheets

    review = {
        "audit_id": "sheet-audit",
        "motion_fingerprint": "sheet-fingerprint",
        "takes": [{
            "take": "take_01",
            "control": "control_01",
            "frames": 4,
            "action_range": [1, 3],
            "event_frame": 2,
        }],
    }
    (tmp_path / "review.json").write_text(json.dumps(review), encoding="utf-8")
    colors = {
        ("take_01", "front"): (240, 20, 20),
        ("take_01", "side"): (20, 20, 240),
        ("control_01", "front"): (20, 240, 20),
        ("control_01", "side"): (240, 240, 20),
    }
    for pair_id in ("take_01", "control_01"):
        for view in ("front", "side"):
            frames = tmp_path / f"{pair_id}_{view}_frames"
            frames.mkdir()
            for frame in range(4):
                Image.new("RGB", (128, 128), colors[(pair_id, view)]).save(
                    frames / f"frame_{frame:05d}.png"
                )

    output = build_sheets(tmp_path, size=96)

    dual = Image.open(output / "take_01_dual.jpg")
    assert dual.width == 960
    assert dual.height == 282
    review_page = Image.open(output / "take_01_review_01.jpg")
    assert review_page.width == 1152
    assert review_page.height == 402
    edit_front = review_page.getpixel((48, 114))
    edit_side = review_page.getpixel((144, 114))
    edit_detail = review_page.getpixel((232, 114))
    source_front = review_page.getpixel((48, 234))
    source_side = review_page.getpixel((144, 234))
    source_detail = review_page.getpixel((232, 234))
    delta_front = review_page.getpixel((48, 354))
    delta_side = review_page.getpixel((144, 354))
    delta_detail = review_page.getpixel((232, 354))
    assert edit_front[0] > 180 and max(edit_front[1:]) < 80
    assert edit_side[2] > 180 and max(edit_side[:2]) < 80
    assert edit_detail[0] > 180 and max(edit_detail[1:]) < 80
    assert source_front[1] > 180 and max(source_front[0], source_front[2]) < 80
    assert min(source_side[:2]) > 180 and source_side[2] < 80
    assert source_detail[1] > 180 and max(source_detail[0], source_detail[2]) < 80
    assert min(delta_front[:2]) > 180 and delta_front[2] < 80
    assert min(delta_side) > 180
    assert min(delta_detail[:2]) > 180 and delta_detail[2] < 80
    review_index = (output / "take_01_review.html").read_text(encoding="utf-8")
    assert (
        "take_01_review_01.jpg?"
        "audit_id=sheet-audit&amp;motion_fingerprint=sheet-fingerprint&amp;take_id=take_01"
    ) in review_index
    assert "edit-minus-source" in review_index
    assert "maestro-motion-audit-comparison-ready" in review_index
    assert '"auditId": "sheet-audit"' in review_index
    assert '"motionFingerprint": "sheet-fingerprint"' in review_index
    assert '"takeId": "take_01"' in review_index
    assert "image.complete&&image.naturalWidth>0" in review_index
    assert "window.opener.postMessage" in review_index


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


def test_rise_reach_has_time_to_read_as_a_planted_level_change():
    """Compressing load, extension, and reach into one beat makes the rise look like a jump."""
    spec = default_motion_bank().resolve("rise_reach")
    assert spec.recommended_beats == 3.0
    assert spec.minimum_frames == 45


@pytest.mark.parametrize(
    "motion_id,minimum_frames",
    [
        ("clap_single", 24),
        ("clap_repeat", 45),
        ("clap_overhead", 30),
        ("jump_two_foot", 36),
        ("jump_arms_up", 36),
        ("chest_pop", 18),
        ("arm_punch", 24),
        ("step_touch", 45),
        ("body_roll", 45),
        ("rise_reach", 45),
    ],
)
def test_blind_failure_actions_keep_their_readable_duration_floor(
    motion_id,
    minimum_frames,
):
    assert default_motion_bank().resolve(motion_id).minimum_frames == minimum_frames


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
    beats = np.array([30.0, 45.0, 60.0, 75.0])
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
    beats = np.array([30.0, 45.0, 60.0, 75.0])
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
    from agentlodge.dance.transition import _matrix_to_axis_angle, _sixd_to_matrix
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
    assert 0.005 < gap < 0.12
    joints = [13, 14, 16, 17, 18, 19]
    base_rot = _sixd_to_matrix(base[a:b, 3:135].reshape(-1, 22, 6))
    neutral_rot = _sixd_to_matrix(neutral[a:b, 3:135].reshape(-1, 22, 6))
    boosted_rot = _sixd_to_matrix(boosted[a:b, 3:135].reshape(-1, 22, 6))
    neutral_delta = np.linalg.norm(_matrix_to_axis_angle(
        neutral_rot[:, joints] @ np.swapaxes(base_rot[:, joints], -1, -2)
    ), axis=-1)
    boosted_delta = np.linalg.norm(_matrix_to_axis_angle(
        boosted_rot[:, joints] @ np.swapaxes(base_rot[:, joints], -1, -2)
    ), axis=-1)
    assert float(np.sqrt(np.mean(boosted_delta ** 2))) > (
        1.005 * float(np.sqrt(np.mean(neutral_delta ** 2)))
    )
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
        ("arm_punch", (13, 14, 16, 17, 18, 19, 20), ()),
    ],
)
def test_mirroring_moves_composition_ownership_to_the_other_side(
    motion_id, mirrored_chain, original_only,
):
    """Mirroring the clip without mirroring its ownership silently discarded the left action."""
    base = _base(180)
    out, _ = default_motion_bank().apply(
        base, motion_id, mirror=True,
        beats=np.arange(0, 180, 15), anchor="beat",
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


@needs_fk
@pytest.mark.parametrize("motion_id", _CLAP_HAND_CONTRACTS)
def test_every_declared_clap_contact_survives_default_splicing(motion_id):
    """A repeated clap must keep every contact, not merely the first beat-anchored one."""
    from server.fk import compute_poses

    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    out, report = bank.apply(_base(180), motion_id, beats=np.arange(0, 180, 15))
    joints = compute_poses(out)["fk_joints"]
    gaps = np.linalg.norm(joints[:, 20] - joints[:, 21], axis=-1)
    start, end = report["action_range"]
    local_event = int(report["event_frame"]) - start

    runs: list[list[int]] = []
    for frame in spec.intensity_lock_frames:
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)

    assert runs
    for run in runs:
        mapped = {
            start + _fit_event_frame(
                frame,
                spec.frames,
                spec.event_frame,
                end - start,
                local_event,
            )
            for frame in run
        }
        contact = min(mapped, key=lambda frame: float(gaps[frame]))
        assert float(gaps[contact]) < 0.08, (motion_id, run, contact, float(gaps[contact]))
        palm_dot, finger_dot = _clap_hand_alignment(out, contact)
        assert palm_dot < -0.95, (motion_id, run, "palms", palm_dot)
        assert finger_dot > 0.95, (motion_id, run, "fingers", finger_dot)


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
    out, report = bank.apply(
        base,
        motion_id,
        beats=np.arange(0, 180, 15),
        anchor="beat",
        direction=spec.canonical_direction,
    )
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
    assert report["action_frames"] == _declared_action_frames(spec, 15), motion_id


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
        direction="forward",
    )

    assert np.array_equal(out[:, :3], base[:, :3])
    assert np.array_equal(out[:, 135:139], base[:, 135:139])
    spec = default_motion_bank().resolve("clap_single")
    owned = set(spec.absolute_joints) | set(spec.additive_joints)
    for joint in set(range(22)) - owned:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        assert np.array_equal(out[:, channels], base[:, channels]), joint

    before = compute_poses(base)["fk_joints"]
    after = compute_poses(out)["fk_joints"]
    assert np.allclose(after[:, (1, 2, 4, 5, 7, 8, 10, 11)],
                       before[:, (1, 2, 4, 5, 7, 8, 10, 11)], atol=1e-6)

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
    assert after_metrics["jerk"] < 2.4 * before_metrics["jerk"]


@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
@pytest.mark.parametrize(
    "motion_id,max_jerk_ratio",
    [
        ("clap_overhead", 3.0),
        ("jump_arms_up", 2.6),
        ("point_side", 1.6),
        ("celebrate_hands_up", 2.0),
        ("rise_reach", 3.6),
    ],
)
def test_beat_compressed_pose_accents_take_the_shortest_path(
    motion_id, max_jerk_ratio,
):
    """Retiming a long ready-stance clip into one beat must not preserve its whole wind-up.

    The semantic pose belongs on the beat, but making the host visit every irrelevant authored
    in-between pose first creates the same acceleration spike that made the original clap feel
    pasted onto the dance. Event-pose joints travel directly from the host to the accent and back;
    joints carrying meaningful internal motion (the jump, rise, punch, or torso response) retain
    their clip trajectory and have phase-specific contracts instead.
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
        motion, 0, 40, "add a clap on the beat here", beats=beats, max_refine=0,
    )
    report = next(step["motion_bank"] for step in result.log if "motion_bank" in step)
    event = int(report["event_frame"])
    joints = compute_poses(result.motion)["fk_joints"]
    gap = float(np.linalg.norm(joints[event, 20] - joints[event, 21]))

    assert report["action_range"] == [4, 34]
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
        motion, 2, 29, "add a clap on the beat here", beats=beats, max_refine=0,
    )
    fitted_report = next(
        step["motion_bank"] for step in fitted.log if "motion_bank" in step
    )
    assert fitted_report["action_frames"] == 24
    assert fitted_report["event_frame"] == 13
    assert fitted_report["beat_error_frames"] == pytest.approx(0.0)
    assert fitted.ok

    impossible = AE.run_agent_edit(
        motion, 5, 32, "add a clap on the beat here", beats=beats, max_refine=0,
    )
    impossible_report = next(
        step["motion_bank"] for step in impossible.log if "motion_bank" in step
    )
    assert impossible_report["beat_error_frames"] > 0.5
    assert not impossible.ok

    beatless = AE.run_agent_edit(
        motion, 5, 32, "add a clap here", beats=np.array([]), max_refine=0,
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


def test_agent_rejects_named_motion_insert_until_insert_specific_audit_exists():
    motion = _base(300)
    with pytest.raises(ValueError, match="insert mode is unavailable"):
        AE.run_agent_edit(
            motion,
            60,
            240,
            "insert a wave before the next move",
            beats=np.arange(0, 300, 15),
        )


def test_direction_repeat_and_intensity_variants_are_data_driven():
    plan = AE.plan_edit("add a big point left twice here", {}, 0.0, 6.0)
    step = plan.steps[0]
    assert step.tool == "motion_bank"
    assert step.params == {
        "motion_id": "point_side", "mode": "replace", "anchor": "beat",
        "mirror": False, "direction": "left", "intensity": 0.8, "repeats": 2,
    }
    # Point is intentionally non-repeatable. The request still lands as a single point with the
    # dropped repetition stated, rather than failing and leaving the window untouched.
    motion = _base(240)
    result = AE.run_agent_edit(motion, 30, 210, "add a big point left twice here")
    assert result.log[0]["status"] != "failed"
    assert result.trace["final"]["checks"][0]["met"]
    assert "does not support repetition" in result.log[0]["note"]


def test_directional_named_motions_publish_their_supported_choices():
    bank = default_motion_bank()
    clap = bank.resolve("clap_single")
    assert clap.directions == ("forward", "left", "right")
    assert clap.canonical_direction == "forward"
    point = bank.resolve("point_side")
    assert point.directions == ("left", "right")
    assert point.canonical_direction == "right"
    public = {motion["id"]: motion for motion in bank.list_public()}
    assert public["clap_single"]["default_direction"] == "auto"
    assert public["clap_single"]["directions"] == ["forward", "left", "right"]


def test_auto_direction_follows_lateral_travel_and_turns():
    from agentlodge.dance.transition import _matrix_to_sixd

    bank = default_motion_bank()
    base = _base(150)
    identity = _matrix_to_sixd(np.eye(3, dtype=np.float32)).reshape(6)
    base[:, 3:9] = identity
    base[:, :3] = 0.0

    left = base.copy()
    left[:, 0] = np.linspace(0.0, 0.4, len(left), dtype=np.float32)
    _out, report = bank.apply(left, "clap_single", direction="auto", anchor="center")
    assert report["direction"] == "left"
    assert report["direction_source"] == "dance_flow"

    right = base.copy()
    right[:, 0] = np.linspace(0.0, -0.4, len(right), dtype=np.float32)
    _out, report = bank.apply(right, "clap_single", direction="auto", anchor="center")
    assert report["direction"] == "right"

    still = base.copy()
    _out, report = bank.apply(still, "clap_single", direction="auto", anchor="center")
    assert report["direction"] == "forward"

    turning = base.copy()
    yaw = np.linspace(0.0, 0.5, len(turning), dtype=np.float32)
    turning[:, 3:9] = _matrix_to_sixd(
        np.stack([
            [[np.cos(v), -np.sin(v), 0.0],
             [np.sin(v), np.cos(v), 0.0],
             [0.0, 0.0, 1.0]]
            for v in yaw
        ], axis=0).astype(np.float32)
    )
    _out, report = bank.apply(turning, "clap_single", direction="auto", anchor="center")
    assert report["direction"] == "left"


@needs_fk
@pytest.mark.parametrize("motion_id", ["clap_single", "clap_repeat", "clap_overhead"])
def test_directional_claps_move_contact_to_the_requested_side(motion_id):
    from server.fk import compute_poses

    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    centers = {}
    for direction in ("left", "right"):
        clip = bank.load_clip(spec, direction=direction)
        joints = compute_poses(clip)["fk_joints"]
        hands = 0.5 * (joints[spec.event_frame, 20] + joints[spec.event_frame, 21])
        centers[direction] = hands - joints[spec.event_frame, 0]
        palm_dot, finger_dot = _clap_hand_alignment(clip, spec.event_frame)
        assert palm_dot < -0.98, (motion_id, direction, palm_dot)
        assert finger_dot > 0.98, (motion_id, direction, finger_dot)
    assert centers["left"][0] > 0.14, (motion_id, centers["left"])
    assert centers["right"][0] < -0.14, (motion_id, centers["right"])


@needs_fk
@pytest.mark.parametrize("motion_id", ["clap_single", "clap_repeat", "clap_overhead"])
def test_clap_regression_keeps_dancing_and_makes_visible_contact(motion_id):
    """Pin the exact live host that exposed the camera-facing, detached-hand clap."""
    from server.fk import compute_poses

    case = np.load(_CLAP_REGRESSION)
    host = case["host"].astype(np.float32)
    bank = default_motion_bank()
    outputs = {}

    for direction in ("forward", "left", "right"):
        out, report = bank.apply(
            host,
            motion_id,
            beats=case["beats"],
            beat_strengths=case["beat_strengths"],
            direction=direction,
        )
        event = int(report["event_frame"])
        start, end = report["action_range"]
        before_global = _global_joint_rotations(host)
        after_global = _global_joint_rotations(out)
        chest_delta = np.arctan2(
            np.sin(
                np.arctan2(after_global[:, 9, 1, 0], after_global[:, 9, 0, 0])
                - np.arctan2(before_global[:, 9, 1, 0], before_global[:, 9, 0, 0])
            ),
            np.cos(
                np.arctan2(after_global[:, 9, 1, 0], after_global[:, 9, 0, 0])
                - np.arctan2(before_global[:, 9, 1, 0], before_global[:, 9, 0, 0])
            ),
        )
        joints = compute_poses(out)["fk_joints"]
        gap = float(np.linalg.norm(joints[event, 20] - joints[event, 21]))
        palm_dot, finger_dot = _clap_hand_alignment(out, event)
        finger_up = 0.5 * (
            float((after_global[event, 20] @ np.array([1.0, 0.0, 0.0]))[2])
            + float((after_global[event, 21] @ np.array([-1.0, 0.0, 0.0]))[2])
        )

        assert gap < 0.01, (motion_id, direction, gap)
        assert palm_dot < -0.98, (motion_id, direction, palm_dot)
        assert finger_dot > 0.98, (motion_id, direction, finger_dot)
        assert 0.20 < finger_up < 0.75, (motion_id, direction, finger_up)
        expected_turn = float(report["counterflow_turn_degrees"])
        observed_turn = float(np.rad2deg(chest_delta[event]))
        peak_turn = float(np.rad2deg(np.max(np.abs(chest_delta[start:end]))))
        if expected_turn:
            assert observed_turn == pytest.approx(expected_turn, abs=3.0)
            assert peak_turn < abs(expected_turn) + 3.0
        else:
            assert peak_turn < 3.0

        owned = (
            set(bank.resolve(motion_id).absolute_joints)
            | set(report["counterflow_turn_joints"])
        )
        for joint in set(range(22)) - owned:
            channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
            assert np.array_equal(out[:, channels], host[:, channels]), (
                motion_id,
                direction,
                joint,
            )
        assert np.array_equal(out[:, :3], host[:, :3])
        assert np.array_equal(out[:, 135:139], host[:, 135:139])

        root_yaw = _root_yaw_series(out[event:event + 1])[0]
        local_left = np.array([np.cos(root_yaw), np.sin(root_yaw)])
        hand_center = 0.5 * (joints[event, 20] + joints[event, 21]) - joints[event, 0]
        outputs[direction] = float(hand_center[:2] @ local_left)

    assert outputs["left"] > outputs["forward"] + 0.14, (motion_id, outputs)
    assert outputs["right"] < outputs["forward"] - 0.14, (motion_id, outputs)


@needs_fk
def test_automatic_clap_direction_reproduces_the_live_left_flow_without_turning_the_torso():
    case = np.load(_CLAP_REGRESSION)
    host = case["host"].astype(np.float32)
    out, report = default_motion_bank().apply(
        host,
        "clap_single",
        beats=case["beats"],
        beat_strengths=case["beat_strengths"],
        direction="auto",
    )

    assert report["direction"] == "left"
    assert report["direction_source"] == "dance_flow"
    before_global = _global_joint_rotations(host)
    after_global = _global_joint_rotations(out)
    event = int(report["event_frame"])
    before_yaw = np.arctan2(before_global[event, 9, 1, 0], before_global[event, 9, 0, 0])
    after_yaw = np.arctan2(after_global[event, 9, 1, 0], after_global[event, 9, 0, 0])
    delta = np.arctan2(np.sin(after_yaw - before_yaw), np.cos(after_yaw - before_yaw))
    assert abs(np.rad2deg(delta)) < 3.0
    assert report["natural_direction"] == "left"
    assert report["counterflow_turn_degrees"] == 0.0

    explicit_left, left_report = default_motion_bank().apply(
        host,
        "clap_single",
        beats=case["beats"],
        beat_strengths=case["beat_strengths"],
        direction="left",
    )
    assert left_report["natural_direction"] == "left"
    assert left_report["counterflow_turn_degrees"] == 0.0

    explicit_right, right_report = default_motion_bank().apply(
        host,
        "clap_single",
        beats=case["beats"],
        beat_strengths=case["beat_strengths"],
        direction="right",
    )
    assert right_report["natural_direction"] == "left"
    assert right_report["counterflow_turn_degrees"] == -14.0
    right_global = _global_joint_rotations(explicit_right)
    right_yaw = np.arctan2(
        right_global[event, 9, 1, 0],
        right_global[event, 9, 0, 0],
    )
    right_delta = np.arctan2(
        np.sin(right_yaw - before_yaw),
        np.cos(right_yaw - before_yaw),
    )
    assert np.rad2deg(right_delta) == pytest.approx(-14.0, abs=3.0)
    assert np.array_equal(explicit_left[:, :3], host[:, :3])
    assert np.array_equal(explicit_right[:, :3], host[:, :3])
    assert np.array_equal(explicit_left[:, 135:139], host[:, 135:139])
    assert np.array_equal(explicit_right[:, 135:139], host[:, 135:139])


@needs_fk
def test_visual_audit_machine_gate_rejects_turning_and_detached_claps():
    from agentlodge.dance.transition import _matrix_to_sixd, _sixd_to_matrix
    from scripts.build_motion_bank_audit import _machine_checks

    case = np.load(_CLAP_REGRESSION)
    host = case["host"].astype(np.float32)
    bank = default_motion_bank()
    out, report = bank.apply(
        host,
        "clap_single",
        beats=case["beats"],
        beat_strengths=case["beat_strengths"],
        direction="auto",
    )
    spec = bank.resolve("clap_single")
    checks, status = _machine_checks(host, out, spec, report)
    assert status == "pass"
    assert all(check["passed"] for check in checks)

    round_tripped = out.copy()
    round_tripped[0, 3] += np.float32(1e-7)
    checks, status = _machine_checks(host, round_tripped, spec, report)
    assert status == "pass"

    unowned_drift = out.copy()
    unowned_drift[0, 3] += np.float32(1e-4)
    checks, status = _machine_checks(host, unowned_drift, spec, report)
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "declared_channel_ownership"
    )["passed"]

    start, end = report["action_range"]
    turning = out.copy()
    spine = _sixd_to_matrix(turning[start:end, 3 + 6 * 9:3 + 6 * 10])
    angle = np.deg2rad(40.0)
    yaw = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    turning[start:end, 3 + 6 * 9:3 + 6 * 10] = _matrix_to_sixd(
        spine @ yaw
    )
    checks, status = _machine_checks(host, turning, spec, report)
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "host_facing_continuity"
    )["passed"]

    detached = out.copy()
    for joint in (14, 17, 19, 21):
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        detached[start:end, channels] = host[start:end, channels]
    checks, status = _machine_checks(host, detached, spec, report)
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "visible_palm_contact"
    )["passed"]


@needs_fk
def test_visual_audit_machine_gate_checks_every_repeated_clap_contact():
    from scripts.build_motion_bank_audit import _machine_checks

    case = np.load(_CLAP_REGRESSION)
    host = case["host"].astype(np.float32)
    bank = default_motion_bank()
    out, report = bank.apply(
        host,
        "clap_repeat",
        beats=case["beats"],
        beat_strengths=case["beat_strengths"],
        direction="forward",
    )
    spec = bank.resolve("clap_repeat")
    first_contact = int(report["event_frame"])
    broken = out.copy()
    for joint in spec.absolute_joints:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        broken[first_contact + 4:, channels] = host[first_contact + 4:, channels]
    checks, status = _machine_checks(host, broken, spec, report)
    assert status == "fail"
    contact = next(check for check in checks if check["name"] == "visible_palm_contact")
    assert not contact["passed"]
    assert contact["detail"].count(",") >= 2


@needs_fk
def test_visual_audit_independently_rejects_a_wrong_automatic_direction_report():
    from scripts.build_motion_bank_audit import _machine_checks

    case = np.load(_CLAP_REGRESSION)
    host = case["host"].astype(np.float32)
    bank = default_motion_bank()
    out, report = bank.apply(
        host,
        "clap_single",
        beats=case["beats"],
        beat_strengths=case["beat_strengths"],
        direction="auto",
    )
    assert report["direction"] == "left"
    wrong = dict(report, direction="right")
    checks, status = _machine_checks(host, out, bank.resolve("clap_single"), wrong)
    assert status == "fail"
    direction = next(check for check in checks if check["name"] == "direction_resolution")
    assert not direction["passed"]
    assert "independently expected 'left'" in direction["detail"]


@needs_fk
def test_visual_audit_derives_counterflow_ownership_instead_of_trusting_the_report():
    from scripts.build_motion_bank_audit import _machine_checks

    case = np.load(_CLAP_REGRESSION)
    host = case["host"].astype(np.float32)
    bank = default_motion_bank()
    spec = bank.resolve("clap_single")

    automatic, auto_report = bank.apply(
        host,
        "clap_single",
        beats=case["beats"],
        beat_strengths=case["beat_strengths"],
        direction="auto",
    )
    forged = dict(
        auto_report,
        counterflow_turn_joints=[3],
        counterflow_turn_degrees=0.0,
    )
    changed = automatic.copy()
    changed[int(auto_report["event_frame"]), 3 + 6 * 3] += np.float32(1e-4)
    checks, status = _machine_checks(host, changed, spec, forged)
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "counterflow_declaration"
    )["passed"]
    assert not next(
        check for check in checks if check["name"] == "declared_channel_ownership"
    )["passed"]

    counterflow, counterflow_report = bank.apply(
        host,
        "clap_single",
        beats=case["beats"],
        beat_strengths=case["beat_strengths"],
        direction="right",
    )
    checks, status = _machine_checks(host, counterflow, spec, counterflow_report)
    assert status == "pass"
    forged = dict(
        counterflow_report,
        counterflow_turn_joints=[],
        counterflow_turn_degrees=0.0,
    )
    checks, status = _machine_checks(host, counterflow, spec, forged)
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "counterflow_declaration"
    )["passed"]
    assert next(
        check for check in checks if check["name"] == "host_facing_continuity"
    )["passed"]


@needs_fk
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_visual_audit_rejects_every_protocol_eight_failure_signature():
    from agentlodge.dance.format import to_editor139
    from scripts.build_motion_bank_audit import _machine_checks

    host = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:180]
    beats = np.arange(0, 180, 15)
    bank = default_motion_bank()

    side, side_report = bank.apply(host, "side_step", beats=beats, direction="left")
    checks, status = _machine_checks(
        host,
        side,
        bank.resolve("side_step"),
        dict(side_report, direction="right"),
    )
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "side_step_direction_signature"
    )["passed"]

    jump, jump_report = bank.apply(host, "jump_arms_up", beats=beats)
    start, end = jump_report["action_range"]
    collapsed_jump = jump.copy()
    for joint in (13, 14, 16, 17, 18, 19, 20, 21):
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        collapsed_jump[start:end, channels] = host[start:end, channels]
    checks, status = _machine_checks(
        host,
        collapsed_jump,
        bank.resolve("jump_arms_up"),
        jump_report,
    )
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "jump_arm_signature"
    )["passed"]

    plain_jump, plain_jump_report = bank.apply(host, "jump_two_foot", beats=beats)
    start, end = plain_jump_report["action_range"]
    event = int(plain_jump_report["event_frame"])
    compressed_jump = plain_jump.copy()
    compressed_jump[start:end, 135:139] = 1.0
    compressed_jump[event - 1:event + 2, 135:139] = 0.0
    checks, status = _machine_checks(
        host,
        compressed_jump,
        bank.resolve("jump_two_foot"),
        plain_jump_report,
    )
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "readable_jump_phases"
    )["passed"]

    chest, chest_report = bank.apply(host, "chest_pop", beats=beats)
    start, end = chest_report["action_range"]
    flat_chest = chest.copy()
    chest_spec = bank.resolve("chest_pop")
    for joint in set(chest_spec.absolute_joints) | set(chest_spec.additive_joints):
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        flat_chest[start:end, channels] = host[start:end, channels]
    checks, status = _machine_checks(host, flat_chest, chest_spec, chest_report)
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "chest_pop_isolation"
    )["passed"]

    rise, rise_report = bank.apply(host, "rise_reach", beats=beats)
    start, end = rise_report["action_range"]
    airborne_rise = rise.copy()
    airborne_rise[start:end, 135:139] = 0.0
    checks, status = _machine_checks(
        host,
        airborne_rise,
        bank.resolve("rise_reach"),
        rise_report,
    )
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "planted_rise_reach_signature"
    )["passed"]

    overhead, overhead_report = bank.apply(
        host,
        "clap_overhead",
        beats=beats,
        direction="right",
    )
    checks, status = _machine_checks(
        host,
        overhead,
        bank.resolve("clap_overhead"),
        overhead_report,
    )
    assert status == "pass"
    assert next(
        check for check in checks
        if check["name"] == "readable_clap_contact_timing"
    )["passed"]
    start, end = overhead_report["action_range"]
    event = int(overhead_report["event_frame"])
    held_clap = overhead.copy()
    for frame in (max(start, event - 6), min(end - 1, event + 6)):
        for joint in (13, 14, 16, 17, 18, 19, 20, 21):
            channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
            held_clap[frame, channels] = overhead[event, channels]
    checks, status = _machine_checks(
        host,
        held_clap,
        bank.resolve("clap_overhead"),
        overhead_report,
    )
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "overhead_clap_approach_recoil"
    )["passed"]

    repeat, repeat_report = bank.apply(
        host,
        "clap_repeat",
        beats=beats,
        direction="left",
    )
    start, end = repeat_report["action_range"]
    merged_repeat = repeat.copy()
    cutoff = start + (end - start) // 3
    for joint in (13, 14, 16, 17, 18, 19, 20, 21):
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        merged_repeat[cutoff:end, channels] = host[cutoff:end, channels]
    checks, status = _machine_checks(
        host,
        merged_repeat,
        bank.resolve("clap_repeat"),
        repeat_report,
    )
    assert status == "fail"
    assert not next(
        check for check in checks
        if check["name"] == "readable_clap_contact_timing"
    )["passed"]

    punch, punch_report = bank.apply(
        host,
        "arm_punch",
        beats=beats,
        direction="right",
    )
    start, end = punch_report["action_range"]
    collapsed_punch = punch.copy()
    for joint in (13, 14, 16, 17, 18, 19, 21):
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        collapsed_punch[start:end, channels] = host[start:end, channels]
    checks, status = _machine_checks(
        host,
        collapsed_punch,
        bank.resolve("arm_punch"),
        punch_report,
    )
    assert status == "fail"
    assert not next(
        check for check in checks
        if check["name"] == "guard_strike_recoil_signature"
    )["passed"]

    touch, touch_report = bank.apply(
        host,
        "step_touch",
        beats=beats,
        direction="left",
    )
    event = int(touch_report["event_frame"])
    obscured_touch = touch.copy()
    for joint in (1, 2, 4, 5, 7, 8, 10, 11):
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        obscured_touch[event, channels] = host[event, channels]
    checks, status = _machine_checks(
        host,
        obscured_touch,
        bank.resolve("step_touch"),
        touch_report,
    )
    assert status == "fail"
    assert not next(
        check for check in checks
        if check["name"] == "step_touch_phase_signature"
    )["passed"]

    roll, roll_report = bank.apply(host, "body_roll", beats=beats)
    start, end = roll_report["action_range"]
    rigid_roll = roll.copy()
    for joint in (6, 9):
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        rigid_roll[start:end, channels] = host[start:end, channels]
    checks, status = _machine_checks(
        host,
        rigid_roll,
        bank.resolve("body_roll"),
        roll_report,
    )
    assert status == "fail"
    assert not next(
        check for check in checks
        if check["name"] == "sequential_body_roll_signature"
    )["passed"]


def test_visual_audit_direction_matrix_rejects_indistinct_or_reversed_claps():
    from scripts.build_motion_bank_audit import _append_clap_direction_matrix_checks

    passing = {
        "clap_single": {
            "left": {"lateral": 0.18, "checks": []},
            "forward": {"lateral": 0.00, "checks": []},
            "right": {"lateral": -0.18, "checks": []},
        }
    }
    _append_clap_direction_matrix_checks(passing)
    assert all(
        record["checks"][-1]["passed"]
        for record in passing["clap_single"].values()
    )

    indistinct = {
        "clap_single": {
            "left": {"lateral": 0.11, "checks": []},
            "forward": {"lateral": 0.00, "checks": []},
            "right": {"lateral": -0.11, "checks": []},
        }
    }
    _append_clap_direction_matrix_checks(indistinct)
    assert not any(
        record["checks"][-1]["passed"]
        for record in indistinct["clap_single"].values()
    )

    reversed_directions = {
        "clap_single": {
            "left": {"lateral": -0.11, "checks": []},
            "forward": {"lateral": 0.00, "checks": []},
            "right": {"lateral": 0.11, "checks": []},
        }
    }
    _append_clap_direction_matrix_checks(reversed_directions)
    assert not any(
        record["checks"][-1]["passed"]
        for record in reversed_directions["clap_single"].values()
    )


@pytest.mark.parametrize("motion_id", ["jump_two_foot", "jump_arms_up"])
def test_jumps_keep_a_readable_real_dance_duration(motion_id):
    bank = default_motion_bank()
    for period in (10, 12, 15, 18):
        _out, report = bank.apply(
            _base(120),
            motion_id,
            beats=np.arange(0, 120, period),
            anchor="beat",
        )
        assert report["action_frames"] >= 36, (motion_id, period, report)
    with pytest.raises(ValueError, match="natural speed"):
        bank.apply(
            _base(26),
            motion_id,
            beats=np.arange(0, 26, 10),
            anchor="beat",
        )


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


@needs_fk
def test_side_step_and_step_touch_have_distinct_named_phases():
    """The recipes were once label-swapped: the side step closed and the touch kept travelling."""
    from server.fk import compute_poses

    bank = default_motion_bank()
    side_spec = bank.resolve("side_step")
    touch_spec = bank.resolve("step_touch")
    side = bank.load_clip(side_spec)
    touch = bank.load_clip(touch_spec)
    side_joints = compute_poses(side)["fk_joints"]
    touch_joints = compute_poses(touch)["fk_joints"]

    side_gap = float(np.linalg.norm(
        side_joints[side_spec.event_frame, 7, :2]
        - side_joints[side_spec.event_frame, 8, :2]
    ))
    touch_gap = float(np.linalg.norm(
        touch_joints[touch_spec.event_frame, 7, :2]
        - touch_joints[touch_spec.event_frame, 8, :2]
    ))
    assert side_gap > 0.38, side_gap
    assert touch_gap < 0.12, touch_gap
    assert side_gap > touch_gap + 0.25

    neighborhood = touch[
        touch_spec.event_frame - 3:touch_spec.event_frame + 4, 0
    ]
    assert float(np.ptp(neighborhood)) < 0.01
    assert np.all(touch[touch_spec.event_frame, 135:139] == 1.0)
    trailing_clearance = float(
        np.max(touch_joints[len(touch) // 2:touch_spec.event_frame, 8, 2])
        - touch_joints[touch_spec.event_frame, 8, 2]
    )
    assert trailing_clearance > 0.04, trailing_clearance

    for motion, joints in ((side, side_joints), (touch, touch_joints)):
        for channels, ankle in ((slice(135, 137), 7), (slice(137, 139), 8)):
            planted = np.all(motion[:, channels] > 0.5, axis=1)
            travel = np.linalg.norm(np.diff(joints[:, ankle, :2], axis=0), axis=1)
            stable = planted[:-1] & planted[1:]
            assert stable.any()
            assert float(np.max(travel[stable])) < 0.012, (ankle, travel[stable].max())


@needs_fk
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_lateral_step_signatures_survive_real_host_composition():
    from agentlodge.dance.format import to_editor139
    from server.fk import compute_poses

    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:180]
    bank = default_motion_bank()
    gaps = {}
    for motion_id in ("side_step", "step_touch"):
        out, report = bank.apply(base, motion_id, beats=np.arange(0, 180, 15))
        event = int(report["event_frame"])
        joints = compute_poses(out)["fk_joints"]
        gaps[motion_id] = float(np.linalg.norm(
            joints[event, 7, :2] - joints[event, 8, :2]
        ))
    assert gaps["side_step"] > 0.32, gaps
    assert gaps["step_touch"] < 0.16, gaps
    assert gaps["side_step"] > gaps["step_touch"] + 0.16, gaps


@needs_fk
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_side_step_root_and_lead_arm_agree_on_dancer_relative_direction():
    from agentlodge.dance.format import to_editor139
    from server.fk import compute_poses

    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:180]
    bank = default_motion_bank()
    for direction, sign in (("left", 1.0), ("right", -1.0)):
        out, report = bank.apply(
            base,
            "side_step",
            beats=np.arange(0, 180, 15),
            direction=direction,
        )
        start, end = report["action_range"]
        event = int(report["event_frame"])
        joints = compute_poses(out)["fk_joints"]
        forward = _body_forward(joints, start)
        left = np.array([-forward[1], forward[0], 0.0])
        root = (out[start:end, :2] - out[start, :2]) @ left[:2]
        lead = 20 if direction == "left" else 21
        trail = 21 if direction == "left" else 20
        lead_offset = sign * float((joints[event, lead] - joints[event, 9]) @ left)
        trail_offset = sign * float((joints[event, trail] - joints[event, 9]) @ left)
        assert float(np.max(sign * root)) > 0.24, (direction, root)
        assert lead_offset > 0.45, (direction, lead_offset)
        assert lead_offset > trail_offset + 0.35, (
            direction,
            lead_offset,
            trail_offset,
        )


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
                base,
                motion_id,
                mode=mode,
                mirror=mirror,
                direction=None if mirror else spec.canonical_direction,
                anchor="center",
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

    # ...and it is still three separate claps, not one long squeeze. Full closure is deliberately
    # held for a few frames so retiming cannot sample between contacts; count those contiguous
    # closure runs, then prove the hands visibly part between each pair.
    full = np.linalg.norm(joints[:, 20] - joints[:, 21], axis=-1)
    closed = full < 0.01
    starts = np.flatnonzero(closed & ~np.r_[False, closed[:-1]])
    ends = np.flatnonzero(closed & ~np.r_[closed[1:], False])
    assert len(starts) == len(ends) == 3, ("expected three claps", starts, ends)
    for left, right in zip(ends[:-1], starts[1:]):
        assert float(full[left + 1:right].max()) > 0.036, (
            "hands do not visibly part between claps",
            left,
            right,
        )


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
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_jump_arms_up_is_a_wide_v_not_an_overhead_clap_on_the_real_host():
    from agentlodge.dance.format import to_editor139
    from server.fk import compute_poses

    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:180]
    bank = default_motion_bank()
    jump, jump_report = bank.apply(
        base,
        "jump_arms_up",
        beats=np.arange(0, 180, 15),
    )
    clap, clap_report = bank.apply(
        base,
        "clap_overhead",
        beats=np.arange(0, 180, 15),
        direction="forward",
    )
    jump_joints = compute_poses(jump)["fk_joints"]
    clap_joints = compute_poses(clap)["fk_joints"]
    jump_event = int(jump_report["event_frame"])
    clap_event = int(clap_report["event_frame"])
    jump_gap = float(np.linalg.norm(
        jump_joints[jump_event, 20] - jump_joints[jump_event, 21]
    ))
    clap_gap = float(np.linalg.norm(
        clap_joints[clap_event, 20] - clap_joints[clap_event, 21]
    ))
    assert jump_gap > 0.65, jump_gap
    assert clap_gap < 0.01, clap_gap
    assert jump_gap > clap_gap + 0.60
    assert float(
        min(jump_joints[jump_event, 20, 2], jump_joints[jump_event, 21, 2])
        - jump_joints[jump_event, 15, 2]
    ) > 0.15


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
    assert not ({4, 5} & set(spec.additive_joints)), (
        "a chest isolation must preserve the host dance's knee cadence"
    )
    joints = compute_poses(bank.load_clip("chest_pop"))["fk_joints"]
    fwd = np.array([0.0, -1.0, 0.0])

    def travel(joint):
        """Displacement relative to the pelvis, so a step or lean cannot fake a pop."""
        return ((joints[spec.event_frame, joint] - joints[spec.event_frame, 0])
                - (joints[0, joint] - joints[0, 0]))

    chest, head = travel(9), travel(15)
    chest_fwd, head_fwd = float(chest @ fwd), float(head @ fwd)
    assert chest_fwd > 0.10, ("the chest pop is too subtle to read at full-body scale", chest_fwd)
    assert abs(head_fwd) < 0.6 * chest_fwd, (
        "the head travels as far as the chest, so this reads as a nod", head_fwd, chest_fwd)
    assert float(head[2]) > -0.06, ("the head dives instead of staying level", float(head[2]))


@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_composed_chest_pop_preserves_source_knee_cadence():
    from agentlodge.dance.format import to_editor139
    from scripts.build_motion_bank_audit import _machine_checks

    host = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:180]
    bank = default_motion_bank()
    spec = bank.resolve("chest_pop")
    edited, report = bank.apply(host, "chest_pop", beats=np.arange(0, 180, 15))
    start, end = report["action_range"]
    knee_channels = slice(3 + 6 * 4, 3 + 6 * 6)
    np.testing.assert_allclose(
        edited[start:end, knee_channels],
        host[start:end, knee_channels],
        atol=5e-7,
        rtol=0,
    )
    checks, status = _machine_checks(host, edited, spec, report)
    assert status == "pass"
    assert next(
        check for check in checks if check["name"] == "chest_pop_isolation"
    )["passed"]

    damaged = edited.copy()
    damaged[start:end, 3 + 6 * 4] += np.float32(0.01)
    checks, status = _machine_checks(host, damaged, spec, report)
    assert status == "fail"
    assert not next(
        check for check in checks if check["name"] == "chest_pop_isolation"
    )["passed"]


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
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
@pytest.mark.parametrize("motion_id", ["jump_two_foot", "jump_arms_up"])
def test_a_composed_jump_has_load_takeoff_apex_and_two_foot_landing(motion_id):
    """A vertical root arc is not a jump unless the real host body performs every jump phase."""
    from agentlodge.dance.format import to_editor139
    from server.fk import compute_poses

    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    assert {1, 2, 4, 5, 7, 8, 10, 11}.issubset(spec.absolute_joints)
    assert set(spec.phase_joints) == {1, 2, 4, 5, 7, 8, 10, 11}
    assert spec.phase_blend_frames is not None and spec.phase_blend_frames <= 3

    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:120]
    out, report = bank.apply(
        base, motion_id, beats=np.arange(0, 120, 15), anchor="beat",
    )
    joints = compute_poses(out)["fk_joints"]
    start, end = report["action_range"]
    event = int(report["event_frame"])
    load = start + int(np.argmin(joints[start:event + 1, 0, 2]))
    landing = event + int(np.argmin(joints[event:end, 0, 2]))

    def joint_angle(a, b, c):
        left = a - b
        right = c - b
        cosine = np.sum(left * right, axis=-1) / (
            np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1) + 1e-9
        )
        return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))

    left_knee = joint_angle(joints[:, 1], joints[:, 4], joints[:, 7])
    right_knee = joint_angle(joints[:, 2], joints[:, 5], joints[:, 8])
    assert max(float(left_knee[load]), float(right_knee[load])) < 145.0
    assert max(float(left_knee[landing]), float(right_knee[landing])) < 145.0

    toes = joints[event, [10, 11]]
    assert float(np.linalg.norm(toes[0, :2] - toes[1, :2])) < 0.32
    assert abs(float(toes[0, 2] - toes[1, 2])) < 0.04
    planted_height = max(
        float(joints[load, [7, 8, 10, 11], 2].min()),
        float(joints[landing, [7, 8, 10, 11], 2].min()),
    )
    assert float(joints[event, [7, 8, 10, 11], 2].min()) - planted_height > 0.20
    assert int(out[load, 135:139].sum()) > 0
    assert int(out[event, 135:139].sum()) == 0
    assert int(out[landing, 135:139].sum()) > 0


@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
@pytest.mark.parametrize("motion_id", ["jump_two_foot", "jump_arms_up"])
def test_jump_validator_rejects_a_root_only_float(motion_id):
    from agentlodge.dance.format import to_editor139

    bank = default_motion_bank()
    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:120]
    out, report = bank.apply(
        base, motion_id, beats=np.arange(0, 120, 15), anchor="beat",
    )
    start, end = report["action_range"]
    impostor = out[start:end].copy()
    reference = base[start:end]
    impostor[:, 3:135] = reference[:, 3:135]

    result = validate_semantics(impostor, bank.resolve(motion_id), reference=reference)
    assert not result["ok"]
    assert "flex" in result["detail"]


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


def test_a_root_only_bob_cannot_impersonate_a_bounce():
    """Contacts and root oscillation alone describe a floating mannequin, not a bounce."""
    bank = default_motion_bank()
    spec = bank.resolve("bounce_in_place")
    impostor = bank.load_clip(spec).copy()
    impostor[:, 3:135] = impostor[:1, 3:135]
    result = validate_semantics(impostor, spec)
    assert not result["ok"]
    assert "bilateral activity" in result["detail"]


@needs_fk
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
@pytest.mark.parametrize("motion_id", ["bounce_in_place", "crouch_drop", "rise_reach"])
def test_planted_level_motions_keep_both_feet_on_the_host_floor(motion_id):
    """Vertical translation must fade with the leg pose instead of floating or sinking the feet."""
    from agentlodge.dance.format import to_editor139
    from server.fk import compute_poses

    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:180]
    bank = default_motion_bank()
    out, report = bank.apply(base, motion_id, beats=np.arange(0, 180, 15))
    start, end = map(int, report["action_range"])
    action = compute_poses(out[start:end])["fk_joints"]
    host = compute_poses(base[start:end])["fk_joints"]
    host_floor = np.min(host[:, (7, 8, 10, 11), 2], axis=1)
    blend = bank.resolve(motion_id).phase_blend_frames or 0
    core = slice(blend, end - start - blend if blend else end - start)

    for foot in ((7, 10), (8, 11)):
        height = np.min(action[:, foot, 2], axis=1) - host_floor
        assert float(np.max(np.abs(height[core]))) < 0.06, (motion_id, foot, height)
    assert np.all(out[start:end, 135:139] == 1.0)
    assert float(out[start, 2] - base[start, 2]) == pytest.approx(0.0, abs=1e-6)
    assert float(out[end - 1, 2] - base[end - 1, 2]) == pytest.approx(0.0, abs=1e-6)


@needs_fk
@pytest.mark.skipif(not os.path.exists(_LODGE_SAMPLE), reason="LODGE sample dance not present")
def test_a_forward_punch_has_guard_strike_and_recoil_phases():
    """A held straight arm reads as a reach; a punch must cock, strike forward, and retract."""
    from agentlodge.dance.format import to_editor139
    from server.fk import compute_poses

    base = to_editor139(np.load(_LODGE_SAMPLE).astype(np.float32))[:180]
    beats = np.arange(0, 180, 15)
    out, report = default_motion_bank().apply(
        base, "arm_punch", beats=beats, direction="right",
    )
    start, end = map(int, report["action_range"])
    joints = compute_poses(out[start:end])["fk_joints"]
    forward = np.stack([_body_forward(joints, frame) for frame in range(len(joints))])
    lateral = np.column_stack([-forward[:, 1], forward[:, 0], np.zeros(len(joints))])
    forward_reach = np.sum((joints[:, 21] - joints[:, 9]) * forward, axis=1)
    side_reach = np.abs(np.sum((joints[:, 21] - joints[:, 17]) * lateral, axis=1))
    arm_length = np.linalg.norm(joints[:, 21] - joints[:, 17], axis=1)
    peak = int(np.argmax(forward_reach))

    assert 1 < peak < len(joints) - 2
    assert float(forward_reach[peak]) > 1.5 * float(side_reach[peak])
    assert float(arm_length[peak] - np.min(arm_length[:peak])) > 0.15
    assert float(arm_length[peak] - np.min(arm_length[peak + 1:])) > 0.15
    assert int(np.count_nonzero(forward_reach > 0.9 * forward_reach[peak])) <= 5
    assert float(np.linalg.norm(joints[peak, 20] - joints[peak, 9])) < 0.30
    assert float(np.linalg.norm(joints[peak, 20] - joints[peak, 16])) < 0.35
    assert window_metrics(out, beats)["jerk"] < 5.0 * window_metrics(base, beats)["jerk"]


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
    from agentlodge.editor.motion_bank import _insertion_host

    song = np.concatenate([_base(n)] * 3, axis=0)
    a, b = n, 2 * n
    base = song[a:b]
    window, report = bank.apply(base, motion_id, mode=mode, anchor="center")
    start, end = map(int, report["action_range"])
    event = int(report["event_frame"])
    host = (
        base
        if mode == "replace"
        else _insertion_host(base, start, end, event, event - start)
    )
    spliced = crossfade_edit(song, a, b, window, blend_frames=8)[a:b]
    return song, host, window, spliced, report


@pytest.mark.parametrize("motion_id", [s.id for s in default_motion_bank().specs])
def test_a_named_motion_still_reads_as_itself_once_it_is_spliced(motion_id):
    """Validating the clip in isolation proves nothing: the splice pins the window's edges
    back to the song, which used to erase a turn or a step from the record entirely."""
    bank = default_motion_bank()
    for mode in ("replace", "insert"):
        _song, host, _window, spliced, report = _spliced(bank, motion_id, mode)
        check = verify_applied_motion(spliced, report, reference=host)
        assert check["ok"], (motion_id, mode, check["detail"])


@pytest.mark.parametrize("motion_id", ["turn_half", "turn_quarter", "step_forward",
                                       "step_backward", "side_step", "crouch_drop", "rise_reach"])
def test_a_travelling_motion_hands_the_root_back_before_the_window_ends(motion_id):
    """Whatever the dancer does inside the window, the next window starts where the song
    left off. Anything still owed at the last frame gets ripped back by the crossfade."""
    bank = default_motion_bank()
    for mode in ("replace", "insert"):
        song, _host, window, _spliced_win, _report = _spliced(bank, motion_id, mode)
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
        _song, _host, _window, spliced, _report = _spliced(bank, motion_id, mode)
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
    expected = _declared_action_frames(spec, 16)
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
