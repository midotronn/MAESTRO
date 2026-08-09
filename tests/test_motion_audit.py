"""Release-gate contracts for the visual motion-bank audit."""

from __future__ import annotations

import json

import pytest

from agentlodge.editor import motion_audit
from agentlodge.editor.motion_audit import (
    REVIEW_PROTOCOL_VERSION,
    motion_fingerprint,
    record_audit_render_receipt,
    required_audit_cases,
    required_phases,
    validate_audit_receipt,
)
from agentlodge.editor.motion_bank import MotionBank
from scripts.finalize_motion_audit import finalize


def _passing_receipt() -> dict:
    bank = MotionBank()
    phases = required_phases()
    return {
        "schema_version": 1,
        "status": "pass",
        "motion_fingerprint": motion_fingerprint(),
        "bank_version": bank.version,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "normal_speed_reviewed": True,
        "source_edit_compared": True,
        "cases": {
            case_id: {
                "blind_recognition": "pass",
                "blind_direction": "pass",
                "human_status": "pass",
                "machine_status": "pass",
                "normal_speed_playback": "pass",
                "source_edit_comparison": "pass",
                "verified_phases": list(phases[case_id.split("@", 1)[0]]),
                "evidence": "Source and edit were compared at normal speed; all phases read clearly.",
            }
            for case_id in required_audit_cases()
        },
    }


def test_complete_current_receipt_is_accepted(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_passing_receipt()), encoding="utf-8")
    assert validate_audit_receipt(path)["status"] == "pass"


def test_required_matrix_includes_automatic_and_explicit_direction_variants():
    cases = set(required_audit_cases())
    assert "clap_single@auto" in cases
    assert "clap_single@forward" in cases
    assert "clap_single@left" in cases
    assert "clap_single@right" in cases
    assert "jump_two_foot" in cases
    assert len(cases) == 43


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda receipt: receipt.update(normal_speed_reviewed=False), "normal-speed"),
        (lambda receipt: receipt.update(source_edit_compared=False), "source and edit"),
        (
            lambda receipt: receipt["cases"].pop(next(iter(receipt["cases"]))),
            "matrix is incomplete",
        ),
        (
            lambda receipt: next(iter(receipt["cases"].values())).update(
                machine_status="fail"
            ),
            "machine visual invariants",
        ),
        (
            lambda receipt: next(iter(receipt["cases"].values())).update(
                human_status="fail"
            ),
            "visual phase review",
        ),
        (
            lambda receipt: next(iter(receipt["cases"].values())).update(
                blind_recognition="fail"
            ),
            "blind recognition",
        ),
        (
            lambda receipt: next(iter(receipt["cases"].values())).update(
                blind_direction="fail"
            ),
            "blind direction",
        ),
        (
            lambda receipt: next(iter(receipt["cases"].values())).update(
                normal_speed_playback="fail"
            ),
            "synchronized normal-speed playback",
        ),
        (
            lambda receipt: next(iter(receipt["cases"].values())).update(
                source_edit_comparison="fail"
            ),
            "source/edit comparison",
        ),
        (
            lambda receipt: next(iter(receipt["cases"].values())).update(
                verified_phases=[]
            ),
            "every required visual phase",
        ),
        (
            lambda receipt: next(iter(receipt["cases"].values())).update(evidence=""),
            "evidence note",
        ),
    ],
)
def test_receipt_cannot_bypass_required_visual_evidence(tmp_path, mutation, match):
    receipt = _passing_receipt()
    mutation(receipt)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match=match):
        validate_audit_receipt(path)


def test_stale_fingerprint_is_rejected(tmp_path):
    receipt = _passing_receipt()
    receipt["motion_fingerprint"] = "0" * 64
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale"):
        validate_audit_receipt(path)


def test_fingerprint_covers_runtime_fk_render_and_review_dependencies():
    required = {
        "agentlodge/dance/format.py",
        "agentlodge/dance/transition.py",
        "agentlodge/editor/window_edit.py",
        "scripts/blender_studio.py",
        "scripts/record_motion_audit_render.py",
        "scripts/render_blender_dance.py",
        "scripts/render_motion_bank_audit.sh",
        "scripts/render_one_ybot.sh",
        "scripts/render_poses_ybot.sh",
        "scripts/render_root_motion.py",
        "server/app.py",
        "server/data/smplx_neu_J_1.npy",
        "server/fk.py",
        "server/rendering.py",
        "server/warm_render.py",
    }
    assert required.issubset(set(motion_audit._AUDITED_FILES))


def test_fingerprint_normalizes_text_line_endings(tmp_path):
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")
    assert motion_audit._fingerprint_content(lf) == motion_audit._fingerprint_content(crlf)


def test_pod_deployment_validates_before_replacing_exact_motion_bank_tree():
    script = (
        motion_audit.ROOT / "scripts" / "host_on_pod.ps1"
    ).read_text(encoding="utf-8")

    assert "validate_audit_receipt()" in script
    assert "assets/motion_bank requirements.txt" in script
    assert 'test -f "$stage/assets/motion_bank/audit_receipt.json"' in script
    staged_validation = script.index("MOTION_AUDIT_RECEIPT_OK")
    old_tree_move = script.index('mv "$live/assets/motion_bank" "$backup"')
    exact_tree_move = script.index(
        'mv "$stage/assets/motion_bank" "$live/assets/motion_bank"'
    )
    launch = script.index('bash "$WORKSPACE/AgentLODGE/scripts/serve_on_pod.sh"')
    assert staged_validation < old_tree_move < exact_tree_move < launch
    assert "OPENAI_API_KEY='$OpenAIKey'" not in script
    assert '$script | & ssh' in script


def _complete_audit_export(tmp_path):
    bank = MotionBank()
    phases = required_phases()
    specs = {spec.id: spec for spec in bank.specs}
    fingerprint = motion_fingerprint()
    takes = []
    answers = {}
    decisions = {}
    for index, case_id in enumerate(required_audit_cases(bank), start=1):
        take = f"take_{index:02d}"
        motion_id, _, requested = case_id.partition("@")
        spec = specs[motion_id]
        resolved = None
        if spec.directions:
            resolved = spec.canonical_direction if requested == "auto" else requested
        takes.append({"take": take, "control": "control_01", "frames": 54})
        answers[take] = {
            "case_id": case_id,
            "id": spec.id,
            "name": spec.name,
            "aliases": list(spec.aliases),
            "requested_direction": requested or None,
            "resolved_direction": resolved,
            "visual_contract": {"required_phases": list(phases[motion_id])},
            "machine_checks": [{"name": "test", "passed": True, "detail": "pass"}],
            "machine_status": "pass",
        }
        decisions[take] = {
            "guess": spec.id,
            "recognized": True,
            "direction_guess": resolved or "none",
            "direction_recognized": True,
            "normal_speed_playback": {
                "completed": True,
                "started_at": "2026-03-26T12:00:00+00:00",
                "completed_at": "2026-03-26T12:00:02+00:00",
                "playback_rate": 1.0,
                "seek_count": 0,
                "pause_count": 0,
                "source_seconds": 1.8,
                "edit_seconds": 1.8,
                "elapsed_seconds": 2.0,
                "max_sync_drift": 0.01,
            },
            "comparison_opened_at": "2026-03-26T12:00:01+00:00",
            "comparison_acknowledgment": {
                "auditId": "test-audit",
                "motionFingerprint": fingerprint,
                "takeId": take,
            },
            "locked_at": "2026-03-26T12:00:03+00:00",
            "status": "pass",
            "evidence": "Compared the exact source and edit at normal speed.",
            "verified_phases": list(phases[motion_id]),
        }
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True)
    review = {
        "audit_id": "test-audit",
        "bank_version": bank.version,
        "motion_fingerprint": fingerprint,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "normalized_facing": True,
        "fixed_camera": True,
        "takes": takes,
    }
    result = {
        "audit_id": "test-audit",
        "motion_fingerprint": fingerprint,
        "normalized_facing": True,
        "normal_speed_reviewed": True,
        "source_edit_compared": True,
        "takes": decisions,
    }
    (audit_dir / "review.json").write_text(json.dumps(review), encoding="utf-8")
    (audit_dir / "answer_key.json").write_text(json.dumps(answers), encoding="utf-8")
    for relative in motion_audit._mandatory_render_artifacts(review):
        path = audit_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{relative}\n".encode())
    for take in takes:
        page = audit_dir / "phase_sheets" / f"{take['take']}_review_01.jpg"
        page.write_bytes(f"{take['take']}\n".encode())
    record_audit_render_receipt(audit_dir)
    result_path = tmp_path / "review-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return audit_dir, result_path, result


def test_finalizer_recomputes_blind_action_and_direction_results(tmp_path):
    audit_dir, result_path, result = _complete_audit_export(tmp_path)
    output = tmp_path / "receipt.json"
    assert finalize(audit_dir, result_path, output)["status"] == "pass"

    first = next(iter(result["takes"].values()))
    first["guess"] = "not the motion"
    first["recognized"] = True
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="blind guess"):
        finalize(audit_dir, result_path, output)

    audit_dir, result_path, result = _complete_audit_export(tmp_path / "direction")
    directional = next(
        decision
        for decision in result["takes"].values()
        if decision["direction_guess"] != "none"
    )
    directional["direction_guess"] = (
        "right" if directional["direction_guess"] != "right" else "left"
    )
    directional["direction_recognized"] = True
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="blind direction"):
        finalize(audit_dir, result_path, tmp_path / "direction-receipt.json")


def test_finalizer_rejects_skipped_visual_phases_even_when_marked_pass(tmp_path):
    audit_dir, result_path, result = _complete_audit_export(tmp_path)
    first = next(iter(result["takes"].values()))
    first["verified_phases"] = first["verified_phases"][:-1]
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="every required visual phase"):
        finalize(audit_dir, result_path, tmp_path / "receipt.json")


def test_finalizer_accepts_rechecked_phases_in_a_different_order(tmp_path):
    audit_dir, result_path, result = _complete_audit_export(tmp_path)
    first = next(iter(result["takes"].values()))
    first["verified_phases"] = list(reversed(first["verified_phases"]))
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert finalize(audit_dir, result_path, tmp_path / "receipt.json")["status"] == "pass"


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda decision: decision["normal_speed_playback"].update(completed=False),
            "playback is incomplete",
        ),
        (
            lambda decision: decision["normal_speed_playback"].update(playback_rate=0.5),
            "1.0x",
        ),
        (
            lambda decision: decision["normal_speed_playback"].update(seek_count=1),
            "without seeking",
        ),
        (
            lambda decision: decision["normal_speed_playback"].update(seek_count=0.5),
            "proof is malformed",
        ),
        (
            lambda decision: decision["normal_speed_playback"].update(pause_count=1),
            "uninterrupted",
        ),
        (
            lambda decision: decision["normal_speed_playback"].update(source_seconds=0.1),
            "full duration",
        ),
        (
            lambda decision: decision["normal_speed_playback"].update(max_sync_drift=0.2),
            "synchronously",
        ),
        (
            lambda decision: decision["normal_speed_playback"].update(
                max_sync_drift=float("nan")
            ),
            "proof is malformed",
        ),
        (
            lambda decision: decision.pop("comparison_opened_at"),
            "comparison page timestamp is missing",
        ),
        (
            lambda decision: decision["comparison_acknowledgment"].update(
                auditId="stale-audit"
            ),
            "acknowledgment is missing or stale",
        ),
        (
            lambda decision: decision.update(
                locked_at="2026-03-26T12:00:01+00:00"
            ),
            "must occur before locking",
        ),
        (
            lambda decision: decision["normal_speed_playback"].update(
                completed_at="2026-03-26T12:00:00.500000+00:00"
            ),
            "timestamps are too short",
        ),
    ],
)
def test_finalizer_rejects_incomplete_per_take_review_proof(
    tmp_path, mutation, match
):
    audit_dir, result_path, result = _complete_audit_export(tmp_path)
    mutation(next(iter(result["takes"].values())))
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        finalize(audit_dir, result_path, tmp_path / "receipt.json")


def test_finalizer_rejects_missing_or_modified_render_evidence(tmp_path):
    audit_dir, result_path, _result = _complete_audit_export(tmp_path)
    (audit_dir / "render_receipt.json").unlink()
    with pytest.raises(ValueError, match="render receipt is missing"):
        finalize(audit_dir, result_path, tmp_path / "missing-receipt.json")

    audit_dir, result_path, _result = _complete_audit_export(tmp_path / "changed")
    video = next((audit_dir / "videos").glob("*.mp4"))
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="render artifact changed"):
        finalize(audit_dir, result_path, tmp_path / "changed-receipt.json")
