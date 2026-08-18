"""Create the only receipt accepted by MAESTRO's production motion-audit gate."""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.editor.motion_audit import (  # noqa: E402
    DEFAULT_RECEIPT,
    REQUIRED_QUALITY_CHECKS,
    REVIEW_PROTOCOL_VERSION,
    REVIEWER_ATTESTATION_STATEMENT,
    motion_fingerprint,
    required_audit_cases,
    validate_audit_render_receipt,
    validate_audit_receipt,
    validate_ybot_metrics_report,
)
from agentlodge.editor.motion_bank import MotionBank, normalize_name  # noqa: E402


def _normalized_direction(value: object) -> str:
    normalized = normalize_name(str(value or ""))
    if normalized in {"none", "no direction", "not directional", "non directional"}:
        return "none"
    return normalized


def _timestamp(value: object, *, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _validate_review_proof(
    take: dict,
    decision: dict,
    *,
    audit_id: str,
    motion_fingerprint_value: str,
    attested_at: datetime,
) -> None:
    take_id = str(take["take"])
    playback = decision.get("normal_speed_playback")
    if not isinstance(playback, dict) or playback.get("completed") is not True:
        raise ValueError(f"{take_id}: synchronized normal-speed playback is incomplete")
    numeric_fields = {
        name: playback.get(name)
        for name in (
            "playback_rate",
            "seek_count",
            "pause_count",
            "max_sync_drift",
            "source_seconds",
            "edit_seconds",
            "elapsed_seconds",
        )
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in numeric_fields.values()
    ):
        raise ValueError(f"{take_id}: normal-speed playback proof is malformed")
    playback_rate = float(numeric_fields["playback_rate"])
    seek_value = float(numeric_fields["seek_count"])
    pause_value = float(numeric_fields["pause_count"])
    if not seek_value.is_integer() or not pause_value.is_integer():
        raise ValueError(f"{take_id}: normal-speed playback proof is malformed")
    seek_count = int(seek_value)
    pause_count = int(pause_value)
    max_sync_drift = float(numeric_fields["max_sync_drift"])
    source_seconds = float(numeric_fields["source_seconds"])
    edit_seconds = float(numeric_fields["edit_seconds"])
    elapsed_seconds = float(numeric_fields["elapsed_seconds"])
    if abs(playback_rate - 1.0) > 1e-6:
        raise ValueError(f"{take_id}: source/edit pair was not played at 1.0x")
    if seek_count != 0:
        raise ValueError(f"{take_id}: source/edit pair was not played without seeking")
    if pause_count != 0:
        raise ValueError(f"{take_id}: source/edit pair was not played uninterrupted")
    if max_sync_drift < 0 or max_sync_drift > 0.12:
        raise ValueError(f"{take_id}: source/edit pair was not played synchronously")
    expected_seconds = int(take.get("frames", 0)) / 30.0
    if expected_seconds <= 0:
        raise ValueError(f"{take_id}: audit take has no valid frame duration")
    if min(source_seconds, edit_seconds, elapsed_seconds) + 0.12 < expected_seconds:
        raise ValueError(f"{take_id}: source/edit pair was not played for its full duration")

    started = _timestamp(
        playback.get("started_at"),
        label=f"{take_id}: playback start timestamp",
    )
    completed = _timestamp(
        playback.get("completed_at"),
        label=f"{take_id}: playback completion timestamp",
    )
    comparison_opened = _timestamp(
        decision.get("comparison_opened_at"),
        label=f"{take_id}: synchronized comparison page timestamp",
    )
    acknowledgment = decision.get("comparison_acknowledgment")
    expected_acknowledgment = {
        "auditId": audit_id,
        "motionFingerprint": motion_fingerprint_value,
        "takeId": take_id,
    }
    if not isinstance(acknowledgment, dict) or any(
        acknowledgment.get(key) != value
        for key, value in expected_acknowledgment.items()
    ):
        raise ValueError(
            f"{take_id}: synchronized comparison acknowledgment is missing or stale"
        )
    locked = _timestamp(
        decision.get("locked_at"),
        label=f"{take_id}: guess lock timestamp",
    )
    if completed < started:
        raise ValueError(f"{take_id}: playback completion predates playback start")
    if (completed - started).total_seconds() + 0.12 < expected_seconds:
        raise ValueError(
            f"{take_id}: playback timestamps are too short for uninterrupted review"
        )
    if locked < completed or locked < comparison_opened:
        raise ValueError(
            f"{take_id}: playback and synchronized comparison must occur before locking"
        )
    if attested_at < locked:
        raise ValueError(f"{take_id}: reviewer attestation predates the locked blind guess")


def _validate_reviewer_attestation(
    result: dict,
    *,
    audit_id: str,
    motion_fingerprint_value: str,
) -> tuple[dict, datetime]:
    attestation = result.get("reviewer_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("review result has no reviewer attestation")
    expected = {
        "audit_id": audit_id,
        "motion_fingerprint": motion_fingerprint_value,
        "statement": REVIEWER_ATTESTATION_STATEMENT,
    }
    if any(attestation.get(key) != value for key, value in expected.items()):
        raise ValueError("reviewer attestation is missing, stale, or incomplete")
    for key in (
        "independent_visual_review",
        "answers_hidden_until_lock",
        "normal_speed_reviewed",
        "source_edit_compared",
    ):
        if attestation.get(key) is not True:
            raise ValueError("reviewer attestation is missing, stale, or incomplete")
    reviewer = attestation.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer attestation needs a named independent reviewer")
    reviewer = reviewer.strip()
    signed_at = _timestamp(
        attestation.get("signed_at"),
        label="reviewer attestation timestamp",
    )
    return {
        "audit_id": audit_id,
        "motion_fingerprint": motion_fingerprint_value,
        "reviewer": reviewer,
        "signed_at": signed_at.isoformat(),
        "independent_visual_review": True,
        "answers_hidden_until_lock": True,
        "normal_speed_reviewed": True,
        "source_edit_compared": True,
        "statement": REVIEWER_ATTESTATION_STATEMENT,
    }, signed_at


def finalize(audit_dir: Path, review_result: Path, output: Path) -> dict:
    audit_dir = Path(audit_dir)
    review = json.loads((audit_dir / "review.json").read_text(encoding="utf-8"))
    answers = json.loads((audit_dir / "answer_key.json").read_text(encoding="utf-8"))
    result = json.loads(Path(review_result).read_text(encoding="utf-8"))
    bank = MotionBank()
    fingerprint = motion_fingerprint(ROOT)

    if review.get("motion_fingerprint") != fingerprint:
        raise ValueError("audit artifacts are stale for the current motion code or assets")
    if review.get("bank_version") != bank.version:
        raise ValueError("audit artifacts belong to a different motion-bank version")
    if int(review.get("review_protocol_version", 0)) < REVIEW_PROTOCOL_VERSION:
        raise ValueError("audit artifacts predate the blocking review protocol")
    if review.get("normalized_facing") is not True:
        raise ValueError("release audit must use normalized semantic front and side views")
    if review.get("fixed_camera") is not True:
        raise ValueError("release audit must use the fixed camera that exposes root travel")
    try:
        validate_audit_render_receipt(audit_dir, review)
        ybot_metrics = validate_ybot_metrics_report(audit_dir, review)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if result.get("audit_id") != review.get("audit_id"):
        raise ValueError("review result belongs to a different audit")
    if result.get("motion_fingerprint") != fingerprint:
        raise ValueError("review result belongs to a different motion fingerprint")
    if result.get("normalized_facing") is not True:
        raise ValueError("review result did not preserve normalized semantic views")
    if result.get("normal_speed_reviewed") is not True:
        raise ValueError("reviewer did not confirm normal-speed playback")
    if result.get("source_edit_compared") is not True:
        raise ValueError("reviewer did not confirm source-versus-edit comparison")
    attestation, attested_at = _validate_reviewer_attestation(
        result,
        audit_id=str(review["audit_id"]),
        motion_fingerprint_value=fingerprint,
    )

    decisions = result.get("takes")
    if not isinstance(decisions, dict):
        raise ValueError("review result has no take decisions")
    takes = review.get("takes") or []
    take_ids = {take["take"] for take in takes}
    if set(decisions) != take_ids:
        raise ValueError("review result does not cover every audit take exactly once")
    if set(answers) != take_ids:
        raise ValueError("answer key does not cover every audit take exactly once")

    cases = {}
    for take in takes:
        take_id = take["take"]
        decision = decisions[take_id]
        answer = answers[take_id]
        case_id = answer["case_id"]
        if ybot_metrics["takes"][take_id].get("case_id") != case_id:
            raise ValueError(f"{take_id}: exact Y-Bot metrics belong to a different case")
        _validate_review_proof(
            take,
            decision,
            audit_id=str(review["audit_id"]),
            motion_fingerprint_value=fingerprint,
            attested_at=attested_at,
        )
        accepted = {
            normalize_name(value)
            for value in (answer["id"], answer["name"], *answer.get("aliases", ()))
        }
        action_matches = normalize_name(str(decision.get("guess", ""))) in accepted
        if decision.get("recognized") is not True or not action_matches:
            raise ValueError(f"{case_id}: blind guess did not match")
        expected_direction = _normalized_direction(answer.get("resolved_direction") or "none")
        direction_matches = (
            _normalized_direction(decision.get("direction_guess")) == expected_direction
        )
        if decision.get("direction_recognized") is not True or not direction_matches:
            raise ValueError(f"{case_id}: blind direction guess did not match")
        if decision.get("status") != "pass":
            raise ValueError(f"{case_id}: visual phase review failed")
        evidence = str(decision.get("evidence", "")).strip()
        if not evidence:
            raise ValueError(f"{case_id}: visual phase review needs an evidence note")
        expected_phases = tuple(answer["visual_contract"]["required_phases"])
        reviewed_phases = tuple(decision.get("verified_phases") or ())
        if (
            len(reviewed_phases) != len(expected_phases)
            or set(reviewed_phases) != set(expected_phases)
        ):
            raise ValueError(f"{case_id}: every required visual phase must be reviewed")
        expected_negative_signatures = tuple(
            answer["visual_contract"]["must_not_read_as"]
        )
        reviewed_negative_signatures = tuple(
            decision.get("verified_negative_signatures") or ()
        )
        if (
            len(reviewed_negative_signatures) != len(expected_negative_signatures)
            or set(reviewed_negative_signatures) != set(expected_negative_signatures)
        ):
            raise ValueError(
                f"{case_id}: every competing silhouette must be explicitly rejected"
            )
        reviewed_quality_checks = tuple(
            decision.get("verified_quality_checks") or ()
        )
        if (
            len(reviewed_quality_checks) != len(REQUIRED_QUALITY_CHECKS)
            or set(reviewed_quality_checks) != set(REQUIRED_QUALITY_CHECKS)
        ):
            raise ValueError(
                f"{case_id}: every biomechanics and continuity check must be reviewed"
            )
        if answer.get("machine_status") != "pass":
            raise ValueError(f"{case_id}: machine visual invariants failed")
        if not all(check.get("passed") is True for check in answer.get("machine_checks", ())):
            raise ValueError(f"{case_id}: a machine visual invariant is not passing")
        if case_id in cases:
            raise ValueError(f"{case_id}: audit contains a duplicate motion-variant case")
        cases[case_id] = {
            "blind_recognition": "pass",
            "blind_direction": "pass",
            "human_status": "pass",
            "machine_status": "pass",
            "normal_speed_playback": "pass",
            "source_edit_comparison": "pass",
            "verified_phases": list(expected_phases),
            "verified_negative_signatures": list(expected_negative_signatures),
            "verified_quality_checks": list(REQUIRED_QUALITY_CHECKS),
            "evidence": evidence,
        }

    required = set(required_audit_cases(bank))
    if set(cases) != required:
        raise ValueError("audit does not cover the complete required motion-variant matrix")

    receipt = {
        "schema_version": 2,
        "status": "pass",
        "motion_fingerprint": fingerprint,
        "bank_version": bank.version,
        "audit_id": review["audit_id"],
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "normal_speed_reviewed": True,
        "source_edit_compared": True,
        "reviewer_attestation": attestation,
        "verification_nonce": secrets.token_hex(16),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    validate_audit_receipt(output, root=ROOT, bank=bank)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = finalize(args.audit, args.review_result, args.output)
    print(
        f"motion audit passed: {len(receipt['cases'])} cases, "
        f"fingerprint {receipt['motion_fingerprint'][:12]}"
    )


if __name__ == "__main__":
    main()
