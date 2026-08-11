"""Commit-bound release gate for MAESTRO's visual motion-bank audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from agentlodge.editor.motion_bank import MotionBank, MotionSpec


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "assets" / "motion_bank" / "audit_receipt.json"
RENDER_RECEIPT_NAME = "render_receipt.json"
REVIEW_PROTOCOL_VERSION = 7
REVIEWER_ATTESTATION_STATEMENT = (
    "I independently reviewed every source/edit pair at normal speed with answers hidden "
    "until all guesses were locked, and I did not waive any failed or ambiguous case."
)
_TEXT_SUFFIXES = {".json", ".py", ".sh"}
_AUDITED_FILES = (
    "agentlodge/dance/format.py",
    "agentlodge/dance/transition.py",
    "agentlodge/editor/agent_edit.py",
    "agentlodge/editor/motion_audit.py",
    "agentlodge/editor/motion_bank.py",
    "agentlodge/editor/remote_generator.py",
    "agentlodge/editor/session.py",
    "agentlodge/editor/window_edit.py",
    "assets/motion_bank/manifest.json",
    "assets/motion_bank/visual_contracts.json",
    "scripts/blender_daemon.py",
    "scripts/blender_studio.py",
    "scripts/blender_render_ybot.py",
    "scripts/build_motion_audit_sheets.py",
    "scripts/build_motion_bank.py",
    "scripts/build_motion_bank_audit.py",
    "scripts/finalize_motion_audit.py",
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
)


def audit_variants(spec: MotionSpec) -> tuple[str | None, ...]:
    """Every behavior variant that must receive its own source/edit review."""
    if not spec.directions:
        return (None,)
    return ("auto", *spec.directions)


def audit_case_id(spec: MotionSpec, direction: str | None) -> str:
    return spec.id if direction is None else f"{spec.id}@{direction}"


def required_audit_cases(bank: MotionBank | None = None) -> tuple[str, ...]:
    bank = bank or MotionBank()
    return tuple(
        audit_case_id(spec, direction)
        for spec in bank.specs
        for direction in audit_variants(spec)
    )


def required_phases(root: Path = ROOT) -> dict[str, tuple[str, ...]]:
    payload = json.loads(
        (Path(root) / "assets" / "motion_bank" / "visual_contracts.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        motion_id: tuple(str(phase) for phase in contract["required_phases"])
        for motion_id, contract in payload["motions"].items()
    }


def _fingerprint_content(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() in _TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def motion_fingerprint(root: Path = ROOT) -> str:
    """Hash every source and asset that can change motion behavior or its review."""
    root = Path(root).resolve()
    paths = [root / relative for relative in _AUDITED_FILES]
    paths.extend(sorted((root / "assets" / "motion_bank" / "clips").glob("*.npy")))
    missing = [str(path.relative_to(root)) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"motion audit fingerprint is missing files: {', '.join(missing)}")

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_fingerprint_content(path))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mandatory_render_artifacts(review: dict) -> set[str]:
    identifiers = {
        control["control"] for control in review.get("controls", ())
    } | {
        take["take"] for take in review.get("takes", ())
    } | {
        take["control"] for take in review.get("takes", ())
    }
    artifacts = {
        relative
        for identifier in identifiers
        for relative in (
            f"{identifier}_front.npz",
            f"{identifier}_side.npz",
            f"videos/{identifier}_front.mp4",
            f"videos/{identifier}_side.mp4",
            f"videos/{identifier}.mp4",
        )
    }
    for take in review.get("takes", ()):
        identifier = take["take"]
        artifacts.update({
            f"phase_sheets/{identifier}_front.jpg",
            f"phase_sheets/{identifier}_side.jpg",
            f"phase_sheets/{identifier}_front_detail.jpg",
            f"phase_sheets/{identifier}_dual.jpg",
            f"phase_sheets/{identifier}_review.html",
        })
    return artifacts


def record_audit_render_receipt(audit_dir: Path) -> dict:
    """Atomically bind a completed fixed-camera render to its audit inputs."""
    audit_dir = Path(audit_dir).resolve()
    review = json.loads((audit_dir / "review.json").read_text(encoding="utf-8"))
    if review.get("fixed_camera") is not True:
        raise RuntimeError("release audit render did not use the fixed camera")
    required = _mandatory_render_artifacts(review)
    artifacts = set(required)
    for take in review.get("takes", ()):
        pages = {
            path.relative_to(audit_dir).as_posix()
            for path in (audit_dir / "phase_sheets").glob(
                f"{take['take']}_review_*.jpg"
            )
        }
        if not pages:
            raise RuntimeError(f"{take['take']}: rendered review pages are missing")
        artifacts.update(pages)
    missing = sorted(relative for relative in artifacts if not (audit_dir / relative).is_file())
    if missing:
        raise RuntimeError(f"motion audit render is incomplete: {missing}")
    payload = {
        "schema_version": 1,
        "audit_id": review["audit_id"],
        "motion_fingerprint": review["motion_fingerprint"],
        "fixed_camera": True,
        "artifacts": {
            relative: _sha256_file(audit_dir / relative)
            for relative in sorted(artifacts)
        },
    }
    output = audit_dir / RENDER_RECEIPT_NAME
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def validate_audit_render_receipt(audit_dir: Path, review: dict | None = None) -> dict:
    """Reject missing, partial, stale, or modified visual render evidence."""
    audit_dir = Path(audit_dir).resolve()
    review = review or json.loads(
        (audit_dir / "review.json").read_text(encoding="utf-8")
    )
    path = audit_dir / RENDER_RECEIPT_NAME
    if not path.is_file():
        raise RuntimeError("motion audit render receipt is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise RuntimeError("motion audit render receipt has an unsupported schema")
    if payload.get("audit_id") != review.get("audit_id"):
        raise RuntimeError("motion audit render receipt belongs to a different audit")
    if payload.get("motion_fingerprint") != review.get("motion_fingerprint"):
        raise RuntimeError("motion audit render receipt has a stale fingerprint")
    if payload.get("fixed_camera") is not True or review.get("fixed_camera") is not True:
        raise RuntimeError("motion audit render did not use the required fixed camera")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("motion audit render receipt has no artifact hashes")
    required = _mandatory_render_artifacts(review)
    if not required.issubset(artifacts):
        raise RuntimeError("motion audit render receipt is missing required artifacts")
    for take in review.get("takes", ()):
        prefix = f"phase_sheets/{take['take']}_review_"
        if not any(relative.startswith(prefix) and relative.endswith(".jpg") for relative in artifacts):
            raise RuntimeError(f"{take['take']}: render receipt has no review pages")
    for relative, expected in artifacts.items():
        candidate = (audit_dir / relative).resolve()
        try:
            candidate.relative_to(audit_dir)
        except ValueError as exc:
            raise RuntimeError("motion audit render receipt contains an unsafe path") from exc
        if not candidate.is_file():
            raise RuntimeError(f"motion audit render artifact is missing: {relative}")
        if _sha256_file(candidate) != expected:
            raise RuntimeError(f"motion audit render artifact changed: {relative}")
    return payload


def validate_audit_receipt(
    receipt_path: Path = DEFAULT_RECEIPT,
    *,
    root: Path = ROOT,
    bank: MotionBank | None = None,
) -> dict:
    """Reject stale, partial, failed, or non-visual audit receipts."""
    receipt_path = Path(receipt_path)
    if not receipt_path.is_file():
        raise RuntimeError(f"motion audit receipt is missing: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 2:
        raise RuntimeError("motion audit receipt has an unsupported schema")
    if payload.get("status") != "pass":
        raise RuntimeError("motion audit receipt is not passing")
    if payload.get("motion_fingerprint") != motion_fingerprint(root):
        raise RuntimeError("motion audit receipt is stale for the current motion code or assets")
    bank = bank or MotionBank(Path(root) / "assets" / "motion_bank")
    if payload.get("bank_version") != bank.version:
        raise RuntimeError("motion audit receipt has the wrong motion-bank version")
    if not payload.get("normal_speed_reviewed"):
        raise RuntimeError("motion audit did not confirm normal-speed playback")
    if not payload.get("source_edit_compared"):
        raise RuntimeError("motion audit did not compare source and edit")
    if int(payload.get("review_protocol_version", 0)) < REVIEW_PROTOCOL_VERSION:
        raise RuntimeError("motion audit receipt predates the blocking review protocol")
    attestation = payload.get("reviewer_attestation")
    if not isinstance(attestation, dict):
        raise RuntimeError("motion audit receipt has no independent reviewer attestation")
    expected_attestation = {
        "audit_id": payload.get("audit_id"),
        "motion_fingerprint": payload.get("motion_fingerprint"),
        "statement": REVIEWER_ATTESTATION_STATEMENT,
    }
    if any(attestation.get(key) != value for key, value in expected_attestation.items()):
        raise RuntimeError("motion audit reviewer attestation is missing, stale, or incomplete")
    for key in (
        "independent_visual_review",
        "answers_hidden_until_lock",
        "normal_speed_reviewed",
        "source_edit_compared",
    ):
        if attestation.get(key) is not True:
            raise RuntimeError("motion audit reviewer attestation is missing, stale, or incomplete")
    reviewer = attestation.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise RuntimeError("motion audit reviewer attestation has no named reviewer")
    try:
        signed_at = datetime.fromisoformat(
            str(attestation.get("signed_at", "")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise RuntimeError("motion audit reviewer attestation timestamp is invalid") from exc
    if signed_at.tzinfo is None:
        raise RuntimeError("motion audit reviewer attestation timestamp has no timezone")
    nonce = str(payload.get("verification_nonce", ""))
    if len(nonce) != 32 or any(character not in "0123456789abcdef" for character in nonce):
        raise RuntimeError("motion audit receipt has no valid verification nonce")

    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise RuntimeError("motion audit receipt has no case results")
    required = set(required_audit_cases(bank))
    phases = required_phases(root)
    actual = set(cases)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise RuntimeError(
            f"motion audit case matrix is incomplete; missing={missing}, extra={extra}"
        )
    for case_id, result in cases.items():
        if result.get("blind_recognition") != "pass":
            raise RuntimeError(f"{case_id}: blind recognition did not pass")
        if result.get("blind_direction") != "pass":
            raise RuntimeError(f"{case_id}: blind direction recognition did not pass")
        if result.get("human_status") != "pass":
            raise RuntimeError(f"{case_id}: visual phase review did not pass")
        if result.get("machine_status") != "pass":
            raise RuntimeError(f"{case_id}: machine visual invariants did not pass")
        if result.get("normal_speed_playback") != "pass":
            raise RuntimeError(f"{case_id}: synchronized normal-speed playback did not pass")
        if result.get("source_edit_comparison") != "pass":
            raise RuntimeError(f"{case_id}: source/edit comparison did not pass")
        motion_id = case_id.split("@", 1)[0]
        if tuple(result.get("verified_phases") or ()) != phases[motion_id]:
            raise RuntimeError(f"{case_id}: not every required visual phase was reviewed")
        if not str(result.get("evidence", "")).strip():
            raise RuntimeError(f"{case_id}: visual review has no evidence note")
    return payload
