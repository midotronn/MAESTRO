"""Build a blind, paired real-host visual audit for every named motion.

Each unlabeled edit is paired with the exact source choreography so the reviewer identifies what
the named edit added instead of accidentally scoring a jump, step, or turn already present in the
host. The review page locks and persists a guess before it can load the separate answer key.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import secrets
import shutil
import sys
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.dance.format import to_editor139  # noqa: E402
from agentlodge.dance.transition import (  # noqa: E402
    _matrix_to_axis_angle,
    _sixd_to_matrix,
)
from agentlodge.editor.motion_audit import (  # noqa: E402
    REVIEW_PROTOCOL_VERSION,
    REVIEWER_ATTESTATION_STATEMENT,
    audit_case_id,
    audit_variants,
    motion_fingerprint,
)
from agentlodge.editor.motion_bank import (  # noqa: E402
    MotionBank,
    _COUNTERFLOW_CLAP_JOINTS,
    _COUNTERFLOW_CLAP_TURN_DEGREES,
    _MIRROR_JOINTS,
    _fit_event_frame,
    _root_yaw_series,
    _yaw_rotate,
)
from scripts.build_motion_bank import build_motion  # noqa: E402
from server.fk import BODY_PARENTS, compute_poses, save_poses_npz  # noqa: E402


CONTRACTS = ROOT / "assets" / "motion_bank" / "visual_contracts.json"


def _new_audit_id(output: Path, seed: int) -> str:
    """Return a fresh browser-storage namespace for one generated audit."""
    return f"{output.name}-seed-{int(seed)}-{secrets.token_hex(8)}"


def _prepare_output(output: Path) -> None:
    """Remove only generated audit artifacts before reusing an output directory."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("videos", "phase_sheets"):
        path = output / name
        if path.is_dir():
            shutil.rmtree(path)
    for path in output.glob("*_frames"):
        if path.is_dir():
            shutil.rmtree(path)
    for pattern in (
        "take_*.npy",
        "take_*_front.npz",
        "take_*_side.npz",
        "take_*_front_ybot.npz",
        "take_*_side_ybot.npz",
        "take_*_front.log",
        "take_*_side.log",
        "control_*.npy",
        "control_*_front.npz",
        "control_*_side.npz",
        "control_*_front_ybot.npz",
        "control_*_side_ybot.npz",
        "control_*_front.log",
        "control_*_side.log",
    ):
        for path in output.glob(pattern):
            if path.is_file():
                path.unlink()
    for name in (
        "answer_key.json",
        "render_receipt.json",
        "ybot_metrics_report.json",
        "review.html",
        "review.json",
    ):
        path = output / name
        if path.is_file():
            path.unlink()


def _load_vector(path: Path | None, fallback: np.ndarray) -> np.ndarray:
    if path is None:
        return fallback
    values = np.asarray(np.load(path)).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values


def _paired_views(
    edited: np.ndarray,
    reference: np.ndarray,
    action_start: int,
    *,
    normalize_facing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return matched front/side edit and reference views.

    A faceless audit rig makes forward and backward impossible to score reliably when every host
    window arrives at an arbitrary world heading. Semantic audits therefore rotate both clips by
    the same amount so root yaw is zero when the action starts. Production-camera audits can opt
    out and retain the song's original heading.
    """
    action_start = int(np.clip(action_start, 0, len(edited) - 1))
    front = np.ascontiguousarray(edited, dtype=np.float32)
    control = np.ascontiguousarray(reference, dtype=np.float32)
    heading = float(_root_yaw_series(front[action_start:action_start + 1])[0])
    if normalize_facing:
        front = _yaw_rotate(front.copy(), -heading, front[action_start, :3].copy())
        control = _yaw_rotate(
            control.copy(),
            -heading,
            control[action_start, :3].copy(),
        )
    side = _yaw_rotate(front.copy(), np.pi / 2.0, front[action_start, :3].copy())
    control_side = _yaw_rotate(
        control.copy(),
        np.pi / 2.0,
        control[action_start, :3].copy(),
    )
    return front, side, control, control_side, heading


def _global_joint_rotations(motion: np.ndarray) -> np.ndarray:
    local = _sixd_to_matrix(np.asarray(motion)[:, 3:135].reshape(-1, 22, 6))
    global_r = np.empty_like(local)
    global_r[:, 0] = local[:, 0]
    for joint in range(1, 22):
        global_r[:, joint] = global_r[:, BODY_PARENTS[joint]] @ local[:, joint]
    return global_r


def _clap_contact_lateral(edited: np.ndarray, event: int) -> float:
    joints = compute_poses(edited)["fk_joints"]
    root_yaw = float(_root_yaw_series(edited[event:event + 1])[0])
    local_left = np.array([np.cos(root_yaw), np.sin(root_yaw)], dtype=np.float32)
    hand_center = 0.5 * (joints[event, 20] + joints[event, 21]) - joints[event, 0]
    return float(hand_center[:2] @ local_left)


def _audit_auto_direction(base: np.ndarray, spec) -> str:
    """Independently derive the direction that an automatic edit should follow."""
    clip = np.asarray(base, dtype=np.float32)
    n = int(clip.shape[0])
    flow = "forward"
    if n >= 4:
        width = int(np.clip(n // 5, 2, 18))
        delta = np.mean(clip[-width:, :2], axis=0) - np.mean(clip[:width, :2], axis=0)
        yaw_series = _root_yaw_series(clip)
        center = yaw_series[
            max(0, n // 2 - width // 2):min(n, n // 2 + width // 2 + 1)
        ]
        yaw = float(np.median(center))
        local_left = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        local_forward = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        lateral = float(delta @ local_left)
        forward = float(delta @ local_forward)
        if abs(lateral) >= 0.04 and abs(lateral) >= 0.45 * abs(forward):
            flow = "left" if lateral > 0.0 else "right"
        else:
            yaw_change = float(yaw_series[-1] - yaw_series[0])
            if abs(yaw_change) >= 0.25:
                flow = "left" if yaw_change > 0.0 else "right"
    if flow in spec.directions:
        return flow
    if "forward" in spec.directions:
        return "forward"
    return spec.canonical_direction


def _declared_clap_contacts(spec, report: dict, gaps: np.ndarray) -> tuple[int, ...]:
    start, end = map(int, report["action_range"])
    local_event = int(report["event_frame"]) - start
    runs: list[list[int]] = []
    for frame in spec.intensity_lock_frames:
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    contacts = []
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
        contacts.append(min(mapped, key=lambda frame: float(gaps[frame])))
    return tuple(contacts)


def _true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return inclusive contiguous runs from a one-dimensional boolean mask."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False])
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _dancer_axes(
    motion: np.ndarray,
    joints: np.ndarray,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dancer-local left and forward axes in the world horizontal plane."""
    frame = int(np.clip(frame, 0, len(motion) - 1))
    toe = (joints[frame, 10] - joints[frame, 7]) + (
        joints[frame, 11] - joints[frame, 8]
    )
    forward = np.asarray(toe[:2], dtype=np.float32)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        yaw = float(_root_yaw_series(motion[frame:frame + 1])[0])
        forward = np.array([np.sin(yaw), -np.cos(yaw)], dtype=np.float32)
    else:
        forward /= norm
    left = np.array([-forward[1], forward[0]], dtype=np.float32)
    return left, forward


def _machine_checks(
    host: np.ndarray,
    edited: np.ndarray,
    spec,
    report: dict,
) -> tuple[list[dict], str]:
    """Evaluate visual invariants that previously passed generic motion metrics."""
    requested_direction = report.get("direction_request")
    resolved_direction = report.get("direction")
    natural_direction = (
        _audit_auto_direction(host, spec)
        if spec.directions
        else None
    )
    expected_counterflow_turn = 0.0
    expected_counterflow_joints: tuple[int, ...] = ()
    if (
        spec.id.startswith("clap_")
        and requested_direction in {"left", "right"}
        and resolved_direction == requested_direction
        and requested_direction != natural_direction
    ):
        expected_counterflow_turn = (
            _COUNTERFLOW_CLAP_TURN_DEGREES
            if requested_direction == "left"
            else -_COUNTERFLOW_CLAP_TURN_DEGREES
        )
        expected_counterflow_joints = _COUNTERFLOW_CLAP_JOINTS

    checks = [{
        "name": "semantic_validator",
        "passed": bool(report["validation"]["ok"]),
        "detail": report["validation"]["detail"],
    }]
    owned = set(spec.absolute_joints) | set(spec.additive_joints)
    owned.update(expected_counterflow_joints)
    if (
        spec.direction_mode == "mirror"
        and resolved_direction != spec.canonical_direction
    ):
        owned = {_MIRROR_JOINTS[joint] for joint in owned}
    unchanged = True
    max_unowned_drift = 0.0
    for joint in set(range(22)) - owned:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        drift = float(np.max(np.abs(edited[:, channels] - host[:, channels])))
        max_unowned_drift = max(max_unowned_drift, drift)
        unchanged = unchanged and drift <= 5e-7
    owned_axes = set(spec.translation_axes)
    if owned_axes.intersection({0, 1}):
        # Authored horizontal travel is dancer-local and rotates into both world-plane axes.
        owned_axes.update({0, 1})
    for axis in set(range(3)) - owned_axes:
        drift = float(np.max(np.abs(edited[:, axis] - host[:, axis])))
        max_unowned_drift = max(max_unowned_drift, drift)
        unchanged = unchanged and drift <= 5e-7
    if not spec.replace_contacts:
        drift = float(np.max(np.abs(edited[:, 135:139] - host[:, 135:139])))
        max_unowned_drift = max(max_unowned_drift, drift)
        unchanged = unchanged and drift <= 5e-7
    checks.append({
        "name": "declared_channel_ownership",
        "passed": bool(unchanged),
        "detail": (
            "all unowned pose, root, and contact channels remain source-identical "
            f"within float32 round-trip tolerance; max drift {max_unowned_drift:.2e}"
        ),
    })

    if spec.directions:
        expected = (
            natural_direction
            if requested_direction == "auto"
            else requested_direction
        )
        valid = resolved_direction in spec.directions and resolved_direction == expected
        checks.append({
            "name": "direction_resolution",
            "passed": bool(valid),
            "detail": (
                f"requested {requested_direction!r}, independently expected {expected!r}, "
                f"resolved {resolved_direction!r}"
            ),
        })

    if spec.minimum_frames > 1:
        action_frames = int(report["action_range"][1] - report["action_range"][0])
        checks.append({
            "name": "readable_action_duration",
            "passed": action_frames >= spec.minimum_frames,
            "detail": (
                f"{action_frames} action frames; minimum readable duration "
                f"{spec.minimum_frames} frames"
            ),
        })

    if spec.id == "side_step":
        start, end = map(int, report["action_range"])
        event = int(report["event_frame"])
        joints = compute_poses(edited)["fk_joints"]
        left_axis, _forward_axis = _dancer_axes(edited, joints, start)
        direction_sign = 1.0 if resolved_direction == "left" else -1.0
        root_lateral = (edited[start:end, :2] - edited[start, :2]) @ left_axis
        signed_root = float(np.max(direction_sign * root_lateral))
        lead_wrist = 20 if direction_sign > 0.0 else 21
        trail_wrist = 21 if direction_sign > 0.0 else 20
        lead_lateral = direction_sign * float(
            (joints[event, lead_wrist, :2] - joints[event, 9, :2]) @ left_axis
        )
        trail_lateral = direction_sign * float(
            (joints[event, trail_wrist, :2] - joints[event, 9, :2]) @ left_axis
        )
        stance_width = abs(float(
            (joints[event, 7, :2] - joints[event, 8, :2]) @ left_axis
        ))
        direction_passed = (
            resolved_direction in {"left", "right"}
            and signed_root > 0.24
            and lead_lateral > 0.45
            and lead_lateral - trail_lateral > 0.35
            and stance_width > 0.40
        )
        checks.append({
            "name": "side_step_direction_signature",
            "passed": bool(direction_passed),
            "detail": (
                f"{resolved_direction} signed root travel {signed_root:.4f} m, "
                f"lead/trailing hand offsets {lead_lateral:.4f}/{trail_lateral:.4f} m, "
                f"stance width {stance_width:.4f} m"
            ),
        })

    if spec.id == "step_touch":
        start, end = map(int, report["action_range"])
        event = int(report["event_frame"])
        joints = compute_poses(edited)["fk_joints"]
        _left_axis, forward_axis = _dancer_axes(edited, joints, start)
        trailing = 8 if resolved_direction == "left" else 7
        event_gap = float(np.linalg.norm(
            joints[event, 7, :2] - joints[event, 8, :2]
        ))
        pre_touch = joints[start + 5:max(start + 6, event - 4)]
        open_gap = float(np.max(np.linalg.norm(
            pre_touch[:, 7, :2] - pre_touch[:, 8, :2],
            axis=1,
        )))
        foot_delta = joints[event, 7, :2] - joints[event, 8, :2]
        sagittal_stagger = abs(float(foot_delta @ forward_axis))
        trailing_clearance = float(
            np.max(joints[start:event, trailing, 2]) - joints[event, trailing, 2]
        )
        touch_passed = (
            resolved_direction in {"left", "right"}
            and open_gap > 0.35
            and event_gap < 0.14
            and sagittal_stagger > 0.06
            and trailing_clearance > 0.08
        )
        checks.append({
            "name": "step_touch_phase_signature",
            "passed": bool(touch_passed),
            "detail": (
                f"open/touch ankle gaps {open_gap:.4f}/{event_gap:.4f} m, "
                f"sagittal touch stagger {sagittal_stagger:.4f} m, "
                f"trailing-foot clearance {trailing_clearance:.4f} m"
            ),
        })

    if spec.id in {"jump_two_foot", "jump_arms_up"}:
        start, end = map(int, report["action_range"])
        event = int(report["event_frame"])
        joints = compute_poses(edited)["fk_joints"]
        host_joints = compute_poses(host)["fk_joints"]
        grounded = np.asarray(edited[start:end, 135:139]).sum(axis=1) > 0.5
        airborne = np.flatnonzero(~grounded)
        if airborne.size:
            takeoff = int(airborne[0])
            landing = int(airborne[-1])
            air_frames = int(airborne.size)
        else:
            takeoff = landing = air_frames = 0
        event_lift = float(
            np.min(joints[event, (7, 8, 10, 11), 2])
            - np.min(host_joints[event, (7, 8, 10, 11), 2])
        )
        phase_passed = (
            airborne.size > 0
            and takeoff >= 8
            and end - start - 1 - landing >= 8
            and air_frames >= 12
            and event_lift > 0.18
        )
        checks.append({
            "name": "readable_jump_phases",
            "passed": bool(phase_passed),
            "detail": (
                f"takeoff at local frame {takeoff}, airborne through {landing} "
                f"({air_frames} frames), event foot lift {event_lift:.4f} m"
            ),
        })
        wrist_gap = float(np.linalg.norm(joints[event, 20] - joints[event, 21]))
        hands_above_head = float(
            min(joints[event, 20, 2], joints[event, 21, 2]) - joints[event, 15, 2]
        )
        if spec.id == "jump_arms_up":
            arm_passed = wrist_gap > 0.65 and hands_above_head > 0.15
            arm_detail = (
                f"airborne V wrist separation {wrist_gap:.4f} m, "
                f"lower hand {hands_above_head:.4f} m above head"
            )
        else:
            wrist_channels = slice(3 + 6 * 20, 3 + 6 * 22)
            wrist_drift = float(np.max(np.abs(
                edited[:, wrist_channels] - host[:, wrist_channels]
            )))
            arm_passed = wrist_drift <= 5e-7
            arm_detail = (
                f"plain jump preserves host wrist channels with maximum drift "
                f"{wrist_drift:.2e}"
            )
        checks.append({
            "name": "jump_arm_signature",
            "passed": bool(arm_passed),
            "detail": arm_detail,
        })

    if spec.id == "arm_punch":
        start, end = map(int, report["action_range"])
        joints = compute_poses(edited[start:end])["fk_joints"]
        if resolved_direction == "left":
            punch_wrist, punch_shoulder = 20, 16
            guard_wrist, guard_shoulder = 21, 17
        else:
            punch_wrist, punch_shoulder = 21, 17
            guard_wrist, guard_shoulder = 20, 16
        extension = np.linalg.norm(
            joints[:, punch_wrist] - joints[:, punch_shoulder],
            axis=1,
        )
        peak = int(np.argmax(extension))
        guard_length = float(np.linalg.norm(
            joints[peak, guard_wrist] - joints[peak, guard_shoulder]
        ))
        guard_chest = float(np.linalg.norm(
            joints[peak, guard_wrist] - joints[peak, 9]
        ))
        before = float(np.min(extension[:peak])) if peak > 0 else float("inf")
        after = (
            float(np.min(extension[peak + 1:]))
            if peak + 1 < len(extension)
            else float("inf")
        )
        punch_passed = (
            resolved_direction in {"left", "right"}
            and 2 < peak < len(extension) - 3
            and float(extension[peak] - before) > 0.25
            and float(extension[peak] - after) > 0.25
            and float(extension[peak] - guard_length) > 0.20
            and guard_chest < 0.32
        )
        checks.append({
            "name": "guard_strike_recoil_signature",
            "passed": bool(punch_passed),
            "detail": (
                f"strike peak at local frame {peak}, arm length "
                f"{float(extension[peak]):.4f} m versus pre/post "
                f"{before:.4f}/{after:.4f} m; guard arm/chest distances "
                f"{guard_length:.4f}/{guard_chest:.4f} m"
            ),
        })

    if spec.id == "body_roll":
        start, end = map(int, report["action_range"])
        host_local = _sixd_to_matrix(
            host[start:end, 3:135].reshape(-1, 22, 6)
        )
        edit_local = _sixd_to_matrix(
            edited[start:end, 3:135].reshape(-1, 22, 6)
        )
        local_delta = np.swapaxes(host_local, -1, -2) @ edit_local
        delta_aa = _matrix_to_axis_angle(local_delta)
        signals = [delta_aa[:, joint, 0] for joint in (3, 6, 9)]
        crest_frames = [int(np.argmax(signal)) for signal in signals]
        release_frames = [int(np.argmin(signal)) for signal in signals]
        roll_passed = (
            crest_frames[0] + 4 <= crest_frames[1]
            and crest_frames[1] + 4 <= crest_frames[2]
            and all(
                float(signal[crest]) > 0.25
                and float(signal[release]) < -0.12
                and release >= crest + 5
                for signal, crest, release in zip(
                    signals,
                    crest_frames,
                    release_frames,
                )
            )
        )
        checks.append({
            "name": "sequential_body_roll_signature",
            "passed": bool(roll_passed),
            "detail": (
                f"lower/mid/upper crest frames {crest_frames}, release frames "
                f"{release_frames}, crest amplitudes "
                f"{[round(float(signal[crest]), 4) for signal, crest in zip(signals, crest_frames)]}"
            ),
        })

    if spec.id == "chest_pop":
        event = int(report["event_frame"])
        joints = compute_poses(edited)["fk_joints"]
        host_joints = compute_poses(host)["fk_joints"]
        _left_axis, forward_axis = _dancer_axes(host, host_joints, event)

        def relative_delta(joint: int) -> np.ndarray:
            return (
                (joints[event, joint] - joints[event, 0])
                - (host_joints[event, joint] - host_joints[event, 0])
            )

        chest_delta = relative_delta(9)
        head_delta = relative_delta(15)
        chest_forward = float(chest_delta[:2] @ forward_axis)
        head_forward = float(head_delta[:2] @ forward_axis)
        local_host = _sixd_to_matrix(host[event, 3:135].reshape(22, 6))
        local_edit = _sixd_to_matrix(edited[event, 3:135].reshape(22, 6))
        local_delta = local_edit @ np.swapaxes(local_host, -1, -2)
        knee_delta = float(np.max(
            np.linalg.norm(_matrix_to_axis_angle(local_delta[[4, 5]]), axis=-1)
        ))
        start, end = map(int, report["action_range"])
        knee_channels = slice(3 + 6 * 4, 3 + 6 * 6)
        knee_channel_drift = float(np.max(np.abs(
            edited[start:end, knee_channels] - host[start:end, knee_channels]
        )))
        isolation_passed = (
            chest_forward > 0.08
            and abs(head_forward) < 0.65 * chest_forward
            and float(head_delta[2]) > -0.07
            and knee_delta < 0.20
            and knee_channel_drift <= 5e-7
        )
        checks.append({
            "name": "chest_pop_isolation",
            "passed": bool(isolation_passed),
            "detail": (
                f"chest/head forward deltas {chest_forward:.4f}/{head_forward:.4f} m, "
                f"head vertical delta {float(head_delta[2]):.4f} m, "
                f"maximum knee rotation delta {knee_delta:.4f} rad, "
                f"source knee-channel drift {knee_channel_drift:.2e}"
            ),
        })

    if spec.id == "rise_reach":
        start, end = map(int, report["action_range"])
        event = int(report["event_frame"])
        joints = compute_poses(edited)["fk_joints"]
        host_joints = compute_poses(host)["fk_joints"]
        _left_axis, forward_axis = _dancer_axes(edited, joints, event)
        chest = joints[event, 9]
        reach = [
            float((joints[event, hand, :2] - chest[:2]) @ forward_axis)
            for hand in (20, 21)
        ]
        lift = [
            float(joints[event, hand, 2] - chest[2])
            for hand in (20, 21)
        ]
        pelvis_delta = joints[start:end, 0, 2] - host_joints[start:end, 0, 2]
        contacts = np.asarray(edited[start:end, 135:139]).sum(axis=1)
        lead_advantage = abs(reach[0] - reach[1])
        rise_passed = (
            float(np.min(pelvis_delta)) < -0.18
            and float(pelvis_delta[event - start]) > -0.05
            and float(np.min(contacts)) >= 4.0
            and min(reach) > 0.08
            and lead_advantage > 0.08
            and min(lift) > 0.35
        )
        checks.append({
            "name": "planted_rise_reach_signature",
            "passed": bool(rise_passed),
            "detail": (
                f"pelvis load/recovery {float(np.min(pelvis_delta)):.4f}/"
                f"{float(pelvis_delta[event - start]):.4f} m, "
                f"hand reach {reach[0]:.4f}/{reach[1]:.4f} m, "
                f"lift {lift[0]:.4f}/{lift[1]:.4f} m, "
                f"minimum contact sum {float(np.min(contacts)):.1f}"
            ),
        })

    if spec.id.startswith("clap_"):
        try:
            reported_counterflow_turn = float(
                report.get("counterflow_turn_degrees", 0.0)
            )
            reported_counterflow_joints = tuple(
                int(joint)
                for joint in report.get("counterflow_turn_joints", ())
            )
        except (TypeError, ValueError):
            reported_counterflow_turn = float("nan")
            reported_counterflow_joints = ()
        declaration_passed = (
            report.get("natural_direction") == natural_direction
            and abs(reported_counterflow_turn - expected_counterflow_turn) < 1e-6
            and reported_counterflow_joints == expected_counterflow_joints
        )
        checks.append({
            "name": "counterflow_declaration",
            "passed": bool(declaration_passed),
            "detail": (
                f"independently expected natural direction {natural_direction!r}, "
                f"turn {expected_counterflow_turn:.2f} degrees on joints "
                f"{list(expected_counterflow_joints)}; report declared "
                f"{report.get('natural_direction')!r}, "
                f"{reported_counterflow_turn:.2f} degrees on joints "
                f"{list(reported_counterflow_joints)}"
            ),
        })
        start, end = map(int, report["action_range"])
        joints = compute_poses(edited)["fk_joints"]
        host_joints = compute_poses(host)["fk_joints"]
        global_r = _global_joint_rotations(edited)
        host_global = _global_joint_rotations(host)
        gaps = np.linalg.norm(joints[:, 20] - joints[:, 21], axis=-1)
        contact_frames = _declared_clap_contacts(spec, report, gaps)
        palm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        left_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        contact_gaps = []
        bilateral_moves = []
        palm_dots = []
        finger_dots = []
        finger_ups = []
        for frame in contact_frames:
            left_palm = global_r[frame, 20] @ palm
            right_palm = global_r[frame, 21] @ palm
            left_finger = global_r[frame, 20] @ left_axis
            right_finger = global_r[frame, 21] @ right_axis
            contact_gaps.append(float(gaps[frame]))
            bilateral_moves.append([
                float(np.linalg.norm(
                    (joints[frame, wrist] - joints[frame, 0])
                    - (host_joints[frame, wrist] - host_joints[frame, 0])
                ))
                for wrist in (20, 21)
            ])
            palm_dots.append(float(left_palm @ right_palm))
            finger_dots.append(float(left_finger @ right_finger))
            finger_ups.append(float(0.5 * (left_finger[2] + right_finger[2])))
        host_yaw = np.arctan2(host_global[:, 9, 1, 0], host_global[:, 9, 0, 0])
        edit_yaw = np.arctan2(global_r[:, 9, 1, 0], global_r[:, 9, 0, 0])
        yaw_delta = np.arctan2(
            np.sin(edit_yaw - host_yaw),
            np.cos(edit_yaw - host_yaw),
        )
        peak_yaw = float(np.rad2deg(np.max(np.abs(yaw_delta[start:end]))))
        event_yaw = float(np.rad2deg(yaw_delta[int(report["event_frame"])]))
        if abs(expected_counterflow_turn) > 1e-6:
            facing_passed = (
                abs(event_yaw - expected_counterflow_turn) < 3.0
                and peak_yaw < abs(expected_counterflow_turn) + 3.0
            )
            facing_detail = (
                f"explicit counter-flow turn {event_yaw:.2f} degrees at contact "
                f"versus expected {expected_counterflow_turn:.2f}; "
                f"peak {peak_yaw:.2f} degrees"
            )
        else:
            facing_passed = peak_yaw < 3.0
            facing_detail = (
                f"peak chest-heading change versus source {peak_yaw:.2f} degrees"
            )
        checks.extend([
            {
                "name": "visible_palm_contact",
                "passed": bool(contact_frames) and all(gap < 0.01 for gap in contact_gaps),
                "detail": (
                    f"contact frames {list(contact_frames)}, wrist gaps "
                    f"{[round(gap, 4) for gap in contact_gaps]} m (all must be < 0.0100 m)"
                ),
            },
            {
                "name": "opposed_palm_planes",
                "passed": bool(contact_frames) and all(
                    palm_dot < -0.98 and finger_dot > 0.98
                    for palm_dot, finger_dot in zip(palm_dots, finger_dots)
                ),
                "detail": (
                    f"palm dots {[round(value, 4) for value in palm_dots]}, "
                    f"finger dots {[round(value, 4) for value in finger_dots]}"
                ),
            },
            {
                "name": "relaxed_hand_angle",
                "passed": bool(contact_frames) and all(
                    0.20 < finger_up < 0.75 for finger_up in finger_ups
                ),
                "detail": (
                    "average world-up finger components "
                    f"{[round(value, 4) for value in finger_ups]}"
                ),
            },
            {
                "name": "host_facing_continuity",
                "passed": facing_passed,
                "detail": facing_detail,
            },
        ])
        action_gaps = gaps[start:end]
        closure_mask = action_gaps < 0.02
        # A one-frame near-contact excursion is still one physical clap, not a second
        # contact. Bridge it only while the hands remain within 10 cm; repeated claps
        # still need a visibly wider separation between their distinct closure runs.
        for frame in range(1, len(closure_mask) - 1):
            if (
                not closure_mask[frame]
                and closure_mask[frame - 1]
                and closure_mask[frame + 1]
                and action_gaps[frame] < 0.10
            ):
                closure_mask[frame] = True
        closure_runs = _true_runs(closure_mask)
        expected_runs = 3 if spec.id == "clap_repeat" else 1
        timing_passed = (
            len(closure_runs) == expected_runs
            and all(end_frame - start_frame + 1 >= 3 for start_frame, end_frame in closure_runs)
        )
        if timing_passed and len(closure_runs) > 1:
            timing_passed = all(
                float(np.max(gaps[start + left_end + 1:start + right_start])) > 0.05
                for (_left_start, left_end), (right_start, _right_end)
                in zip(closure_runs, closure_runs[1:])
            )
        checks.extend([
            {
                "name": "readable_clap_contact_timing",
                "passed": bool(timing_passed),
                "detail": (
                    f"closure runs {list(closure_runs)} at < 0.0200 m; expected "
                    f"{expected_runs} run(s), each at least 3 frames with visible separation"
                ),
            },
            {
                "name": "bilateral_clap_approach",
                "passed": bool(bilateral_moves) and all(
                    min(movements) > 0.10 for movements in bilateral_moves
                ),
                "detail": (
                    "left/right wrist movement versus source at contact "
                    f"{[[round(value, 4) for value in pair] for pair in bilateral_moves]} m"
                ),
            },
        ])
        if spec.id == "clap_overhead":
            event = int(report["event_frame"])
            pre = max(start, event - 6)
            post = min(end - 1, event + 6)
            above_head = float(
                min(joints[event, 20, 2], joints[event, 21, 2])
                - joints[event, 15, 2]
            )
            local = _sixd_to_matrix(edited[start:end, 3:135].reshape(-1, 22, 6))
            wrist_step = 0.0
            for wrist in (20, 21):
                step = (
                    local[1:, wrist]
                    @ np.swapaxes(local[:-1, wrist], -1, -2)
                )
                wrist_step = max(
                    wrist_step,
                    float(np.max(np.linalg.norm(
                        _matrix_to_axis_angle(step),
                        axis=-1,
                    ))),
                )
            overhead_passed = (
                gaps[pre] > 0.10
                and gaps[post] > 0.10
                and above_head > 0.12
                and wrist_step < 0.80
            )
            checks.append({
                "name": "overhead_clap_approach_recoil",
                "passed": bool(overhead_passed),
                "detail": (
                    f"wrist gaps {float(gaps[pre]):.4f}/{float(gaps[event]):.4f}/"
                    f"{float(gaps[post]):.4f} m before/contact/after, "
                    f"lower hand {above_head:.4f} m above head, "
                    f"maximum wrist step {wrist_step:.4f} rad"
                ),
            })
    return checks, ("pass" if all(check["passed"] for check in checks) else "fail")


def _append_clap_direction_matrix_checks(records: dict[str, dict[str, dict]]) -> None:
    """Require visible left/right separation from the same motion's forward contact."""
    for motion_id, directions in records.items():
        if not {"forward", "left", "right"}.issubset(directions):
            continue
        forward = float(directions["forward"]["lateral"])
        left = float(directions["left"]["lateral"])
        right = float(directions["right"]["lateral"])
        minimum_separation = 0.14
        passed = (
            left > forward + minimum_separation
            and right < forward - minimum_separation
        )
        detail = (
            f"local-left offsets: left={left:.4f}, forward={forward:.4f}, "
            f"right={right:.4f} m; each side must separate by "
            f"> {minimum_separation:.4f} m"
        )
        for direction in ("forward", "left", "right"):
            directions[direction]["checks"].append({
                "name": "directional_contact_matrix",
                "passed": bool(passed),
                "detail": f"{motion_id}: {detail}",
            })
        automatic = directions.get("auto")
        if automatic is not None:
            resolved = automatic["resolved"]
            reference = directions.get(resolved)
            auto_passed = (
                passed
                and reference is not None
                and abs(float(automatic["lateral"]) - float(reference["lateral"])) < 0.01
            )
            automatic["checks"].append({
                "name": "automatic_contact_direction",
                "passed": bool(auto_passed),
                "detail": (
                    f"{motion_id}: auto resolved {resolved!r} at "
                    f"{float(automatic['lateral']):.4f} m local-left; "
                    f"explicit {resolved!r} is "
                    f"{float(reference['lateral']):.4f} m"
                    if reference is not None
                    else f"{motion_id}: auto resolved unsupported direction {resolved!r}"
                ),
            })


def _review_html(
    takes: list[dict],
    rule: str,
    *,
    audit_id: str,
    normalized_facing: bool,
    motion_fingerprint_value: str = "",
) -> str:
    vocabulary = [(spec.id, spec.name) for spec in MotionBank().specs]
    vocabulary_options = "".join(
        f'<option value="{html.escape(motion_id)}">{html.escape(name)}</option>'
        for motion_id, name in vocabulary
    )
    vocabulary_items = "".join(
        f"<li><code>{html.escape(motion_id)}</code>: {html.escape(name)}</li>"
        for motion_id, name in vocabulary
    )
    cards = []
    for take in takes:
        key = take["take"]
        control = take["control"]
        cache_query = html.escape(
            urlencode({
                "audit_id": str(audit_id),
                "motion_fingerprint": str(motion_fingerprint_value),
                "take_id": str(key),
            }),
            quote=True,
        )
        cards.append(
            f"""
            <article class="take" data-take="{html.escape(key)}">
              <h2>{html.escape(key)}</h2>
              <div class="pair">
                <section>
                  <h3>Source choreography: front left, side right</h3>
                  <video class="control" controls muted playsinline preload="metadata"
                         src="videos/{html.escape(control)}.mp4?{cache_query}"></video>
                </section>
                <section>
                  <h3>Unlabeled edit: front left, side right</h3>
                  <video class="edited" controls muted playsinline preload="metadata"
                         src="videos/{html.escape(key)}.mp4?{cache_query}"></video>
                </section>
              </div>
              <button class="play-pair" type="button">Play both from the start</button>
              <p class="playback-status" aria-live="polite"></p>
              <p><a class="review-link"
                    href="phase_sheets/{html.escape(key)}_review.html?{cache_query}"
                    target="_blank">Open synchronized views and edit-minus-source strip</a></p>
              <label>What action did you see?
                <input class="guess" type="text" list="motion-vocabulary"
                       autocomplete="off" spellcheck="false">
              </label>
              <label>What dancer-relative direction did the added action face or travel?
                <select class="direction-guess">
                  <option value="">Choose a direction</option>
                  <option value="none">No distinct direction</option>
                  <option value="forward">Forward</option>
                  <option value="left">Left (dancer's left)</option>
                  <option value="right">Right (dancer's right)</option>
                </select>
              </label>
              <div class="actions">
                <button class="lock" type="button" disabled>Lock guess</button>
              </div>
              <p class="status" aria-live="polite"></p>
              <section class="answer" hidden>
                <p><strong class="answer-name"></strong> (<code class="answer-id"></code>)</p>
                <p class="direction-detail"></p>
                <p>Must read as: <span class="recognizable"></span></p>
                <ul class="phases"></ul>
                <p><strong>Machine visual invariants</strong></p>
                <ul class="machine-checks"></ul>
                <section class="phase-review">
                  <label>Final visual verdict
                    <select class="visual-verdict">
                      <option value="">Choose pass or fail</option>
                      <option value="pass">Pass</option>
                      <option value="fail">Fail</option>
                    </select>
                  </label>
                  <label>Required evidence note
                    <textarea class="visual-evidence" rows="3"
                              placeholder="Describe contact, hand angle, host continuity, and required phases."></textarea>
                  </label>
                </section>
              </section>
            </article>
            """
        )
    facing_note = (
        "<p>The views are normalized to the dancer's heading at action start. "
        "In the front view, dancer-left appears on screen right and dancer-right appears on "
        "screen left. In the side view, the dancer faces screen right, so forward is screen "
        "right and backward is screen left.</p>"
        if normalized_facing
        else (
            "<p>The views preserve the production heading from the source choreography. "
            "All left/right choices remain relative to the dancer, never the viewer.</p>"
        )
    )
    audit_json = json.dumps(str(audit_id))
    take_ids_json = json.dumps([take["take"] for take in takes])
    normalized_json = "true" if normalized_facing else "false"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MAESTRO blind motion audit</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: system-ui, sans-serif;
            color: #eee; background: #17171b; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(640px, 1fr));
            gap: 20px; }}
    .instructions {{ max-width: 1000px; margin: 0 auto 24px; line-height: 1.5; }}
    .take {{ background: #26262d; border: 1px solid #444; border-radius: 12px;
             padding: 16px; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    h3 {{ margin: 0 0 8px; font-size: 0.95rem; color: #ccc; }}
    video {{ width: 100%; background: #000; }}
    label {{ display: block; margin: 12px 0; }}
    input, select, textarea {{ box-sizing: border-box; display: block; width: 100%; margin-top: 6px;
             padding: 10px; color: #fff; background: #111; border: 1px solid #666; }}
    input[type="checkbox"] {{ display: inline; width: auto; margin-right: 8px; }}
    button {{ margin-right: 8px; padding: 8px 12px; cursor: pointer; }}
    button:disabled {{ cursor: not-allowed; opacity: 0.55; }}
    .status {{ min-height: 1.5em; color: #d8ccff; }}
    .answer {{ margin-top: 12px; padding: 12px; border: 1px solid #6555a5;
               border-radius: 8px; background: #1c1928; }}
    code {{ color: #d8ccff; }}
    @media (max-width: 760px) {{
      main {{ grid-template-columns: 1fr; }}
      .pair {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <section class="instructions">
    <h1>MAESTRO blind motion audit</h1>
    <p><strong>Pass rule:</strong> {html.escape(rule)}</p>
    <p>Compare the source choreography with the edited take, then name only the action that was
       added. The answer key is not loaded until every take has a non-empty locked guess.</p>
    <p><strong>Direction scoring:</strong> use the dancer's observed direction, never the camera's.
       For mirrored gestures choose the active arm or lead side; for claps choose the side where
       the hands meet or <em>forward</em> for a centered contact; for turns and lateral travel use
       dancer-relative rotation or travel. Forward/backward steps use <em>No distinct direction</em>
       because that direction is already part of the action name. An automatic edit is scored by
       the direction it visibly resolved to, not by the word "auto".</p>
    {facing_note}
    <details>
      <summary>Supported motion vocabulary ({len(vocabulary)})</summary>
      <ul>{vocabulary_items}</ul>
    </details>
    <datalist id="motion-vocabulary">{vocabulary_options}</datalist>
    <ol>
      <li>Use <strong>Play both from the start</strong>. The lock remains disabled until both
          videos finish together at 1.0x without pausing, seeking, or changing speed.</li>
      <li>Open the synchronized source/edit/difference review for every take before locking it.</li>
      <li>Record and lock both the action and its observed direction.</li>
      <li>If the action is unclear or guessed incorrectly, mark it failed.</li>
      <li>After revealing, inspect every required phase from both front and side.</li>
    </ol>
    <p id="playback-progress" class="status" aria-live="polite"></p>
    <label>Independent reviewer name
      <input id="reviewer-name" type="text" autocomplete="name"
             placeholder="Name of the person who performed this visual review">
    </label>
    <label>
      <input id="independent-review" type="checkbox">
      I independently reviewed every source/edit pair and did not inherit another reviewer's
      guesses or verdicts.
    </label>
    <label>
      <input id="answers-hidden" type="checkbox">
      I confirm the answer key stayed hidden until every action and direction guess was locked.
    </label>
    <button id="export" type="button" disabled>Export blocking review result</button>
    <button id="reveal-all" type="button" disabled>Lock every guess before reveal</button>
    <span id="reveal-status" class="status" aria-live="polite"></span>
  </section>
  <main>
    {''.join(cards)}
  </main>
  <script>
    const auditId = {audit_json};
    const takeIds = {take_ids_json};
    const normalizedFacing = {normalized_json};
    const motionFingerprint = {json.dumps(str(motion_fingerprint_value))};
    const reviewerStatement = {json.dumps(REVIEWER_ATTESTATION_STATEMENT)};
    const storageKey = `maestro-motion-audit:${{auditId}}`;
    let state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    let answersPromise = null;
    const comparisonWindows = new Map();

    function save() {{
      localStorage.setItem(storageKey, JSON.stringify(state));
    }}

    function normalize(value) {{
      return String(value || "").toLowerCase().replace(/[_-]+/g, " ")
        .replace(/[^a-z0-9 ]+/g, "").replace(/\\s+/g, " ").trim();
    }}

    function normalizeDirection(value) {{
      const normalized = normalize(value);
      if (["none", "no direction", "not directional", "non directional"].includes(normalized)) {{
        return "none";
      }}
      return normalized;
    }}

    function loadAnswers() {{
      if (!answersPromise) {{
        answersPromise = fetch("answer_key.json", {{cache: "no-store"}}).then(response => {{
          if (!response.ok) throw new Error(`answer key request failed: ${{response.status}}`);
          return response.json();
        }});
      }}
      return answersPromise;
    }}

    function playbackComplete(take) {{
      return state[take]
        && state[take].normalSpeedPlayback
        && state[take].normalSpeedPlayback.completed === true;
    }}

    function comparisonOpened(take) {{
      const saved = state[take] || {{}};
      const acknowledgment = saved.comparisonAcknowledgment || {{}};
      return Boolean(
        saved.comparisonOpenedAt
        && acknowledgment.auditId === auditId
        && acknowledgment.motionFingerprint === motionFingerprint
        && acknowledgment.takeId === take
      );
    }}

    function validComparisonAcknowledgment(event, take) {{
      const acknowledgment = event.data || {{}};
      return (
        event.origin === window.location.origin
        && event.source === comparisonWindows.get(take)
        && acknowledgment.type === "maestro-motion-audit-comparison-ready"
        && acknowledgment.auditId === auditId
        && acknowledgment.motionFingerprint === motionFingerprint
        && acknowledgment.takeId === take
      );
    }}

    function takeReadyToLock(take) {{
      return playbackComplete(take) && comparisonOpened(take);
    }}

    function allTakePlaybackComplete() {{
      return takeIds.length > 0 && takeIds.every(playbackComplete);
    }}

    function allTakeComparisonsOpened() {{
      return takeIds.length > 0 && takeIds.every(comparisonOpened);
    }}

    function allGuessesLocked() {{
      return takeIds.length > 0 && takeIds.every(take =>
        takeReadyToLock(take) && state[take] && String(state[take].guess || "").trim()
        && String(state[take].directionGuess || "").trim() && state[take].lockedAt
      );
    }}

    function allAnswersRevealed() {{
      return takeIds.length > 0 && takeIds.every(take =>
        state[take] && state[take].revealedAt
      );
    }}

    function allVisualReviewsComplete() {{
      return takeIds.length > 0 && takeIds.every(take =>
        state[take] && ["pass", "fail"].includes(state[take].visualStatus)
        && String(state[take].visualEvidence || "").trim()
        && Array.isArray(state[take].requiredPhases)
        && state[take].requiredPhases.length > 0
        && state[take].requiredPhases.every(phase =>
          (state[take].verifiedPhases || []).includes(phase)
        )
      );
    }}

    function reviewerAttestationComplete() {{
      const reviewer = state.__reviewer || {{}};
      return Boolean(
        String(reviewer.name || "").trim()
        && reviewer.independentVisualReview === true
        && reviewer.answersHiddenUntilLock === true
      );
    }}

    function refreshExportAvailability() {{
      document.querySelector("#export").disabled = !(
        allAnswersRevealed()
        && allVisualReviewsComplete()
        && allTakePlaybackComplete()
        && allTakeComparisonsOpened()
        && reviewerAttestationComplete()
      );
    }}

    function refreshPlaybackProgress() {{
      const played = takeIds.filter(playbackComplete).length;
      const compared = takeIds.filter(comparisonOpened).length;
      document.querySelector("#playback-progress").textContent =
        `Verified playback: ${{played}}/${{takeIds.length}}; `
        + `synchronized comparison opened: ${{compared}}/${{takeIds.length}}.`;
    }}

    function refreshCardReadiness(card) {{
      const take = card.dataset.take;
      const saved = state[take] || {{}};
      const lock = card.querySelector(".lock");
      const playbackStatus = card.querySelector(".playback-status");
      const played = playbackComplete(take);
      const compared = comparisonOpened(take);
      if (saved.lockedAt) {{
        lock.disabled = true;
      }} else {{
        lock.disabled = !(played && compared);
      }}
      if (played && compared) {{
        playbackStatus.textContent =
          "Ready to lock: full synchronized playback and comparison review completed.";
      }} else if (played) {{
        playbackStatus.textContent =
          "Playback complete. Open the synchronized comparison before locking.";
      }} else if (saved.normalSpeedPlayback && saved.normalSpeedPlayback.reason) {{
        playbackStatus.textContent =
          `Playback incomplete: ${{saved.normalSpeedPlayback.reason}}`;
      }} else if (compared) {{
        playbackStatus.textContent =
          "Comparison opened. Complete uninterrupted synchronized playback before locking.";
      }} else {{
        playbackStatus.textContent =
          "Lock blocked until uninterrupted 1.0x playback and comparison review are complete.";
      }}
    }}

    function refreshRevealAvailability() {{
      const revealAll = document.querySelector("#reveal-all");
      const remaining = takeIds.filter(take =>
        !(takeReadyToLock(take) && state[take] && String(state[take].guess || "").trim()
          && String(state[take].directionGuess || "").trim() && state[take].lockedAt)
      ).length;
      revealAll.disabled = !allGuessesLocked() || allAnswersRevealed();
      if (allAnswersRevealed()) {{
        revealAll.textContent = "Answers revealed";
      }} else if (remaining === 0) {{
        revealAll.textContent = "Reveal all answers";
      }} else {{
        revealAll.textContent =
          `Lock ${{remaining}} remaining guess${{remaining === 1 ? "" : "es"}} before reveal`;
      }}
    }}

    function restore(card) {{
      const take = card.dataset.take;
      const saved = state[take];
      const input = card.querySelector(".guess");
      const direction = card.querySelector(".direction-guess");
      if (saved) {{
        input.value = saved.guess || "";
        direction.value = saved.directionGuess || "";
        if (saved.lockedAt) {{
          input.disabled = true;
          direction.disabled = true;
          card.querySelector(".status").textContent = saved.revealedAt
            ? `Locked: "${{saved.guess}}" - answer already revealed`
            : `Locked: "${{saved.guess}}"`;
        }}
      }}
      refreshCardReadiness(card);
    }}

    function renderAnswer(card, answer) {{
      const take = card.dataset.take;
      if (!answer) throw new Error(`missing answer for ${{take}}`);
      const accepted = [answer.id, answer.name, ...(answer.aliases || [])].map(normalize);
      const recognized = accepted.includes(normalize(state[take].guess));
      const expectedDirection = normalizeDirection(answer.resolved_direction || "none");
      const directionRecognized =
        normalizeDirection(state[take].directionGuess) === expectedDirection;
      state[take] = {{
        ...state[take],
        actual: answer.id,
        recognized,
        expectedDirection,
        directionRecognized,
        requiredPhases: [...answer.visual_contract.required_phases],
        verifiedPhases: state[take].verifiedPhases || [],
        revealedAt: state[take].revealedAt || new Date().toISOString(),
      }};
      card.querySelector(".answer-name").textContent = answer.name;
      card.querySelector(".answer-id").textContent = answer.id;
      card.querySelector(".recognizable").textContent =
        answer.visual_contract.recognizable_as;
      const phases = card.querySelector(".phases");
      phases.replaceChildren();
      answer.visual_contract.required_phases.forEach(phase => {{
        const item = document.createElement("li");
        const label = document.createElement("label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "phase-reviewed";
        checkbox.checked = state[take].verifiedPhases.includes(phase);
        checkbox.addEventListener("change", () => {{
          const reviewed = new Set(state[take].verifiedPhases || []);
          if (checkbox.checked) reviewed.add(phase);
          else reviewed.delete(phase);
          state[take] = {{...state[take], verifiedPhases: [...reviewed]}};
          save();
          refreshExportAvailability();
        }});
        label.append(checkbox, ` Reviewed: ${{phase}}`);
        item.appendChild(label);
        phases.appendChild(item);
      }});
      const direction = answer.requested_direction || "not directional";
      card.querySelector(".direction-detail").textContent =
        `Requested direction: ${{direction}}; resolved direction: ${{answer.resolved_direction || "none"}}`;
      const machineChecks = card.querySelector(".machine-checks");
      machineChecks.replaceChildren();
      (answer.machine_checks || []).forEach(check => {{
        const item = document.createElement("li");
        item.textContent = `${{check.passed ? "PASS" : "FAIL"}} - ${{check.name}}: ${{check.detail}}`;
        machineChecks.appendChild(item);
      }});
      card.querySelector(".answer").hidden = false;
      card.querySelector(".visual-verdict").value = state[take].visualStatus || "";
      card.querySelector(".visual-evidence").value = state[take].visualEvidence || "";
      card.querySelector(".status").textContent =
        recognized && directionRecognized
          ? `PASS - action and direction matched ${{answer.id}} (${{expectedDirection}})`
          : `FAIL - guessed "${{state[take].guess}}" (${{state[take].directionGuess}}); `
            + `answer is ${{answer.id}} (${{expectedDirection}})`;
      save();
      refreshExportAvailability();
    }}

    window.addEventListener("message", event => {{
      const acknowledgment = event.data || {{}};
      const take = acknowledgment.takeId;
      if (!takeIds.includes(take) || !validComparisonAcknowledgment(event, take)) {{
        return;
      }}
      state[take] = {{
        ...(state[take] || {{}}),
        comparisonOpenedAt: new Date().toISOString(),
        comparisonAcknowledgment: {{
          auditId: acknowledgment.auditId,
          motionFingerprint: acknowledgment.motionFingerprint,
          takeId: acknowledgment.takeId,
        }},
      }};
      comparisonWindows.delete(take);
      save();
      const card = document.querySelector(`.take[data-take="${{take}}"]`);
      if (card) refreshCardReadiness(card);
      refreshPlaybackProgress();
      refreshRevealAvailability();
      refreshExportAvailability();
    }});

    document.querySelectorAll(".take").forEach(card => {{
      const take = card.dataset.take;
      const input = card.querySelector(".guess");
      const direction = card.querySelector(".direction-guess");
      const lock = card.querySelector(".lock");
      const status = card.querySelector(".status");
      const verdict = card.querySelector(".visual-verdict");
      const evidence = card.querySelector(".visual-evidence");
      const reviewLink = card.querySelector(".review-link");
      const videos = [...card.querySelectorAll("video")];
      const controlVideo = card.querySelector(".control");
      const editedVideo = card.querySelector(".edited");
      let playbackRun = null;

      function updateTake(patch) {{
        state[take] = {{...(state[take] || {{}}), ...patch}};
        save();
        refreshCardReadiness(card);
        refreshPlaybackProgress();
        refreshRevealAvailability();
        refreshExportAvailability();
      }}

      function invalidatePlayback(reason) {{
        if (!playbackRun || playbackRun.invalid) return;
        playbackRun.invalid = true;
        updateTake({{
          normalSpeedPlayback: {{
            completed: false,
            reason,
            started_at: playbackRun.startedAt,
            playback_rate: 1,
            seek_count: playbackRun.seekCount,
            pause_count: playbackRun.pauseCount,
            max_sync_drift: playbackRun.maxSyncDrift,
          }},
        }});
      }}

      function maybeCompletePlayback() {{
        if (!playbackRun || playbackRun.invalid || !videos.every(video => video.ended)) {{
          return;
        }}
        const run = playbackRun;
        if (run.maxSyncDrift > 0.12) {{
          invalidatePlayback(
            `Source/edit synchronization drifted by ${{run.maxSyncDrift.toFixed(3)}}s; `
              + "replay the pair.",
          );
          return;
        }}
        playbackRun = null;
        updateTake({{
          normalSpeedPlayback: {{
            completed: true,
            started_at: run.startedAt,
            completed_at: new Date().toISOString(),
            playback_rate: 1,
            seek_count: run.seekCount,
            pause_count: run.pauseCount,
            source_seconds: Number(controlVideo.currentTime.toFixed(3)),
            edit_seconds: Number(editedVideo.currentTime.toFixed(3)),
            elapsed_seconds: Number(((performance.now() - run.startedMs) / 1000).toFixed(3)),
            max_sync_drift: Number(run.maxSyncDrift.toFixed(3)),
          }},
        }});
      }}

      videos.forEach(video => {{
        video.addEventListener("timeupdate", () => {{
          if (!playbackRun || playbackRun.invalid) return;
          playbackRun.maxSyncDrift = Math.max(
            playbackRun.maxSyncDrift,
            Math.abs(controlVideo.currentTime - editedVideo.currentTime),
          );
        }});
        video.addEventListener("ratechange", () => {{
          if (playbackRun && Math.abs(video.playbackRate - 1) > 1e-6) {{
            invalidatePlayback("Playback speed changed; replay both videos at 1.0x.");
          }}
        }});
        video.addEventListener("seeking", () => {{
          if (playbackRun) {{
            playbackRun.seekCount += 1;
            invalidatePlayback("Seeking invalidated this run; replay both videos from the start.");
          }}
        }});
        video.addEventListener("pause", () => {{
          if (
            playbackRun
            && Number.isFinite(video.duration)
            && video.currentTime + 0.05 < video.duration
          ) {{
            playbackRun.pauseCount += 1;
            invalidatePlayback("Pausing invalidated this run; replay both videos uninterrupted.");
          }}
        }});
        video.addEventListener("ended", maybeCompletePlayback);
      }});

      function metadataReady(video) {{
        if (video.readyState >= 1) return Promise.resolve();
        return new Promise((resolve, reject) => {{
          video.addEventListener("loadedmetadata", resolve, {{once: true}});
          video.addEventListener(
            "error",
            () => reject(new Error("video metadata could not be loaded")),
            {{once: true}},
          );
        }});
      }}

      function rewind(video) {{
        video.pause();
        video.playbackRate = 1;
        if (video.currentTime <= 0.01) {{
          return Promise.resolve();
        }}
        return new Promise(resolve => {{
          video.addEventListener("seeked", resolve, {{once: true}});
          video.currentTime = 0;
        }});
      }}

      card.querySelector(".play-pair").addEventListener("click", async () => {{
        playbackRun = null;
        try {{
          await Promise.all(videos.map(metadataReady));
          await Promise.all(videos.map(rewind));
          const startedAt = new Date().toISOString();
          playbackRun = {{
            startedAt,
            startedMs: performance.now(),
            invalid: false,
            seekCount: 0,
            pauseCount: 0,
            maxSyncDrift: 0,
          }};
          updateTake({{
            normalSpeedPlayback: {{
              completed: false,
              reason: "Playback in progress.",
              started_at: startedAt,
              playback_rate: 1,
              seek_count: 0,
              pause_count: 0,
              max_sync_drift: 0,
            }},
          }});
          await Promise.all(videos.map(video => video.play()));
        }} catch (error) {{
          if (playbackRun) {{
            invalidatePlayback(`Playback could not start: ${{error.message}}`);
          }} else {{
            updateTake({{
              normalSpeedPlayback: {{
                completed: false,
                reason: `Playback could not start: ${{error.message}}`,
              }},
            }});
          }}
        }}
      }});

      reviewLink.addEventListener("click", event => {{
        event.preventDefault();
        const child = window.open(reviewLink.href, `maestro-motion-audit-${{take}}`);
        if (!child) {{
          status.textContent =
            "Comparison window was blocked. Allow pop-ups and open the comparison again.";
          return;
        }}
        comparisonWindows.set(take, child);
      }});

      lock.addEventListener("click", () => {{
        if (!takeReadyToLock(take)) {{
          status.textContent =
            "Complete uninterrupted paired playback and open the synchronized comparison first.";
          return;
        }}
        const guess = input.value.trim();
        if (!guess) {{
          status.textContent = "Enter a guess before locking.";
          input.focus();
          return;
        }}
        if (!direction.value) {{
          status.textContent = "Choose an observed direction before locking.";
          direction.focus();
          return;
        }}
        state[take] = {{
          ...(state[take] || {{}}),
          guess,
          directionGuess: direction.value,
          lockedAt: new Date().toISOString(),
        }};
        save();
        input.disabled = true;
        direction.disabled = true;
        lock.disabled = true;
        status.textContent = `Locked: "${{guess}}"`;
        refreshCardReadiness(card);
        refreshPlaybackProgress();
        refreshRevealAvailability();
        refreshExportAvailability();
      }});

      verdict.addEventListener("change", () => {{
        state[take] = {{...(state[take] || {{}}), visualStatus: verdict.value}};
        save();
        refreshExportAvailability();
      }});
      evidence.addEventListener("input", () => {{
        state[take] = {{...(state[take] || {{}}), visualEvidence: evidence.value}};
        save();
        refreshExportAvailability();
      }});
      restore(card);
    }});

    const revealAll = document.querySelector("#reveal-all");
    const revealStatus = document.querySelector("#reveal-status");
    revealAll.addEventListener("click", async () => {{
      if (!allGuessesLocked()) {{
        revealStatus.textContent = "Lock every guess before loading the answer key.";
        refreshRevealAvailability();
        return;
      }}
      revealAll.disabled = true;
      revealStatus.textContent = "Loading answer key...";
      try {{
        const answers = await loadAnswers();
        document.querySelectorAll(".take").forEach(card => {{
          renderAnswer(card, answers[card.dataset.take]);
        }});
        save();
        revealStatus.textContent = "All answers revealed. Inspect every required phase.";
        refreshRevealAvailability();
        refreshExportAvailability();
      }} catch (error) {{
        revealStatus.textContent = `Could not reveal answers: ${{error.message}}`;
        refreshRevealAvailability();
      }}
    }});

    refreshRevealAvailability();
    if (allAnswersRevealed()) {{
      loadAnswers().then(answers => {{
        document.querySelectorAll(".take").forEach(card => {{
          renderAnswer(card, answers[card.dataset.take]);
        }});
        revealStatus.textContent = "Previously revealed answers restored.";
        refreshExportAvailability();
      }}).catch(error => {{
        revealStatus.textContent = `Could not restore answers: ${{error.message}}`;
      }});
    }}

    document.querySelectorAll(".take").forEach(refreshCardReadiness);
    const reviewerName = document.querySelector("#reviewer-name");
    const independentReview = document.querySelector("#independent-review");
    const answersHidden = document.querySelector("#answers-hidden");
    const savedReviewer = state.__reviewer || {{}};
    reviewerName.value = savedReviewer.name || "";
    independentReview.checked = savedReviewer.independentVisualReview === true;
    answersHidden.checked = savedReviewer.answersHiddenUntilLock === true;
    function saveReviewerAttestation() {{
      state.__reviewer = {{
        name: reviewerName.value.trim(),
        independentVisualReview: independentReview.checked,
        answersHiddenUntilLock: answersHidden.checked,
      }};
      save();
      refreshExportAvailability();
    }}
    reviewerName.addEventListener("input", saveReviewerAttestation);
    independentReview.addEventListener("change", saveReviewerAttestation);
    answersHidden.addEventListener("change", saveReviewerAttestation);
    refreshPlaybackProgress();
    refreshExportAvailability();

    document.querySelector("#export").addEventListener("click", () => {{
      const signedAt = new Date().toISOString();
      const payload = {{
        audit_id: auditId,
        motion_fingerprint: motionFingerprint,
        normalized_facing: normalizedFacing,
        normal_speed_reviewed: allTakePlaybackComplete(),
        source_edit_compared: allTakeComparisonsOpened(),
        exported_at: new Date().toISOString(),
        reviewer_attestation: {{
          audit_id: auditId,
          motion_fingerprint: motionFingerprint,
          reviewer: state.__reviewer.name,
          signed_at: signedAt,
          independent_visual_review: state.__reviewer.independentVisualReview === true,
          answers_hidden_until_lock: state.__reviewer.answersHiddenUntilLock === true,
          normal_speed_reviewed: allTakePlaybackComplete(),
          source_edit_compared: allTakeComparisonsOpened(),
          statement: reviewerStatement,
        }},
        takes: Object.fromEntries(takeIds.map(take => [take, {{
          guess: state[take].guess,
          recognized: state[take].recognized === true,
          direction_guess: state[take].directionGuess,
          direction_recognized: state[take].directionRecognized === true,
          normal_speed_playback: state[take].normalSpeedPlayback,
          comparison_opened_at: state[take].comparisonOpenedAt,
          comparison_acknowledgment: state[take].comparisonAcknowledgment,
          locked_at: state[take].lockedAt,
          status: state[take].visualStatus,
          evidence: state[take].visualEvidence,
          verified_phases: state[take].verifiedPhases,
        }}])),
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${{auditId}}-review-result.json`;
      link.click();
      URL.revokeObjectURL(link.href);
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True, help="real 139-channel host dance")
    parser.add_argument("--beats", type=Path, help="optional beat-frame .npy")
    parser.add_argument("--beat-strengths", type=Path, help="optional beat-strength .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--preserve-heading",
        action="store_true",
        help="keep the source world heading instead of normalizing the semantic review views",
    )
    parser.add_argument(
        "--motions",
        nargs="+",
        help="optional motion ids to re-audit instead of rendering the full bank",
    )
    args = parser.parse_args()

    payload = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    contracts = payload["motions"]
    bank = MotionBank()
    bank_ids = {spec.id for spec in bank.specs}
    if set(contracts) != bank_ids:
        missing = sorted(bank_ids - set(contracts))
        extra = sorted(set(contracts) - bank_ids)
        raise ValueError(f"visual contracts do not match the bank; missing={missing}, extra={extra}")

    host = to_editor139(np.load(args.host).astype(np.float32))
    frame_count = min(max(24, int(args.frames)), host.shape[0])
    host = np.ascontiguousarray(host[:frame_count], dtype=np.float32)
    default_beats = np.arange(0, frame_count, 15, dtype=np.int64)
    beats = _load_vector(args.beats, default_beats).astype(np.int64)
    valid = (beats >= 0) & (beats < frame_count)
    beats = beats[valid]
    if beats.size < 2:
        raise ValueError("the audit needs at least two in-range beats to preserve action tempo")
    strengths = None
    if args.beat_strengths is not None:
        strengths = _load_vector(args.beat_strengths, np.ones_like(beats, dtype=np.float32))
        if strengths.shape != valid.shape:
            raise ValueError("beat strengths must contain one value per unfiltered beat")
        strengths = strengths[valid].astype(np.float32)

    specs = list(bank.specs)
    if args.motions:
        requested = {motion.strip() for value in args.motions for motion in value.split(",")}
        unknown = sorted(requested - bank_ids)
        if unknown:
            raise ValueError(f"unknown motion ids: {unknown}")
        specs = [spec for spec in specs if spec.id in requested]
    for spec in specs:
        authored_directions = (
            spec.directions if spec.direction_mode == "clip" else (None,)
        )
        for direction in authored_directions:
            actual = bank.load_clip(spec, direction=direction)
            expected = build_motion(
                spec.id,
                spec.frames,
                direction=direction or "forward",
            )
            if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
                suffix = "" if direction is None else f"[{direction}]"
                raise ValueError(
                    f"{spec.id}{suffix}: committed clip is stale; "
                    "run scripts/build_motion_bank.py first"
                )
    cases = [
        (spec, direction)
        for spec in specs
        for direction in audit_variants(spec)
    ]
    random.Random(args.seed).shuffle(cases)
    _prepare_output(args.output)
    audit_id = _new_audit_id(args.output, args.seed)
    fingerprint = motion_fingerprint(ROOT)
    takes: list[dict] = []
    controls: list[dict] = []
    control_keys: dict[tuple[int, int, int], str] = {}
    answers: dict[str, dict] = {}
    clap_direction_records: dict[str, dict[str, dict]] = {}
    context = max(0, int(args.context))
    normalize_facing = not bool(args.preserve_heading)
    for index, (spec, direction) in enumerate(cases, start=1):
        take = f"take_{index:02d}"
        case_id = audit_case_id(spec, direction)
        motion, report = bank.apply(
            host,
            spec.id,
            beats=beats,
            beat_strengths=strengths,
            direction=direction,
        )
        machine_checks, machine_status = _machine_checks(host, motion, spec, report)
        start, end = map(int, report["action_range"])
        lo = max(0, start - context)
        hi = min(len(motion), end + context)
        action_start = start - lo
        front, side, control_front, control_side, heading = _paired_views(
            motion[lo:hi],
            host[lo:hi],
            action_start,
            normalize_facing=normalize_facing,
        )
        control_key = (lo, hi, int(round(heading * 1_000_000)))
        control = control_keys.get(control_key)
        if control is None:
            control = f"control_{len(control_keys) + 1:02d}"
            control_keys[control_key] = control
            np.save(args.output / f"{control}.npy", control_front)
            save_poses_npz(control_front, args.output / f"{control}_front.npz")
            save_poses_npz(control_side, args.output / f"{control}_side.npz")
            controls.append({"control": control, "frames": int(len(control_front))})
        np.save(args.output / f"{take}.npy", front)
        save_poses_npz(front, args.output / f"{take}_front.npz")
        save_poses_npz(side, args.output / f"{take}_side.npz")

        takes.append({
            "take": take,
            "control": control,
            "frames": int(len(front)),
            "action_range": [start - lo, end - lo],
            "event_frame": int(report["event_frame"]) - lo,
        })
        answers[take] = {
            "case_id": case_id,
            "id": spec.id,
            "name": spec.name,
            "aliases": list(spec.aliases),
            "requested_direction": direction,
            "resolved_direction": report.get("direction"),
            "visual_contract": contracts[spec.id],
            "machine_checks": machine_checks,
            "machine_status": machine_status,
            "report": report,
        }
        if spec.id.startswith("clap_") and direction in {"auto", "forward", "left", "right"}:
            clap_direction_records.setdefault(spec.id, {})[direction] = {
                "lateral": _clap_contact_lateral(motion, int(report["event_frame"])),
                "resolved": report.get("direction"),
                "checks": machine_checks,
                "answer": answers[take],
            }

    _append_clap_direction_matrix_checks(clap_direction_records)
    for answer in answers.values():
        answer["machine_status"] = (
            "pass"
            if all(check["passed"] for check in answer["machine_checks"])
            else "fail"
        )

    review = {
        "audit_id": audit_id,
        "fps": 30,
        "fixed_camera": True,
        "normalized_facing": normalize_facing,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "seed": int(args.seed),
        "bank_version": bank.version,
        "motion_fingerprint": fingerprint,
        "acceptance_rule": payload["acceptance_rule"],
        "controls": controls,
        "takes": takes,
    }
    (args.output / "review.json").write_text(
        json.dumps(review, indent=2), encoding="utf-8"
    )
    (args.output / "answer_key.json").write_text(
        json.dumps(answers, indent=2), encoding="utf-8"
    )
    (args.output / "review.html").write_text(
        _review_html(
            takes,
            payload["acceptance_rule"],
            audit_id=audit_id,
            normalized_facing=normalize_facing,
            motion_fingerprint_value=fingerprint,
        ),
        encoding="utf-8",
    )
    failures = [
        answer["case_id"]
        for answer in answers.values()
        if answer["machine_status"] != "pass"
    ]
    if failures:
        raise ValueError(f"machine visual invariants failed: {failures}")
    print(f"built {len(takes)} blind audit takes in {args.output}")


if __name__ == "__main__":
    main()
