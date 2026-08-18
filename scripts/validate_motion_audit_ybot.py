"""Validate exact posed Y-Bot geometry captured during a motion-bank audit render."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.editor.motion_audit import YBOT_METRICS_REPORT_NAME  # noqa: E402


_JOINT_NAMES = (
    "m_avg_Pelvis",
    "m_avg_L_Hip",
    "m_avg_R_Hip",
    "m_avg_Spine1",
    "m_avg_L_Knee",
    "m_avg_R_Knee",
    "m_avg_Spine2",
    "m_avg_L_Ankle",
    "m_avg_R_Ankle",
    "m_avg_Spine3",
    "m_avg_L_Foot",
    "m_avg_R_Foot",
    "m_avg_Neck",
    "m_avg_L_Collar",
    "m_avg_R_Collar",
    "m_avg_Head",
    "m_avg_L_Shoulder",
    "m_avg_R_Shoulder",
    "m_avg_L_Elbow",
    "m_avg_R_Elbow",
    "m_avg_L_Wrist",
    "m_avg_R_Wrist",
)
_GROUNDED_ACTIONS = {
    "bounce_in_place",
    "crouch_drop",
    "rise_reach",
    "side_step",
    "step_touch",
}


def _true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False])
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _load_metrics(path: Path, expected_frames: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"exact Y-Bot metrics are missing: {path.name}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "joints",
            "projected",
            "mesh_floor",
            "joint_names",
            "bone_heads",
            "bone_tails",
            "bone_reference_axes",
            "bone_names",
            "rendered_frames",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"{path.name}: missing arrays {missing}")
        values = {name: np.asarray(payload[name]) for name in required}
    names = tuple(str(value) for value in values["joint_names"].tolist())
    if names != _JOINT_NAMES:
        raise RuntimeError(f"{path.name}: unexpected Y-Bot joint order")
    bone_names = tuple(str(value) for value in values["bone_names"].tolist())
    missing_bones = [name for name in _JOINT_NAMES if name not in bone_names]
    if missing_bones:
        raise RuntimeError(f"{path.name}: missing Y-Bot bones {missing_bones}")
    joints = values["joints"]
    projected = values["projected"]
    floor = values["mesh_floor"]
    bone_heads = values["bone_heads"]
    bone_tails = values["bone_tails"]
    bone_reference_axes = values["bone_reference_axes"]
    rendered = values["rendered_frames"]
    if joints.shape != (expected_frames, len(_JOINT_NAMES), 3):
        raise RuntimeError(f"{path.name}: invalid joint shape {joints.shape}")
    if projected.shape != joints.shape:
        raise RuntimeError(f"{path.name}: invalid projected-joint shape {projected.shape}")
    if floor.shape != (expected_frames,):
        raise RuntimeError(f"{path.name}: invalid mesh-floor shape {floor.shape}")
    expected_bones = (expected_frames, len(bone_names), 3)
    if (
        bone_heads.shape != expected_bones
        or bone_tails.shape != expected_bones
        or bone_reference_axes.shape != expected_bones
    ):
        raise RuntimeError(f"{path.name}: invalid posed-bone shapes")
    if not np.array_equal(rendered, np.arange(expected_frames, dtype=rendered.dtype)):
        raise RuntimeError(f"{path.name}: metrics do not cover every rendered frame")
    if not all(
        np.isfinite(array).all()
        for array in (
            joints,
            projected,
            floor,
            bone_heads,
            bone_tails,
            bone_reference_axes,
        )
    ):
        raise RuntimeError(f"{path.name}: metrics contain non-finite values")
    axis_norms = np.linalg.norm(bone_reference_axes, axis=-1)
    if not np.allclose(axis_norms, 1.0, atol=1e-4):
        raise RuntimeError(f"{path.name}: posed-bone reference axes are not normalized")
    ordered_axes = np.stack(
        [
            bone_reference_axes[:, bone_names.index(name)]
            for name in _JOINT_NAMES
        ],
        axis=1,
    )
    return {
        "joints": joints.astype(np.float32, copy=False),
        "projected": projected.astype(np.float32, copy=False),
        "mesh_floor": floor.astype(np.float32, copy=False),
        "bone_axes": ordered_axes.astype(np.float32, copy=False),
    }


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _clap_checks(
    motion_id: str,
    front: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
    start: int,
    end: int,
    event: int,
) -> tuple[list[dict], float]:
    joints = front["joints"]
    host = control["joints"]
    wrist_gap = np.linalg.norm(joints[:, 20] - joints[:, 21], axis=1)
    shoulder_width = np.linalg.norm(joints[:, 16] - joints[:, 17], axis=1)
    ratio = wrist_gap / np.maximum(shoulder_width, 1e-6)
    threshold = 0.30 if motion_id == "clap_overhead" else 0.40
    runs = _true_runs(ratio[start:end] < threshold)
    expected = 3 if motion_id == "clap_repeat" else 1
    timing_passed = (
        len(runs) == expected
        and all(run_end - run_start + 1 >= 3 for run_start, run_end in runs)
    )
    if timing_passed and len(runs) > 1:
        timing_passed = all(
            int(np.count_nonzero(
                ratio[start + left_end + 1:start + right_start]
                > threshold + 0.12
            )) >= 3
            for (_left_start, left_end), (right_start, _right_end)
            in zip(runs, runs[1:])
        )
    contact_frames = [
        start + (run_start + run_end) // 2 for run_start, run_end in runs
    ]
    bilateral = []
    for frame in contact_frames:
        bilateral.append([
            float(np.linalg.norm(
                (joints[frame, wrist] - joints[frame, 0])
                - (host[frame, wrist] - host[frame, 0])
            ))
            for wrist in (20, 21)
        ])
    bilateral_passed = bool(bilateral) and all(
        min(movements) > 0.08 for movements in bilateral
    )
    checks = [
        _check(
            "rendered_clap_contact_timing",
            timing_passed,
            (
                f"Y-Bot wrist/shoulder closure runs {list(runs)} below ratio "
                f"{threshold:.2f}; expected {expected} readable run(s) of at "
                "least 3 frames with at least 3 visibly open frames between contacts"
            ),
        ),
        _check(
            "rendered_bilateral_clap",
            bilateral_passed,
            (
                "Y-Bot left/right wrist movement versus source at closure "
                f"{[[round(value, 4) for value in pair] for pair in bilateral]} m"
            ),
        ),
    ]
    if motion_id == "clap_overhead":
        above_head = float(
            min(joints[event, 20, 2], joints[event, 21, 2])
            - joints[event, 15, 2]
        )
        checks.append(_check(
            "rendered_overhead_clearance",
            above_head > 0.12,
            f"lower Y-Bot wrist is {above_head:.4f} m above the head at contact",
        ))
        alignment = []
        for frame in (
            max(start, event - 6),
            event,
            min(end - 1, event + 6),
        ):
            for wrist, elbow in ((20, 18), (21, 19)):
                forearm = joints[frame, wrist] - joints[frame, elbow]
                forearm /= np.linalg.norm(forearm) + 1e-9
                alignment.append(float(front["bone_axes"][frame, wrist] @ forearm))
        checks.append(_check(
            "rendered_overhead_wrist_continuity",
            min(alignment, default=-1.0) > np.cos(np.deg2rad(55.0)),
            (
                "minimum Y-Bot hand-bone/forearm alignment at approach, contact, "
                f"and recoil {min(alignment, default=float('nan')):.4f}"
            ),
        ))
    if contact_frames:
        lateral = float(np.mean([
            (0.5 * (joints[frame, 20] + joints[frame, 21]) - joints[frame, 0])[0]
            for frame in contact_frames
        ]))
    else:
        lateral = float("nan")
    return checks, lateral


def _jump_check(floor: np.ndarray, start: int, end: int) -> dict:
    action = floor[start:end]
    runs = _true_runs(action > 0.08)
    passed = (
        len(runs) == 1
        and runs[0][1] - runs[0][0] + 1 >= 12
        and runs[0][0] >= 8
        and len(action) - 1 - runs[0][1] >= 8
        and float(np.max(action)) > 0.18
    )
    return _check(
        "rendered_jump_floor_clearance",
        passed,
        (
            f"Y-Bot airborne runs {list(runs)} above 0.08 m; "
            f"maximum mesh clearance {float(np.max(action)):.4f} m"
        ),
    )


def _grounded_check(floor: np.ndarray, start: int, end: int) -> dict:
    maximum = float(np.max(np.abs(floor[start:end])))
    return _check(
        "rendered_ground_contact",
        maximum < 0.02,
        f"maximum absolute Y-Bot mesh-floor error {maximum:.4f} m",
    )


def _body_axes(joints: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    toe = (
        joints[frame, 10, :2] - joints[frame, 7, :2]
        + joints[frame, 11, :2] - joints[frame, 8, :2]
    )
    norm = float(np.linalg.norm(toe))
    if norm < 1e-6:
        lateral = joints[frame, 16, :2] - joints[frame, 17, :2]
        lateral /= np.linalg.norm(lateral) + 1e-9
        forward = np.array([lateral[1], -lateral[0]], dtype=np.float32)
    else:
        forward = toe / norm
    left = np.array([-forward[1], forward[0]], dtype=np.float32)
    return left, forward


def _planted_foot_check(
    joints: np.ndarray,
    start: int,
    end: int,
) -> dict:
    trim = min(4, max(0, (end - start - 2) // 2))
    action = joints[start + trim:end - trim]
    drift = [
        float(np.max(np.linalg.norm(
            action[:, ankle, :2] - action[0, ankle, :2],
            axis=1,
        )))
        for ankle in (7, 8)
    ]
    passed = max(drift) < 0.08
    return _check(
        "rendered_planted_foot_stability",
        passed,
        (
            f"maximum core-phase left/right ankle-plane drift "
            f"{drift[0]:.4f}/{drift[1]:.4f} m"
        ),
    )


def _bounce_cycle_check(
    joints: np.ndarray,
    start: int,
    end: int,
) -> dict:
    pelvis = joints[start:end, 0, 2]
    travel = float(np.ptp(pelvis))
    threshold = float(np.min(pelvis) + 0.42 * travel)
    runs = _true_runs(pelvis < threshold) if travel > 1e-6 else ()
    rebound = (
        float(np.max(pelvis[runs[0][1] + 1:runs[1][0]]) - np.min(pelvis))
        if len(runs) == 2 and runs[0][1] + 1 < runs[1][0]
        else 0.0
    )
    finish_recovery = float(pelvis[-1] - np.min(pelvis))
    passed = (
        travel > 0.10
        and len(runs) == 2
        and all(run_end - run_start + 1 >= 2 for run_start, run_end in runs)
        and rebound > 0.08
        and finish_recovery > 0.08
    )
    return _check(
        "rendered_bounce_cycle_count",
        passed,
        (
            f"pelvis range {travel:.4f} m with low-phase runs {list(runs)} "
            f"below {threshold:.4f} m; inter-pulse rebound {rebound:.4f} m and "
            f"final recovery {finish_recovery:.4f} m"
        ),
    )


def _level_change_check(
    motion_id: str,
    joints: np.ndarray,
    control: np.ndarray,
    start: int,
    end: int,
    event: int,
) -> dict:
    pelvis = joints[start:end, 0, 2] - control[start:end, 0, 2]
    event_local = int(np.clip(event - start, 0, len(pelvis) - 1))
    travel = float(np.ptp(pelvis))
    if motion_id == "crouch_drop":
        low = int(np.argmin(pelvis))
        minimum = float(pelvis[low])
        drop = -minimum
        event_position = float(pelvis[event_local] - minimum)
        low_hold = int(np.count_nonzero(pelvis <= minimum + 0.02))
        recovery = float(pelvis[-1] - minimum)
        anticipation_end = max(3, int(round(0.22 * len(pelvis))))
        descent_start = max(anticipation_end + 1, int(round(0.28 * len(pelvis))))
        anticipation_low = int(np.argmin(pelvis[:anticipation_end]))
        anticipation_drop = -float(pelvis[anticipation_low])
        anticipation_release = float(
            np.max(pelvis[anticipation_low:descent_start])
            - pelvis[anticipation_low]
        )
        _left, forward = _body_axes(control, start + low)
        torso = joints[start + low, 9] - joints[start + low, 0]
        host_torso = control[start + low, 9] - control[start + low, 0]
        torso_forward = float((torso[:2] - host_torso[:2]) @ forward)
        passed = (
            0.18 < drop < 0.30
            and anticipation_drop > 0.008
            and anticipation_release > 0.008
            and event_position < 0.025
            and 2 <= low_hold <= max(8, int(np.ceil(0.30 * len(pelvis))))
            and recovery > 0.12
            and 0.02 < torso_forward < 0.20
        )
        detail = (
            f"pelvis drop {drop:.4f} m; event is {event_position:.4f} m "
            f"above the lowest pose; preload/release "
            f"{anticipation_drop:.4f}/{anticipation_release:.4f} m; low hold "
            f"{low_hold} frames; recovery {recovery:.4f} m; torso forward delta "
            f"{torso_forward:.4f} m"
        )
    else:
        passed = travel > 0.20
        detail = f"pelvis range {travel:.4f} m"
    return _check("rendered_level_change_signature", passed, detail)


def _rise_phase_check(
    joints: np.ndarray,
    start: int,
    end: int,
) -> dict:
    action = joints[start:end]
    pelvis = action[:, 0, 2]
    low = int(np.argmin(pelvis))
    hand_lift = np.minimum(
        action[:, 20, 2] - action[:, 9, 2],
        action[:, 21, 2] - action[:, 9, 2],
    )
    high = np.flatnonzero(hand_lift > 0.35)
    first_high = int(high[0]) if high.size else -1
    passed = (
        first_high >= low + 3
        and float(hand_lift[low]) < 0.25
        and float(np.max(hand_lift)) > 0.45
    )
    return _check(
        "rendered_rise_before_reach",
        passed,
        (
            f"lowest pelvis frame {low}, first dual-hand high frame {first_high}, "
            f"hand lift at low/peak {float(hand_lift[low]):.4f}/"
            f"{float(np.max(hand_lift)):.4f} m"
        ),
    )


def _side_step_check(
    joints: np.ndarray,
    control: np.ndarray,
    start: int,
    end: int,
    event: int,
    direction: str,
) -> dict:
    left, _forward = _body_axes(joints, start)
    sign = 1.0 if direction == "left" else -1.0
    lead_ankle = 7 if direction == "left" else 8
    trail_ankle = 8 if direction == "left" else 7
    root = (joints[start:end, 0, :2] - joints[start, 0, :2]) @ left
    foot = (
        joints[event, lead_ankle, :2] - joints[start, lead_ankle, :2]
    ) @ left
    stance = abs(float(
        (joints[event, 7, :2] - joints[event, 8, :2]) @ left
    ))
    signed_root = float(np.max(sign * root))
    signed_foot = sign * float(foot)
    lead_position = sign * float(joints[event, lead_ankle, :2] @ left)
    trail_position = sign * float(joints[event, trail_ankle, :2] @ left)
    pelvis_position = sign * float(joints[event, 0, :2] @ left)
    support_span = max(lead_position - trail_position, 1e-6)
    support_u = (pelvis_position - trail_position) / support_span
    root_foot_ratio = signed_root / max(signed_foot, 1e-6)
    pelvis_vertical = (
        joints[start:end, 0, 2] - control[start:end, 0, 2]
    )
    relative = (
        (joints[start:end, 12:22] - joints[start:end, :1])
        - (control[start:end, 12:22] - control[start:end, :1])
    )
    upper_body_delta = float(np.max(np.linalg.norm(relative, axis=-1)))
    passed = (
        direction in {"left", "right"}
        and 0.16 < signed_root < 0.36
        and 0.28 < signed_foot < 0.52
        and 0.40 < stance < 0.66
        and 0.45 < root_foot_ratio < 0.95
        and 0.45 < support_u < 0.90
        and float(np.ptp(pelvis_vertical)) < 0.16
        and upper_body_delta < 0.18
    )
    return _check(
        "rendered_side_step_weight_transfer",
        passed,
        (
            f"{direction} signed root/lead-foot travel "
            f"{signed_root:.4f}/{signed_foot:.4f} m, root/foot ratio "
            f"{root_foot_ratio:.3f}, stance {stance:.4f} m, support position "
            f"{support_u:.3f}, pelvis vertical range "
            f"{float(np.ptp(pelvis_vertical)):.4f} m, upper-body relative delta "
            f"{upper_body_delta:.4f} m"
        ),
    )


def _step_direction_check(
    motion_id: str,
    joints: np.ndarray,
    start: int,
    end: int,
) -> dict:
    _left, forward = _body_axes(joints, start)
    sign = 1.0 if motion_id == "step_forward" else -1.0
    action = joints[start:end]
    root = (action[:, 0, :2] - action[0, 0, :2]) @ forward
    lead = (action[:, 7, :2] - action[0, 7, :2]) @ forward
    signed_root = sign * float(root[-1])
    signed_foot = float(np.max(sign * lead))
    wrong_way = float(np.max(-sign * root))
    plant_candidates = np.flatnonzero(sign * lead >= 0.90 * signed_foot)
    plant = int(plant_candidates[0]) if plant_candidates.size else len(root) - 1
    follow_through = sign * float(root[-1] - root[plant])
    passed = (
        signed_root > 0.35
        and signed_foot > 0.50
        and follow_through > 0.10
        and wrong_way < 0.08
    )
    return _check(
        "rendered_step_direction_signature",
        passed,
        (
            f"{motion_id} signed root/lead-foot travel "
            f"{signed_root:.4f}/{signed_foot:.4f} m; root follow-through after "
            f"lead-foot plant {follow_through:.4f} m; wrong-way root excursion "
            f"{wrong_way:.4f} m"
        ),
    )


def _turn_check(
    motion_id: str,
    joints: np.ndarray,
    start: int,
    end: int,
    direction: str,
) -> dict:
    lateral = joints[start:end, 16, :2] - joints[start:end, 17, :2]
    angles = np.unwrap(np.arctan2(lateral[:, 1], lateral[:, 0]))
    sign = 1.0 if direction == "left" else -1.0
    progress = sign * (angles - angles[0])
    target = np.pi / 2.0 if motion_id == "turn_quarter" else np.pi
    increments = np.diff(progress)
    backward_fraction = float(np.mean(increments < -0.03)) if increments.size else 1.0
    final = float(progress[-1])
    completion = np.flatnonzero(progress >= 0.85 * target)
    completion_frame = int(completion[0]) if completion.size else len(progress) - 1
    completion_fraction = completion_frame / max(1, len(progress) - 1)
    hold_frame = int(np.ceil(0.80 * max(0, len(progress) - 1)))
    tail_span = float(np.ptp(progress[hold_frame:]))
    passed = (
        direction in {"left", "right"}
        and 0.78 * target < final < 1.22 * target
        and float(np.max(progress)) < 1.28 * target
        and backward_fraction < 0.20
        and completion_fraction < 0.82
        and float(progress[hold_frame]) > 0.85 * target
        and tail_span < 0.08 * target
    )
    return _check(
        "rendered_turn_progression",
        passed,
        (
            f"{direction} final/peak heading progress "
            f"{np.rad2deg(final):.2f}/{np.rad2deg(float(np.max(progress))):.2f} "
            f"degrees; 85%-complete at {completion_fraction:.3f} of the action with "
            f"{np.rad2deg(tail_span):.2f} degrees of drift in the final 20%; backward-step "
            f"fraction {backward_fraction:.3f}"
        ),
    )


def _punch_check(
    joints: np.ndarray,
    start: int,
    end: int,
    event: int,
    direction: str,
) -> dict:
    action = joints[start:end]
    if direction == "left":
        wrist, shoulder, guard_wrist, guard_shoulder = 20, 16, 21, 17
    else:
        wrist, shoulder, guard_wrist, guard_shoulder = 21, 17, 20, 16
    extension = np.linalg.norm(action[:, wrist] - action[:, shoulder], axis=1)
    peak = int(np.argmax(extension))
    before = float(np.min(extension[:peak])) if peak > 0 else float("inf")
    after = (
        float(np.min(extension[peak + 1:]))
        if peak + 1 < len(extension)
        else float("inf")
    )
    guard = float(np.linalg.norm(
        action[peak, guard_wrist] - action[peak, guard_shoulder]
    ))
    event_local = int(np.clip(event - start, 0, len(action) - 1))
    forward_reach = np.asarray([
        float(
            (action[frame, wrist, :2] - action[frame, 9, :2])
            @ _body_axes(action, frame)[1]
        )
        for frame in range(len(action))
    ])
    reach_peak = int(np.argmax(forward_reach))
    near_peak = int(np.count_nonzero(
        forward_reach >= 0.995 * float(forward_reach[reach_peak])
    ))
    _left, event_forward = _body_axes(action, event_local)
    guard_forward = float(
        (
            action[event_local, guard_wrist, :2]
            - action[event_local, 9, :2]
        )
        @ event_forward
    )
    guard_forward_ratio = guard_forward / max(
        float(forward_reach[event_local]),
        1e-6,
    )
    passed = (
        direction in {"left", "right"}
        and 1 < peak < len(extension) - 2
        and float(extension[peak] - before) > 0.18
        and float(extension[peak] - after) > 0.18
        and float(extension[peak] - guard) > 0.18
        and abs(reach_peak - event_local) <= 1
        and float(forward_reach[event_local])
        >= 0.97 * float(forward_reach[reach_peak])
        and near_peak <= 5
        and guard_forward_ratio < 0.36
    )
    return _check(
        "rendered_guard_strike_recoil",
        passed,
        (
            f"Y-Bot strike peak local frame {peak}, arm length "
            f"{float(extension[peak]):.4f} m versus pre/post "
            f"{before:.4f}/{after:.4f} m and guard {guard:.4f} m; "
            f"declared event/forward-reach peak {event_local}/{reach_peak}, "
            f"event/peak reach {float(forward_reach[event_local]):.4f}/"
            f"{float(forward_reach[reach_peak]):.4f} m, near-peak span "
            f"{near_peak} frames, guard/strike forward ratio "
            f"{guard_forward_ratio:.3f}"
        ),
    )


def _punch_plane_check(
    joints: np.ndarray,
    start: int,
    end: int,
    direction: str,
) -> dict:
    action = joints[start:end]
    wrist, shoulder = ((20, 16) if direction == "left" else (21, 17))
    extension = np.linalg.norm(action[:, wrist] - action[:, shoulder], axis=1)
    peak = int(np.argmax(extension))
    left, forward = _body_axes(action, peak)
    reach = action[peak, wrist, :2] - action[peak, shoulder, :2]
    forward_reach = float(reach @ forward)
    lateral_reach = abs(float(reach @ left))
    passed = (
        direction in {"left", "right"}
        and forward_reach > 0.35
        and forward_reach > 1.5 * lateral_reach
    )
    return _check(
        "rendered_forward_punch_plane",
        passed,
        (
            f"strike shoulder-to-wrist forward/lateral reach "
            f"{forward_reach:.4f}/{lateral_reach:.4f} m"
        ),
    )


def _chest_pop_check(
    joints: np.ndarray,
    control: np.ndarray,
    event: int,
) -> dict:
    _left, forward = _body_axes(control, event)

    def delta(joint: int) -> np.ndarray:
        return (
            (joints[event, joint] - joints[event, 0])
            - (control[event, joint] - control[event, 0])
        )

    chest = delta(9)
    head = delta(15)
    chest_forward = float(chest[:2] @ forward)
    head_forward = float(head[:2] @ forward)
    passed = (
        chest_forward > 0.08
        and abs(head_forward) < 0.60 * chest_forward
        and float(head[2]) > -0.06
    )
    return _check(
        "rendered_chest_pop_isolation",
        passed,
        (
            f"chest/head forward delta {chest_forward:.4f}/{head_forward:.4f} m; "
            f"head vertical delta {float(head[2]):.4f} m"
        ),
    )


def _step_touch_check(
    joints: np.ndarray,
    start: int,
    event: int,
) -> dict:
    gap = np.linalg.norm(joints[:, 7, :2] - joints[:, 8, :2], axis=1)
    pre = gap[start + 5:max(start + 6, event - 4)]
    open_gap = float(np.max(pre))
    touch_gap = float(gap[event])
    sagittal = abs(float(joints[event, 7, 1] - joints[event, 8, 1]))
    passed = open_gap > 0.30 and touch_gap < 0.15 and sagittal > 0.05
    return _check(
        "rendered_step_touch_separation",
        passed,
        (
            f"Y-Bot open/touch ankle gaps {open_gap:.4f}/{touch_gap:.4f} m; "
            f"side-view stagger {sagittal:.4f} m"
        ),
    )


def build_report(audit_dir: Path) -> dict:
    audit_dir = Path(audit_dir).resolve()
    review = json.loads((audit_dir / "review.json").read_text(encoding="utf-8"))
    answers = json.loads((audit_dir / "answer_key.json").read_text(encoding="utf-8"))
    take_items = review.get("takes") or []
    take_ids = {item["take"] for item in take_items}
    if set(answers) != take_ids:
        raise RuntimeError("answer key does not cover every Y-Bot audit take")

    cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def metrics(identifier: str, view: str, frames: int) -> dict[str, np.ndarray]:
        key = (identifier, view)
        if key not in cache:
            cache[key] = _load_metrics(
                audit_dir / f"{identifier}_{view}_ybot.npz",
                frames,
            )
        return cache[key]

    records: dict[str, dict] = {}
    clap_directions: dict[str, dict[str, tuple[str, float, str]]] = {}
    for item in take_items:
        take = str(item["take"])
        control_id = str(item["control"])
        frames = int(item["frames"])
        start, end = map(int, item["action_range"])
        event = int(item["event_frame"])
        answer = answers[take]
        motion_id = str(answer["id"])
        resolved = str(answer.get("resolved_direction") or "none")
        requested = str(answer.get("requested_direction") or "none")
        front = metrics(take, "front", frames)
        metrics(take, "side", frames)
        control = metrics(control_id, "front", frames)
        metrics(control_id, "side", frames)
        checks = [_check(
            "exact_rig_metrics_complete",
            True,
            f"{frames} finite front/side Y-Bot frames captured for edit and source",
        )]

        if motion_id in _GROUNDED_ACTIONS:
            checks.append(_grounded_check(front["mesh_floor"], start, end))
        if motion_id in {"jump_two_foot", "jump_arms_up"}:
            checks.append(_jump_check(front["mesh_floor"], start, end))
        if motion_id == "bounce_in_place":
            checks.extend([
                _bounce_cycle_check(front["joints"], start, end),
                _planted_foot_check(front["joints"], start, end),
            ])
        if motion_id == "crouch_drop":
            checks.extend([
                _level_change_check(
                    motion_id,
                    front["joints"],
                    control["joints"],
                    start,
                    end,
                    event,
                ),
                _planted_foot_check(front["joints"], start, end),
            ])
        if motion_id == "rise_reach":
            joints = front["joints"]
            travel = float(np.ptp(joints[start:end, 0, 2]))
            lift = [
                float(joints[event, hand, 2] - joints[event, 9, 2])
                for hand in (20, 21)
            ]
            checks.append(_check(
                "rendered_planted_rise",
                travel > 0.20 and min(lift) > 0.30,
                f"Y-Bot pelvis range {travel:.4f} m; hand lift {lift}",
            ))
            checks.extend([
                _planted_foot_check(joints, start, end),
                _rise_phase_check(joints, start, end),
            ])
        if motion_id.startswith("clap_"):
            clap_checks, lateral = _clap_checks(
                motion_id,
                front,
                control,
                start,
                end,
                event,
            )
            checks.extend(clap_checks)
            clap_directions.setdefault(motion_id, {})[requested] = (
                take,
                lateral,
                resolved,
            )
        if motion_id == "arm_punch":
            checks.extend([
                _punch_check(front["joints"], start, end, event, resolved),
                _punch_plane_check(front["joints"], start, end, resolved),
            ])
        if motion_id == "side_step":
            checks.append(_side_step_check(
                front["joints"],
                control["joints"],
                start,
                end,
                event,
                resolved,
            ))
        if motion_id == "step_touch":
            checks.append(_step_touch_check(front["joints"], start, event))
        if motion_id in {"step_forward", "step_backward"}:
            checks.append(_step_direction_check(
                motion_id,
                front["joints"],
                start,
                end,
            ))
        if motion_id in {"turn_quarter", "turn_half"}:
            checks.append(_turn_check(
                motion_id,
                front["joints"],
                start,
                end,
                resolved,
            ))
        if motion_id == "chest_pop":
            checks.append(_chest_pop_check(
                front["joints"],
                control["joints"],
                event,
            ))

        records[take] = {
            "case_id": answer["case_id"],
            "status": "pass" if all(check["passed"] for check in checks) else "fail",
            "checks": checks,
        }

    for motion_id, directions in clap_directions.items():
        explicit = {"forward", "left", "right"}
        if not explicit.issubset(directions):
            continue
        forward = directions["forward"][1]
        left = directions["left"][1]
        right = directions["right"][1]
        matrix_passed = (
            np.isfinite([left, forward, right]).all()
            and left > forward + 0.10
            and right < forward - 0.10
        )
        detail = (
            f"Y-Bot contact offsets left/forward/right "
            f"{left:.4f}/{forward:.4f}/{right:.4f} m"
        )
        for direction in explicit:
            take = directions[direction][0]
            records[take]["checks"].append(_check(
                "rendered_directional_contact_matrix",
                matrix_passed,
                detail,
            ))
        if "auto" in directions:
            take, lateral, resolved = directions["auto"]
            reference = directions.get(resolved)
            automatic_passed = (
                matrix_passed
                and reference is not None
                and abs(lateral - reference[1]) < 0.03
            )
            records[take]["checks"].append(_check(
                "rendered_automatic_contact_direction",
                automatic_passed,
                (
                    f"Y-Bot automatic contact {lateral:.4f} m resolved "
                    f"{resolved!r}; explicit contact "
                    f"{reference[1]:.4f} m"
                    if reference is not None
                    else f"automatic clap resolved unsupported direction {resolved!r}"
                ),
            ))

    for record in records.values():
        record["status"] = (
            "pass"
            if all(check.get("passed") is True for check in record["checks"])
            else "fail"
        )
    return {
        "schema_version": 3,
        "audit_id": review["audit_id"],
        "motion_fingerprint": review["motion_fingerprint"],
        "status": (
            "pass"
            if all(record["status"] == "pass" for record in records.values())
            else "fail"
        ),
        "takes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.audit)
    output = args.audit / YBOT_METRICS_REPORT_NAME
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    failures = [
        take for take, result in report["takes"].items()
        if result["status"] != "pass"
    ]
    print(
        f"YBOT_METRICS_{report['status'].upper()} "
        f"{len(report['takes'])} takes; failures={failures}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
