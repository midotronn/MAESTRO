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
            "rendered_frames",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"{path.name}: missing arrays {missing}")
        values = {name: np.asarray(payload[name]) for name in required}
    names = tuple(str(value) for value in values["joint_names"].tolist())
    if names != _JOINT_NAMES:
        raise RuntimeError(f"{path.name}: unexpected Y-Bot joint order")
    joints = values["joints"]
    projected = values["projected"]
    floor = values["mesh_floor"]
    rendered = values["rendered_frames"]
    if joints.shape != (expected_frames, len(_JOINT_NAMES), 3):
        raise RuntimeError(f"{path.name}: invalid joint shape {joints.shape}")
    if projected.shape != joints.shape:
        raise RuntimeError(f"{path.name}: invalid projected-joint shape {projected.shape}")
    if floor.shape != (expected_frames,):
        raise RuntimeError(f"{path.name}: invalid mesh-floor shape {floor.shape}")
    if not np.array_equal(rendered, np.arange(expected_frames, dtype=rendered.dtype)):
        raise RuntimeError(f"{path.name}: metrics do not cover every rendered frame")
    if not all(np.isfinite(array).all() for array in (joints, projected, floor)):
        raise RuntimeError(f"{path.name}: metrics contain non-finite values")
    return {
        "joints": joints.astype(np.float32, copy=False),
        "projected": projected.astype(np.float32, copy=False),
        "mesh_floor": floor.astype(np.float32, copy=False),
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
        and all(run_end - run_start + 1 >= 2 for run_start, run_end in runs)
    )
    if timing_passed and len(runs) > 1:
        timing_passed = all(
            float(np.max(ratio[start + left_end + 1:start + right_start]))
            > threshold + 0.12
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
                f"{threshold:.2f}; expected {expected} readable run(s)"
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


def _punch_check(
    joints: np.ndarray,
    start: int,
    end: int,
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
    passed = (
        direction in {"left", "right"}
        and 1 < peak < len(extension) - 2
        and float(extension[peak] - before) > 0.18
        and float(extension[peak] - after) > 0.18
        and float(extension[peak] - guard) > 0.18
    )
    return _check(
        "rendered_guard_strike_recoil",
        passed,
        (
            f"Y-Bot strike peak local frame {peak}, arm length "
            f"{float(extension[peak]):.4f} m versus pre/post "
            f"{before:.4f}/{after:.4f} m and guard {guard:.4f} m"
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
            travel = float(np.ptp(front["joints"][start:end, 0, 2]))
            checks.append(_check(
                "rendered_bounce_height",
                travel > 0.07,
                f"Y-Bot pelvis height range {travel:.4f} m while grounded",
            ))
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
            checks.append(_punch_check(
                front["joints"],
                start,
                end,
                resolved,
            ))
        if motion_id == "step_touch":
            checks.append(_step_touch_check(front["joints"], start, event))

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
        "schema_version": 1,
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
