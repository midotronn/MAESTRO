"""Build MAESTRO's redistributable procedural named-motion bank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.dance.transition import (  # noqa: E402
    _axis_angle_to_matrix,
    _matrix_to_axis_angle,
    _matrix_to_sixd,
    to_zup,
)
from agentlodge.editor.motion_bank import MotionBank  # noqa: E402
from server.fk import _template  # noqa: E402

BANK = ROOT / "assets" / "motion_bank"
MANIFEST = BANK / "manifest.json"


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _pulse(t, center=0.5, width=0.16):
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def _identity_clip(n: int):
    aa = np.zeros((n, 22, 3), dtype=np.float32)
    trans = np.zeros((n, 3), dtype=np.float32)
    contacts = np.ones((n, 4), dtype=np.float32)
    return aa, trans, contacts


def _align_vector(source, target):
    a = source / (np.linalg.norm(source) + 1e-9)
    b = target / (np.linalg.norm(target) + 1e-9)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    norm = float(np.linalg.norm(cross))
    if norm < 1e-7:
        if dot > 0:
            return np.eye(3, dtype=np.float32)
        axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        if np.linalg.norm(axis) < 1e-7:
            axis = np.cross(a, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    else:
        axis = cross / norm
    angle = np.arccos(dot)
    return _axis_angle_to_matrix((axis * angle).astype(np.float32))


def _solve_arm(aa, side: str, targets: np.ndarray):
    joints = (16, 18, 20) if side == "left" else (17, 19, 21)
    shoulder, elbow, wrist = joints
    J = _template()
    upper = J[elbow] - J[shoulder]
    fore = J[wrist] - J[elbow]
    l1, l2 = float(np.linalg.norm(upper)), float(np.linalg.norm(fore))
    sign = 1.0 if side == "left" else -1.0
    for frame, hand in enumerate(targets):
        origin = J[shoulder]
        delta = hand - origin
        distance = float(np.clip(np.linalg.norm(delta), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
        direction = delta / (np.linalg.norm(delta) + 1e-9)
        a = (l1 * l1 - l2 * l2 + distance * distance) / (2.0 * distance)
        height = np.sqrt(max(0.0, l1 * l1 - a * a))
        hint = np.array([sign, 0.15, 0.0], dtype=np.float32)
        bend = hint - np.dot(hint, direction) * direction
        bend /= np.linalg.norm(bend) + 1e-9
        elbow_target = origin + a * direction + height * bend
        upper_global = _align_vector(upper, elbow_target - origin)
        fore_global = _align_vector(fore, hand - elbow_target)
        elbow_local = upper_global.T @ fore_global
        aa[frame, shoulder] = _matrix_to_axis_angle(upper_global)
        aa[frame, elbow] = _matrix_to_axis_angle(elbow_local)


def _arm_targets(side: str, signal: np.ndarray, target: np.ndarray) -> np.ndarray:
    wrist = 20 if side == "left" else 21
    rest = _template()[wrist]
    return rest[None, :] + signal[:, None] * (np.asarray(target) - rest)[None, :]


def _clap_arms(aa, signal, *, overhead=False):
    target = np.array([0.0, 0.42 if overhead else 0.04, -0.24], dtype=np.float32)
    _solve_arm(aa, "left", _arm_targets("left", signal, target))
    _solve_arm(aa, "right", _arm_targets("right", signal, target))


def _legs(aa, phase, amount=0.45):
    aa[:, 1, 0] += amount * np.sin(phase)
    aa[:, 2, 0] -= amount * np.sin(phase)
    aa[:, 4, 0] += amount * np.maximum(0.0, -np.sin(phase))
    aa[:, 5, 0] += amount * np.maximum(0.0, np.sin(phase))


def build_motion(motion_id: str, n: int) -> np.ndarray:
    aa, trans, contacts = _identity_clip(n)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    phase = 2.0 * np.pi * t

    if motion_id in {"clap_single", "clap_overhead"}:
        hit = _pulse(t, 0.54, 0.15)
        _clap_arms(aa, hit, overhead=motion_id == "clap_overhead")
        aa[:, 3, 0] += 0.12 * hit
    elif motion_id == "clap_repeat":
        hit = np.maximum.reduce([_pulse(t, c, 0.07) for c in (0.27, 0.52, 0.77)])
        _clap_arms(aa, hit)
        trans[:, 2] += 0.025 * np.sin(6.0 * np.pi * t)
    elif motion_id in {"jump_two_foot", "jump_arms_up"}:
        air = np.sin(np.pi * _smoothstep((t - 0.18) / 0.64))
        air = np.maximum(air, 0.0)
        trans[:, 2] += (0.22 if motion_id == "jump_two_foot" else 0.27) * air
        crouch = _pulse(t, 0.17, 0.10) + 0.45 * _pulse(t, 0.85, 0.10)
        aa[:, 1, 0] -= 0.42 * crouch
        aa[:, 2, 0] -= 0.42 * crouch
        aa[:, 4, 0] += 0.75 * crouch
        aa[:, 5, 0] += 0.75 * crouch
        contacts[air > 0.22] = 0.0
        if motion_id == "jump_arms_up":
            left = np.array([0.25, 0.48, -0.04], dtype=np.float32)
            right = np.array([-0.25, 0.48, -0.04], dtype=np.float32)
            _solve_arm(aa, "left", _arm_targets("left", air, left))
            _solve_arm(aa, "right", _arm_targets("right", air, right))
    elif motion_id == "bounce_in_place":
        bounce = 0.5 - 0.5 * np.cos(6.0 * np.pi * t)
        trans[:, 2] += 0.075 * bounce
        aa[:, 4, 0] += 0.32 * (1.0 - bounce)
        aa[:, 5, 0] += 0.32 * (1.0 - bounce)
        aa[:, 3, 0] += 0.1 * np.sin(6.0 * np.pi * t)
    elif motion_id == "wave":
        raise_arm = _smoothstep(t / 0.28) * (1.0 - _smoothstep((t - 0.78) / 0.22))
        target = np.column_stack([
            -0.30 + 0.08 * np.sin(8.0 * np.pi * t),
            np.full(n, 0.43),
            np.full(n, -0.08),
        ]).astype(np.float32)
        target = _template()[21][None, :] + raise_arm[:, None] * (target - _template()[21])
        _solve_arm(aa, "right", target)
        aa[:, 21, 0] += 0.65 * np.sin(8.0 * np.pi * t) * raise_arm
    elif motion_id == "point_side":
        point = _smoothstep(t / 0.42) * (1.0 - 0.35 * _smoothstep((t - 0.82) / 0.18))
        target = np.array([-0.52, 0.13, -0.46], dtype=np.float32)
        _solve_arm(aa, "right", _arm_targets("right", point, target))
        aa[:, 9, 2] += 0.16 * point
    elif motion_id == "celebrate_hands_up":
        rise = _smoothstep(t / 0.55)
        _solve_arm(aa, "left", _arm_targets(
            "left", rise, np.array([0.28, 0.52, -0.02], dtype=np.float32)))
        _solve_arm(aa, "right", _arm_targets(
            "right", rise, np.array([-0.28, 0.52, -0.02], dtype=np.float32)))
        trans[:, 2] += 0.06 * rise
    elif motion_id == "chest_pop":
        hit = _pulse(t, 0.52, 0.10)
        aa[:, 3, 0] -= 0.28 * hit
        aa[:, 6, 0] += 0.38 * hit
        aa[:, 9, 0] += 0.32 * hit
        aa[:, 16, 2] -= 0.2 * hit
        aa[:, 17, 2] += 0.2 * hit
    elif motion_id == "arm_punch":
        hit = _pulse(t, 0.55, 0.14)
        target = np.array([-0.12, 0.04, -0.72], dtype=np.float32)
        _solve_arm(aa, "right", _arm_targets("right", hit, target))
        aa[:, 9, 2] += 0.24 * hit
        trans[:, 1] += 0.06 * hit
    elif motion_id in {"side_step", "step_touch"}:
        distance = 0.42 if motion_id == "side_step" else 0.30
        trans[:, 0] += distance * _smoothstep(t)
        cycles = 1.0 if motion_id == "side_step" else 2.0
        _legs(aa, cycles * 2.0 * np.pi * t, 0.5)
        contacts[:, 0:2] = (np.sin(cycles * 2.0 * np.pi * t) <= 0)[:, None]
        contacts[:, 2:4] = (np.sin(cycles * 2.0 * np.pi * t) >= 0)[:, None]
        aa[:, 3, 2] += 0.13 * np.sin(cycles * 2.0 * np.pi * t)
    elif motion_id in {"step_forward", "step_backward"}:
        direction = 1.0 if motion_id == "step_forward" else -1.0
        trans[:, 1] += direction * 0.46 * _smoothstep(t)
        _legs(aa, 2.0 * np.pi * t, 0.55 * direction)
        contacts[:, 0:2] = (np.sin(2.0 * np.pi * t) <= 0)[:, None]
        contacts[:, 2:4] = (np.sin(2.0 * np.pi * t) >= 0)[:, None]
    elif motion_id in {"turn_quarter", "turn_half"}:
        angle = (np.pi / 2.0) if motion_id == "turn_quarter" else np.pi
        aa[:, 0, 1] = angle * _smoothstep(t)
        _legs(aa, 2.0 * np.pi * t, 0.28)
        trans[:, 2] += 0.025 * np.sin(2.0 * np.pi * t) ** 2
    elif motion_id == "body_roll":
        for joint, center in ((3, 0.30), (6, 0.48), (9, 0.66)):
            aa[:, joint, 0] += 0.48 * _pulse(t, center, 0.15)
        aa[:, 12, 0] -= 0.18 * _pulse(t, 0.72, 0.16)
        trans[:, 2] += 0.035 * np.sin(2.0 * np.pi * t)
    elif motion_id == "crouch_drop":
        down = _smoothstep(t / 0.68)
        trans[:, 2] -= 0.28 * down
        aa[:, 1, 0] -= 0.55 * down
        aa[:, 2, 0] -= 0.55 * down
        aa[:, 4, 0] += 1.05 * down
        aa[:, 5, 0] += 1.05 * down
        aa[:, 3, 0] += 0.3 * down
    elif motion_id == "rise_reach":
        rise = _smoothstep(t / 0.72)
        trans[:, 2] = -0.24 + 0.28 * rise
        aa[:, 1, 0] = -0.45 * (1.0 - rise)
        aa[:, 2, 0] = -0.45 * (1.0 - rise)
        aa[:, 4, 0] = 0.9 * (1.0 - rise)
        aa[:, 5, 0] = 0.9 * (1.0 - rise)
        _solve_arm(aa, "left", _arm_targets(
            "left", rise, np.array([0.18, 0.53, -0.08], dtype=np.float32)))
        _solve_arm(aa, "right", _arm_targets(
            "right", rise, np.array([-0.18, 0.53, -0.08], dtype=np.float32)))
    else:
        raise KeyError(f"no procedural authoring recipe for {motion_id}")

    rotations = _matrix_to_sixd(_axis_angle_to_matrix(aa)).reshape(n, 132)
    # Author against the native Y-up SMPL template, then convert the complete clip to MAESTRO's
    # Z-up editing frame. ``trans`` above is expressed in desired Z-up coordinates.
    native_trans = np.stack([trans[:, 0], trans[:, 2], -trans[:, 1]], axis=1)
    native = np.concatenate([native_trans, rotations, contacts], axis=1).astype(np.float32)
    return to_zup(native)


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = BANK / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for raw in payload["motions"]:
        motion = build_motion(raw["id"], int(raw["frames"]))
        np.save(BANK / raw["clip"], motion)
        print(f"{raw['id']}: {motion.shape}")
    bank = MotionBank(BANK)
    for spec in bank.specs:
        bank.load_clip(spec)
    print(f"validated {len(bank.specs)} named motions")


if __name__ == "__main__":
    main()
