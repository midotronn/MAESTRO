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
from server.fk import _template, compute_poses  # noqa: E402

BANK = ROOT / "assets" / "motion_bank"
MANIFEST = BANK / "manifest.json"
_FOOT_JOINTS = (7, 8, 10, 11)


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


# The SMPL rest pose is a T-pose. Authoring straight off it makes every clip read as a mannequin
# rather than a dancer, so all clips start from a common "ready" stance: arms carried below
# horizontal with softly flexed elbows, and slightly bent knees.
_STANCE_ARM = 0.95
_STANCE_ELBOW = 0.28
_STANCE_KNEE = 0.10


def _apply_stance(aa):
    aa[:, 16, 2] -= _STANCE_ARM                 # carry the arms down out of the T-pose
    aa[:, 17, 2] += _STANCE_ARM
    aa[:, 18, 1] += _STANCE_ELBOW               # soft elbows
    aa[:, 19, 1] -= _STANCE_ELBOW
    aa[:, 4, 0] += _STANCE_KNEE                 # athletic knee bend
    aa[:, 5, 0] += _STANCE_KNEE


def _stance_wrist(side: str) -> np.ndarray:
    """Where the wrist sits in the ready stance, so IK-driven arms rest there instead of a T-pose."""
    J = _template()
    shoulder, wrist = (16, 20) if side == "left" else (17, 21)
    sign = -1.0 if side == "left" else 1.0
    rot = _axis_angle_to_matrix(np.array([0.0, 0.0, sign * _STANCE_ARM], dtype=np.float32))
    return (J[shoulder] + rot @ (J[wrist] - J[shoulder])).astype(np.float32)


def _performance_layer(aa, trans, t, *, sway=1.0, breath=1.0):
    """Breathing, weight shift, and torso counter-rotation so no clip is a frozen pose.

    Every term is multiplied by an envelope that vanishes at t=0 and t=1, so the first and last
    frame are untouched and the manifest validators that compare clip endpoints (root displacement,
    root yaw, root level, vertical peak) measure exactly what the authoring recipe produced.
    """
    env = np.sin(np.pi * t)
    cycle = np.sin(2.0 * np.pi * t) * env
    aa[:, 3, 0] += 0.022 * breath * cycle
    aa[:, 6, 0] -= 0.014 * breath * cycle
    aa[:, 12, 0] += 0.016 * breath * cycle
    aa[:, 0, 2] += 0.030 * sway * cycle
    aa[:, 9, 2] -= 0.022 * sway * cycle


def _arm_swing(aa, phase, amount=0.34):
    """Counter-phase arm swing driven by the gait.

    A single sign of shoulder yaw swings the two arms in opposite world directions (they start on
    opposite sides of the body), which is exactly the opposition a walking gait needs.
    """
    s = np.sin(phase)
    aa[:, 16, 1] -= amount * s
    aa[:, 17, 1] -= amount * s
    aa[:, 18, 1] += 0.40 * amount * np.abs(s)
    aa[:, 19, 1] -= 0.40 * amount * np.abs(s)
    aa[:, 9, 1] += 0.16 * amount * s                # torso counter-rotates against the arms


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
    rest = _stance_wrist(side)
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
    _apply_stance(aa)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    phase = 2.0 * np.pi * t

    if motion_id in {"clap_single", "clap_overhead"}:
        overhead = motion_id == "clap_overhead"
        hit = _pulse(t, 0.54, 0.15)
        wind = _pulse(t, 0.30, 0.11)                     # anticipation before the hit
        settle = _pulse(t, 0.80, 0.13)                   # follow-through after it
        _clap_arms(aa, np.clip(hit - 0.22 * wind, 0.0, None), overhead=overhead)
        aa[:, 3, 0] += (0.20 if overhead else 0.13) * hit - 0.10 * wind
        aa[:, 9, 0] += (0.22 if overhead else 0.08) * hit
        aa[:, 12, 0] -= (0.26 if overhead else 0.06) * hit    # look up into an overhead clap
        aa[:, 4, 0] += 0.16 * wind + 0.10 * settle           # load and absorb through the knees
        aa[:, 5, 0] += 0.16 * wind + 0.10 * settle
    elif motion_id == "clap_repeat":
        hit = np.maximum.reduce([_pulse(t, c, 0.07) for c in (0.27, 0.52, 0.77)])
        _clap_arms(aa, hit)
        trans[:, 2] += 0.025 * np.sin(6.0 * np.pi * t)
        aa[:, 3, 0] += 0.10 * hit
        aa[:, 0, 2] += 0.10 * np.sin(3.0 * np.pi * t)        # groove side to side across the claps
        aa[:, 4, 0] += 0.14 * (1.0 - hit)
        aa[:, 5, 0] += 0.14 * (1.0 - hit)
    elif motion_id in {"jump_two_foot", "jump_arms_up"}:
        air = np.sin(np.pi * _smoothstep((t - 0.18) / 0.64))
        air = np.maximum(air, 0.0)
        trans[:, 2] += (0.22 if motion_id == "jump_two_foot" else 0.27) * air
        crouch = _pulse(t, 0.17, 0.10) + 0.45 * _pulse(t, 0.85, 0.10)
        aa[:, 1, 0] -= 0.42 * crouch
        aa[:, 2, 0] -= 0.42 * crouch
        aa[:, 4, 0] += 0.75 * crouch
        aa[:, 5, 0] += 0.75 * crouch
        aa[:, 3, 0] += 0.24 * crouch - 0.12 * air            # fold to load, extend in the air
        aa[:, 12, 0] -= 0.14 * air
        contacts[air > 0.22] = 0.0
        if motion_id == "jump_arms_up":
            left = np.array([0.25, 0.48, -0.04], dtype=np.float32)
            right = np.array([-0.25, 0.48, -0.04], dtype=np.float32)
            _solve_arm(aa, "left", _arm_targets("left", air, left))
            _solve_arm(aa, "right", _arm_targets("right", air, right))
        else:
            aa[:, 16, 1] -= 0.55 * crouch - 0.30 * air       # arms drive the two-foot takeoff
            aa[:, 17, 1] += 0.55 * crouch - 0.30 * air
    elif motion_id == "bounce_in_place":
        bounce = 0.5 - 0.5 * np.cos(6.0 * np.pi * t)
        trans[:, 2] += 0.075 * bounce
        aa[:, 4, 0] += 0.32 * (1.0 - bounce)
        aa[:, 5, 0] += 0.32 * (1.0 - bounce)
        aa[:, 3, 0] += 0.1 * np.sin(6.0 * np.pi * t)
        aa[:, 0, 2] += 0.13 * np.sin(3.0 * np.pi * t)        # hips answer every other bounce
        aa[:, 16, 1] -= 0.16 * np.sin(3.0 * np.pi * t)
        aa[:, 17, 1] -= 0.16 * np.sin(3.0 * np.pi * t)
    elif motion_id == "wave":
        raise_arm = _smoothstep(t / 0.28) * (1.0 - _smoothstep((t - 0.78) / 0.22))
        target = np.column_stack([
            -0.30 + 0.08 * np.sin(8.0 * np.pi * t),
            np.full(n, 0.43),
            np.full(n, -0.08),
        ]).astype(np.float32)
        rest = _stance_wrist("right")
        target = rest[None, :] + raise_arm[:, None] * (target - rest[None, :])
        _solve_arm(aa, "right", target)
        aa[:, 21, 0] += 0.65 * np.sin(8.0 * np.pi * t) * raise_arm
        aa[:, 14, 2] += 0.20 * raise_arm                     # right collar lifts with the arm
        aa[:, 9, 2] += 0.12 * raise_arm                      # torso leans toward the wave
        aa[:, 12, 2] += 0.14 * raise_arm                     # head tilts with it
        aa[:, 12, 1] -= 0.16 * raise_arm
    elif motion_id == "point_side":
        point = _smoothstep(t / 0.42) * (1.0 - 0.35 * _smoothstep((t - 0.82) / 0.18))
        # Reach across the body's side, not diagonally forward: the arm is roughly 0.58 long, so
        # a target 0.55 out and 0.18 ahead of the shoulder reads as a side point while staying
        # legible to a front-on camera. A larger forward term makes it a forward reach instead.
        target = np.array([-0.72, 0.12, -0.18], dtype=np.float32)
        _solve_arm(aa, "right", _arm_targets("right", point, target))
        aa[:, 9, 2] += 0.16 * point
        aa[:, 12, 1] -= 0.38 * point                         # sustained: the head follows the point
        aa[:, 0, 1] -= 0.16 * point
        aa[:, 0, 2] -= 0.10 * point                          # weight settles onto the far leg
    elif motion_id == "celebrate_hands_up":
        rise = _smoothstep(t / 0.55)
        pump = 0.5 - 0.5 * np.cos(4.0 * np.pi * np.clip((t - 0.45) / 0.55, 0.0, 1.0))
        # A wide, open V held high with a double pump -- distinct from the single continuous
        # rise_reach, which travels up from a crouch instead.
        _solve_arm(aa, "left", _arm_targets(
            "left", rise, np.array([0.46, 0.50, 0.06], dtype=np.float32)))
        _solve_arm(aa, "right", _arm_targets(
            "right", rise, np.array([-0.46, 0.50, 0.06], dtype=np.float32)))
        trans[:, 2] += 0.06 * rise + 0.035 * pump * rise
        aa[:, 3, 0] -= 0.20 * rise                           # chest opens upward
        aa[:, 12, 0] -= 0.30 * rise
        aa[:, 0, 2] += 0.14 * np.sin(4.0 * np.pi * t) * rise
        aa[:, 4, 0] += 0.20 * (1.0 - rise)
        aa[:, 5, 0] += 0.20 * (1.0 - rise)
    elif motion_id == "chest_pop":
        hit = _pulse(t, 0.52, 0.09)
        wind = _pulse(t, 0.33, 0.10)
        settle = _pulse(t, 0.74, 0.12)
        drive = hit - 0.35 * wind - 0.18 * settle            # wind up, snap, then release
        aa[:, 3, 0] -= 0.62 * drive
        aa[:, 6, 0] += 0.78 * drive
        aa[:, 9, 0] += 0.66 * drive
        aa[:, 12, 0] -= 0.34 * drive
        aa[:, 0, 0] += 0.26 * drive                          # hips counter the chest
        aa[:, 16, 2] -= 0.30 * hit
        aa[:, 17, 2] += 0.30 * hit
        aa[:, 4, 0] += 0.18 * hit
        aa[:, 5, 0] += 0.18 * hit
        trans[:, 1] -= 0.035 * drive
    elif motion_id == "arm_punch":
        hit = _pulse(t, 0.55, 0.10)
        recoil = _pulse(t, 0.78, 0.12)
        target = np.array([-0.12, 0.04, -0.72], dtype=np.float32)
        _solve_arm(aa, "right", _arm_targets(
            "right", np.clip(hit - 0.30 * recoil, 0.0, None), target))
        aa[:, 9, 1] -= 0.42 * hit                            # sharp: the whole torso rotates in
        aa[:, 0, 1] -= 0.22 * hit
        aa[:, 16, 1] += 0.40 * hit                           # opposite arm pulls back
        aa[:, 12, 1] -= 0.12 * hit
        aa[:, 4, 0] += 0.20 * hit
        trans[:, 1] += 0.06 * hit
    elif motion_id == "side_step":
        # A lateral weight transfer, NOT a walking gait: the feet stay square to the front, the
        # hips carry the weight sideways, and the torso counter-leans. This is what separates it
        # from step_forward, which uses an actual alternating gait.
        shift = _smoothstep(t)
        reach = np.sin(np.pi * t)
        trans[:, 0] += 0.42 * shift
        aa[:, 1, 2] += 0.34 * reach                          # lead leg reaches out of the stance
        aa[:, 2, 2] -= 0.20 * reach                          # trailing leg spreads the other way
        aa[:, 4, 0] += 0.30 * reach
        aa[:, 5, 0] += 0.16 * reach
        aa[:, 0, 2] += 0.26 * reach                          # hips ride the weight across
        aa[:, 9, 2] -= 0.24 * reach                          # shoulders counter-lean
        aa[:, 16, 2] -= 0.28 * reach
        aa[:, 17, 2] -= 0.28 * reach                         # both arms drift with the momentum
        contacts[:, 0:2] = (reach < 0.55)[:, None]
    elif motion_id == "step_touch":
        # Step out and close: two lateral steps that each end with the feet together.
        cycle = np.sin(2.0 * np.pi * t)
        trans[:, 0] += 0.30 * _smoothstep(t)
        trans[:, 0] += 0.06 * np.sin(4.0 * np.pi * t)        # the out-and-close within each step
        out = np.abs(np.sin(2.0 * np.pi * t))
        aa[:, 1, 2] += 0.30 * out                            # the stance opens, then closes again
        aa[:, 2, 2] -= 0.14 * out
        aa[:, 4, 0] += 0.26 * out
        aa[:, 5, 0] += 0.14 * out
        aa[:, 0, 2] += 0.22 * cycle
        aa[:, 9, 2] -= 0.18 * cycle
        aa[:, 16, 2] -= 0.24 * out
        aa[:, 17, 2] -= 0.24 * out
        aa[:, 12, 2] += 0.10 * cycle
        contacts[:, 0:2] = (cycle <= 0)[:, None]
        contacts[:, 2:4] = (cycle >= 0)[:, None]
    elif motion_id in {"step_forward", "step_backward"}:
        direction = 1.0 if motion_id == "step_forward" else -1.0
        trans[:, 1] += direction * 0.46 * _smoothstep(t)
        _legs(aa, 2.0 * np.pi * t, 0.55 * direction)
        _arm_swing(aa, 2.0 * np.pi * t, 0.34 * direction)    # opposite arm to the leading leg
        aa[:, 3, 0] += direction * 0.10 * np.sin(np.pi * t)  # lean into the travel direction
        contacts[:, 0:2] = (np.sin(2.0 * np.pi * t) <= 0)[:, None]
        contacts[:, 2:4] = (np.sin(2.0 * np.pi * t) >= 0)[:, None]
    elif motion_id in {"turn_quarter", "turn_half"}:
        angle = (np.pi / 2.0) if motion_id == "turn_quarter" else np.pi
        spin = _smoothstep(t)
        aa[:, 0, 1] = angle * spin
        _legs(aa, 2.0 * np.pi * t, 0.28)
        _arm_swing(aa, 2.0 * np.pi * t, 0.22)
        trans[:, 2] += 0.025 * np.sin(2.0 * np.pi * t) ** 2
        spot = np.sin(np.pi * t)
        aa[:, 12, 1] += 0.30 * spot                          # head leads, then spots the turn
        aa[:, 9, 1] -= 0.18 * spot
        aa[:, 4, 0] += 0.18 * spot
        aa[:, 5, 0] += 0.18 * spot
    elif motion_id == "body_roll":
        for joint, center in ((3, 0.30), (6, 0.48), (9, 0.66)):
            aa[:, joint, 0] += 0.48 * _pulse(t, center, 0.15)
        aa[:, 12, 0] -= 0.18 * _pulse(t, 0.72, 0.16)
        trans[:, 2] += 0.035 * np.sin(2.0 * np.pi * t)
        aa[:, 0, 0] += 0.30 * _pulse(t, 0.20, 0.14)          # the wave starts in the hips
        aa[:, 4, 0] += 0.26 * _pulse(t, 0.24, 0.16)
        aa[:, 5, 0] += 0.26 * _pulse(t, 0.24, 0.16)
        aa[:, 16, 2] -= 0.18 * _pulse(t, 0.62, 0.20)         # arms open as the chest arrives
        aa[:, 17, 2] += 0.18 * _pulse(t, 0.62, 0.20)
    elif motion_id == "crouch_drop":
        down = _smoothstep(t / 0.68)
        # The root height is NOT authored here: _ground solves it from the pose so the feet stay
        # planted. Hand-picking a drop distance sinks the feet through the floor whenever it
        # disagrees with how much the bent legs actually shorten.
        aa[:, 1, 0] -= 0.95 * down
        aa[:, 2, 0] -= 0.95 * down
        aa[:, 4, 0] += 1.85 * down
        aa[:, 5, 0] += 1.85 * down
        aa[:, 3, 0] += 0.3 * down
        aa[:, 12, 0] += 0.22 * down                          # head drops with the body
        aa[:, 16, 1] -= 0.45 * down                          # arms sweep down and forward
        aa[:, 17, 1] -= 0.45 * down
        aa[:, 18, 1] += 0.30 * down
        aa[:, 19, 1] -= 0.30 * down
    elif motion_id == "rise_reach":
        rise = _smoothstep(t / 0.72)
        fall = 1.0 - rise
        aa[:, 1, 0] -= 0.95 * fall
        aa[:, 2, 0] -= 0.95 * fall
        aa[:, 4, 0] += 1.85 * fall
        aa[:, 5, 0] += 1.85 * fall
        # One continuous reach up and slightly forward, with the leading arm ahead of the other,
        # so it reads as a rise rather than the symmetric held V of celebrate_hands_up.
        _solve_arm(aa, "left", _arm_targets(
            "left", rise, np.array([0.14, 0.55, -0.20], dtype=np.float32)))
        _solve_arm(aa, "right", _arm_targets(
            "right", _smoothstep(np.clip((t - 0.12) / 0.72, 0.0, 1.0)),
            np.array([-0.20, 0.47, -0.10], dtype=np.float32)))
        aa[:, 3, 0] -= 0.26 * rise                           # spine extends through the rise
        aa[:, 12, 0] -= 0.24 * rise
    else:
        raise KeyError(f"no procedural authoring recipe for {motion_id}")

    _performance_layer(aa, trans, t)
    rotations = _matrix_to_sixd(_axis_angle_to_matrix(aa)).reshape(n, 132)
    # Author against the native Y-up SMPL template, then convert the complete clip to MAESTRO's
    # Z-up editing frame. ``trans`` above is expressed in desired Z-up coordinates.
    native_trans = np.stack([trans[:, 0], trans[:, 2], -trans[:, 1]], axis=1)
    native = np.concatenate([native_trans, rotations, contacts], axis=1).astype(np.float32)
    return _ground(to_zup(native))


def _standing_floor() -> float:
    """Foot height of the neutral standing pose: the floor every planted foot should rest on."""
    n = 2
    rest = _matrix_to_sixd(_axis_angle_to_matrix(np.zeros((n, 22, 3), dtype=np.float32)))
    neutral = np.concatenate([
        np.zeros((n, 3), dtype=np.float32),
        rest.reshape(n, 132),
        np.ones((n, 4), dtype=np.float32),
    ], axis=1).astype(np.float32)
    joints = compute_poses(to_zup(neutral))["fk_joints"]
    return float(np.min(joints[:, _FOOT_JOINTS, 2]))


def _ground(clip: np.ndarray) -> np.ndarray:
    """Solve root height from the pose so planted feet rest on the floor.

    Hand-authoring a root drop alongside a knee bend silently sinks the feet through the ground
    whenever the two disagree. Instead the vertical offset is derived per frame: every frame with a
    foot contact is placed exactly on the standing floor, and airborne spans keep their authored arc
    by interpolating the correction across them.
    """
    joints = compute_poses(clip)["fk_joints"]
    lowest = np.min(joints[:, _FOOT_JOINTS, 2], axis=1)
    grounded = np.asarray(clip[:, 135:139]).sum(axis=1) > 0
    if not grounded.any():
        return clip
    idx = np.arange(clip.shape[0])
    delta = np.interp(idx, idx[grounded], (_standing_floor() - lowest)[grounded])
    out = np.array(clip, dtype=np.float32, copy=True)
    out[:, 2] += delta.astype(np.float32)
    return out


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
