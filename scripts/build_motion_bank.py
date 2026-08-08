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
from server.fk import BODY_PARENTS, _template, compute_poses  # noqa: E402

BANK = ROOT / "assets" / "motion_bank"
MANIFEST = BANK / "manifest.json"
_FOOT_JOINTS = (7, 8, 10, 11)

# Which way is forward? The two frames in this file disagree, and getting it wrong is silent:
# the manifest's ``root_displacement`` validator is unsigned, so a clip that travels backwards
# satisfies the forward step's contract exactly.
#
#   * IK targets are absolute positions in the native SMPL template frame, which faces +Z.
#     A target with a POSITIVE z reaches in FRONT of the body.
#   * ``trans`` is in MAESTRO's Z-up editing frame, where y = -z_native. Its +Y therefore
#     points BEHIND the dancer, so travelling forward means SUBTRACTING from ``trans[:, 1]``.
#
# Both were once assumed to be the other way round, which authored every clap, punch and reach
# behind the body and made step_forward moonwalk. ``_travel`` exists so the sign is stated once.


def _travel(trans, distance):
    """Move the root along the way the dancer faces; positive distance is forwards."""
    trans[:, 1] -= distance


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


def _native_joint_positions(axis_angles: np.ndarray) -> np.ndarray:
    """Forward-kinematic joint positions in the native SMPL authoring frame."""
    local = _axis_angle_to_matrix(np.asarray(axis_angles, dtype=np.float32))
    template = _template()[:22]
    global_r = np.empty_like(local)
    positions = np.empty((22, 3), dtype=np.float32)
    global_r[0] = local[0]
    positions[0] = template[0]
    for joint in range(1, 22):
        parent = BODY_PARENTS[joint]
        positions[joint] = (
            positions[parent]
            + global_r[parent] @ (template[joint] - template[parent])
        )
        global_r[joint] = global_r[parent] @ local[joint]
    return positions


def _stance_ankle(side: str) -> np.ndarray:
    """Ankle position of the shared ready stance in the native authoring frame."""
    stance = np.zeros((1, 22, 3), dtype=np.float32)
    _apply_stance(stance)
    ankle = 7 if side == "left" else 8
    return _native_joint_positions(stance[0])[ankle]


def _phase_ease(t: np.ndarray, start: float, end: float) -> np.ndarray:
    return _smoothstep(np.clip((t - start) / max(end - start, 1e-6), 0.0, 1.0))


def _foot_swing(
    t: np.ndarray,
    start: float,
    end: float,
    origin: np.ndarray,
    target: np.ndarray,
    *,
    lift: float,
) -> np.ndarray:
    """Move one ankle between planted targets with a visible airborne arc."""
    raw = np.clip((t - start) / max(end - start, 1e-6), 0.0, 1.0)
    progress = _smoothstep(raw)
    origin = np.asarray(origin, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    out = origin[None, :] + progress[:, None] * (target - origin)[None, :]
    active = (t > start) & (t < end)
    out[:, 1] += float(lift) * np.sin(np.pi * raw) * active
    return out


def _solve_leg(aa: np.ndarray, side: str, ankle_targets: np.ndarray) -> None:
    """Two-bone leg IK with a forward knee and a level foot."""
    hip, knee, ankle = (1, 4, 7) if side == "left" else (2, 5, 8)
    template = _template()[:22]
    upper = template[knee] - template[hip]
    lower = template[ankle] - template[knee]
    upper_len = float(np.linalg.norm(upper))
    lower_len = float(np.linalg.norm(lower))
    knee_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for frame, foot in enumerate(np.asarray(ankle_targets, dtype=np.float32)):
        root_global = _axis_angle_to_matrix(aa[frame, 0])
        origin = template[0] + root_global @ (template[hip] - template[0])
        delta = foot - origin
        raw_distance = float(np.linalg.norm(delta))
        distance = float(np.clip(
            raw_distance,
            abs(upper_len - lower_len) + 1e-4,
            upper_len + lower_len - 1e-4,
        ))
        direction = delta / (raw_distance + 1e-9)
        along = (
            upper_len * upper_len
            - lower_len * lower_len
            + distance * distance
        ) / (2.0 * distance)
        bend_height = np.sqrt(max(0.0, upper_len * upper_len - along * along))
        bend = knee_hint - np.dot(knee_hint, direction) * direction
        bend /= np.linalg.norm(bend) + 1e-9
        knee_target = origin + along * direction + bend_height * bend

        upper_global = _align_vector(upper, knee_target - origin)
        lower_global = _align_vector(lower, foot - knee_target)
        aa[frame, hip] = _matrix_to_axis_angle(root_global.T @ upper_global)
        aa[frame, knee] = _matrix_to_axis_angle(upper_global.T @ lower_global)
        aa[frame, ankle] = _matrix_to_axis_angle(lower_global.T)


# Where the elbow swings to when the wrist target leaves it free to choose. Two positions of
# the arm reach the same hand target -- elbow out to the side, or elbow tucked down by the ribs
# -- and the IK picks between them with this hint alone. The default carries the elbow outward,
# which suits a reach or a punch; a motion that works in front of the chest must say otherwise
# or it comes out flapping its arms. ``x`` is signed per side inside ``_solve_arm``.
_ELBOW_OUT = np.array([1.0, 0.15, 0.0], dtype=np.float32)


def _clap_wrist_rotation(side: str, forearm_global: np.ndarray) -> np.ndarray:
    """Return a local wrist rotation with upright fingers and inward-facing palms."""
    sign = 1.0 if side == "left" else -1.0
    rest_finger = np.array([sign, 0.0, 0.0], dtype=np.float32)
    rest_palm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    rest_width = np.cross(rest_finger, rest_palm)
    desired_finger = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    desired_palm = np.array([-sign, 0.0, 0.0], dtype=np.float32)
    desired_width = np.cross(desired_finger, desired_palm)
    rest_basis = np.stack([rest_finger, rest_palm, rest_width], axis=1)
    desired_basis = np.stack([desired_finger, desired_palm, desired_width], axis=1)
    hand_global = desired_basis @ rest_basis.T
    return forearm_global.T @ hand_global


def _solve_arm(
    aa,
    side: str,
    targets: np.ndarray,
    *,
    elbow_hint=_ELBOW_OUT,
    clap_orientation: np.ndarray | None = None,
):
    joints = (16, 18, 20) if side == "left" else (17, 19, 21)
    shoulder, elbow, wrist = joints
    J = _template()
    upper = J[elbow] - J[shoulder]
    fore = J[wrist] - J[elbow]
    l1, l2 = float(np.linalg.norm(upper)), float(np.linalg.norm(fore))
    sign = 1.0 if side == "left" else -1.0
    elbow_hint = np.asarray(elbow_hint, dtype=np.float32)
    for frame, hand in enumerate(targets):
        origin = J[shoulder]
        delta = hand - origin
        distance = float(np.clip(np.linalg.norm(delta), abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
        direction = delta / (np.linalg.norm(delta) + 1e-9)
        a = (l1 * l1 - l2 * l2 + distance * distance) / (2.0 * distance)
        height = np.sqrt(max(0.0, l1 * l1 - a * a))
        hint = np.array([sign * elbow_hint[0], elbow_hint[1], elbow_hint[2]], dtype=np.float32)
        bend = hint - np.dot(hint, direction) * direction
        bend /= np.linalg.norm(bend) + 1e-9
        elbow_target = origin + a * direction + height * bend
        upper_global = _align_vector(upper, elbow_target - origin)
        fore_global = _align_vector(fore, hand - elbow_target)
        elbow_local = upper_global.T @ fore_global
        aa[frame, shoulder] = _matrix_to_axis_angle(upper_global)
        aa[frame, elbow] = _matrix_to_axis_angle(elbow_local)
        if clap_orientation is not None:
            wrist_local = _clap_wrist_rotation(side, fore_global)
            weight = float(np.clip(clap_orientation[frame], 0.0, 1.0))
            aa[frame, wrist] = _matrix_to_axis_angle(wrist_local) * weight


# Half the IK target separation at contact. Wrist rotation moves the rendered hand surfaces
# inward from those targets, so a small positive separation produces palm contact without asking
# the forearms to cross the body's midline.
_CLAP_HALF_GAP = 0.004

# Where the palms meet. In the template frame the shoulders sit at y=+0.083 and the chest
# (spine3) at y=-0.057, so a clap belongs a little BELOW the chest line. This was authored at
# y=+0.04 -- above the chest, level with the collarbones -- which renders as hands pressed
# together under the chin, closer to a bow than a clap.
_CLAP_POINT = (-0.045, 0.26)                        # (height, distance in front)

# Clapping hands are carried in front of the sternum, so the elbows hang DOWN and only a little
# outward. Left to the default outward hint the solver lifts them to shoulder height and the
# clip reads as flapping wings.
_CLAP_ELBOW = np.array([0.55, -1.0, -0.1], dtype=np.float32)

# How far towards the meeting point the hands stay BETWEEN repeated claps: 1.0 would leave them
# touching, 0.0 drops them to the hips. Sets the parting of the hands, so it trades readability
# of each clap (needs travel) against the hands staying up (needs little).
_CLAP_GUARD = 0.68


def _arm_targets(side: str, signal: np.ndarray, target: np.ndarray) -> np.ndarray:
    rest = _stance_wrist(side)
    return rest[None, :] + signal[:, None] * (np.asarray(target) - rest)[None, :]


def _clap_arms(aa, signal, *, overhead=False):
    # The overhead variant has to clear the head to read as overhead, and reaching forward
    # costs height, so it trades a little of the forward travel back for lift.
    #
    # The palms meet at the midline, so each WRIST stops half a hand short of it. Aiming both
    # hands at x=0 asks two wrists to occupy one point: the solver obliges, the forearms pass
    # through each other, and the clap reads as folded, tangled arms rather than a clap.
    half = _CLAP_HALF_GAP
    y, z = (0.62, 0.18) if overhead else _CLAP_POINT
    hint = _CLAP_ELBOW
    _solve_arm(aa, "left", _arm_targets("left", signal, np.array([half, y, z], dtype=np.float32)),
               elbow_hint=hint, clap_orientation=signal)
    _solve_arm(aa, "right", _arm_targets("right", signal, np.array([-half, y, z], dtype=np.float32)),
               elbow_hint=hint, clap_orientation=signal)


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
    leg_targets = None

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
        # Someone clapping three times keeps their hands UP between the claps and parts them a
        # few inches. Driving the arms straight off ``hit`` sends them all the way back to the
        # hips after every clap, and a 0.84m swing three times over reads as flapping, not
        # clapping. So raise them once, hold that guard across the phrase, and let each hit
        # close the last of the gap. The envelope still vanishes at both ends, which is what
        # keeps the clip's first and last frame on the shared stance the splice hands over on.
        ready = _smoothstep(t / 0.18) * _smoothstep((1.0 - t) / 0.18)
        _clap_arms(aa, ready * (_CLAP_GUARD + (1.0 - _CLAP_GUARD) * hit))
        trans[:, 2] += 0.025 * np.sin(6.0 * np.pi * t)
        aa[:, 3, 0] += 0.10 * hit
        aa[:, 0, 2] += 0.10 * np.sin(3.0 * np.pi * t)        # groove side to side across the claps
        aa[:, 4, 0] += 0.14 * (1.0 - hit)
        aa[:, 5, 0] += 0.14 * (1.0 - hit)
    elif motion_id in {"jump_two_foot", "jump_arms_up"}:
        air = np.sin(np.pi * _smoothstep((t - 0.22) / 0.58))
        air = np.maximum(air, 0.0)
        trans[:, 2] += (0.30 if motion_id == "jump_two_foot" else 0.36) * air
        load = _pulse(t, 0.18, 0.09)
        land = _pulse(t, 0.82, 0.09)
        crouch = load + 0.72 * land
        aa[:, 1, 0] -= 0.58 * crouch
        aa[:, 2, 0] -= 0.58 * crouch
        aa[:, 4, 0] += 1.02 * crouch
        aa[:, 5, 0] += 1.02 * crouch
        aa[:, 7, 0] -= 0.18 * air
        aa[:, 8, 0] -= 0.18 * air
        aa[:, 3, 0] += 0.32 * load + 0.20 * land - 0.16 * air
        aa[:, 12, 0] -= 0.18 * air
        contacts[air > 0.12] = 0.0
        if motion_id == "jump_arms_up":
            left = np.array([0.25, 0.48, 0.04], dtype=np.float32)
            right = np.array([-0.25, 0.48, 0.04], dtype=np.float32)
            _solve_arm(aa, "left", _arm_targets("left", air, left))
            _solve_arm(aa, "right", _arm_targets("right", air, right))
        else:
            aa[:, 16, 1] -= 0.70 * load + 0.38 * land - 0.48 * air
            aa[:, 17, 1] += 0.70 * load + 0.38 * land - 0.48 * air
    elif motion_id == "bounce_in_place":
        compression = 0.5 - 0.5 * np.cos(6.0 * np.pi * t)
        # No authored root rise here: _ground re-solves pelvis height from the leg pose on every
        # grounded frame, so a hand-written trans[:, 2] term is silently cancelled. The bounce has
        # to come from the knees. Start and end tall, with three internal compressions, so the
        # splice hands over on the shared stance instead of entering already crouched.
        aa[:, 4, 0] += 0.82 * compression
        aa[:, 5, 0] += 0.82 * compression
        aa[:, 3, 0] += 0.1 * np.sin(6.0 * np.pi * t)
        aa[:, 0, 2] += 0.13 * np.sin(3.0 * np.pi * t)        # hips answer every other bounce
        aa[:, 16, 1] -= 0.16 * np.sin(3.0 * np.pi * t)
        aa[:, 17, 1] -= 0.16 * np.sin(3.0 * np.pi * t)
    elif motion_id == "wave":
        raise_arm = _smoothstep(t / 0.28) * (1.0 - _smoothstep((t - 0.78) / 0.22))
        target = np.column_stack([
            -0.30 + 0.08 * np.sin(8.0 * np.pi * t),
            np.full(n, 0.43),
            np.full(n, 0.08),
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
        target = np.array([-0.72, 0.12, 0.18], dtype=np.float32)
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
        # A chest pop is an isolation: the ribcage shoots forward while the head stays level.
        # Only joints BELOW the chest can move the chest -- rotating spine3 or the neck swings
        # the head instead. The old recipe drove spine3 (+0.66) and leaned on the neck to
        # counter it, so the chest sat at exactly its rest offset while the head speared 0.27 m
        # forward and 0.13 m down: a pigeon peck, not a pop. The forward drive now lives at
        # pelvis/spine1/spine2, and spine3 plus the neck give back that same total so the
        # shoulders open and the head arrives level instead of leading.
        pop_scale = 1.65
        base = pop_scale * 0.07 * drive                      # hips stay under the ribcage
        lower = pop_scale * 0.26 * drive
        mid = pop_scale * 0.23 * drive                       # the chest travels off these three
        # Cancelling only the rotation is not enough to hold the head: the chain below has
        # already carried it forward, so the counter has to over-rotate to bring it back. The
        # extra 0.34 is the arcsin of that leftover head travel over the spine3->head length.
        counter = base + lower + mid + pop_scale * 0.34 * drive
        aa[:, 0, 0] += base
        aa[:, 3, 0] += lower
        aa[:, 6, 0] += mid
        aa[:, 9, 0] -= 0.58 * counter                        # shoulders open rather than dive
        aa[:, 12, 0] -= 0.42 * counter                       # head holds its place
        aa[:, 16, 2] -= 0.48 * hit
        aa[:, 17, 2] += 0.48 * hit
        aa[:, 4, 0] += 0.30 * hit
        aa[:, 5, 0] += 0.30 * hit
        _travel(trans, -0.055 * drive)                       # hips settle back off the chest
    elif motion_id == "arm_punch":
        ready = _smoothstep(t / 0.24) * _smoothstep((1.0 - t) / 0.18)
        wind = _pulse(t, 0.34, 0.09)
        hit = _pulse(t, 0.55, 0.12)
        right_rest = _stance_wrist("right")
        right_guard = np.array([-0.24, 0.16, 0.20], dtype=np.float32)
        right_strike = np.array([-0.14, 0.08, 0.72], dtype=np.float32)
        right_targets = (
            right_rest[None, :]
            + ready[:, None] * (right_guard - right_rest)[None, :]
            + hit[:, None] * (right_strike - right_guard)[None, :]
        )
        left_rest = _stance_wrist("left")
        left_guard = np.array([0.18, 0.12, 0.22], dtype=np.float32)
        left_tuck = np.array([0.12, 0.10, 0.18], dtype=np.float32)
        left_targets = (
            left_rest[None, :]
            + ready[:, None] * (left_guard - left_rest)[None, :]
            + hit[:, None] * (left_tuck - left_guard)[None, :]
        )
        _solve_arm(aa, "right", right_targets)
        _solve_arm(aa, "left", left_targets)
        aa[:, 9, 1] += 0.12 * wind - 0.56 * hit              # wind up, then rotate into the hit
        aa[:, 0, 1] += 0.08 * wind - 0.26 * hit
        aa[:, 12, 1] -= 0.12 * hit
        aa[:, 3, 0] += 0.14 * hit
        aa[:, 4, 0] += 0.28 * hit
        _travel(trans, 0.075 * hit)                          # a short step into the punch
    elif motion_id == "side_step":
        # The lead foot opens and the pelvis settles over it while the trailing foot stays
        # planted. Holding the wide stance is what distinguishes this from a step-and-touch.
        left_start = _stance_ankle("left")
        right_start = _stance_ankle("right")
        left_end = left_start.copy()
        left_end[0] += 0.34
        trans[:, 0] += 0.24 * _phase_ease(t, 0.30, 0.64)
        left_targets = _foot_swing(
            t, 0.05, 0.45, left_start, left_end, lift=0.09,
        )
        right_targets = np.repeat(right_start[None, :], n, axis=0)
        leg_targets = (left_targets, right_targets)

        settle = _phase_ease(t, 0.25, 0.62)
        aa[:, 0, 2] += 0.18 * settle
        aa[:, 9, 2] -= 0.15 * settle
        reach = np.sin(np.pi * t)
        aa[:, 16, 2] -= 0.25 * reach
        aa[:, 17, 2] -= 0.25 * reach
        contacts[:, 0:2] = ((t <= 0.05) | (t >= 0.42))[:, None]
        contacts[:, 2:4] = 1.0
    elif motion_id == "step_touch":
        # The lead foot accepts the weight before the trailing foot lifts and taps beside it.
        # The pelvis stops during the tap instead of continuing into another lateral step.
        left_start = _stance_ankle("left")
        right_start = _stance_ankle("right")
        left_end = left_start.copy()
        left_end[0] += 0.31
        right_touch = left_end.copy()
        right_touch[0] -= 0.085
        trans[:, 0] += 0.29 * _phase_ease(t, 0.34, 0.62)

        left_targets = _foot_swing(
            t, 0.05, 0.46, left_start, left_end, lift=0.085,
        )
        right_targets = _foot_swing(
            t, 0.50, 0.77, right_start, right_touch, lift=0.075,
        )
        right_finish = right_start.copy()
        right_finish[0] += float(trans[-1, 0])
        right_targets += (
            _phase_ease(t, 0.88, 1.0)[:, None]
            * (right_finish - right_touch)[None, :]
        )
        leg_targets = (left_targets, right_targets)

        groove = np.sin(np.pi * t)
        aa[:, 0, 2] += 0.10 * groove
        aa[:, 9, 2] -= 0.10 * groove
        aa[:, 16, 2] -= 0.20 * groove
        aa[:, 17, 2] -= 0.20 * groove
        aa[:, 12, 2] += 0.08 * _pulse(t, 0.77, 0.12)
        contacts[:, 0:2] = ((t <= 0.05) | (t >= 0.42))[:, None]
        contacts[:, 2:4] = (
            (t <= 0.50) | ((t >= 0.76) & (t <= 0.86))
        )[:, None]
    elif motion_id in {"step_forward", "step_backward"}:
        direction = 1.0 if motion_id == "step_forward" else -1.0
        _travel(trans, direction * 0.46 * _smoothstep(t))
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
        load = _smoothstep(t / 0.24)
        rise = _smoothstep(np.clip((t - 0.28) / 0.44, 0.0, 1.0))
        crouch = load * (1.0 - rise)
        aa[:, 1, 0] -= 0.95 * crouch
        aa[:, 2, 0] -= 0.95 * crouch
        aa[:, 4, 0] += 1.85 * crouch
        aa[:, 5, 0] += 1.85 * crouch
        # One continuous reach up and slightly forward, with the leading arm ahead of the other,
        # so it reads as a rise rather than the symmetric held V of celebrate_hands_up. Both
        # targets stay dominantly vertical -- the forward term only keeps the hands off the back.
        _solve_arm(aa, "left", _arm_targets(
            "left", rise, np.array([0.14, 0.55, 0.44], dtype=np.float32)))
        _solve_arm(aa, "right", _arm_targets(
            "right", _smoothstep(np.clip((t - 0.12) / 0.72, 0.0, 1.0)),
            np.array([-0.20, 0.47, 0.30], dtype=np.float32)))
        aa[:, 3, 0] -= 0.26 * rise                           # spine extends through the rise
        aa[:, 12, 0] -= 0.24 * rise
    else:
        raise KeyError(f"no procedural authoring recipe for {motion_id}")

    _performance_layer(aa, trans, t)
    if leg_targets is not None:
        root_offset = np.zeros((n, 3), dtype=np.float32)
        root_offset[:, 0] = trans[:, 0]
        _solve_leg(aa, "left", leg_targets[0] - root_offset)
        _solve_leg(aa, "right", leg_targets[1] - root_offset)
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
