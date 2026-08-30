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
    temporal_smooth,
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


def _smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _pulse(t, center=0.5, width=0.16):
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def _pin_signal_peaks(
    signal: np.ndarray,
    frames: tuple[int, ...],
    *,
    shoulder_frames: int = 9,
    plateau_radius: int = 2,
) -> np.ndarray:
    """Give each contact a short full-closure plateau that survives retiming."""
    out = np.asarray(signal, dtype=np.float32).copy()
    shoulder_frames = max(int(shoulder_frames), int(plateau_radius) + 1)
    plateau_radius = max(0, int(plateau_radius))
    span = shoulder_frames - plateau_radius
    for frame in frames:
        frame = int(np.clip(frame, 0, len(out) - 1))
        for distance in range(1, span):
            q = distance / float(span)
            weight = q * q * (3.0 - 2.0 * q)
            for index in (
                frame - shoulder_frames + distance,
                frame + shoulder_frames - distance,
            ):
                if 0 <= index < len(out):
                    out[index] += weight * (1.0 - out[index])
        out[
            max(0, frame - plateau_radius):
            min(len(out), frame + plateau_radius + 1)
        ] = 1.0
    return np.clip(out, 0.0, 1.0)


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


def _native_joint_pose(axis_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-kinematic joint rotations and positions in the native authoring frame."""
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
    return global_r, positions


def _native_joint_positions(axis_angles: np.ndarray) -> np.ndarray:
    """Forward-kinematic joint positions in the native SMPL authoring frame."""
    _global_r, positions = _native_joint_pose(axis_angles)
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


def _clap_wrist_rotation(
    side: str,
    forearm_global: np.ndarray,
    body_global: np.ndarray,
    *,
    overhead: bool = False,
) -> np.ndarray:
    """Return a local wrist rotation with relaxed raised fingers and inward-facing palms."""
    sign = 1.0 if side == "left" else -1.0
    rest_finger = np.array([sign, 0.0, 0.0], dtype=np.float32)
    rest_palm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    rest_width = np.cross(rest_finger, rest_palm)
    # Chest-level claps carry relaxed forward-up fingers. Reusing that hand plane overhead
    # points the fingers mostly forward while the forearms travel upward, forcing a severe
    # wrist bend and an axis-angle flip on approach and recoil. Above the head the fingers
    # instead continue upward with only a small forward cant.
    desired_finger = np.array(
        [0.0, 0.97, 0.24] if overhead else [0.0, 0.55, 0.835],
        dtype=np.float32,
    )
    desired_finger /= np.linalg.norm(desired_finger)
    desired_palm = np.array([-sign, 0.0, 0.0], dtype=np.float32)
    desired_width = np.cross(desired_finger, desired_palm)
    rest_basis = np.stack([rest_finger, rest_palm, rest_width], axis=1)
    desired_basis = np.stack([desired_finger, desired_palm, desired_width], axis=1)
    hand_global = body_global @ desired_basis @ rest_basis.T
    return forearm_global.T @ hand_global


def _solve_arm(
    aa,
    side: str,
    targets: np.ndarray,
    *,
    elbow_hint=_ELBOW_OUT,
    clap_orientation: np.ndarray | None = None,
    clap_overhead: bool = False,
    respect_parent_pose: bool = False,
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
        body_global = np.eye(3, dtype=np.float32)
        if respect_parent_pose:
            current_global, current_positions = _native_joint_pose(aa[frame])
            parent_global = current_global[BODY_PARENTS[shoulder]]
            origin = current_positions[shoulder]
            body_global = parent_global
            hand = origin + body_global @ (hand - J[shoulder])
        else:
            parent_global = np.eye(3, dtype=np.float32)
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
        aa[frame, shoulder] = _matrix_to_axis_angle(parent_global.T @ upper_global)
        aa[frame, elbow] = _matrix_to_axis_angle(elbow_local)
        if clap_orientation is not None:
            wrist_local = _clap_wrist_rotation(
                side,
                fore_global,
                body_global,
                overhead=clap_overhead,
            )
            weight = float(np.clip(clap_orientation[frame], 0.0, 1.0))
            aa[frame, wrist] = _matrix_to_axis_angle(wrist_local) * weight


# Both wrist targets meet at contact. A positive separation left a visible gap after the arm pose
# was composed onto asymmetric real choreography, even though the canonical stance still passed
# its wrist-distance check.
_CLAP_HALF_GAP = 0.003
# Overhead palms need more room because their wrist centers sit side by side above the head.
# Reusing the chest-clap gap makes the hand meshes pass through one another even when the wrist
# joints technically remain ordered.
_CLAP_OVERHEAD_HALF_GAP = 0.025

# Where the palms meet. In the template frame the shoulders sit at y=+0.083 and the chest
# (spine3) at y=-0.057, so a clap belongs a little BELOW the chest line. This was authored at
# y=+0.04 -- above the chest, level with the collarbones -- which renders as hands pressed
# together under the chin, closer to a bow than a clap.
_CLAP_POINT = (-0.045, 0.26)                        # (height, distance in front)
_CLAP_SIDE_OFFSET = 0.22
_CLAP_OVERHEAD_SIDE_OFFSET = 0.20
_CLAP_OVERHEAD_POINT = (0.495, 0.12)
_CLAP_OVERHEAD_SIDE_POINT = (0.435, 0.12)

# Clapping hands are carried in front of the sternum, so the elbows hang DOWN and only a little
# outward. Left to the default outward hint the solver lifts them to shoulder height and the
# clip reads as flapping wings.
_CLAP_ELBOW = np.array([0.55, -1.0, -0.1], dtype=np.float32)

# How far towards the meeting point the hands stay BETWEEN repeated claps: 1.0 would leave them
# touching, 0.0 drops them to the hips. Sets the parting of the hands, so it trades readability
# of each clap (needs travel) against the hands staying up (needs little).
_CLAP_GUARD = 0.485


def _smooth_clap_wrists(
    clip: np.ndarray,
    contacts: tuple[int, ...],
    *,
    amount: float = 0.5,
) -> np.ndarray:
    """Remove retiming-amplified wrist snaps without moving a clap's contact planes."""
    out = np.asarray(clip, dtype=np.float32).copy()
    smoothed = temporal_smooth(out, amount=amount)
    wrists = slice(3 + 6 * 20, 3 + 6 * 22)
    out[:, wrists] = smoothed[:, wrists]
    locked = {0, 1, 2, len(out) - 3, len(out) - 2, len(out) - 1}
    for contact in contacts:
        locked.update(range(max(0, contact - 4), min(len(out), contact + 5)))
    locked = sorted(frame for frame in locked if 0 <= frame < len(out))
    out[locked, wrists] = clip[locked, wrists]
    return out


def _smooth_single_clap_arms(clip: np.ndarray, contact: int) -> np.ndarray:
    """Ease the whole arm into contact while preserving the authored five-frame closure."""
    out = np.asarray(clip, dtype=np.float32).copy()
    smoothed = temporal_smooth(out, amount=0.65)
    channels = np.concatenate([
        np.arange(3 + 6 * joint, 3 + 6 * (joint + 1))
        for joint in (13, 14, 16, 17, 18, 19, 20, 21)
    ])
    # Quaternion smoothing can differ by ~2e-5 between Windows and Linux BLAS builds. Quantizing
    # only the eased, non-contact arm channels keeps committed clips reproducible on the render pod
    # without weakening the stale-asset gate.
    out[:, channels] = np.trunc(smoothed[:, channels] * 1_000.0) / 1_000.0
    locked = {0, 1, 2, len(out) - 3, len(out) - 2, len(out) - 1}
    locked.update(range(max(0, contact - 2), min(len(out), contact + 3)))
    locked = sorted(frame for frame in locked if 0 <= frame < len(out))
    out[np.ix_(locked, channels)] = clip[np.ix_(locked, channels)]
    return out


def _smooth_punch_pose(clip: np.ndarray, event: int) -> np.ndarray:
    """Ease guard, strike, and recoil without blunting the forward hit."""
    out = np.asarray(clip, dtype=np.float32).copy()
    smoothed = temporal_smooth(out, amount=0.35)
    channels = np.concatenate([
        np.arange(3 + 6 * joint, 3 + 6 * (joint + 1))
        for joint in (0, 3, 4, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
    ])
    out[:, channels] = smoothed[:, channels]
    locked = range(max(0, event - 1), min(len(out), event + 2))
    out[np.ix_(list(locked), channels)] = clip[np.ix_(list(locked), channels)]
    return out


def _arm_targets(side: str, signal: np.ndarray, target: np.ndarray) -> np.ndarray:
    rest = _stance_wrist(side)
    return rest[None, :] + signal[:, None] * (np.asarray(target) - rest)[None, :]


def _clap_arms(
    aa,
    signal,
    *,
    overhead=False,
    center_x=0.0,
    orientation_signal=None,
    respect_parent_pose=False,
):
    # The overhead variant has to clear the head to read as overhead, and reaching forward
    # costs height, so it trades a little of the forward travel back for lift.
    #
    # The palms meet at the midline, so each WRIST stops half a hand short of it. Aiming both
    # hands at x=0 asks two wrists to occupy one point: the solver obliges, the forearms pass
    # through each other, and the clap reads as folded, tangled arms rather than a clap.
    half = _CLAP_OVERHEAD_HALF_GAP if overhead else _CLAP_HALF_GAP
    # Keep the overhead target above the head but inside both arms' reachable intersection.
    # The old (0.62, 0.18) target sat about 7 cm beyond full extension, so each arm stopped on
    # its own reach boundary and the palms could never meet.
    y, z = (
        _CLAP_OVERHEAD_SIDE_POINT
        if overhead and abs(center_x) > 1e-6
        else (_CLAP_OVERHEAD_POINT if overhead else _CLAP_POINT)
    )
    orientation = signal if orientation_signal is None else orientation_signal
    hint = _CLAP_ELBOW
    _solve_arm(aa, "left", _arm_targets(
        "left", signal, np.array([center_x + half, y, z], dtype=np.float32)),
               elbow_hint=hint, clap_orientation=orientation,
               clap_overhead=overhead,
               respect_parent_pose=respect_parent_pose)
    _solve_arm(aa, "right", _arm_targets(
        "right", signal, np.array([center_x - half, y, z], dtype=np.float32)),
               elbow_hint=hint, clap_orientation=orientation,
               clap_overhead=overhead,
               respect_parent_pose=respect_parent_pose)


def _legs(aa, phase, amount=0.45):
    aa[:, 1, 0] += amount * np.sin(phase)
    aa[:, 2, 0] -= amount * np.sin(phase)
    aa[:, 4, 0] += amount * np.maximum(0.0, -np.sin(phase))
    aa[:, 5, 0] += amount * np.maximum(0.0, np.sin(phase))


def build_motion(motion_id: str, n: int, *, direction: str = "forward") -> np.ndarray:
    direction = str(direction).strip().lower()
    if direction not in {"forward", "left", "right"}:
        raise ValueError(f"unsupported authored direction: {direction!r}")
    if direction != "forward" and not motion_id.startswith("clap_"):
        raise ValueError(f"{motion_id} has no authored {direction} variant")
    aa, trans, contacts = _identity_clip(n)
    _apply_stance(aa)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    phase = 2.0 * np.pi * t
    leg_targets = None
    clap_recipe = None
    side = 1.0 if direction == "left" else (-1.0 if direction == "right" else 0.0)

    if motion_id in {"clap_single", "clap_overhead"}:
        overhead = motion_id == "clap_overhead"
        hit = _pulse(t, 0.54, 0.15)
        wind = _pulse(t, 0.30, 0.11)                     # anticipation before the hit
        settle = _pulse(t, 0.80, 0.13)                   # follow-through after it
        if overhead:
            clap_signal = np.clip(hit - 0.22 * wind, 0.0, None)
            clap_orientation = (
                _smoothstep(t / 0.38)
                * _smoothstep((1.0 - t) / 0.28)
            )
        else:
            # Lift both hands into an open chest-level guard before closing them. A side clap
            # launched directly from the hips gives one arm a long strike path and reads as a
            # punch; the visible two-hand guard makes the approach unmistakably bilateral.
            ready = _smoothstep(t / 0.24) * _smoothstep((1.0 - t) / 0.24)
            clap_signal = ready * (0.42 + 0.46 * hit)
            clap_orientation = ready
        contact = 30 if overhead else 24
        clap_signal = _pin_signal_peaks(clap_signal, (contact,))
        center_x = side * (
            _CLAP_OVERHEAD_SIDE_OFFSET if overhead else _CLAP_SIDE_OFFSET
        )
        _clap_arms(
            aa,
            clap_signal,
            overhead=overhead,
            center_x=center_x,
            orientation_signal=clap_orientation,
        )
        clap_recipe = (clap_signal, clap_orientation, overhead, center_x)
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
        clap_signal = ready * (_CLAP_GUARD + (1.0 - _CLAP_GUARD) * hit)
        # Repeated contacts are only 17-18 authored frames apart. The broad shoulder used by a
        # single clap overlaps adjacent hits and leaves the hands visibly closed for almost the
        # entire phrase. Keep the five-frame contact holds, but shorten each approach/recoil so
        # every clap has a sustained open interval that survives retiming.
        clap_signal = _pin_signal_peaks(
            clap_signal,
            (20, 37, 55),
            shoulder_frames=5,
        )
        _clap_arms(
            aa,
            clap_signal,
            center_x=side * _CLAP_SIDE_OFFSET,
            orientation_signal=ready,
        )
        clap_recipe = (
            clap_signal,
            ready,
            False,
            side * _CLAP_SIDE_OFFSET,
        )
        trans[:, 2] += 0.010 * np.sin(6.0 * np.pi * t)
        aa[:, 3, 0] += 0.07 * hit
        aa[:, 0, 2] += 0.05 * np.sin(3.0 * np.pi * t)        # groove without hiding hand recoils
        aa[:, 4, 0] += 0.06 * (1.0 - hit)
        aa[:, 5, 0] += 0.06 * (1.0 - hit)
    elif motion_id in {"jump_two_foot", "jump_arms_up"}:
        air = np.sin(np.pi * _smoothstep((t - 0.22) / 0.58))
        air = np.maximum(air, 0.0) ** 0.78
        trans[:, 2] += (0.38 if motion_id == "jump_two_foot" else 0.42) * air
        load = _pulse(t, 0.18, 0.11)
        land = _pulse(t, 0.82, 0.11)
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
            # Keep a wide airborne V. Hands converging above the head turn the silhouette into
            # an overhead clap, especially after the action is composed onto a moving host.
            left = np.array([0.46, 0.52, 0.08], dtype=np.float32)
            right = np.array([-0.46, 0.52, 0.08], dtype=np.float32)
            _solve_arm(aa, "left", _arm_targets("left", air, left))
            _solve_arm(aa, "right", _arm_targets("right", air, right))
        else:
            aa[:, 16, 1] -= 0.70 * load + 0.38 * land - 0.48 * air
            aa[:, 17, 1] += 0.70 * load + 0.38 * land - 0.48 * air
    elif motion_id == "bounce_in_place":
        compression = 0.5 - 0.5 * np.cos(4.0 * np.pi * t)
        # Keep both ankles visibly planted and apart while the pelvis performs two full pulses.
        # Three faster dips survived the numeric cycle check but collapsed into one crouch at
        # normal playback speed after composition onto a real host.
        # Solving the legs against fixed feet avoids the tucked silhouette that can make a deep
        # grounded bend look airborne on a featureless floor.
        trans[:, 2] -= 0.16 * compression
        left = _stance_ankle("left")
        right = _stance_ankle("right")
        leg_targets = (
            np.repeat(left[None, :], n, axis=0),
            np.repeat(right[None, :], n, axis=0),
        )
        aa[:, 3, 0] += 0.13 * compression
        aa[:, 0, 2] += 0.12 * np.sin(4.0 * np.pi * t)
        aa[:, 16, 1] -= 0.18 * compression
        aa[:, 17, 1] -= 0.18 * compression
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
        pop_scale = 1.80
        base = pop_scale * 0.07 * drive                      # hips stay under the ribcage
        lower = pop_scale * 0.26 * drive
        mid = pop_scale * 0.23 * drive                       # the chest travels off these three
        # Cancelling only the rotation is not enough to hold the head: the chain below has
        # already carried it forward, so the counter has to over-rotate to bring it back. The
        # extra term compensates for that leftover head travel over the spine3->head length.
        counter = base + lower + mid + pop_scale * 0.24 * drive
        aa[:, 0, 0] += base
        aa[:, 3, 0] += lower
        aa[:, 6, 0] += mid
        aa[:, 9, 0] -= 0.73 * counter                        # shoulders open rather than dive
        aa[:, 12, 0] -= 0.27 * counter                       # head holds its place
        aa[:, 16, 2] -= 0.20 * hit
        aa[:, 17, 2] += 0.20 * hit
        aa[:, 4, 0] += 0.08 * hit
        aa[:, 5, 0] += 0.08 * hit
        _travel(trans, -0.015 * drive)                       # keep the pelvis under the pop
    elif motion_id == "arm_punch":
        ready = _smoothstep(t / 0.24) * _smoothstep((1.0 - t) / 0.18)
        wind = _pulse(t, 0.34, 0.09)
        hit = _pin_signal_peaks(
            _pulse(t, 26.0 / 47.0, 0.045),
            (26,),
            shoulder_frames=4,
            plateau_radius=1,
        )
        right_rest = _stance_wrist("right")
        right_guard = np.array([-0.22, 0.09, 0.20], dtype=np.float32)
        # Keep the strike just inside the arm's reachable radius. The old target was far past
        # full extension, so IK locked the elbow several frames before the declared beat and
        # held a long straight-arm plateau.
        right_strike = np.array([-0.10, 0.07, 0.50], dtype=np.float32)
        right_targets = (
            right_rest[None, :]
            + ready[:, None] * (right_guard - right_rest)[None, :]
            + hit[:, None] * (right_strike - right_guard)[None, :]
        )
        left_rest = _stance_wrist("left")
        left_guard = np.array([0.07, 0.15, 0.10], dtype=np.float32)
        left_tuck = np.array([0.02, 0.18, 0.06], dtype=np.float32)
        left_targets = (
            left_rest[None, :]
            + ready[:, None] * (left_guard - left_rest)[None, :]
            + hit[:, None] * (left_tuck - left_guard)[None, :]
        )
        _solve_arm(aa, "right", right_targets)
        _solve_arm(
            aa,
            "left",
            left_targets,
            elbow_hint=np.array([0.40, -0.90, -0.12], dtype=np.float32),
        )
        aa[:, 9, 1] += 0.12 * wind - 0.56 * hit              # wind up, then rotate into the hit
        aa[:, 0, 1] += 0.08 * wind - 0.26 * hit
        aa[:, 12, 1] -= 0.12 * hit
        aa[:, 3, 0] += 0.14 * hit
        aa[:, 4, 0] += 0.28 * hit
        _travel(trans, 0.075 * hit)                          # a short step into the punch
    elif motion_id == "side_step":
        # A compact lateral weight transfer: the lead foot opens, the pelvis follows after the
        # foot commits, and the upper body keeps the host dance instead of being replaced by a
        # large directional arm pose.
        left_start = _stance_ankle("left")
        right_start = _stance_ankle("right")
        left_end = left_start.copy()
        left_end[0] += 0.44
        anticipation = _pulse(t, 0.18, 0.08)
        transfer = _phase_ease(t, 0.30, 0.68)
        trans[:, 0] += -0.025 * anticipation + 0.29 * transfer
        left_targets = _foot_swing(
            t, 0.12, 0.55, left_start, left_end, lift=0.075,
        )
        right_targets = np.repeat(right_start[None, :], n, axis=0)
        leg_targets = (left_targets, right_targets)

        settle = _pulse(t, 0.64, 0.13)
        trans[:, 2] -= 0.035 * settle
        aa[:, 0, 2] -= 0.10 * transfer
        aa[:, 3, 2] += 0.03 * transfer
        aa[:, 9, 2] += 0.02 * transfer
        contacts[:, 0:2] = ((t <= 0.12) | (t >= 0.55))[:, None]
        contacts[:, 2:4] = 1.0
    elif motion_id == "step_touch":
        # The lead foot accepts the weight before the trailing foot lifts and taps beside it.
        # The pelvis stops during the tap instead of continuing into another lateral step.
        left_start = _stance_ankle("left")
        right_start = _stance_ankle("right")
        left_end = left_start.copy()
        left_end[0] += 0.31
        right_touch = left_end.copy()
        right_touch[0] -= 0.060
        # A perfectly superimposed touch disappears in both audit views. Keep the feet close
        # laterally while staggering the tapping foot slightly forward so the side view can
        # still show which foot lifted and touched.
        right_touch[2] += 0.10
        trans[:, 0] += 0.29 * _phase_ease(t, 0.34, 0.62)

        left_targets = _foot_swing(
            t, 0.05, 0.46, left_start, left_end, lift=0.085,
        )
        right_targets = _foot_swing(
            t, 0.48, 0.68, right_start, right_touch, lift=0.085,
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
        _arm_swing(aa, 2.0 * np.pi * t, 0.16)
        contacts[:, 0:2] = ((t <= 0.05) | (t >= 0.42))[:, None]
        contacts[:, 2:4] = (
            (t <= 0.48) | ((t >= 0.68) & (t <= 0.88))
        )[:, None]
    elif motion_id in {"step_forward", "step_backward"}:
        direction = 1.0 if motion_id == "step_forward" else -1.0
        left_start = _stance_ankle("left")
        right_start = _stance_ankle("right")
        left_end = left_start.copy()
        left_end[2] += 0.62 * direction
        travel = _phase_ease(t, 0.28, 0.74)
        _travel(trans, direction * 0.46 * travel)
        left_targets = _foot_swing(
            t,
            0.08,
            0.48,
            left_start,
            left_end,
            lift=0.12,
        )
        right_targets = np.repeat(right_start[None, :], n, axis=0)
        leg_targets = (left_targets, right_targets)
        _arm_swing(aa, np.pi * _phase_ease(t, 0.08, 0.66), 0.42 * direction)
        aa[:, 3, 0] += direction * 0.18 * np.sin(np.pi * travel)
        contacts[:, 0:2] = ((t <= 0.08) | (t >= 0.48))[:, None]
        contacts[:, 2:4] = 1.0
    elif motion_id in {"turn_quarter", "turn_half"}:
        angle = (np.pi / 2.0) if motion_id == "turn_quarter" else np.pi
        # Resolve the new facing before the clip ends and hold it long enough to read. A turn
        # spread over every frame has no stable "before" or "after" silhouette at normal speed.
        spin = _phase_ease(t, 0.08, 0.66)
        aa[:, 0, 1] = angle * spin
        _legs(aa, np.pi * _phase_ease(t, 0.05, 0.66), 0.32)
        _arm_swing(aa, np.pi * _phase_ease(t, 0.05, 0.66), 0.30)
        trans[:, 2] += 0.025 * np.sin(2.0 * np.pi * t) ** 2
        spot = np.sin(np.pi * _phase_ease(t, 0.02, 0.68))
        aa[:, 12, 1] += 0.38 * spot                          # head leads, then spots the turn
        aa[:, 9, 1] -= 0.22 * spot
        aa[:, 4, 0] += 0.18 * spot
        aa[:, 5, 0] += 0.18 * spot
    elif motion_id == "body_roll":
        # A roll needs a travelling S-curve, not three same-sign bends that accumulate into a
        # rigid bow. Each spine segment crests and releases before the next one arrives.
        for joint, center in ((3, 0.26), (6, 0.46), (9, 0.66)):
            crest = _pulse(t, center, 0.10)
            release = _pulse(t, center + 0.16, 0.11)
            aa[:, joint, 0] += 0.60 * crest - 0.42 * release
        hip = _pulse(t, 0.18, 0.11)
        aa[:, 0, 0] += 0.34 * hip - 0.24 * _pulse(t, 0.36, 0.12)
        aa[:, 12, 0] -= 0.28 * _pulse(t, 0.76, 0.13)
        trans[:, 2] += 0.04 * np.sin(2.0 * np.pi * t)
        aa[:, 4, 0] += 0.16 * hip
        aa[:, 5, 0] += 0.16 * hip
        aa[:, 16, 2] -= 0.20 * _pulse(t, 0.64, 0.17)         # arms open as the chest arrives
        aa[:, 17, 2] += 0.20 * _pulse(t, 0.64, 0.17)
    elif motion_id == "crouch_drop":
        descend = _smootherstep(np.clip((t - 0.14) / 0.48, 0.0, 1.0))
        recover = _smootherstep(np.clip((t - 0.68) / 0.32, 0.0, 1.0))
        down = descend * (1.0 - recover)
        anticipation = _pulse(t, 0.10, 0.055) * (1.0 - descend)
        leg_load = np.clip(down + 0.14 * anticipation, 0.0, 1.0)
        trans[:, 2] += -0.028 * anticipation - 0.25 * down
        left = _stance_ankle("left")
        right = _stance_ankle("right")
        leg_targets = (
            np.repeat(left[None, :], n, axis=0),
            np.repeat(right[None, :], n, axis=0),
        )
        aa[:, 0, 0] += 0.20 * leg_load                       # hips fold back into the squat
        aa[:, 3, 0] += 0.22 * down
        aa[:, 6, 0] -= 0.05 * down
        aa[:, 9, 0] -= 0.05 * down
        aa[:, 12, 0] -= 0.14 * down                         # keep the gaze forward
    elif motion_id == "rise_reach":
        load = _smoothstep(t / 0.22)
        rise = _smoothstep(np.clip((t - 0.46) / 0.36, 0.0, 1.0))
        crouch = load * (1.0 - rise)
        trans[:, 2] -= 0.31 * crouch
        left = _stance_ankle("left")
        right = _stance_ankle("right")
        leg_targets = (
            np.repeat(left[None, :], n, axis=0),
            np.repeat(right[None, :], n, axis=0),
        )
        # One continuous reach up and slightly forward, with the leading arm ahead of the other,
        # so it reads as a rise rather than a symmetric held V. Both targets stay dominantly
        # vertical -- the forward term only keeps the hands off the back.
        lead_reach = _smootherstep(np.clip((t - 0.34) / 0.46, 0.0, 1.0))
        trail_reach = _smootherstep(np.clip((t - 0.42) / 0.40, 0.0, 1.0))
        _solve_arm(aa, "left", _arm_targets(
            "left", lead_reach, np.array([0.16, 0.60, 0.64], dtype=np.float32)))
        _solve_arm(aa, "right", _arm_targets(
            "right", trail_reach, np.array([-0.24, 0.60, 0.35], dtype=np.float32)))
        aa[:, 3, 0] -= 0.26 * rise                           # spine extends through the rise
        aa[:, 12, 0] -= 0.24 * rise
    else:
        raise KeyError(f"no procedural authoring recipe for {motion_id}")

    _performance_layer(
        aa,
        trans,
        t,
        sway=0.0 if motion_id == "chest_pop" else 1.0,
    )
    if clap_recipe is not None:
        signal, orientation, overhead, center_x = clap_recipe
        _clap_arms(
            aa,
            signal,
            overhead=overhead,
            center_x=center_x,
            orientation_signal=orientation,
            respect_parent_pose=True,
        )
    if leg_targets is not None:
        root_offset = np.zeros((n, 3), dtype=np.float32)
        root_offset[:, 0] = trans[:, 0]
        root_offset[:, 1] = trans[:, 2]
        root_offset[:, 2] = -trans[:, 1]
        _solve_leg(aa, "left", leg_targets[0] - root_offset)
        _solve_leg(aa, "right", leg_targets[1] - root_offset)
    rotations = _matrix_to_sixd(_axis_angle_to_matrix(aa)).reshape(n, 132)
    # Author against the native Y-up SMPL template, then convert the complete clip to MAESTRO's
    # Z-up editing frame. ``trans`` above is expressed in desired Z-up coordinates.
    native_trans = np.stack([trans[:, 0], trans[:, 2], -trans[:, 1]], axis=1)
    native = np.concatenate([native_trans, rotations, contacts], axis=1).astype(np.float32)
    out = _ground(to_zup(native))
    if motion_id == "clap_single":
        out = _smooth_single_clap_arms(out, 24)
    elif motion_id == "clap_repeat":
        out = _smooth_clap_wrists(out, (20, 37, 55))
    elif motion_id == "clap_overhead":
        out = _smooth_clap_wrists(out, (30,), amount=0.65)
    elif motion_id == "arm_punch":
        out = _smooth_punch_pose(out, 26)
    return out


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
        direction = raw.get("direction", {})
        if direction.get("mode") == "clip":
            canonical = str(direction["canonical"])
            for name, path in direction["clips"].items():
                if name == canonical and path == raw["clip"]:
                    continue
                variant = build_motion(raw["id"], int(raw["frames"]), direction=name)
                np.save(BANK / path, variant)
                print(f"{raw['id']}[{name}]: {variant.shape}")
    bank = MotionBank(BANK)
    for spec in bank.specs:
        bank.load_clip(spec)
    print(f"validated {len(bank.specs)} named motions")


if __name__ == "__main__":
    main()
