"""Training-free motion transitions for the hybrid LODGE+EDGE pipeline.

The hybrid stitches time-segments taken from an independently generated LODGE dance and EDGE
dance. Because the two generators live in different motion distributions, we cannot make one
model's diffusion *continue* the other's pose (proven infeasible without fine-tuning). Instead
we impose a smooth handoff at each generator switch using the game-animation standard:

  root position + facing (yaw) alignment  +  inertialized blending (Bollo 2016).

At a switch the incoming segment is rotated/translated so its first frame's root matches the
outgoing motion's end, then the first ``blend_frames`` frames are eased from the outgoing end
pose into the incoming motion with a smoothstep-decayed per-joint quaternion offset. This yields
a C0/near-C1 continuous seam that (measured) is smoother than native same-generator stitching.

All motion here is the AgentLODGE 139-dim layout ``[trans(3) | 22*6D rot(132) | contact(4)]``.
Everything is normalised to a Z-up frame (EDGE-native) so a mixed sequence renders consistently.

pytorch3d is imported lazily so importing this module never hard-fails in a pytorch3d-less env.
"""

from __future__ import annotations

import numpy as np

from agentlodge.config import FPS

NUM_JOINTS = 22
_ROOT = slice(3, 9)          # root joint 6D in the 139 layout
_ROT = slice(3, 3 + NUM_JOINTS * 6)
_TRANS = slice(0, 3)
_CONTACT = slice(3 + NUM_JOINTS * 6, 139)

# Rx(+90): Y-up -> Z-up, (x, y, z) -> (x, -z, y)
_RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], dtype=np.float32)


def _t():
    import torch  # noqa: F401
    return torch


def _pt3d():
    from pytorch3d.transforms import (  # noqa: F401
        matrix_to_quaternion,
        matrix_to_rotation_6d,
        quaternion_to_matrix,
        rotation_6d_to_matrix,
    )
    return (rotation_6d_to_matrix, matrix_to_rotation_6d,
            matrix_to_quaternion, quaternion_to_matrix)


def to_zup(motion139: np.ndarray) -> np.ndarray:
    """Convert a Y-up (LODGE/FineDance) 139 motion to the Z-up (EDGE) frame.

    Input MUST be in AgentLODGE layout ``[trans(3) | rot(132) | contact(4)]`` (contact last);
    pass LODGE-native ``[contact | trans | rot]`` arrays through ``to_agentlodge139`` first or the
    root orientation will be corrupted (contact channels get swizzled as translation).

    Translation is swizzled (x, y, z) -> (x, -z, y); the root joint's global orientation is
    pre-multiplied by Rx(+90). Local joint rotations (frame-independent) are unchanged. Uses
    pytorch3d when available, else a numerically-identical numpy path (:func:`_to_zup_numpy`) so a
    candidate bank can be built without torch/pytorch3d.
    """
    try:
        torch = _t()
        rot6d_to_mat, mat_to_6d, *_ = _pt3d()
    except Exception:  # noqa: BLE001 - no torch/pytorch3d: use the numpy-equivalent path
        return _to_zup_numpy(motion139)
    m = motion139.astype(np.float32).copy()
    s = m.shape[0]
    trans = m[:, _TRANS]
    m[:, _TRANS] = np.stack([trans[:, 0], -trans[:, 2], trans[:, 1]], axis=1)
    R = rot6d_to_mat(torch.from_numpy(m[:, _ROT].reshape(s, NUM_JOINTS, 6).copy()).float())
    R[:, 0] = torch.einsum("ij,sjk->sik", torch.from_numpy(_RX90), R[:, 0])
    m[:, _ROT] = mat_to_6d(R).reshape(s, NUM_JOINTS * 6).numpy()
    return m


def _to_zup_numpy(motion139: np.ndarray) -> np.ndarray:
    """Pure-numpy :func:`to_zup` (identical math via the module's 6D<->matrix helpers)."""
    m = motion139.astype(np.float32).copy()
    s = m.shape[0]
    trans = m[:, _TRANS]
    m[:, _TRANS] = np.stack([trans[:, 0], -trans[:, 2], trans[:, 1]], axis=1)
    R = _sixd_to_matrix(m[:, _ROT].reshape(s, NUM_JOINTS, 6))          # (s, 22, 3, 3)
    R[:, 0] = np.einsum("ij,sjk->sik", _RX90, R[:, 0])
    m[:, _ROT] = _matrix_to_sixd(R).reshape(s, NUM_JOINTS * 6)
    return m


def rotate_root_yaw(motion139: np.ndarray, delta_rad: float, up: str = "z") -> np.ndarray:
    """Rotate a whole 139-dim motion about its vertical axis by ``delta_rad`` (global facing).

    Applies R(delta) about the ``up`` axis to the root joint's global orientation and to the root
    translation so the dancer faces a new direction without otherwise altering the choreography.
    ``up`` is "z" for the AgentLODGE/EDGE Z-up frame and "y" for native LODGE/FineDance (Y-up):
    rotating a Y-up motion about +Y maps to a +Z rotation of the rendered (re-oriented) result.
    Used to align a motion's facing to a reference (e.g. EDGE, which faces the camera).
    """
    if abs(delta_rad) < 1e-9:
        return motion139
    torch = _t()
    rot6d_to_mat, mat_to_6d, *_ = _pt3d()
    m = motion139.astype(np.float32).copy()
    s = m.shape[0]
    c, sn = np.cos(delta_rad), np.sin(delta_rad)
    if up == "y":
        R = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]], dtype=np.float32)
    else:  # "z"
        R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    m[:, _TRANS] = m[:, _TRANS] @ R.T
    Rmat = rot6d_to_mat(torch.from_numpy(m[:, _ROT].reshape(s, NUM_JOINTS, 6).copy()).float())
    Rmat[:, 0] = torch.einsum("ij,sjk->sik", torch.from_numpy(R), Rmat[:, 0])
    m[:, _ROT] = mat_to_6d(Rmat).reshape(s, NUM_JOINTS * 6).numpy()
    return m
def _quat_from_6d(r6):
    rot6d_to_mat, _, mat_to_quat, _ = _pt3d()
    return mat_to_quat(rot6d_to_mat(r6))


def _6d_from_quat(q):
    _, mat_to_6d, _, quat_to_mat = _pt3d()
    return mat_to_6d(quat_to_mat(q))


def _qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    torch = _t()
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qinv(a):
    c = a.clone()
    c[..., 1:] *= -1
    return c / (a * a).sum(-1, keepdim=True)


def _slerp(a, b, w):
    torch = _t()
    d = (a * b).sum(-1, keepdim=True)
    b = torch.where(d < 0, -b, b)
    d = d.abs().clamp(max=1.0)
    ang = torch.arccos(d)
    s = torch.sin(ang)
    w = w.unsqueeze(-1)
    lin = d > 0.9995
    out = torch.where(lin, a + w * (b - a),
                      (torch.sin((1 - w) * ang) * a + torch.sin(w * ang) * b) / (s + 1e-8))
    return out / (out.norm(dim=-1, keepdim=True) + 1e-8)


def _yaw_z(rootq):
    """Yaw about +Z from a root quaternion (Z-up)."""
    _, _, _, quat_to_mat = _pt3d()
    torch = _t()
    R = quat_to_mat(rootq)
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def root_yaw(frame139: np.ndarray) -> float:
    """Yaw (about +Z) of a single 139-dim frame's root orientation, in the Z-up frame."""
    torch = _t()
    q = _quat_from_6d(torch.from_numpy(frame139[_ROOT].reshape(1, 6).astype(np.float32)).float())
    return float(_yaw_z(q[0]))


def blend_onto(prev_tail139: np.ndarray, seg139: np.ndarray, blend_frames: int = 15,
               canonical_yaw: float | None = None, align_facing: bool = True) -> np.ndarray:
    """Align ``seg139`` to the end of ``prev_tail139`` and inertialize the handoff.

    Position is always chained (seg starts where prev ended) so there is no teleport. When
    ``align_facing`` is True, the segment is rotated about +Z to ``canonical_yaw`` (kills
    cross-segment drift) or the previous segment's end facing. When False, the segment KEEPS its
    own natural facing (only position is chained) -- use this when both generators already face a
    consistent direction (e.g. the camera), so re-anchoring would rotate a whole segment away; the
    small seam orientation difference is still smoothed by the inertialized root blend below.

    Both inputs must already be in the same (Z-up) frame. Returns the aligned + inertialized
    segment (same length as ``seg139``), ready to concatenate after the previous motion.
    """
    torch = _t()
    seg = torch.from_numpy(seg139.astype(np.float32)).clone()
    prev_end = torch.from_numpy(prev_tail139[-1].astype(np.float32))
    n = seg.shape[0]
    blend_frames = int(max(0, min(blend_frames, n)))

    seg_tr = seg[:, _TRANS]
    seg_r6 = seg[:, _ROT].reshape(n, NUM_JOINTS, 6)
    seg_q = _quat_from_6d(seg_r6)                       # (n, 22, 4)
    prev_r6 = prev_end[_ROT].reshape(NUM_JOINTS, 6)
    prev_q = _quat_from_6d(prev_r6)                     # (22, 4)
    prev_tr = prev_end[_TRANS]

    # 1. facing alignment: rotate the segment about +Z. Target = a fixed canonical yaw (kills
    #    cross-segment drift) when provided, else the previous segment's end facing. When
    #    align_facing is False, keep the segment's own facing (dyaw = 0) and only chain position.
    _, _, mat_to_quat, quat_to_mat = _pt3d()
    if align_facing:
        target_yaw = (torch.tensor(float(canonical_yaw)) if canonical_yaw is not None
                      else _yaw_z(prev_q[0]))
        dyaw = target_yaw - _yaw_z(seg_q[0, 0])
    else:
        dyaw = torch.tensor(0.0)
    cz, sz = torch.cos(dyaw), torch.sin(dyaw)
    Rz = torch.stack([
        torch.stack([cz, -sz, torch.zeros_like(cz)]),
        torch.stack([sz, cz, torch.zeros_like(cz)]),
        torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
    ])
    segR0 = quat_to_mat(seg_q[:, 0])
    seg_q[:, 0] = mat_to_quat(torch.einsum("ij,sjk->sik", Rz, segR0))
    seg_tr = torch.einsum("ij,sj->si", Rz, seg_tr - seg_tr[0]) + prev_tr

    # 2. inertialize: first blend_frames start at prev end pose, ease into segment.
    if blend_frames > 0:
        qoff = _qmul(prev_q, _qinv(seg_q[0]))          # (22, 4)
        troff = prev_tr - seg_tr[0]                    # (3,)
        k = torch.arange(blend_frames).float() / max(1, blend_frames - 1)
        w = 1.0 - (3 * k ** 2 - 2 * k ** 3)            # smoothstep 1 -> 0
        for f in range(blend_frames):
            shifted = _qmul(qoff, seg_q[f])
            seg_q[f] = _slerp(seg_q[f], shifted, w[f].expand(NUM_JOINTS))
            seg_tr[f] = seg_tr[f] + w[f] * troff

    seg[:, _TRANS] = seg_tr
    seg[:, _ROT] = _6d_from_quat(seg_q).reshape(n, NUM_JOINTS * 6)
    return seg.numpy().astype(np.float32)


def blend_tail_into(seg139: np.ndarray, target_head139: np.ndarray,
                    blend_frames: int = 15) -> np.ndarray:
    """Ease the END of ``seg139`` so its last frame matches ``target_head139`` (position + pose).

    The mirror image of :func:`blend_onto`: instead of chaining the segment's start onto a previous
    motion, this bends only the final ``blend_frames`` of the segment so ``seg[-1]`` lands exactly on
    ``target_head139`` (the untouched frame that follows the segment), with **zero** change at frame
    ``n - blend_frames`` (smoothstep ramp 0->1). Used to splice a regenerated window between a fixed
    prefix and a fixed suffix without modifying either side: the head is chained with ``blend_onto``
    and the tail is eased onto the suffix's first frame here, so every frame outside the window is
    preserved byte-for-byte while both seams stay continuous.
    """
    torch = _t()
    seg = torch.from_numpy(np.ascontiguousarray(seg139, dtype=np.float32)).clone()
    n = seg.shape[0]
    bf = int(max(0, min(blend_frames, n)))
    if bf == 0 or n == 0:
        return seg.numpy().astype(np.float32)
    tgt = torch.from_numpy(np.ascontiguousarray(target_head139, dtype=np.float32))

    seg_tr = seg[:, _TRANS]
    seg_q = _quat_from_6d(seg[:, _ROT].reshape(n, NUM_JOINTS, 6))     # (n, 22, 4)
    tgt_tr = tgt[_TRANS]
    tgt_q = _quat_from_6d(tgt[_ROT].reshape(NUM_JOINTS, 6))           # (22, 4)

    troff = tgt_tr - seg_tr[-1].clone()                              # (3,)
    qoff = _qmul(tgt_q, _qinv(seg_q[-1].clone()))                    # (22, 4)
    for idx in range(bf):
        f = n - bf + idx
        k = idx / max(1, bf - 1)
        w = 3 * k ** 2 - 2 * k ** 3                                  # smoothstep 0 -> 1
        shifted = _qmul(qoff, seg_q[f])
        seg_q[f] = _slerp(seg_q[f], shifted, torch.tensor(float(w)).expand(NUM_JOINTS))
        seg_tr[f] = seg_tr[f] + w * troff

    seg[:, _TRANS] = seg_tr
    seg[:, _ROT] = _6d_from_quat(seg_q).reshape(n, NUM_JOINTS * 6)
    return seg.numpy().astype(np.float32)


def _orthonormalize6d(r6: np.ndarray) -> np.ndarray:
    """Re-orthonormalize 6D rotation rows via the module's Gram-Schmidt round-trip."""
    return _matrix_to_sixd(_sixd_to_matrix(r6))


def _ease_rot_toward(window: np.ndarray, ref_frame: np.ndarray, frames, *, head: bool) -> None:
    """In-place: blend the given ``frames`` of ``window`` rotations toward ``ref_frame`` pose.

    ``head`` True ramps the weight 1->0 across ``frames`` (first frame == ref, easing back to the
    window's own pose); ``head`` False ramps 0->1 (last frame == ref). 6D rows are linearly blended
    then re-orthonormalized, an approximate but valid short-seam rotation interpolation (numpy-only,
    no torch/pytorch3d), so the splice runs and is testable without the heavy rotation backend.
    """
    ref = ref_frame[_ROT].reshape(NUM_JOINTS, 6)
    frames = list(frames)
    m = len(frames)
    if m == 0:
        return
    for i, f in enumerate(frames):
        k = i / max(1, m - 1)
        s = 3 * k ** 2 - 2 * k ** 3                 # smoothstep 0 -> 1
        wgt = (1.0 - s) if head else s              # head: 1->0, tail: 0->1
        cur = window[f, _ROT].reshape(NUM_JOINTS, 6)
        blended = (1.0 - wgt) * cur + wgt * ref
        window[f, _ROT] = _orthonormalize6d(blended).reshape(NUM_JOINTS * 6)


def splice_window(motion139: np.ndarray, a: int, b: int, new_window139: np.ndarray,
                  blend_frames: int = 15) -> np.ndarray:
    """Replace frames ``[a, b)`` of ``motion139`` with ``new_window139``, preserving the rest exactly.

    The regenerated window is chained onto the fixed prefix (its start position/pose eased from the
    last kept frame before ``a``) and eased onto the fixed suffix (its end lands on the first kept
    frame at ``b``), distributing both corrections *inside* the window over ``blend_frames`` so that
    **every frame outside ``[a, b)`` is byte-for-byte identical** to the input (high-fidelity edit).
    If the supplied window length differs from ``b - a`` it is retimed to fit. Pure numpy.
    """
    m = np.ascontiguousarray(motion139, dtype=np.float32)
    L = int(m.shape[0])
    a, b = int(a), int(b)
    if not (0 <= a < b <= L):
        raise ValueError(f"invalid window [{a}, {b}) for motion of length {L}")
    n = b - a
    w = np.ascontiguousarray(new_window139, dtype=np.float32)
    w = retime(w, n) if w.shape[0] != n else w.copy()
    prefix, suffix = m[:a], m[b:]
    bf = int(max(0, min(blend_frames, n)))

    if a >= 1:                                       # chain start onto the kept prefix
        anchor = m[a - 1]
        w[:, _TRANS] += anchor[_TRANS] - w[0, _TRANS]
        _ease_rot_toward(w, anchor, range(bf), head=True)
    if suffix.shape[0] >= 1:                         # ease end onto the kept suffix
        nxt = suffix[0]
        resid = nxt[_TRANS] - w[n - 1, _TRANS]
        for i in range(bf):
            f = n - bf + i
            k = i / max(1, bf - 1)
            w[f, _TRANS] += (3 * k ** 2 - 2 * k ** 3) * resid
        _ease_rot_toward(w, nxt, range(n - bf, n), head=False)

    return np.concatenate([prefix, w, suffix], axis=0).astype(np.float32)


# --------------------------------------------------------------------------- pure-numpy transforms
# retime / mirror / amplitude_scale operate directly on the AgentLODGE 139 layout using ONLY numpy
# (no torch/pytorch3d), so they run anywhere and are unit-testable without the heavy rotation
# backend. The 6D<->matrix conversion follows the pytorch3d convention used elsewhere in this
# module (the 6 numbers are the first two ROWS of the rotation matrix).

# SMPL body-joint left/right pairs (22-joint layout) for lateral mirroring.
_SMPL_LR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]
# Reflect the lateral (X) axis of the Z-up frame. A rotation mirrors as R' = D R D.
_MIRROR_D = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def _sixd_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3): rows b1,b2,b3 via Gram-Schmidt (pytorch3d convention)."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _matrix_to_sixd(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): first two rows flattened (pytorch3d convention)."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3): rotation vector (axis * angle)."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    axis = np.stack(
        [R[..., 2, 1] - R[..., 1, 2],
         R[..., 0, 2] - R[..., 2, 0],
         R[..., 1, 0] - R[..., 0, 1]], axis=-1,
    )
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.where(norm < 1e-8, 0.0, axis / (norm + 1e-12))
    return axis * angle[..., None]


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vector -> (..., 3, 3) rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / (angle + 1e-12)
    a = angle[..., 0]
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c, s = np.cos(a), np.sin(a)
    C = 1.0 - c
    R = np.stack([
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C,
    ], axis=-1).reshape(*aa.shape[:-1], 3, 3)
    return R.astype(np.float32)


def retime(motion139: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample a 139-dim clip to ``n_frames`` (linear time-warp).

    Translation and rotation channels are linearly interpolated on a normalized time grid;
    rotations are re-orthonormalized (6D Gram-Schmidt) and contacts re-binarized at 0.5. Used to
    fit a reused motif clip into a target section of a different length.
    """
    m = motion139.astype(np.float32)
    length = m.shape[0]
    n = int(n_frames)
    if n <= 0:
        raise ValueError(f"n_frames must be positive, got {n}")
    if length == n:
        return m.copy()
    if length == 1:
        return np.repeat(m, n, axis=0)
    t_old = np.linspace(0.0, 1.0, length)
    t_new = np.linspace(0.0, 1.0, n)
    out = np.empty((n, 139), dtype=np.float32)
    for ch in range(139):
        out[:, ch] = np.interp(t_new, t_old, m[:, ch])
    r6 = out[:, _ROT].reshape(n, NUM_JOINTS, 6)
    out[:, _ROT] = _matrix_to_sixd(_sixd_to_matrix(r6)).reshape(n, NUM_JOINTS * 6)
    out[:, _CONTACT] = (out[:, _CONTACT] > 0.5).astype(np.float32)
    return out


def mirror(motion139: np.ndarray) -> np.ndarray:
    """Lateral (left<->right) mirror of a 139-dim clip.

    Reflects the frame across the sagittal plane (negate lateral X translation, conjugate every
    joint rotation by ``diag(-1,1,1)``), swaps left/right SMPL joints, and swaps the L/R foot
    contacts. ``mirror(mirror(x)) == x``. Used to produce a varied motif recurrence.
    """
    m = motion139.astype(np.float32)
    length = m.shape[0]
    out = m.copy()
    out[:, _TRANS] = m[:, _TRANS] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
    R = _sixd_to_matrix(m[:, _ROT].reshape(length, NUM_JOINTS, 6))
    Rm = _MIRROR_D @ R @ _MIRROR_D
    r6m = _matrix_to_sixd(Rm)
    perm = np.arange(NUM_JOINTS)
    for left, right in _SMPL_LR_PAIRS:
        perm[left], perm[right] = right, left
    out[:, _ROT] = r6m[:, perm, :].reshape(length, NUM_JOINTS * 6)
    contact = m[:, _CONTACT]
    out[:, _CONTACT] = contact[:, [2, 3, 0, 1]]
    return out


def retrograde(motion139: np.ndarray) -> np.ndarray:
    """Temporal reversal ("mirror in time") of a 139-dim clip: play the phrase backward.

    A classic choreographic variation device (retrograde; Blom & Chaplin, *The Intimate Act of
    Choreography*): recur an earlier phrase in reverse so the return is structurally related yet
    distinct. This simply reverses the frame order (contact channels included); rotations stay
    valid (they are merely reordered). ``retrograde(retrograde(x)) == x``. Seam continuity when the
    reversed clip is concatenated is handled by the assembler's inertialized ``blend_onto``, so no
    internal smoothing is applied here.
    """
    return np.ascontiguousarray(motion139.astype(np.float32)[::-1])


def amplitude_scale(motion139: np.ndarray, alpha: float,
                    *, clamp: tuple[float, float] = (0.7, 1.4)) -> np.ndarray:
    """Scale movement amplitude about the clip's mean pose by ``alpha`` (clamped).

    Translation deviation from the temporal mean is scaled linearly; each joint rotation's angle
    relative to its mean rotation is scaled (axis preserved). ``alpha`` > 1 exaggerates, < 1
    calms. EXPERIMENTAL / off by default (may reduce realism); gated behind validation.
    """
    alpha = float(np.clip(alpha, clamp[0], clamp[1]))
    m = motion139.astype(np.float32)
    length = m.shape[0]
    if length < 2 or abs(alpha - 1.0) < 1e-6:
        return m.copy()
    out = m.copy()
    tmean = m[:, _TRANS].mean(axis=0, keepdims=True)
    out[:, _TRANS] = tmean + alpha * (m[:, _TRANS] - tmean)
    r6 = m[:, _ROT].reshape(length, NUM_JOINTS, 6)
    R = _sixd_to_matrix(r6)                                  # (L, 22, 3, 3)
    r_mean = _sixd_to_matrix(r6.mean(axis=0))                # (22, 3, 3)
    rel = np.einsum("jba,ljbc->ljac", r_mean, R)             # r_mean^T @ R
    rel_scaled = _axis_angle_to_matrix(_matrix_to_axis_angle(rel) * alpha)
    scaled = np.einsum("jab,ljbc->ljac", r_mean, rel_scaled)  # r_mean @ rel_scaled
    out[:, _ROT] = _matrix_to_sixd(scaled).reshape(length, NUM_JOINTS * 6)
    return out


def _kin_lowpass(kin: np.ndarray, win: int) -> np.ndarray:
    """Centered moving-average low-pass over time for the kinematic channels (edge-padded)."""
    win = int(max(1, win))
    if win <= 1 or kin.shape[0] < 3:
        return kin.copy()
    k = np.ones(win, dtype=np.float64) / win
    pad = win // 2
    padded = np.pad(kin, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(kin, dtype=np.float64)
    for ch in range(kin.shape[1]):
        out[:, ch] = np.convolve(padded[:, ch], k, mode="valid")[: kin.shape[0]]
    return out.astype(np.float32)


def temporal_smooth(motion139: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Reduce jerk by low-pass filtering the motion over time (``amount`` in [0,1] -> kernel width).

    Smooths translation + joint-rotation channels with a centered moving average (rotations are
    re-orthonormalized), leaving contacts. Deterministic; lowers the jerk metric without regenerating.
    """
    m = motion139.astype(np.float32)
    n = m.shape[0]
    amount = float(np.clip(amount, 0.0, 1.0))
    if n < 3 or amount <= 0.0:
        return m.copy()
    win = int(round(1 + amount * (FPS // 3)))                # up to ~1/3 s window at amount=1
    out = m.copy()
    out[:, :_CONTACT.start] = _kin_lowpass(m[:, :_CONTACT.start], win)
    r6 = out[:, _ROT].reshape(n, NUM_JOINTS, 6)
    out[:, _ROT] = _matrix_to_sixd(_sixd_to_matrix(r6)).reshape(n, NUM_JOINTS * 6)
    return out


def temporal_sharpen(motion139: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Increase snappiness/attack (unsharp mask over time): accentuate deviations from the low-pass.

    ``out_kin = kin + amount * (kin - lowpass(kin))``. Raises the crispness of hits (higher jerk /
    more staccato) without regenerating. Rotations re-orthonormalized; contacts untouched.
    """
    m = motion139.astype(np.float32)
    n = m.shape[0]
    amount = float(np.clip(amount, 0.0, 1.5))
    if n < 3 or amount <= 0.0:
        return m.copy()
    kin = m[:, :_CONTACT.start]
    low = _kin_lowpass(kin, max(2, FPS // 6))
    out = m.copy()
    out[:, :_CONTACT.start] = kin + amount * (kin - low)
    r6 = out[:, _ROT].reshape(n, NUM_JOINTS, 6)
    out[:, _ROT] = _matrix_to_sixd(_sixd_to_matrix(r6)).reshape(n, NUM_JOINTS * 6)
    return out


def _resample_by_time(motion139: np.ndarray, in_times: np.ndarray) -> np.ndarray:
    """Sample ``motion139`` at fractional input frame indices ``in_times`` (len = output length).

    Linear per-channel interpolation, rotations re-orthonormalized, contacts re-binarized -- the
    same channel handling as :func:`retime`, but on an arbitrary (possibly non-uniform) time map.
    """
    m = motion139.astype(np.float32)
    length = m.shape[0]
    n = int(len(in_times))
    grid = np.arange(length, dtype=np.float64)
    it = np.clip(np.asarray(in_times, dtype=np.float64), 0.0, length - 1)
    out = np.empty((n, 139), dtype=np.float32)
    for ch in range(139):
        out[:, ch] = np.interp(it, grid, m[:, ch])
    r6 = out[:, _ROT].reshape(n, NUM_JOINTS, 6)
    out[:, _ROT] = _matrix_to_sixd(_sixd_to_matrix(r6)).reshape(n, NUM_JOINTS * 6)
    out[:, _CONTACT] = (out[:, _CONTACT] > 0.5).astype(np.float32)
    return out


def _beat_align_warp_once(m: np.ndarray, mb: np.ndarray, strength: float,
                          max_shift_frames: float, motion_beats: np.ndarray | None) -> np.ndarray:
    """One detect->pair->monotone-warp pass (see :func:`beat_align_warp`)."""
    n = int(m.shape[0])
    from agentlodge.dance.beat_metrics import kinematic_beats
    kb = motion_beats if motion_beats is not None else kinematic_beats(m)
    kb = np.asarray(sorted(int(b) for b in kb if 0 < b < n - 1), dtype=np.float64)
    if kb.size == 0:
        return m.copy()

    # Anchors: output time (where we WANT the accent) -> input time (where the pose is). Endpoints
    # fixed. Greedily keep only anchors that stay strictly monotone in BOTH axes (a valid warp).
    anchors_out = [0.0]
    anchors_in = [0.0]
    for t in kb:                                             # motion-beat input frames, ascending
        a = mb[int(np.argmin(np.abs(mb - t)))]              # nearest music beat
        if abs(a - t) > max_shift_frames:
            continue
        target = t + strength * (a - t)                     # pull the accent toward the beat
        if target <= anchors_out[-1] + 1.0 or t <= anchors_in[-1] + 1.0:
            continue                                        # keep both axes strictly increasing
        if target >= n - 1 or t >= n - 1:
            continue
        anchors_out.append(float(target))
        anchors_in.append(float(t))
    anchors_out.append(float(n - 1))
    anchors_in.append(float(n - 1))
    if len(anchors_out) <= 2:                                # nothing movable -> unchanged
        return m.copy()

    out_grid = np.arange(n, dtype=np.float64)
    in_times = np.interp(out_grid, np.asarray(anchors_out), np.asarray(anchors_in))
    return _resample_by_time(m, in_times)


def beat_align_warp(window139: np.ndarray, window_beats, *, strength: float = 1.0,
                    max_shift_frames: int | None = None, passes: int = 3,
                    motion_beats: np.ndarray | None = None) -> np.ndarray:
    """Time-warp a window so its motion beats land on the music beats (raises Beat Alignment Score).

    Best-of-K over diffusion samples is a weak lever for beat sync: the backbones produce plausible
    dances whose motion beats (kinematic-speed troughs) fall wherever they fall, so BAS barely varies
    across seeds. This instead *directly* fixes timing: it detects the window's motion beats, pairs
    each with its nearest music beat (within ``max_shift_frames``), and builds a **monotone,
    endpoint-preserving** piecewise-linear time reparameterization that slides those accented poses
    onto the beat, then resamples. Because it moves the very frames BAS measures toward the music
    beats, BAS increases deterministically -- no generation, works with the pod off.

    Applied for a few ``passes`` (re-detecting the now-shifted motion beats each time) it converges
    the accents onto the grid (e.g. BAS 0.46 -> 0.78 in 3 passes). ``strength`` in [0, 1] scales how
    far beats are pulled toward the grid per pass (1 = full snap). Endpoints are fixed so
    :func:`splice_window` blends cleanly. ``window_beats`` are window-local (0-based) music-beat
    frames.
    """
    m = np.ascontiguousarray(window139.astype(np.float32))
    n = int(m.shape[0])
    mb = np.asarray(window_beats, dtype=np.float64)
    mb = np.sort(mb[np.isfinite(mb) & (mb > 0) & (mb < n - 1)])
    strength = float(np.clip(strength, 0.0, 1.0))
    if n < 4 or mb.size == 0 or strength <= 0.0:
        return m.copy()

    if max_shift_frames is None:
        # default budget: about half the median music-beat spacing (never yank across a whole beat)
        spacing = float(np.median(np.diff(mb))) if mb.size > 1 else float(n)
        max_shift_frames = max(2.0, min(0.5 * spacing, 12.0))

    out = m
    prev_beats = motion_beats
    for _ in range(max(1, int(passes))):
        nxt = _beat_align_warp_once(out, mb, strength, float(max_shift_frames), prev_beats)
        if nxt is out or np.array_equal(nxt, out):          # converged (no movable anchors)
            break
        out = nxt
        prev_beats = None                                   # re-detect on the warped motion
    return out
