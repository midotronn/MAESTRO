"""Motion format helpers."""

from __future__ import annotations

import numpy as np


def edge_to_lodge139(motion: np.ndarray) -> np.ndarray:
    """Convert EDGE (L, 151) representation to Lodge (L, 139)."""
    if motion.shape[-1] != 151:
        raise ValueError(f"Expected EDGE motion with 151 dims, got {motion.shape[-1]}")
    trans = motion[:, :3]
    rot22 = motion[:, 3 : 3 + 22 * 6]
    contact = motion[:, 147:151]
    return np.concatenate([trans, rot22, contact], axis=-1).astype(np.float32)


def ensure_lodge139(motion: np.ndarray) -> np.ndarray:
    if motion.shape[-1] == 139:
        return motion.astype(np.float32)
    if motion.shape[-1] == 151:
        return edge_to_lodge139(motion)
    raise ValueError(f"Unsupported motion dimension: {motion.shape[-1]}")


def _looks_like_contact(values: np.ndarray) -> bool:
    return float(np.mean((values >= -0.01) & (values <= 1.01))) > 0.9


def _rotation_layout_score(values: np.ndarray) -> float:
    """Lower is better for a block of 22 valid 6D rotations.

    This resolves the ambiguous stationary-pose case where zero translation makes the first four
    AgentLODGE channels look contact-like even though the real contacts are at the end.
    """
    r6 = np.asarray(values, dtype=np.float32).reshape(-1, 22, 6)
    a, b = r6[..., :3], r6[..., 3:]
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1) / (na * nb + 1e-8)
    return float(np.mean(np.abs(na - 1.0) + np.abs(nb - 1.0) + np.abs(dot)))


def _is_native_finedance139(motion: np.ndarray) -> bool:
    """Whether a 139-channel array uses LODGE/FineDance's contact-first, Y-up layout."""
    start_contact = _looks_like_contact(motion[:, :4])
    end_contact = _looks_like_contact(motion[:, 135:139])
    native_score = _rotation_layout_score(motion[:, 7:139])
    agent_score = _rotation_layout_score(motion[:, 3:135])
    return bool(start_contact and (not end_contact or native_score + 1e-4 < agent_score))


def to_agentlodge139(motion: np.ndarray) -> np.ndarray:
    """Normalize a 139-dim motion to AgentLODGE layout ``[trans(3) | rot(132) | contact(4)]``.

    LODGE/FineDance emits the native layout ``[contact(4) | trans(3) | rot(132)]`` (contact
    first). The hybrid/transition code (``to_zup``, ``assemble``) assumes the AgentLODGE
    layout (contact last), so callers must normalize contact-first motions before using them.
    Detects a contact-first array and reorders it; already-AgentLODGE arrays pass through.
    """
    if motion.shape[-1] != 139:
        raise ValueError(f"Expected motion with 139 dims, got {motion.shape[-1]}")
    motion = motion.astype(np.float32)
    if _is_native_finedance139(motion):
        # native [contact(4) | trans(3) | rot(132)] -> [trans(3) | rot(132) | contact(4)]
        return np.concatenate(
            [motion[:, 4:7], motion[:, 7:139], motion[:, 0:4]], axis=1
        ).astype(np.float32)
    return motion


def to_editor139(motion: np.ndarray) -> np.ndarray:
    """Normalize EDGE or LODGE output to the editor's contact-last, Z-up 139 layout.

    ``ensure_lodge139`` only normalizes channel count. Raw LODGE output is still contact-first
    and Y-up after that call, which is renderable because the renderer auto-detects its layout but
    is not editable: the editor interprets four contacts as root translation and shifts every
    rotation by four channels. That stayed hidden until a Z-up canonical motion was mixed into a
    LODGE window. Normalize both the channel order and frame convention at the system boundary.
    """
    m = ensure_lodge139(np.asarray(motion))
    native = _is_native_finedance139(m)
    out = to_agentlodge139(m)
    if native:
        # Lazy import avoids making the lightweight format module own transition dependencies.
        from agentlodge.dance.transition import to_zup

        out = to_zup(out)
    return np.ascontiguousarray(out, dtype=np.float32)


def to_native_finedance139(motion: np.ndarray) -> np.ndarray:
    """Convert AgentLODGE layout to native FineDance 139-dim layout for FK/rendering.

    Native layout: contact (4) + root translation (3) + 22-joint 6D rotation (132).
    AgentLODGE layout: root translation (3) + rotation (132) + contact (4).
    """
    if motion.shape[-1] != 139:
        raise ValueError(f"Expected motion with 139 dims, got {motion.shape[-1]}")
    motion = motion.astype(np.float32)
    if not _is_native_finedance139(motion):
        return np.concatenate(
            [motion[:, 135:139], motion[:, :3], motion[:, 3:135]],
            axis=1,
        )
    return motion
