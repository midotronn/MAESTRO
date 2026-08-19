"""Normalize external baseline outputs into MAESTRO's 30 FPS Z-up 139 layout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentlodge.dance.format import to_editor139
from agentlodge.dance.transition import to_zup

TARGET_FPS = 30
TARGET_DIMS = 139
BODY_JOINTS = 22


@dataclass(frozen=True)
class ConversionResult:
    motion: np.ndarray
    report: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resample_frames(values: np.ndarray, source_fps: float) -> tuple[np.ndarray, dict]:
    source_fps = float(source_fps)
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    values = np.asarray(values)
    if values.shape[0] < 2 or abs(source_fps - TARGET_FPS) < 1e-6:
        return values, {"source_fps": source_fps, "target_fps": TARGET_FPS, "policy": "identity"}
    target_frames = max(2, int(round(values.shape[0] * TARGET_FPS / source_fps)))
    indices = np.rint(
        np.linspace(0, values.shape[0] - 1, target_frames)
    ).astype(np.int64)
    return values[indices], {
        "source_fps": source_fps,
        "target_fps": TARGET_FPS,
        "policy": "nearest_source_frame",
        "source_frames": int(values.shape[0]),
        "target_frames": int(target_frames),
    }


def _matrix_to_rotation_6d(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float32)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotation matrices ending in (3, 3), got {matrices.shape}")
    return matrices[..., :2, :].reshape(*matrices.shape[:-2], 6)


def _validate_motion(motion: np.ndarray) -> np.ndarray:
    motion = np.ascontiguousarray(np.asarray(motion, dtype=np.float32))
    if motion.ndim != 2 or motion.shape[1] != TARGET_DIMS:
        raise ValueError(f"expected (frames, {TARGET_DIMS}) motion, got {motion.shape}")
    if motion.shape[0] < 2:
        raise ValueError("motion must contain at least two frames")
    if not np.isfinite(motion).all():
        raise ValueError("motion contains NaN or infinite values")
    return motion


def _finish(
    motion: np.ndarray,
    *,
    method: str,
    source_fps: float,
    source_up_axis: str,
    contacts: str,
    resampling: dict,
) -> ConversionResult:
    source_up_axis = source_up_axis.lower()
    if source_up_axis not in {"y", "z"}:
        raise ValueError("source_up_axis must be 'y' or 'z'")
    motion = _validate_motion(motion)
    if source_up_axis == "y":
        motion = to_zup(motion)
    motion = _validate_motion(motion)
    report = {
        "method": method,
        "target_layout": "trans3_rot6d22_contact4",
        "target_fps": TARGET_FPS,
        "target_up_axis": "z",
        "source_fps": float(source_fps),
        "source_up_axis": source_up_axis,
        "frames": int(motion.shape[0]),
        "duration_seconds": round(motion.shape[0] / TARGET_FPS, 6),
        "contacts": contacts,
        "resampling": resampling,
        "root_min": motion[:, :3].min(axis=0).astype(float).tolist(),
        "root_max": motion[:, :3].max(axis=0).astype(float).tolist(),
    }
    return ConversionResult(motion=motion, report=report)


def convert_agentlodge_motion(
    motion: np.ndarray,
    *,
    method: str,
    source_fps: float = TARGET_FPS,
) -> ConversionResult:
    normalized = to_editor139(np.asarray(motion))
    normalized, resampling = _resample_frames(normalized, source_fps)
    return _finish(
        normalized,
        method=method,
        source_fps=source_fps,
        source_up_axis="z",
        contacts="provided",
        resampling=resampling,
    )


def convert_finedance_motion(
    motion319: np.ndarray,
    *,
    source_fps: float = TARGET_FPS,
    source_up_axis: str = "y",
) -> ConversionResult:
    """Convert FineDance's contact4 + trans3 + SMPL-H 52x6D output."""
    motion319 = np.asarray(motion319, dtype=np.float32)
    if motion319.ndim != 2 or motion319.shape[1] != 319:
        raise ValueError(f"expected FineDance (frames, 319), got {motion319.shape}")
    contacts = motion319[:, :4]
    translation = motion319[:, 4:7]
    rotations = motion319[:, 7:].reshape(-1, 52, 6)[:, :BODY_JOINTS]
    normalized = np.concatenate(
        [translation, rotations.reshape(-1, BODY_JOINTS * 6), contacts],
        axis=1,
    )
    normalized, resampling = _resample_frames(normalized, source_fps)
    return _finish(
        normalized,
        method="finedance",
        source_fps=source_fps,
        source_up_axis=source_up_axis,
        contacts="provided",
        resampling=resampling,
    )


def convert_bailando_motion(
    translation: np.ndarray,
    rotation_matrices: np.ndarray,
    *,
    contacts: np.ndarray | None = None,
    source_fps: float = 60.0,
    source_up_axis: str = "y",
) -> ConversionResult:
    """Convert Bailando++ SMPL translation and rotation matrices."""
    translation = np.asarray(translation, dtype=np.float32)
    matrices = np.asarray(rotation_matrices, dtype=np.float32)
    if translation.ndim != 2 or translation.shape[1] != 3:
        raise ValueError(f"expected translation (frames, 3), got {translation.shape}")
    if matrices.ndim != 4 or matrices.shape[0] != translation.shape[0]:
        raise ValueError(
            "rotation_matrices must have shape (frames, joints, 3, 3) "
            "with the same frame count as translation"
        )
    if matrices.shape[1] < BODY_JOINTS:
        raise ValueError(f"Bailando output must contain at least {BODY_JOINTS} joints")
    rotations = _matrix_to_rotation_6d(matrices[:, :BODY_JOINTS])
    if contacts is None:
        contact_values = np.zeros((translation.shape[0], 4), dtype=np.float32)
        contact_status = "unavailable_zero_filled"
    else:
        contact_values = np.asarray(contacts, dtype=np.float32)
        if contact_values.shape != (translation.shape[0], 4):
            raise ValueError(
                f"contacts must have shape {(translation.shape[0], 4)}, "
                f"got {contact_values.shape}"
            )
        contact_status = "provided"
    normalized = np.concatenate(
        [translation, rotations.reshape(-1, BODY_JOINTS * 6), contact_values],
        axis=1,
    )
    normalized, resampling = _resample_frames(normalized, source_fps)
    return _finish(
        normalized,
        method="bailando_plus_plus",
        source_fps=source_fps,
        source_up_axis=source_up_axis,
        contacts=contact_status,
        resampling=resampling,
    )


def save_conversion(
    result: ConversionResult,
    *,
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result.motion)
    report = {
        **result.report,
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
