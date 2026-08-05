"""Manifest-driven named motions for semantic window edits.

The bank is deliberately data-driven: the agent sees one generic operation while names, aliases,
capabilities, provenance, and validator contracts live in the manifest. Canonical clips use the
MAESTRO 139-channel layout at 30 FPS.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from agentlodge.dance.transition import (
    _matrix_to_axis_angle,
    _sixd_to_matrix,
    accentuate,
    mirror as mirror_motion,
    retime,
)

_ROOT = slice(3, 9)
_ROT = slice(3, 135)
_CONTACT = slice(135, 139)
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "assets" / "motion_bank"


def normalize_name(value: str) -> str:
    """Normalize a user-facing name for exact alias matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


@dataclass(frozen=True)
class MotionSpec:
    id: str
    name: str
    aliases: tuple[str, ...]
    category: str
    clip: str
    fps: int
    frames: int
    source: str
    license: str
    attribution: str
    stationary: bool
    travel_axis: str | None
    mirrorable: bool
    repeatable: bool
    event_frame: int
    recommended_beats: float
    validator: dict

    @classmethod
    def from_dict(cls, raw: dict) -> "MotionSpec":
        required = {
            "id", "name", "aliases", "category", "clip", "fps", "frames", "source", "license",
            "attribution", "stationary", "mirrorable", "repeatable", "event_frame",
            "recommended_beats", "validator",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"motion-bank entry missing fields: {', '.join(missing)}")
        motion_id = str(raw["id"]).strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", motion_id):
            raise ValueError(f"invalid motion id: {motion_id!r}")
        aliases = tuple(str(x).strip() for x in raw["aliases"] if str(x).strip())
        if not aliases:
            raise ValueError(f"{motion_id}: aliases cannot be empty")
        return cls(
            id=motion_id,
            name=str(raw["name"]).strip(),
            aliases=aliases,
            category=str(raw["category"]).strip(),
            clip=str(raw["clip"]).strip(),
            fps=int(raw["fps"]),
            frames=int(raw["frames"]),
            source=str(raw["source"]).strip(),
            license=str(raw["license"]).strip(),
            attribution=str(raw["attribution"]).strip(),
            stationary=bool(raw["stationary"]),
            travel_axis=(str(raw["travel_axis"]).strip() if raw.get("travel_axis") else None),
            mirrorable=bool(raw["mirrorable"]),
            repeatable=bool(raw["repeatable"]),
            event_frame=int(raw["event_frame"]),
            recommended_beats=float(raw["recommended_beats"]),
            validator=dict(raw["validator"]),
        )

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "aliases": list(self.aliases),
            "category": self.category,
            "frames": self.frames,
            "fps": self.fps,
            "mirrorable": self.mirrorable,
            "repeatable": self.repeatable,
            "recommended_beats": self.recommended_beats,
            "source": self.source,
            "license": self.license,
            "attribution": self.attribution,
        }


class MotionBank:
    """Load, resolve, fit, and verify canonical named motions."""

    def __init__(self, root: str | Path = _DEFAULT_ROOT):
        self.root = Path(root)
        payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported motion-bank manifest schema")
        self.version = str(payload.get("bank_version", ""))
        self.specs = tuple(MotionSpec.from_dict(x) for x in payload.get("motions", []))
        if not self.specs:
            raise ValueError("motion bank is empty")
        self._by_id = {spec.id: spec for spec in self.specs}
        if len(self._by_id) != len(self.specs):
            raise ValueError("motion-bank ids must be unique")
        self._aliases: dict[str, MotionSpec] = {}
        for spec in self.specs:
            for value in (spec.id, spec.name, *spec.aliases):
                key = normalize_name(value)
                prior = self._aliases.get(key)
                if prior is not None and prior.id != spec.id:
                    raise ValueError(f"duplicate motion alias {value!r}: {prior.id} and {spec.id}")
                self._aliases[key] = spec

    def list_public(self) -> list[dict]:
        return [spec.public_dict() for spec in self.specs]

    def resolve(self, value: str) -> MotionSpec:
        key = normalize_name(value)
        spec = self._aliases.get(key)
        if spec is None:
            raise KeyError(f"unknown named motion: {value}")
        return spec

    def match_instruction(self, instruction: str) -> MotionSpec | None:
        text = f" {normalize_name(instruction)} "
        matches = [
            (len(alias), spec)
            for alias, spec in self._aliases.items()
            if alias and f" {alias} " in text
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]

    def load_clip(self, spec_or_id: MotionSpec | str) -> np.ndarray:
        spec = spec_or_id if isinstance(spec_or_id, MotionSpec) else self.resolve(spec_or_id)
        path = (self.root / spec.clip).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"{spec.id}: clip path escapes the motion bank")
        if not path.is_file():
            raise FileNotFoundError(f"{spec.id}: missing clip {path}")
        clip = np.asarray(np.load(path), dtype=np.float32)
        validate_canonical_clip(clip, spec)
        return clip

    def apply(
        self,
        base_clip: np.ndarray,
        motion_id: str,
        *,
        beats: np.ndarray | None = None,
        mode: str = "replace",
        anchor: str = "center",
        mirror: bool = False,
        intensity: float = 0.5,
        repeats: int = 1,
        blend_frames: int = 8,
    ) -> tuple[np.ndarray, dict]:
        """Fit a named motion into ``base_clip`` without changing its frame count."""
        base = np.ascontiguousarray(base_clip, dtype=np.float32)
        if base.ndim != 2 or base.shape[1] != 139:
            raise ValueError(f"base clip must have shape (frames, 139), got {base.shape}")
        spec = self.resolve(motion_id)
        raw = self.load_clip(spec)
        if mirror:
            if not spec.mirrorable:
                raise ValueError(f"{spec.name} does not support mirroring")
            raw = mirror_motion(raw)
        count = int(np.clip(repeats, 1, 8))
        if count > 1:
            if not spec.repeatable:
                raise ValueError(f"{spec.name} does not support repetition")
            raw = np.concatenate([raw] * count, axis=0)
        source_event = (count // 2) * spec.frames + spec.event_frame
        gain = 0.7 + 0.6 * float(np.clip(intensity, 0.0, 1.0))
        if abs(gain - 1.0) > 1e-6:
            raw = accentuate(
                raw, gain, baseline_win=min(13, max(3, raw.shape[0] // 4 * 2 + 1)),
                taper_frames=min(4, max(1, raw.shape[0] // 8)), trans_gain=0.6,
            )

        n = int(base.shape[0])
        target_event = _target_event(n, beats, anchor)
        if mode == "replace":
            fitted = _fit_event(raw, source_event, n, target_event)
            fitted = _align_translation(fitted, base[0, :3])
            action_range = [0, n]
        elif mode == "insert":
            fitted, action_range = _insert_fixed_duration(
                base, raw, source_event, target_event, blend_frames=blend_frames,
            )
        else:
            raise ValueError(f"unsupported motion-bank mode: {mode!r}")

        action = fitted[action_range[0]:action_range[1]]
        validation = validate_semantics(action, spec)
        if not validation["ok"]:
            raise ValueError(f"{spec.name} failed semantic validation: {validation['detail']}")
        report = {
            "id": spec.id,
            "name": spec.name,
            "category": spec.category,
            "mode": mode,
            "anchor": anchor,
            "event_frame": int(target_event),
            "action_range": action_range,
            "mirror": bool(mirror),
            "intensity": float(np.clip(intensity, 0.0, 1.0)),
            "repeats": count,
            "source": spec.source,
            "license": spec.license,
            "attribution": spec.attribution,
            "validation": validation,
        }
        return np.ascontiguousarray(fitted, dtype=np.float32), report


@lru_cache(maxsize=1)
def default_motion_bank() -> MotionBank:
    return MotionBank()


def validate_canonical_clip(clip: np.ndarray, spec: MotionSpec) -> None:
    if clip.shape != (spec.frames, 139):
        raise ValueError(f"{spec.id}: expected {(spec.frames, 139)}, got {clip.shape}")
    if spec.fps != 30:
        raise ValueError(f"{spec.id}: only 30 FPS clips are supported")
    if not np.isfinite(clip).all():
        raise ValueError(f"{spec.id}: clip contains non-finite values")
    contacts = clip[:, _CONTACT]
    if not np.all((contacts == 0.0) | (contacts == 1.0)):
        raise ValueError(f"{spec.id}: contacts must be binary")
    matrices = _sixd_to_matrix(clip[:, _ROT].reshape(-1, 22, 6))
    ident = np.eye(3, dtype=np.float32)
    err = float(np.max(np.abs(matrices @ np.swapaxes(matrices, -1, -2) - ident)))
    det_err = float(np.max(np.abs(np.linalg.det(matrices) - 1.0)))
    if err > 2e-4 or det_err > 2e-4:
        raise ValueError(f"{spec.id}: invalid 6D rotations")
    if not (0 <= spec.event_frame < spec.frames):
        raise ValueError(f"{spec.id}: event frame lies outside clip")
    result = validate_semantics(clip, spec)
    if not result["ok"]:
        raise ValueError(f"{spec.id}: canonical semantic validation failed: {result['detail']}")


def validate_semantics(clip: np.ndarray, spec: MotionSpec) -> dict:
    """Run the declarative validator contract stored in the manifest."""
    contract = spec.validator
    kind = str(contract.get("type", "")).strip()
    threshold = float(contract.get("threshold", 0.0))
    metric = 0.0
    detail = ""

    if clip.shape[0] < 2:
        return {"ok": False, "type": kind, "metric": 0.0, "threshold": threshold,
                "detail": "clip is too short"}
    if kind == "joint_activity":
        joints = [int(j) for j in contract.get("joints", [])]
        aa = _matrix_to_axis_angle(_sixd_to_matrix(clip[:, _ROT].reshape(-1, 22, 6)))
        metric = float(np.max(np.linalg.norm(aa[:, joints], axis=-1))) if joints else 0.0
        detail = f"joint activity {metric:.3f}"
    elif kind == "vertical_peak":
        z = clip[:, 2]
        rise = float(np.max(z) - min(float(z[0]), float(z[-1])))
        airborne = float(np.mean(np.sum(clip[:, _CONTACT], axis=1) == 0))
        required_air = float(contract.get("airborne_fraction", 0.0))
        metric = min(rise / max(threshold, 1e-6), airborne / max(required_air, 1e-6))
        detail = f"rise {rise:.3f}, airborne {airborne:.3f}"
        threshold = 1.0
    elif kind == "vertical_cycles":
        z = clip[:, 2] - float(np.mean(clip[:, 2]))
        turns = np.count_nonzero(np.diff(np.sign(np.diff(z))) < 0)
        metric = float(turns)
        detail = f"vertical peaks {turns}"
    elif kind == "root_yaw":
        root = _sixd_to_matrix(clip[:, _ROOT].reshape(-1, 6))
        yaw = np.unwrap(np.arctan2(root[:, 1, 0], root[:, 0, 0]))
        metric = float(abs(yaw[-1] - yaw[0]))
        detail = f"yaw change {metric:.3f} rad"
    elif kind == "root_level":
        direction = str(contract.get("direction", "down"))
        delta = float(clip[-1, 2] - clip[0, 2])
        metric = -delta if direction == "down" else delta
        detail = f"root level delta {delta:.3f}"
    elif kind == "root_displacement":
        axes = {"x": 0, "y": 1, "z": 2}
        axis = axes.get(str(contract.get("axis", "x")), 0)
        metric = float(abs(clip[-1, axis] - clip[0, axis]))
        detail = f"root displacement {metric:.3f}"
    elif kind == "articulation_chain":
        joints = [int(j) for j in contract.get("joints", [])]
        aa = _matrix_to_axis_angle(_sixd_to_matrix(clip[:, _ROT].reshape(-1, 22, 6)))
        activity = np.max(np.linalg.norm(aa[:, joints], axis=-1), axis=0) if joints else np.array([0.0])
        metric = float(np.min(activity))
        detail = f"minimum chain activity {metric:.3f}"
    else:
        return {"ok": False, "type": kind, "metric": 0.0, "threshold": threshold,
                "detail": f"unknown validator type {kind!r}"}
    return {"ok": metric >= threshold, "type": kind, "metric": metric, "threshold": threshold,
            "detail": detail}


def verify_applied_motion(window: np.ndarray, report: dict, bank: MotionBank | None = None) -> dict:
    bank = bank or default_motion_bank()
    spec = bank.resolve(report["id"])
    start, end = (int(x) for x in report["action_range"])
    start = max(0, min(start, window.shape[0] - 1))
    end = max(start + 1, min(end, window.shape[0]))
    return validate_semantics(np.asarray(window[start:end], dtype=np.float32), spec)


def _target_event(n: int, beats: np.ndarray | None, anchor: str) -> int:
    ratios = {"start": 0.25, "early": 0.3, "center": 0.5, "beat": 0.5, "late": 0.7, "end": 0.75}
    target = int(round((n - 1) * ratios.get(str(anchor).lower(), 0.5)))
    if beats is not None:
        valid = np.asarray(beats, dtype=float)
        valid = valid[(valid >= 0) & (valid < n)]
        if valid.size:
            target = int(round(valid[np.argmin(np.abs(valid - target))]))
    return int(np.clip(target, 0, n - 1))


def _fit_event(clip: np.ndarray, event: int, n: int, target: int) -> np.ndarray:
    event = int(np.clip(event, 0, clip.shape[0] - 1))
    target = int(np.clip(target, 0, n - 1))
    if target == 0:
        return retime(clip[event:], n)
    if target == n - 1:
        return retime(clip[:event + 1], n)
    left = retime(clip[:event + 1], target + 1)
    right = retime(clip[event:], n - target)
    return np.concatenate([left[:-1], right], axis=0).astype(np.float32)


def _align_translation(clip: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = clip.copy()
    out[:, :3] += np.asarray(target, dtype=np.float32) - out[0, :3]
    return out


def _ease_join(left: np.ndarray, right: np.ndarray, frames: int) -> np.ndarray:
    if left.shape[0] == 0:
        return right
    if right.shape[0] == 0:
        return left
    out = right.copy()
    out[:, :3] += left[-1, :3] - out[0, :3]
    count = int(max(0, min(frames, out.shape[0])))
    if count:
        ref = np.repeat(left[-1:, :], count, axis=0)
        k = np.linspace(0.0, 1.0, count, dtype=np.float32)
        smooth = (3 * k ** 2 - 2 * k ** 3)[:, None]
        out[:count, :3] = (1.0 - smooth) * ref[:, :3] + smooth * out[:count, :3]
        r_ref = _sixd_to_matrix(ref[:, _ROT].reshape(count, 22, 6))
        r_out = _sixd_to_matrix(out[:count, _ROT].reshape(count, 22, 6))
        aa = _matrix_to_axis_angle(r_out @ np.swapaxes(r_ref, -1, -2))
        from agentlodge.dance.transition import _axis_angle_to_matrix, _matrix_to_sixd
        blended = _axis_angle_to_matrix(aa * smooth[:, None, :]) @ r_ref
        out[:count, _ROT] = _matrix_to_sixd(blended).reshape(count, 132)
        out[:count, _CONTACT] = np.where(smooth >= 0.5, out[:count, _CONTACT], ref[:, _CONTACT])
    return np.concatenate([left, out], axis=0).astype(np.float32)


def _insert_fixed_duration(
    base: np.ndarray,
    action: np.ndarray,
    event: int,
    target_event: int,
    *,
    blend_frames: int,
) -> tuple[np.ndarray, list[int]]:
    n = int(base.shape[0])
    if n < 24:
        raise ValueError("selected window is too short for insertion; use replace or select at least 0.8s")
    action_len = int(np.clip(round(0.45 * n), 12, max(12, n - 12)))
    event_ratio = float(np.clip(event / max(1, action.shape[0] - 1), 0.15, 0.85))
    action_event = int(round(event_ratio * (action_len - 1)))
    start = int(np.clip(target_event - action_event, 4, n - action_len - 4))
    end = start + action_len
    fitted_action = _fit_event(action, event, action_len, target_event - start)

    split = int(round((n - 1) * target_event / max(1, n - 1)))
    prefix = retime(base[:split + 1], start) if start > 0 else base[:0].copy()
    post_len = n - end
    suffix = retime(base[split:], post_len) if post_len > 0 else base[:0].copy()
    fitted_action = _align_translation(fitted_action, prefix[-1, :3] if prefix.shape[0] else base[0, :3])
    combined = _ease_join(prefix, fitted_action, min(blend_frames, max(1, action_len // 4)))
    combined = _ease_join(combined, suffix, min(blend_frames, max(1, post_len // 3))) if post_len else combined
    if combined.shape[0] != n:
        combined = retime(combined, n)
    return combined.astype(np.float32), [start, end]
