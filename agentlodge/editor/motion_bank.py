"""Manifest-driven named motions for semantic window edits.

The bank is deliberately data-driven: the agent sees one generic operation while names, aliases,
capabilities, provenance, and validator contracts live in the manifest. Canonical clips use the
MAESTRO 139-channel layout at 30 FPS.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from agentlodge.dance.format import to_editor139
from agentlodge.dance.transition import (
    _axis_angle_to_matrix,
    _matrix_to_axis_angle,
    _matrix_to_sixd,
    _sixd_to_matrix,
    accentuate,
    mirror as mirror_motion,
    retime,
)

_ROOT = slice(3, 9)
_ROT = slice(3, 135)
# Peak rate, in radians per frame, at which a seam is allowed to close a pose gap. Calibrated
# against the joint speeds the songs themselves reach: above this the hand-over outruns the dance.
_JOIN_MAX_RATE = 0.05
# ...but never for longer than about a beat. A hand-over borrows frames from the song's own
# choreography to fade the clip's pose offset out in, so letting it grow with the selection would
# quietly rewrite seconds of dance either side of a one-second gesture.
_JOIN_MAX_FRAMES = 16
_CONTACT = slice(135, 139)
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "assets" / "motion_bank"
_VALID_ANCHORS = {"start", "early", "center", "beat", "late", "end"}
DEFAULT_MOTION_INTENSITY = 0.65
_POSE_INTENSITY_SHARE = 0.65
_INTENSITY_EDGE_FRAMES = 4
_INTENSITY_LOCK_CORE = 1
_INTENSITY_LOCK_RADIUS = 1
_COMPOSITION_LOCK_SIGMA = 2.0

_FPS = 30.0
_CLOSE_YAW_RATE = 2.5      # rad/s the root may turn while giving an offset back
_CLOSE_TRANS_RATE = 1.0    # m/s the root may travel while giving an offset back
_CLOSE_LIFT_RATE = 0.6     # m/s for height, which reads as a bob rather than a step
_SMOOTHSTEP_PEAK = 1.5     # a smoothstep ramp peaks at 1.5x its average rate
_CLOSE_MAX_RATIO = 0.35    # never spend more than this share of the window closing
_INSERT_HOST_RATE = 0.75   # host choreography keeps moving while time is made for an insertion
_MIRROR_JOINTS = (0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20)
_DIRECTIONS = {"forward", "left", "right"}
_COUNTERFLOW_CLAP_JOINTS = (3, 6, 9)
_COUNTERFLOW_CLAP_TURN_DEGREES = 14.0
_COUNTERFLOW_CLAP_TURN_SHARES = (0.2, 0.3, 0.5)


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
    default_anchor: str
    validator: dict
    absolute_joints: tuple[int, ...]
    additive_joints: tuple[int, ...]
    translation_axes: tuple[int, ...]
    replace_contacts: bool
    carry_root_rotation: bool
    event_pose_joints: tuple[int, ...]
    intensity_lock_frames: tuple[int, ...]
    phase_joints: tuple[int, ...]
    phase_blend_frames: int | None
    minimum_frames: int
    direction_mode: str | None
    directions: tuple[str, ...]
    canonical_direction: str | None
    direction_clips: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, raw: dict) -> "MotionSpec":
        required = {
            "id", "name", "aliases", "category", "clip", "fps", "frames", "source", "license",
            "attribution", "stationary", "mirrorable", "repeatable", "event_frame",
            "recommended_beats", "default_anchor", "validator", "composition",
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
        composition = dict(raw["composition"])
        absolute = tuple(int(x) for x in composition.get("absolute_joints", ()))
        additive = tuple(int(x) for x in composition.get("additive_joints", ()))
        if any(j < 0 or j >= 22 for j in (*absolute, *additive)):
            raise ValueError(f"{motion_id}: composition joint lies outside the 22-joint body")
        if set(absolute) & set(additive):
            raise ValueError(f"{motion_id}: a joint cannot be both absolute and additive")
        event_pose = tuple(int(x) for x in composition.get("event_pose_joints", ()))
        if not set(event_pose).issubset(absolute):
            raise ValueError(
                f"{motion_id}: event-pose joints must also be declared as absolute joints"
            )
        intensity_lock_frames = tuple(
            int(x) for x in composition.get("intensity_lock_frames", ())
        )
        if any(frame < 0 or frame >= int(raw["frames"]) for frame in intensity_lock_frames):
            raise ValueError(f"{motion_id}: intensity lock frame lies outside clip")
        phase_joints = tuple(int(x) for x in composition.get("phase_joints", ()))
        if not set(phase_joints).issubset(set(absolute) | set(additive)):
            raise ValueError(f"{motion_id}: phase joints must be owned by the motion")
        phase_blend_frames = composition.get("phase_blend_frames")
        if phase_blend_frames is not None:
            phase_blend_frames = int(phase_blend_frames)
            if phase_blend_frames < 1:
                raise ValueError(f"{motion_id}: phase_blend_frames must be positive")
            if not phase_joints:
                raise ValueError(f"{motion_id}: phase_blend_frames requires phase_joints")
        minimum_frames = int(raw.get("minimum_frames", 1))
        if minimum_frames < 1 or minimum_frames > int(raw["frames"]):
            raise ValueError(
                f"{motion_id}: minimum_frames must be within the authored clip"
            )
        direction = raw.get("direction")
        direction_mode = None
        directions: tuple[str, ...] = ()
        canonical_direction = None
        direction_clips: tuple[tuple[str, str], ...] = ()
        if direction is not None:
            if not isinstance(direction, dict):
                raise ValueError(f"{motion_id}: direction must be an object")
            direction_mode = str(direction.get("mode", "")).strip().lower()
            if direction_mode not in {"clip", "mirror"}:
                raise ValueError(
                    f"{motion_id}: direction mode must be 'clip' or 'mirror'"
                )
            directions = tuple(
                str(value).strip().lower() for value in direction.get("supported", ())
            )
            if not directions or len(set(directions)) != len(directions):
                raise ValueError(
                    f"{motion_id}: supported directions must be non-empty and unique"
                )
            if set(directions) - _DIRECTIONS:
                raise ValueError(f"{motion_id}: unsupported direction name")
            canonical_direction = str(direction.get("canonical", "")).strip().lower()
            if canonical_direction not in directions:
                raise ValueError(
                    f"{motion_id}: canonical direction must be supported"
                )
            clips = direction.get("clips", {})
            if direction_mode == "clip":
                if not isinstance(clips, dict) or set(clips) != set(directions):
                    raise ValueError(
                        f"{motion_id}: clip directions need one clip per supported direction"
                    )
                direction_clips = tuple(
                    (name, str(clips[name]).strip()) for name in directions
                )
                if any(not path for _name, path in direction_clips):
                    raise ValueError(f"{motion_id}: direction clip path cannot be empty")
            elif not bool(raw["mirrorable"]):
                raise ValueError(
                    f"{motion_id}: mirror direction mode requires mirrorable=true"
                )
        axes = {"x": 0, "y": 1, "z": 2}
        try:
            translation_axes = tuple(axes[str(x).lower()] for x in composition.get(
                "translation_axes", ()))
        except KeyError as exc:
            raise ValueError(f"{motion_id}: invalid translation axis {exc.args[0]!r}") from exc
        carry_root_rotation = bool(composition.get("carry_root_rotation", False))
        if carry_root_rotation and 0 not in additive:
            raise ValueError(f"{motion_id}: carrying root rotation requires additive joint 0")
        default_anchor = str(raw["default_anchor"]).strip().lower()
        if default_anchor not in {"center", "beat"}:
            raise ValueError(
                f"{motion_id}: default_anchor must be 'center' or 'beat', got {default_anchor!r}"
            )
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
            default_anchor=default_anchor,
            validator=dict(raw["validator"]),
            absolute_joints=absolute,
            additive_joints=additive,
            translation_axes=translation_axes,
            replace_contacts=bool(composition.get("replace_contacts", False)),
            carry_root_rotation=carry_root_rotation,
            event_pose_joints=event_pose,
            intensity_lock_frames=intensity_lock_frames,
            phase_joints=phase_joints,
            phase_blend_frames=phase_blend_frames,
            minimum_frames=minimum_frames,
            direction_mode=direction_mode,
            directions=directions,
            canonical_direction=canonical_direction,
            direction_clips=direction_clips,
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
            "default_anchor": self.default_anchor,
            "minimum_frames": self.minimum_frames,
            "minimum_seconds": self.minimum_frames / self.fps,
            "directions": list(self.directions),
            "default_direction": "auto" if self.directions else None,
            "composition": {
                "absolute_joints": list(self.absolute_joints),
                "additive_joints": list(self.additive_joints),
                "translation_axes": ["xyz"[x] for x in self.translation_axes],
                "replace_contacts": self.replace_contacts,
                "carry_root_rotation": self.carry_root_rotation,
                "event_pose_joints": list(self.event_pose_joints),
                "intensity_lock_frames": list(self.intensity_lock_frames),
                "phase_joints": list(self.phase_joints),
                "phase_blend_frames": self.phase_blend_frames,
            },
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

    def load_clip(
        self,
        spec_or_id: MotionSpec | str,
        *,
        direction: str | None = None,
    ) -> np.ndarray:
        spec = spec_or_id if isinstance(spec_or_id, MotionSpec) else self.resolve(spec_or_id)
        clip_path = spec.clip
        if direction is not None and spec.direction_mode == "clip":
            clip_path = dict(spec.direction_clips).get(str(direction), "")
            if not clip_path:
                raise ValueError(
                    f"{spec.name} does not support direction {direction!r}"
                )
        path = (self.root / clip_path).resolve()
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
        beat_strengths: np.ndarray | None = None,
        mode: str = "replace",
        anchor: str | None = None,
        mirror: bool = False,
        direction: str | None = None,
        intensity: float | None = None,
        repeats: int = 1,
        blend_frames: int = 8,
    ) -> tuple[np.ndarray, dict]:
        """Fit a named motion into ``base_clip`` without changing its frame count."""
        base = to_editor139(base_clip)
        if base.ndim != 2 or base.shape[1] != 139:
            raise ValueError(f"base clip must have shape (frames, 139), got {base.shape}")
        spec = self.resolve(motion_id)
        resolved_anchor = spec.default_anchor if anchor is None else str(anchor).strip().lower()
        if resolved_anchor not in _VALID_ANCHORS:
            raise ValueError(f"unsupported motion-bank anchor: {anchor!r}")
        resolved_intensity = float(np.clip(
            DEFAULT_MOTION_INTENSITY if intensity is None else intensity,
            0.0,
            1.0,
        ))
        legacy_mirror = bool(mirror) and direction is None
        direction_request = (
            spec.canonical_direction
            if legacy_mirror and spec.directions
            else direction
        )
        requested_direction, resolved_direction, direction_source = _resolve_direction(
            base,
            spec,
            direction_request,
        )
        natural_direction = _dance_flow_direction(base) if spec.directions else None
        raw = self.load_clip(spec, direction=resolved_direction)
        compose_spec = spec
        direction_mirror = bool(
            spec.direction_mode == "mirror"
            and resolved_direction != spec.canonical_direction
        )
        # Semantic direction wins when both controls are supplied. ``mirror`` remains a backwards-
        # compatible low-level modifier only for callers that did not provide direction.
        effective_mirror = legacy_mirror ^ direction_mirror
        if effective_mirror:
            if not spec.mirrorable:
                raise ValueError(f"{spec.name} does not support mirroring")
            raw = mirror_motion(raw)
            compose_spec = replace(
                spec,
                absolute_joints=tuple(_MIRROR_JOINTS[j] for j in spec.absolute_joints),
                additive_joints=tuple(_MIRROR_JOINTS[j] for j in spec.additive_joints),
                event_pose_joints=tuple(_MIRROR_JOINTS[j] for j in spec.event_pose_joints),
                phase_joints=tuple(_MIRROR_JOINTS[j] for j in spec.phase_joints),
            )
            if resolved_direction in {"left", "right"} and legacy_mirror:
                resolved_direction = "right" if resolved_direction == "left" else "left"
                direction_source = "mirror"
        count = int(np.clip(repeats, 1, 8))
        if count > 1:
            if not spec.repeatable:
                raise ValueError(f"{spec.name} does not support repetition")
            raw = np.concatenate([raw] * count, axis=0)
            if compose_spec.event_pose_joints:
                compose_spec = replace(compose_spec, event_pose_joints=())
        intensity_lock_frames = tuple(
            copy * spec.frames + frame
            for copy in range(count)
            for frame in spec.intensity_lock_frames
        )
        source_event = (count // 2) * spec.frames + spec.event_frame
        gain = 0.7 + 0.6 * resolved_intensity
        if abs(gain - 1.0) > 1e-6:
            neutral_raw = raw
            raw = accentuate(
                raw, gain, baseline_win=min(13, max(3, raw.shape[0] // 4 * 2 + 1)),
                taper_frames=min(4, max(1, raw.shape[0] // 8)), trans_gain=0.6,
            )
            raw = _protect_intensity_frames(raw, neutral_raw, intensity_lock_frames)

        n = int(base.shape[0])
        if mode not in {"replace", "insert"}:
            raise ValueError(f"unsupported motion-bank mode: {mode!r}")
        if mode == "insert" and n < 24:
            raise ValueError("selected window is too short for insertion; use replace or select at least 0.8s")
        required_minimum = spec.minimum_frames * count
        if n - _join_tail(n) < required_minimum:
            minimum_window = _minimum_window_frames(required_minimum)
            raise ValueError(
                f"selected window is too short for {spec.name}; select at least "
                f"{minimum_window / _FPS:.1f}s so it can play at a natural speed"
            )
        action_len = _beat_locked_length(
            raw.shape[0], n, beats, recommended_beats=spec.recommended_beats * count,
            minimum_frames=required_minimum,
        )
        local_event = _action_local_event(raw.shape[0], source_event, action_len)
        latest = max(0, n - action_len - _join_tail(n))
        target_event = _target_event(
            n,
            beats,
            resolved_anchor,
            beat_strengths=beat_strengths,
            minimum=local_event,
            maximum=latest + local_event,
        )
        start, end, actual_event, local_event = _action_placement(
            n, action_len, local_event, target_event,
        )
        protected_frames = tuple(
            start + _fit_event_frame(
                frame,
                raw.shape[0],
                source_event,
                action_len,
                local_event,
            )
            for frame in intensity_lock_frames
        )
        protected_centers = tuple(
            start + _fit_event_frame(
                frame,
                raw.shape[0],
                source_event,
                action_len,
                local_event,
            )
            for frame in _lock_run_centers(intensity_lock_frames)
        )
        host = (
            base
            if mode == "replace"
            else _insertion_host(base, start, end, actual_event, local_event)
        )
        fitted, action_range, actual_event = _compose_beat_locked(
            host,
            raw,
            source_event,
            actual_event,
            action_len,
            compose_spec,
            blend_frames=blend_frames,
            protected_frames=tuple(frame - start for frame in protected_centers),
        )
        fitted = _exaggerate_owned_delta(
            fitted,
            host,
            compose_spec,
            action_range,
            gain=1.0 + _POSE_INTENSITY_SHARE * (gain - 1.0),
            protected_frames=protected_frames,
        )
        counterflow_turn = 0.0
        counterflow_turn_joints: tuple[int, ...] = ()
        if (
            spec.id.startswith("clap_")
            and direction_source == "explicit"
            and resolved_direction in {"left", "right"}
            and resolved_direction != natural_direction
        ):
            counterflow_turn = (
                _COUNTERFLOW_CLAP_TURN_DEGREES
                if resolved_direction == "left"
                else -_COUNTERFLOW_CLAP_TURN_DEGREES
            )
            counterflow_turn_joints = _COUNTERFLOW_CLAP_JOINTS
            fitted = _apply_counterflow_clap_turn(
                fitted,
                action_range,
                actual_event,
                counterflow_turn,
                blend_frames=blend_frames,
            )

        fitted = _close_root_residual(fitted, base, earliest=action_range[1])
        action = fitted[action_range[0]:action_range[1]]
        validation = validate_semantics(action, spec, reference=host[action_range[0]:action_range[1]])
        if not validation["ok"]:
            raise ValueError(f"{spec.name} failed semantic validation: {validation['detail']}")
        report = {
            "id": spec.id,
            "name": spec.name,
            "category": spec.category,
            "mode": mode,
            "anchor": resolved_anchor,
            "event_frame": int(actual_event),
            "action_range": action_range,
            "action_frames": int(action_range[1] - action_range[0]),
            "recommended_beats": float(spec.recommended_beats * count),
            "beat_error_frames": _beat_error(actual_event, beats, n),
            "mirror": effective_mirror,
            "direction_request": requested_direction,
            "direction": resolved_direction,
            "direction_source": direction_source,
            "natural_direction": natural_direction,
            "counterflow_turn_degrees": counterflow_turn,
            "counterflow_turn_joints": list(counterflow_turn_joints),
            "intensity": resolved_intensity,
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


def validate_semantics(
    clip: np.ndarray,
    spec: MotionSpec,
    *,
    reference: np.ndarray | None = None,
) -> dict:
    """Run the declarative validator contract stored in the manifest.

    Root contracts measure the largest excursion from the opening pose rather than the
    difference between the first and last frame. A window is spliced with its edges pinned
    to the surrounding dance, so an endpoint measure reports zero for a turn or a step that
    plainly happened -- the same reason ``vertical_peak`` already measures a peak.
    """
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
        rotations = _sixd_to_matrix(clip[:, _ROT].reshape(-1, 22, 6))
        aa = _matrix_to_axis_angle(rotations)
        activity = float(np.max(np.linalg.norm(aa[:, joints], axis=-1))) if joints else 0.0
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float32)
            if ref.shape != clip.shape:
                raise ValueError(
                    f"semantic reference must match clip shape {clip.shape}, got {ref.shape}"
                )
            ref_r = _sixd_to_matrix(ref[:, _ROT].reshape(-1, 22, 6))
            delta = _matrix_to_axis_angle(rotations @ np.swapaxes(ref_r, -1, -2))
            activity = max(
                activity,
                float(np.max(np.linalg.norm(delta[:, joints], axis=-1))) if joints else 0.0,
            )
        metric = activity
        detail = f"joint activity {metric:.3f}"
    elif kind == "jump_phases":
        z = np.asarray(clip[:, 2], dtype=np.float32)
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float32)
            if ref.shape != clip.shape:
                raise ValueError(
                    f"semantic reference must match clip shape {clip.shape}, got {ref.shape}"
                )
            z = z - ref[:, 2]
        else:
            z = z - np.linspace(float(z[0]), float(z[-1]), clip.shape[0], dtype=np.float32)
        apex = int(np.argmax(z))
        if apex < 2 or apex > clip.shape[0] - 3:
            return {
                "ok": False,
                "type": kind,
                "metric": 0.0,
                "threshold": 1.0,
                "detail": f"apex frame {apex} leaves no takeoff or landing phase",
            }
        load = int(np.argmin(z[:apex + 1]))
        landing = apex + int(np.argmin(z[apex:]))
        baseline = 0.5 * (float(z[0]) + float(z[-1]))
        rise = float(z[apex] - min(float(z[0]), float(z[-1])))
        load_drop = float(baseline - z[load])
        airborne = float(np.mean(np.sum(clip[:, _CONTACT], axis=1) == 0))

        rotations = _sixd_to_matrix(clip[:, _ROT].reshape(-1, 22, 6))
        load_delta = _matrix_to_axis_angle(
            rotations[load, [4, 5]]
            @ np.swapaxes(rotations[apex, [4, 5]], -1, -2)
        )
        landing_delta = _matrix_to_axis_angle(
            rotations[landing, [4, 5]]
            @ np.swapaxes(rotations[apex, [4, 5]], -1, -2)
        )
        load_flexion = float(np.min(np.linalg.norm(load_delta, axis=-1)))
        landing_flexion = float(np.min(np.linalg.norm(landing_delta, axis=-1)))
        grounded_before = bool(np.any(np.sum(clip[:apex, _CONTACT], axis=1) > 0))
        grounded_after = bool(np.any(np.sum(clip[apex + 1:, _CONTACT], axis=1) > 0))
        apex_airborne = bool(np.sum(clip[apex, _CONTACT]) == 0)

        requirements = {
            "rise": (rise, float(contract.get("height", 0.0))),
            "load drop": (load_drop, float(contract.get("load_drop", 0.0))),
            "load knee flexion": (
                load_flexion, float(contract.get("load_flexion", 0.0))
            ),
            "landing knee flexion": (
                landing_flexion, float(contract.get("landing_flexion", 0.0))
            ),
            "airborne fraction": (
                airborne, float(contract.get("airborne_fraction", 0.0))
            ),
        }
        ratios = [
            value / max(required, 1e-6)
            for value, required in requirements.values()
            if required > 0.0
        ]
        metric = min(ratios) if ratios else 1.0
        contacts_ok = grounded_before and apex_airborne and grounded_after
        if not contacts_ok:
            metric = 0.0
        detail = (
            f"load f{load} drop {load_drop:.3f} flex {load_flexion:.3f}; "
            f"apex f{apex} rise {rise:.3f}; landing f{landing} flex "
            f"{landing_flexion:.3f}; airborne {airborne:.3f}; "
            f"contacts {'ground-air-ground' if contacts_ok else 'invalid'}"
        )
        threshold = 1.0
    elif kind == "vertical_peak":
        z = clip[:, 2]
        rise = float(np.max(z) - min(float(z[0]), float(z[-1])))
        airborne = float(np.mean(np.sum(clip[:, _CONTACT], axis=1) == 0))
        required_air = float(contract.get("airborne_fraction", 0.0))
        metric = min(rise / max(threshold, 1e-6), airborne / max(required_air, 1e-6))
        detail = f"rise {rise:.3f}, airborne {airborne:.3f}"
        threshold = 1.0
    elif kind == "grounded_vertical_cycles":
        z = np.asarray(clip[:, 2], dtype=np.float32)
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float32)
            if ref.shape != clip.shape:
                raise ValueError(
                    f"semantic reference must match clip shape {clip.shape}, got {ref.shape}"
                )
            z = z - ref[:, 2]
        turns = np.diff(np.sign(np.diff(z)))
        pulses = int(np.count_nonzero(turns > 0))
        height = float(np.max(z) - np.min(z))
        grounded = float(np.mean(np.all(clip[:, _CONTACT] > 0.5, axis=1)))
        joints = [int(j) for j in contract.get("joints", ())]
        rotations = _sixd_to_matrix(clip[:, _ROT].reshape(-1, 22, 6))
        activity = (
            np.max(np.linalg.norm(_matrix_to_axis_angle(rotations[:, joints]), axis=-1), axis=0)
            if joints else np.array([np.inf], dtype=np.float32)
        )
        if reference is not None and joints:
            ref_r = _sixd_to_matrix(ref[:, _ROT].reshape(-1, 22, 6))
            delta = _matrix_to_axis_angle(
                rotations[:, joints] @ np.swapaxes(ref_r[:, joints], -1, -2)
            )
            activity = np.maximum(
                activity,
                np.max(np.linalg.norm(delta, axis=-1), axis=0),
            )
        bilateral_activity = float(np.min(activity))
        required_pulses = max(1, int(contract.get("pulses", threshold or 1)))
        required_height = max(1e-6, float(contract.get("height", 0.0)))
        required_grounded = max(1e-6, float(contract.get("grounded_fraction", 1.0)))
        required_activity = max(1e-6, float(contract.get("joint_activity", 0.0)))
        metric = min(
            pulses / required_pulses,
            height / required_height,
            grounded / required_grounded,
            bilateral_activity / required_activity,
        )
        threshold = 1.0
        detail = (
            f"down pulses {pulses}; vertical travel {height:.3f}; "
            f"fully planted {grounded:.3f}; bilateral activity {bilateral_activity:.3f}"
        )
    elif kind == "vertical_cycles":
        z = clip[:, 2] - float(np.mean(clip[:, 2]))
        turns = np.count_nonzero(np.diff(np.sign(np.diff(z))) < 0)
        metric = float(turns)
        detail = f"vertical peaks {turns}"
    elif kind == "root_yaw":
        yaw = _root_yaw_series(clip)
        metric = float(np.max(np.abs(yaw - yaw[0])))
        detail = f"yaw excursion {metric:.3f} rad"
    elif kind == "rise_phases":
        z = np.asarray(clip[:, 2], dtype=np.float32)
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float32)
            if ref.shape != clip.shape:
                raise ValueError(
                    f"semantic reference must match clip shape {clip.shape}, got {ref.shape}"
                )
            z = z - ref[:, 2]
        low = int(np.argmin(z))
        high = low + int(np.argmax(z[low:]))
        rise = float(z[high] - z[low])
        grounded = float(np.mean(np.all(clip[:, _CONTACT] > 0.5, axis=1)))
        required_height = max(1e-6, float(contract.get("height", threshold)))
        required_grounded = max(1e-6, float(contract.get("grounded_fraction", 1.0)))
        metric = min(rise / required_height, grounded / required_grounded)
        threshold = 1.0
        detail = (
            f"low f{low}; high f{high}; rise {rise:.3f}; "
            f"fully planted {grounded:.3f}"
        )
    elif kind == "root_level":
        direction = str(contract.get("direction", "down"))
        z = np.asarray(clip[:, 2], dtype=np.float32)
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float32)
            if ref.shape != clip.shape:
                raise ValueError(
                    f"semantic reference must match clip shape {clip.shape}, got {ref.shape}"
                )
            delta = z - ref[:, 2]
        else:
            delta = z - float(z[0])
        metric = float(np.max(-delta if direction == "down" else delta))
        detail = f"root level excursion {metric:.3f}"
    elif kind == "root_displacement":
        # Measured in the dancer's own frame, not the world's. `insert` yaw-aligns the action to
        # whatever direction the dancer happens to be facing at the splice point, so a world-axis
        # component collapses as the dancer turns: the identical edit passed at 0 and 180 degrees
        # but failed at 90 and 270 purely because the travel had rotated onto the other world
        # axis. "Stepped 0.28 forward" is a claim about the dancer, so it is read off the dancer.
        axes = {"x": 0, "y": 1, "z": 2}
        axis = axes.get(str(contract.get("axis", "x")), 0)
        if axis == 2:
            metric = float(np.max(np.abs(clip[:, 2] - float(clip[0, 2]))))
        else:
            yaw = float(_root_yaw_series(clip[:1])[0])
            basis = (np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32) if axis == 0
                     else np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32))
            metric = float(np.max(np.abs((clip[:, :2] - clip[0, :2]) @ basis)))
        detail = f"root displacement {metric:.3f}"
    elif kind == "articulation_chain":
        joints = [int(j) for j in contract.get("joints", [])]
        rotations = _sixd_to_matrix(clip[:, _ROT].reshape(-1, 22, 6))
        aa = _matrix_to_axis_angle(rotations)
        activity = np.max(np.linalg.norm(aa[:, joints], axis=-1), axis=0) if joints else np.array([0.0])
        if reference is not None:
            ref = np.asarray(reference, dtype=np.float32)
            if ref.shape != clip.shape:
                raise ValueError(
                    f"semantic reference must match clip shape {clip.shape}, got {ref.shape}"
                )
            ref_r = _sixd_to_matrix(ref[:, _ROT].reshape(-1, 22, 6))
            delta = _matrix_to_axis_angle(rotations @ np.swapaxes(ref_r, -1, -2))
            relative = (
                np.max(np.linalg.norm(delta[:, joints], axis=-1), axis=0)
                if joints else np.array([0.0])
            )
            activity = np.maximum(activity, relative)
        metric = float(np.min(activity))
        detail = f"minimum chain activity {metric:.3f}"
    else:
        return {"ok": False, "type": kind, "metric": 0.0, "threshold": threshold,
                "detail": f"unknown validator type {kind!r}"}
    return {"ok": metric >= threshold, "type": kind, "metric": metric, "threshold": threshold,
            "detail": detail}


def verify_applied_motion(
    window: np.ndarray,
    report: dict,
    bank: MotionBank | None = None,
    *,
    reference: np.ndarray | None = None,
) -> dict:
    bank = bank or default_motion_bank()
    spec = bank.resolve(report["id"])
    start, end = (int(x) for x in report["action_range"])
    start = max(0, min(start, window.shape[0] - 1))
    end = max(start + 1, min(end, window.shape[0]))
    ref = None
    if reference is not None:
        ref_window = np.asarray(reference, dtype=np.float32)
        if ref_window.shape != window.shape:
            raise ValueError(
                f"verification reference must match window shape {window.shape}, got {ref_window.shape}"
            )
        ref = ref_window[start:end]
    return validate_semantics(
        np.asarray(window[start:end], dtype=np.float32),
        spec,
        reference=ref,
    )


def _target_event(
    n: int,
    beats: np.ndarray | None,
    anchor: str,
    *,
    beat_strengths: np.ndarray | None = None,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    ratios = {"start": 0.25, "early": 0.3, "center": 0.5, "beat": 0.5, "late": 0.7, "end": 0.75}
    target = int(round((n - 1) * ratios.get(str(anchor).lower(), 0.5)))
    lo = int(np.clip(minimum, 0, max(0, n - 1)))
    hi = int(np.clip(n - 1 if maximum is None else maximum, lo, max(lo, n - 1)))
    if beats is not None:
        all_beats = np.asarray(beats, dtype=float).reshape(-1)
        strengths = None
        if beat_strengths is not None:
            strengths = np.asarray(beat_strengths, dtype=float).reshape(-1)
            if strengths.size != all_beats.size:
                raise ValueError(
                    "beat_strengths must contain one value for every beat "
                    f"({strengths.size} != {all_beats.size})"
                )
        valid_mask = (all_beats >= 0) & (all_beats < n)
        valid = all_beats[valid_mask]
        valid_strengths = strengths[valid_mask] if strengths is not None else None
        if valid.size:
            feasible_mask = (valid >= lo) & (valid <= hi)
            if np.any(feasible_mask):
                candidates = valid[feasible_mask]
                candidate_strengths = (
                    valid_strengths[feasible_mask] if valid_strengths is not None else None
                )
            else:
                candidates = valid
                candidate_strengths = valid_strengths
            if str(anchor).lower() == "beat" and candidate_strengths is not None:
                finite = np.isfinite(candidate_strengths)
                if np.any(finite):
                    strongest = float(np.max(candidate_strengths[finite]))
                    strongest_mask = finite & np.isclose(candidate_strengths, strongest)
                    candidates = candidates[strongest_mask]
            target = int(round(candidates[np.argmin(np.abs(candidates - target))]))
    return int(np.clip(target, lo, hi))


def _beat_error(target: int, beats: np.ndarray | None, n: int) -> float | None:
    if beats is None:
        return None
    valid = np.asarray(beats, dtype=float)
    if not valid.size:
        return None
    return float(np.min(np.abs(valid - int(target))))


def _intensity_release(n: int, locked_frames: tuple[int, ...]) -> np.ndarray:
    """Return zero at protected poses and ease back to full intensity around them."""
    release = np.ones(int(n), dtype=np.float32)
    positions = np.arange(int(n), dtype=np.float32)
    for frame in locked_frames:
        distance = np.abs(positions - float(frame)) - _INTENSITY_LOCK_CORE
        t = np.clip(distance / _INTENSITY_LOCK_RADIUS, 0.0, 1.0)
        release = np.minimum(release, t * t * (3.0 - 2.0 * t))
    return release


def _lock_run_centers(locked_frames: tuple[int, ...]) -> tuple[int, ...]:
    runs: list[list[int]] = []
    for frame in locked_frames:
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return tuple((run[0] + run[-1] + 1) // 2 for run in runs)


def _protect_intensity_frames(
    accented: np.ndarray,
    neutral: np.ndarray,
    locked_frames: tuple[int, ...],
) -> np.ndarray:
    """Preserve authored contact geometry while accenting its approach and release."""
    if not locked_frames:
        return accented
    out = np.asarray(accented, dtype=np.float32).copy()
    reference = np.asarray(neutral, dtype=np.float32)
    release = _intensity_release(out.shape[0], locked_frames)
    out[:, :3] = reference[:, :3] + release[:, None] * (
        out[:, :3] - reference[:, :3]
    )
    reference_r = _sixd_to_matrix(reference[:, _ROT].reshape(-1, 22, 6))
    accented_r = _sixd_to_matrix(out[:, _ROT].reshape(-1, 22, 6))
    delta = accented_r @ np.swapaxes(reference_r, -1, -2)
    out[:, _ROT] = _matrix_to_sixd(
        _fractional_rotation(delta, release[:, None]) @ reference_r
    ).reshape(-1, 22 * 6)
    return out


def _fit_event_frame(
    frame: int,
    source_frames: int,
    event: int,
    output_frames: int,
    target: int,
) -> int:
    """Map a source pose to the same output frame used by ``_fit_event``."""
    frame = int(np.clip(frame, 0, source_frames - 1))
    event = int(np.clip(event, 0, source_frames - 1))
    target = int(np.clip(target, 0, output_frames - 1))
    if frame <= event:
        return int(round(target * frame / max(1, event)))
    tail = int(output_frames - 1 - target)
    return int(target + round(
        tail * (frame - event) / max(1, source_frames - 1 - event)
    ))


def _exaggerate_owned_delta(
    motion: np.ndarray,
    reference: np.ndarray,
    spec: MotionSpec,
    action_range: tuple[int, int],
    *,
    gain: float,
    protected_frames: tuple[int, ...] = (),
) -> np.ndarray:
    """Scale only the named action's owned delta, leaving the host dance and seams intact."""
    if abs(float(gain) - 1.0) <= 1e-6:
        return motion
    out = np.asarray(motion, dtype=np.float32).copy()
    ref = np.asarray(reference, dtype=np.float32)
    a, b = map(int, action_range)
    if b <= a:
        return out

    frame_gain = np.full(b - a, float(gain), dtype=np.float32)
    edge = min(_INTENSITY_EDGE_FRAMES, max(1, (b - a) // 4))
    if edge:
        t = np.linspace(0.0, 1.0, edge + 2, dtype=np.float32)[1:-1]
        taper = t * t * (3.0 - 2.0 * t)
        frame_gain[:edge] = 1.0 + (gain - 1.0) * taper
        frame_gain[-edge:] = 1.0 + (gain - 1.0) * taper[::-1]
    local_locks = tuple(frame - a for frame in protected_frames if a <= frame < b)
    if local_locks:
        frame_gain = 1.0 + (frame_gain - 1.0) * _intensity_release(
            b - a, local_locks
        )
    active = np.flatnonzero(np.abs(frame_gain - 1.0) > 1e-6)
    if not active.size:
        return out

    # Root yaw has its own bounded close-out and wrists carry authored hand planes. Scaling either
    # here can create a seam whip or make a correct palm pose look detached from its forearm.
    owned = sorted(
        (set(spec.absolute_joints) | set(spec.additive_joints)) - {0, 20, 21}
    )
    if owned:
        ref_rot = _sixd_to_matrix(ref[a:b, _ROT].reshape(-1, 22, 6))
        out_rot = _sixd_to_matrix(out[a:b, _ROT].reshape(-1, 22, 6))
        delta = (
            out_rot[active][:, owned]
            @ np.swapaxes(ref_rot[active][:, owned], -1, -2)
        )
        delta_aa = _matrix_to_axis_angle(delta) * frame_gain[active, None, None]
        scaled = _matrix_to_sixd(
            _axis_angle_to_matrix(delta_aa) @ ref_rot[active][:, owned]
        )
        for index, joint in enumerate(owned):
            channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
            out[a + active, channels] = scaled[:, index]

    for axis in spec.translation_axes:
        frames = a + active
        out[frames, axis] = ref[frames, axis] + frame_gain[active] * (
            out[frames, axis] - ref[frames, axis]
        )
    return out


def _join_tail(n: int) -> int:
    """Frames of dance kept after an action so the hand-back has somewhere to happen."""
    return int(min(8, max(0, int(n) // 8)))


def _minimum_window_frames(action_frames: int) -> int:
    """Smallest selection that leaves both the action and its blend-back room."""
    n = int(max(1, action_frames))
    while n - _join_tail(n) < action_frames:
        n += 1
    return n


def _dance_flow_direction(base: np.ndarray) -> str:
    """Infer left/right flow in the dancer's frame, falling back to forward.

    Translation is the strongest cue. A mostly stationary turning phrase uses yaw direction as
    its fallback. Tiny lateral noise is ignored so a centered dance does not make gestures flip
    sides unpredictably.
    """
    clip = np.asarray(base, dtype=np.float32)
    n = int(clip.shape[0])
    if n < 4:
        return "forward"
    width = int(np.clip(n // 5, 2, 18))
    start = np.mean(clip[:width, :2], axis=0)
    end = np.mean(clip[-width:, :2], axis=0)
    delta = end - start
    yaw_series = _root_yaw_series(clip)
    center = yaw_series[max(0, n // 2 - width // 2):min(n, n // 2 + width // 2 + 1)]
    yaw = float(np.median(center))
    left_axis = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    forward_axis = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
    lateral = float(delta @ left_axis)
    forward = float(delta @ forward_axis)
    if abs(lateral) >= 0.04 and abs(lateral) >= 0.45 * abs(forward):
        return "left" if lateral > 0.0 else "right"
    yaw_change = float(yaw_series[-1] - yaw_series[0])
    if abs(yaw_change) >= 0.25:
        return "left" if yaw_change > 0.0 else "right"
    return "forward"


def _resolve_direction(
    base: np.ndarray,
    spec: MotionSpec,
    requested: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not spec.directions:
        if requested not in (None, "", "auto"):
            raise ValueError(f"{spec.name} does not support direction")
        return None, None, None
    normalized = "auto" if requested is None else normalize_name(requested).replace(" ", "_")
    if normalized == "straight_ahead":
        normalized = "forward"
    if normalized != "auto" and normalized not in spec.directions:
        raise ValueError(
            f"{spec.name} supports directions: {', '.join(spec.directions)}"
        )
    if normalized != "auto":
        return normalized, normalized, "explicit"
    flow = _dance_flow_direction(base)
    if flow in spec.directions:
        return "auto", flow, "dance_flow"
    if "forward" in spec.directions:
        return "auto", "forward", "dance_flow"
    return "auto", spec.canonical_direction, "canonical_fallback"


def _action_local_event(action_frames: int, event: int, action_len: int) -> int:
    ratio = float(np.clip(event / max(1, action_frames - 1), 0.0, 1.0))
    return int(round(ratio * max(0, action_len - 1)))


def _action_placement(
    n: int,
    action_len: int,
    local_event: int,
    target_event: int,
) -> tuple[int, int, int, int]:
    action_len = int(np.clip(action_len, 1, n))
    local_event = int(np.clip(local_event, 0, action_len - 1))
    latest = max(0, n - action_len - _join_tail(n))
    start = int(np.clip(target_event - local_event, 0, latest))
    end = start + action_len
    local_event = int(np.clip(target_event - start, 0, action_len - 1))
    return start, end, start + local_event, local_event


def _insertion_host(
    base: np.ndarray,
    start: int,
    end: int,
    target_event: int,
    local_event: int,
) -> np.ndarray:
    """Time-warp the host around a fixed-duration insertion without freezing its other joints.

    The total song length cannot grow, so insertion necessarily makes time somewhere. The host
    keeps moving through the action at 75% speed while its prefix and suffix are compressed just
    enough to keep both window endpoints unchanged. All source frames remain represented in order,
    and the named action is then layered onto this continuous host rather than replacing it.
    """
    n = int(base.shape[0])
    action_len = int(end - start)
    source_intervals = int(np.clip(
        round(max(1, action_len - 1) * _INSERT_HOST_RATE),
        1,
        max(1, n - 1),
    ))
    source_event = int(round(
        local_event * source_intervals / max(1, action_len - 1),
    ))
    source_start = int(np.clip(
        target_event - source_event,
        0,
        max(0, n - 1 - source_intervals),
    ))
    source_end = source_start + source_intervals
    source_event = int(np.clip(target_event - source_start, 0, source_intervals))

    prefix = retime(base[:source_start + 1], start + 1)
    action_host = _fit_event(
        base[source_start:source_end + 1],
        source_event,
        action_len,
        local_event,
    )
    suffix = retime(base[source_end:], n - end + 1)
    host = np.concatenate([prefix[:-1], action_host, suffix[1:]], axis=0)
    if host.shape[0] != n:
        raise RuntimeError(f"insertion host changed frame count: expected {n}, got {host.shape[0]}")
    return np.ascontiguousarray(host, dtype=np.float32)


def _beat_locked_length(
    natural: int,
    n: int,
    beats: np.ndarray | None,
    *,
    recommended_beats: float | None = None,
    minimum_frames: int = 1,
) -> int:
    """How many frames the action gets: its authored duration, snapped to a whole number of beats.

    The window says *where* an action goes, not how fast it runs. Retiming the clip to the whole
    selection makes its speed a function of how wide the user happened to drag: a 1.5s clap
    dropped on a 4s selection plays at 0.4x, smeared across nearly eight beats, which reads as
    slow motion fighting the song rather than dancing with it. The authored duration is the speed
    the motion was designed at, so it is what gets used; snapping it to whole beats lands the
    start and the finish on the groove instead of drifting against it.
    """
    natural = int(max(1, natural))
    # Always keep a few frames of dance after the action to hand back into. Filling the window to
    # its last frame leaves the join no room, and the whole pose gap gets closed in one or two
    # frames: measured joint speed hit nearly six times anything in the song when the action ended
    # two frames from the edge. Dropping a beat is far cheaper than that lurch.
    room = int(max(1, n - _join_tail(n)))
    minimum = int(np.clip(minimum_frames, 1, room))
    if beats is None:
        return int(np.clip(natural, minimum, room))
    b = np.unique(np.asarray(beats, dtype=float))
    if b.size < 2:
        return int(np.clip(natural, minimum, room))
    period = float(np.median(np.diff(b)))
    if not np.isfinite(period) or period < 1.0:
        return int(np.clip(natural, minimum, room))
    if recommended_beats is None:
        beats_wide = max(1.0, float(round(natural / period)))
    else:
        beats_wide = max(0.25, float(recommended_beats))
    fitted = int(round(beats_wide * period))
    while beats_wide > 1.0 and fitted > room:
        beats_wide -= 1.0
        fitted = int(round(beats_wide * period))
    while fitted < minimum:
        candidate = int(round((beats_wide + 1.0) * period))
        if candidate > room or candidate <= fitted:
            break
        beats_wide += 1.0
        fitted = candidate
    return int(np.clip(fitted, minimum, room))


def _composition_envelope(n: int, event: int, blend_frames: int) -> np.ndarray:
    """Weight of the named action inside its slot, pinned to the host dance at both ends."""
    n = int(n)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    event = int(np.clip(event, 0, n - 1))
    width = int(max(1, min(blend_frames, event, n - 1 - event)))
    weight = np.ones(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, width + 1, dtype=np.float32)
    ramp = 3.0 * ramp ** 2 - 2.0 * ramp ** 3
    weight[:width + 1] = ramp
    weight[n - 1 - width:] = np.minimum(weight[n - 1 - width:], ramp[::-1])
    return weight


def _fractional_rotation(delta: np.ndarray, weight: np.ndarray) -> np.ndarray:
    aa = _matrix_to_axis_angle(delta)
    return _axis_angle_to_matrix(aa * np.asarray(weight, dtype=np.float32)[..., None])


def _apply_counterflow_clap_turn(
    clip: np.ndarray,
    action_range: tuple[int, int],
    event_frame: int,
    turn_degrees: float,
    *,
    blend_frames: int,
) -> np.ndarray:
    """Turn only the upper spine when an explicit clap opposes the host's natural flow."""
    out = np.asarray(clip, dtype=np.float32).copy()
    start, end = map(int, action_range)
    if end <= start:
        return out
    local_event = int(np.clip(event_frame - start, 0, end - start - 1))
    weight = _composition_envelope(end - start, local_event, blend_frames)
    rotations = _sixd_to_matrix(out[start:end, _ROT].reshape(-1, 22, 6))
    radians = np.deg2rad(float(turn_degrees))
    parent_global = rotations[:, 0]
    for joint, share in zip(
        _COUNTERFLOW_CLAP_JOINTS,
        _COUNTERFLOW_CLAP_TURN_SHARES,
    ):
        current_global = parent_global @ rotations[:, joint]
        desired_global = (
            _yaw_rotation(weight * np.float32(radians * share))
            @ current_global
        )
        rotations[:, joint] = np.swapaxes(parent_global, -1, -2) @ desired_global
        parent_global = desired_global
    sixd = _matrix_to_sixd(rotations)
    for joint in _COUNTERFLOW_CLAP_JOINTS:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        out[start:end, channels] = sixd[:, joint]
    return out


def _rotate_translation_into_facing(
    delta: np.ndarray,
    action: np.ndarray,
    base_frame: np.ndarray,
) -> np.ndarray:
    """Rotate canonical X/Y travel into the direction the host dancer currently faces."""
    action_yaw = float(_root_yaw_series(action[:1])[0])
    base_yaw = float(_root_yaw_series(base_frame[None, :])[0])
    angle = base_yaw - action_yaw
    c, s = float(np.cos(angle)), float(np.sin(angle))
    out = np.asarray(delta, dtype=np.float32).copy()
    x, y = out[:, 0].copy(), out[:, 1].copy()
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def _yaw_rotation(angle: np.ndarray | float) -> np.ndarray:
    """Rotation matrix for a world-up yaw, preserving any leading angle dimensions."""
    angle = np.asarray(angle, dtype=np.float32)
    flat = angle.reshape(-1)
    cos, sin = np.cos(flat), np.sin(flat)
    spin = np.zeros((flat.size, 3, 3), dtype=np.float32)
    spin[:, 0, 0], spin[:, 0, 1] = cos, -sin
    spin[:, 1, 0], spin[:, 1, 1] = sin, cos
    spin[:, 2, 2] = 1.0
    return spin.reshape(angle.shape + (3, 3))


def _compose_beat_locked(
    base: np.ndarray,
    action: np.ndarray,
    event: int,
    target_event: int,
    action_len: int,
    spec: MotionSpec,
    *,
    blend_frames: int,
    protected_frames: tuple[int, ...] = (),
) -> tuple[np.ndarray, list[int], int]:
    """Layer the named action onto the host instead of replacing its entire choreography.

    Canonical clips share one ready stance. Replacing all 22 joints with that stance erased the
    song's groove, footwork, contacts, and body style for the duration of every named action. A
    clap changed 152 foot-contact bits and made non-arm velocity correlation fall to 0.09 even
    though no leg or root motion was requested. Composition metadata now states exactly which
    joints and root channels each action owns; everything else remains bit-identical to the host.
    """
    n = int(base.shape[0])
    action_len = int(np.clip(action_len, 1, n))
    local_event = _action_local_event(action.shape[0], event, action_len)
    start, end, actual_event, local_event = _action_placement(
        n, action_len, local_event, target_event,
    )

    fitted = _fit_event(action, event, action_len, local_event)
    host = base[start:end]
    out = base.copy()
    weight = _composition_envelope(action_len, local_event, blend_frames)
    if protected_frames:
        positions = np.arange(action_len, dtype=np.float32)
        for frame in protected_frames:
            bump = np.exp(
                -0.5 * ((positions - float(frame)) / _COMPOSITION_LOCK_SIGMA) ** 2
            )
            weight = np.maximum(weight, bump.astype(np.float32))
    phase_weight = weight
    if spec.phase_joints and spec.phase_blend_frames is not None:
        phase_weight = _composition_envelope(
            action_len,
            local_event,
            min(int(blend_frames), spec.phase_blend_frames),
        )
    phase_joints = set(spec.phase_joints)
    base_r = _sixd_to_matrix(host[:, _ROT].reshape(action_len, 22, 6))
    action_r = _sixd_to_matrix(fitted[:, _ROT].reshape(action_len, 22, 6))
    composed = base_r.copy()

    if spec.absolute_joints:
        event_pose_joints = set(spec.event_pose_joints)
        for joint in spec.absolute_joints:
            absolute_r = action_r[:, joint]
            if joint == 0:
                # A planted level change may need the authored root tilt so its canonical leg
                # pose still reaches the floor, but it must not reset the dancer's heading.
                action_yaw = _root_yaw_series(fitted)
                host_yaw = _root_yaw_series(host)
                absolute_r = _yaw_rotation(host_yaw - action_yaw) @ absolute_r
            if joint in event_pose_joints:
                absolute_r = np.repeat(
                    action_r[local_event:local_event + 1, joint],
                    action_len,
                    axis=0,
                )
            offset = absolute_r @ np.swapaxes(base_r[:, joint], -1, -2)
            joint_weight = phase_weight if joint in phase_joints else weight
            composed[:, joint] = (
                _fractional_rotation(offset, joint_weight) @ base_r[:, joint]
            )

    if spec.additive_joints:
        reference = action_r[:1]
        for joint in spec.additive_joints:
            joint_weight = phase_weight if joint in phase_joints else weight
            if joint == 0:
                root_weight = joint_weight
                if spec.carry_root_rotation:
                    width = int(max(1, min(blend_frames, local_event)))
                    root_weight = np.ones(action_len, dtype=np.float32)
                    ramp = np.linspace(0.0, 1.0, width + 1, dtype=np.float32)
                    root_weight[:width + 1] = 3.0 * ramp ** 2 - 2.0 * ramp ** 3
                    # A named turn owns yaw. Adding its yaw to every host frame lets an
                    # opposite host turn cancel it, so the requested quarter turn can arrive as
                    # only 62 degrees. Preserve the host's tilt, but drive its world yaw from the
                    # heading at which the action began.
                    action_yaw = _root_yaw_series(fitted)
                    host_yaw = _root_yaw_series(host)
                    yaw_delta = action_yaw - action_yaw[0]
                    desired = host_yaw[0] + root_weight * yaw_delta
                    composed[:, 0] = _yaw_rotation(desired - host_yaw) @ base_r[:, 0]
                    if end < n:
                        suffix = _sixd_to_matrix(out[end:, _ROOT].reshape(-1, 6))
                        carry_yaw = float(desired[-1] - host_yaw[-1])
                        out[end:, _ROOT] = _matrix_to_sixd(
                            _yaw_rotation(carry_yaw) @ suffix
                        ).reshape(-1, 6)
                else:
                    delta = action_r[:, 0] @ np.swapaxes(reference[:, 0], -1, -2)
                    composed[:, 0] = _fractional_rotation(delta, root_weight) @ base_r[:, 0]
            else:
                delta = np.swapaxes(reference[:, joint], -1, -2) @ action_r[:, joint]
                composed[:, joint] = (
                    base_r[:, joint] @ _fractional_rotation(delta, joint_weight)
                )

    composed_6d = _matrix_to_sixd(composed)
    owned_joints = sorted(set(spec.absolute_joints) | set(spec.additive_joints))
    for joint in owned_joints:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        out[start:end, channels] = composed_6d[:, joint]

    if spec.translation_axes:
        canonical = fitted[:, :3] - fitted[:1, :3]
        selected = np.zeros_like(canonical)
        selected[:, list(spec.translation_axes)] = canonical[:, list(spec.translation_axes)]
        if 2 in spec.translation_axes:
            # Vertical root motion and the lower-body pose describe the same level change.
            # Fading the pose while carrying the full root offset makes a rise float and a
            # crouch sink at the hand-back. Use the lower-body phase envelope for both.
            selected[:, 2] *= phase_weight
        selected = _rotate_translation_into_facing(selected, fitted, host[0])
        translated = host[:, :3].copy()

        planar = [axis for axis in spec.translation_axes if axis < 2]
        if len(planar) == 2:
            translated[:, :2] = host[0, :2] + selected[:, :2]
        elif len(planar) == 1:
            action_yaw = float(_root_yaw_series(fitted[:1])[0])
            base_yaw = float(_root_yaw_series(host[:1])[0])
            angle = base_yaw - action_yaw
            canonical_axis = np.zeros(2, dtype=np.float32)
            canonical_axis[planar[0]] = 1.0
            c, s = float(np.cos(angle)), float(np.sin(angle))
            basis = np.array(
                [c * canonical_axis[0] - s * canonical_axis[1],
                 s * canonical_axis[0] + c * canonical_axis[1]],
                dtype=np.float32,
            )
            host_delta = host[:, :2] - host[:1, :2]
            host_owned = (host_delta @ basis)[:, None] * basis[None, :]
            translated[:, :2] = host[:, :2] - host_owned + selected[:, :2]
        if 2 in spec.translation_axes:
            translated[:, 2] = host[:, 2] + selected[:, 2]

        out[start:end, :3] = translated
        if end < n:
            residual = translated[-1] - host[-1, :3]
            if float(np.linalg.norm(residual)) > 1e-6:
                out[end:, :3] = base[end:, :3] + residual

    if spec.replace_contacts:
        out[start:end, _CONTACT] = fitted[:, _CONTACT]

    return np.ascontiguousarray(out, dtype=np.float32), [start, end], actual_event


def _replace_beat_locked(
    base: np.ndarray,
    action: np.ndarray,
    event: int,
    target_event: int,
    action_len: int,
    *,
    blend_frames: int,
) -> tuple[np.ndarray, list[int]]:
    """Place the action at its own speed and leave the rest of the window as the original dance.

    Only the frames the action actually occupies are replaced. Everything around it is the song's
    own choreography, unretimed, so the groove either side of the gesture is the one the dancer
    was already in.

    The root offset an action leaves behind is deliberately *not* settled here. Closing it inside
    the action span means ramping the root back over the very frames the action is happening in,
    and for a turn -- whose whole excursion sits at its final frame -- that cancels the thing
    being spliced: a quarter turn arrives as a 68-degree lean. The dance after the action follows
    the new heading instead, exactly as fixed-duration insertion already does, and the caller
    settles the residual across the whole window, where there is room to give it back gently.
    """
    n = int(base.shape[0])
    ratio = float(np.clip(event / max(1, action.shape[0] - 1), 0.0, 1.0))
    latest = max(0, n - action_len - _join_tail(n))
    start = int(np.clip(target_event - int(round(ratio * (action_len - 1))), 0, latest))
    end = start + action_len

    anchor = max(0, start - 1)
    fitted = _fit_event(action, event, action_len, target_event - start)
    fitted = _align_translation(fitted, base[anchor, :3])
    fitted = _align_yaw(fitted, _root_yaw_series(base[anchor:anchor + 1])[0])

    combined = _ease_join(base[:start], fitted,
                          _join_frames(base[max(0, start - 1)], fitted[0],
                                       min(blend_frames, max(1, action_len // 4)), action_len // 2))
    suffix = base[end:]
    if suffix.shape[0]:
        suffix = _align_yaw(suffix, _root_yaw_series(combined[-1:])[0])
        combined = _ease_join(combined, suffix,
                              _join_frames(combined[-1], suffix[0],
                                           min(blend_frames, max(1, suffix.shape[0] // 3)),
                                           max(1, (2 * suffix.shape[0]) // 3)))
    if combined.shape[0] != n:                       # defensive; the pieces already sum to n
        combined = retime(combined, n)
    return np.ascontiguousarray(combined, dtype=np.float32), [start, end]


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


def _root_yaw_series(clip: np.ndarray) -> np.ndarray:
    root = _sixd_to_matrix(np.asarray(clip[:, _ROOT], dtype=np.float32).reshape(-1, 6))
    return np.unwrap(np.arctan2(root[:, 1, 0], root[:, 0, 0]))


def _yaw_rotate(clip: np.ndarray, angle: np.ndarray | float, pivot: np.ndarray) -> np.ndarray:
    """Turn a clip about the world up axis, per frame, around a fixed ground pivot."""
    n = int(clip.shape[0])
    ang = np.broadcast_to(np.asarray(angle, dtype=np.float32), (n,))
    cos, sin = np.cos(ang), np.sin(ang)
    out = clip.copy()
    spin = _yaw_rotation(ang)
    root = _sixd_to_matrix(np.asarray(out[:, _ROOT], dtype=np.float32).reshape(-1, 6))
    out[:, _ROOT] = _matrix_to_sixd(spin @ root).reshape(n, 6)
    rel = out[:, :3] - np.asarray(pivot, dtype=np.float32)
    out[:, 0] = pivot[0] + cos * rel[:, 0] - sin * rel[:, 1]
    out[:, 1] = pivot[1] + sin * rel[:, 0] + cos * rel[:, 1]
    return out


def _align_yaw(clip: np.ndarray, target_yaw: float) -> np.ndarray:
    """Face a clip the way its predecessor ended, so the join has nothing to snap through."""
    yaw = _root_yaw_series(clip[:1])[0]
    delta = float((target_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)
    if abs(delta) < 1e-4:
        return clip
    return _yaw_rotate(clip, delta, clip[0, :3].copy())


def _closing_frames(yaw_gap: float, trans_gap: np.ndarray, n: int) -> int:
    """Frames needed to give an offset back at a rate a dancer could actually move."""
    seconds = _SMOOTHSTEP_PEAK * max(
        abs(yaw_gap) / _CLOSE_YAW_RATE,
        float(np.linalg.norm(trans_gap[:2])) / _CLOSE_TRANS_RATE,
        abs(float(trans_gap[2])) / _CLOSE_LIFT_RATE,
    )
    return int(np.clip(round(seconds * _FPS), 0, int(_CLOSE_MAX_RATIO * n)))


def _closing_ramp(n: int, tail: int) -> np.ndarray:
    k = np.linspace(0.0, 1.0, tail + 1, dtype=np.float32)
    ramp = np.zeros(n, dtype=np.float32)
    ramp[n - 1 - tail:] = 3.0 * k ** 2 - 2.0 * k ** 3
    return ramp


def _close_root_residual(
    clip: np.ndarray,
    base: np.ndarray,
    *,
    earliest: int = 0,
) -> np.ndarray:
    """Land the root where the surrounding dance expects it, using the window's own tail.

    The splice pins the window's edges back to the song, so a motion that ends somewhere
    other than it started gets yanked back across a handful of blend frames -- a half turn
    snapping 180 degrees in a quarter second. Handing the offset back here instead, spread
    over a danceable tail, leaves the action intact and the seam calm. A turn is continued
    into a full revolution whenever that is no further than reversing it, because a dancer
    finishes a spin rather than rewinding one.
    """
    n = int(clip.shape[0])
    if n < 4:
        return clip
    yaw = _root_yaw_series(clip)
    trans_gap = np.asarray(base[-1, :3], dtype=np.float32) - clip[-1, :3]
    gap = float(_root_yaw_series(base[-1:])[0] - yaw[-1])
    wrapped = (gap + np.pi) % (2.0 * np.pi) - np.pi
    turning = float(yaw[-1] - yaw[max(0, n - 6)])
    yaw_gap = min(
        (wrapped, wrapped + 2.0 * np.pi, wrapped - 2.0 * np.pi),
        key=lambda d: abs(d) * (0.6 if turning and (d > 0) == (turning > 0) else 1.0),
    )

    available = max(0, n - 1 - int(np.clip(earliest, 0, n - 1)))
    tail = min(_closing_frames(yaw_gap, trans_gap, n), available)
    if tail < 2:
        return clip
    ramp = _closing_ramp(n, tail)
    out = clip.copy()
    if abs(yaw_gap) > 1e-4:
        root = _sixd_to_matrix(np.asarray(out[:, _ROOT], dtype=np.float32).reshape(-1, 6))
        out[:, _ROOT] = _matrix_to_sixd(
            _yaw_rotation(ramp * yaw_gap) @ root
        ).reshape(n, 6)
    out[:, :3] += ramp[:, None] * trans_gap[None, :]
    return np.ascontiguousarray(out, dtype=np.float32)


def _join_frames(left_end: np.ndarray, right_start: np.ndarray, requested: int, available: int) -> int:
    """How long to take handing over at a seam, given how far apart the two poses are.

    A fixed blend closes a small gap gently and a large one violently: the same eight frames that
    suit a clap returning to a neutral stance have to fling the body through a much bigger change
    when a bounce hands back mid-cycle, and the rush read as a hitch at twice any speed in the
    song. Scaling the hand-over with the gap keeps the rate roughly constant instead.
    """
    a = _sixd_to_matrix(np.asarray(left_end[_ROT], dtype=np.float32).reshape(1, 22, 6))
    b = _sixd_to_matrix(np.asarray(right_start[_ROT], dtype=np.float32).reshape(1, 22, 6))
    gap = float(np.abs(_matrix_to_axis_angle(a @ np.swapaxes(b, -1, -2))).max())
    # Smoothstep's steepest slope is 1.5/count, so this bounds the rate the gap is closed at.
    need = int(np.ceil(1.5 * gap / _JOIN_MAX_RATE))
    return int(np.clip(max(int(requested), need), 1, max(1, min(int(available), _JOIN_MAX_FRAMES))))


def _ease_join(left: np.ndarray, right: np.ndarray, frames: int) -> np.ndarray:
    if left.shape[0] == 0:
        return right
    if right.shape[0] == 0:
        return left
    out = right.copy()
    # Hand over onto where the outgoing motion was *going*, not onto where it stopped. Landing on
    # `left[-1]` itself makes the first joined frame a pose-for-pose copy of the last one, so a
    # frame of time passes with the dancer perfectly still: measured joint speed was exactly 0.000
    # at every seam, read as a stutter, and the catch-up afterwards overshot the song's own peak.
    step = left[-1] - left[-2] if left.shape[0] > 1 else np.zeros_like(left[-1])
    out[:, :3] += (left[-1, :3] + step[:3]) - out[0, :3]
    count = int(max(0, min(frames, out.shape[0])))
    if count:
        k = np.linspace(0.0, 1.0, count, dtype=np.float32)
        smooth = (3 * k ** 2 - 2 * k ** 3)[:, None]
        # Decay the pose difference while carrying the incoming motion, rather than easing toward
        # a frozen pose. Smoothstep is flat at k=0, so the join leaves at the clip's own speed.
        r_pred = _sixd_to_matrix(left[-1:, _ROT].reshape(1, 22, 6))
        if left.shape[0] > 1:
            r_prev = _sixd_to_matrix(left[-2:-1, _ROT].reshape(1, 22, 6))
            r_pred = (r_pred @ np.swapaxes(r_prev, -1, -2)) @ r_pred
        r_out = _sixd_to_matrix(out[:count, _ROT].reshape(count, 22, 6))
        offset = _matrix_to_axis_angle(r_pred @ np.swapaxes(r_out[:1], -1, -2))
        from agentlodge.dance.transition import _axis_angle_to_matrix, _matrix_to_sixd
        blended = _axis_angle_to_matrix(offset * (1.0 - smooth)[:, None, :]) @ r_out
        out[:count, _ROT] = _matrix_to_sixd(blended).reshape(count, 132)
        held = np.repeat(left[-1:, _CONTACT], count, axis=0)
        out[:count, _CONTACT] = np.where(smooth >= 0.5, out[:count, _CONTACT], held)
    return np.concatenate([left, out], axis=0).astype(np.float32)


def _insert_fixed_duration(
    base: np.ndarray,
    action: np.ndarray,
    event: int,
    target_event: int,
    *,
    beats: np.ndarray | None = None,
    blend_frames: int,
) -> tuple[np.ndarray, list[int]]:
    n = int(base.shape[0])
    if n < 24:
        raise ValueError("selected window is too short for insertion; use replace or select at least 0.8s")
    # The action runs at its authored speed snapped to whole beats, not at a fixed fraction of
    # however wide the selection happens to be, so the inserted gesture stays in time with the song.
    action_len = int(np.clip(_beat_locked_length(action.shape[0], n, beats), 12, max(12, n - 12)))
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
    if prefix.shape[0]:
        fitted_action = _align_yaw(fitted_action, _root_yaw_series(prefix[-1:])[0])
    combined = _ease_join(prefix, fitted_action, min(blend_frames, max(1, action_len // 4)))
    if post_len:
        suffix = _align_yaw(suffix, _root_yaw_series(combined[-1:])[0])
    combined = _ease_join(combined, suffix, min(blend_frames, max(1, post_len // 3))) if post_len else combined
    if combined.shape[0] != n:
        combined = retime(combined, n)
    return combined.astype(np.float32), [start, end]
