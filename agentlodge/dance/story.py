"""Structure-aware ("story") dance assembly.

Generalizes the hybrid assembler: instead of segmenting on raw downbeats and optimizing only local
coherence, it assembles one dance over the storyboard's musical **sections**, choosing per section
which material best realizes the plan (preferred generator + target energy) while staying smooth,
optionally reusing a recurring motif (retimed / mirrored), and joining source changes with the same
training-free inertialized transition used by the hybrid.

Two stages are kept separate so the decision logic is testable without the heavy rotation backend:
  * :func:`select_sources` -- pure-numpy per-section material selection (no torch),
  * :func:`assemble_story` -- concatenation + inertialized blending at source changes (uses torch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from agentlodge.audio.structure import MusicStructure, Section
from agentlodge.agent.storyboard import CommonMotionCue, Storyboard, SectionPlan
from agentlodge.agent.segment_caption import caption_segment, plan_realization_alignment
from agentlodge.dance.format import ensure_lodge139, to_agentlodge139
from agentlodge.dance.transition import amplitude_scale, blend_onto, mirror, retime, retrograde, to_zup
from agentlodge.editor.motion_bank import (
    MotionBank,
    MotionSpec,
    default_motion_bank,
    minimum_window_frames,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agentlodge.audio.preprocess import SongMetadata

logger = logging.getLogger(__name__)

_KIN = 135  # translation(3) + rotation(132); excludes the 4 contact labels
DEFAULT_EXPRESSIVENESS = 4.0

# Per-section selection weights. The storyboard encodes the musical intent (form alignment, energy
# arc, sectional contrast), so plan adherence (arc + generator bias) drives selection; local
# coherence is a lighter tie-breaker (seam smoothness is separately handled by inertialization).
_W_COH = 0.4     # weight on the (bounded, relative) local-coherence penalty
_W_ARC = 0.6     # penalty for missing the plan's target energy
_W_BIAS = 0.5    # bonus for matching the plan's preferred generator
_W_REUSE = 0.3   # bonus for reusing a motif the plan asked for
_W_RECAP = 0.6   # decisive bonus for an enabled recapitulation (ABA) close

# Reusing an entire long phrase inside a much shorter section used to create extreme playback
# speeds (for example, 32 seconds compressed into 12 seconds). Storyboard ``retime`` is an output
# duration / source duration ratio. Reused motifs may receive a modest tempo lift, but shorter
# sections crop a coherent excerpt rather than squeezing the complete phrase.
MAX_REUSE_PLAYBACK_SPEED = 1.2
MIN_REUSE_TIME_SCALE = 1.0 / MAX_REUSE_PLAYBACK_SPEED
MAX_REUSE_TIME_SCALE = 1.25


@dataclass
class StoryResult:
    motion: np.ndarray                                   # (L, 139) Z-up assembled dance
    schedule: list = field(default_factory=list)         # [(a, b, source, role), ...]
    storyboard: Storyboard | None = None
    structure: MusicStructure | None = None
    reasoning: str = ""
    section_scores: list = field(default_factory=list)   # per-section chosen source + costs

    def schedule_summary(self) -> str:
        from agentlodge.config import FPS
        return ", ".join(f"{a / FPS:.1f}-{b / FPS:.1f}s:{src}[{role}]"
                         for a, b, src, role in self.schedule)


# --------------------------------------------------------------------------- scoring helpers
def _energy_mean(clip: np.ndarray) -> float:
    if clip.shape[0] < 2:
        return 0.0
    rv = np.linalg.norm(np.diff(clip[:, :3], axis=0, prepend=clip[:1, :3]), axis=1)
    jm = np.linalg.norm(np.diff(clip[:, :_KIN], axis=0, prepend=clip[:1, :_KIN]), axis=1)
    return float(np.mean(0.6 * rv + 0.4 * jm))


def _coh_badness(clip: np.ndarray) -> float:
    """Standalone local coherence penalty (jerk + foot skating); lower = smoother."""
    if clip.shape[0] < 6:
        return 0.0
    kin = clip[:, :_KIN]
    jerk = float(np.mean(np.linalg.norm(np.diff(kin, n=3, axis=0), axis=1)))
    contact = clip[:, _KIN:139].mean(axis=1)[1:]
    horiz = np.linalg.norm(np.diff(clip[:, [0, 1]], axis=0), axis=1)
    foot = float(np.mean(horiz * contact))
    return jerk + foot


def _coh_penalty(values: dict) -> dict:
    """Bounded RELATIVE coherence penalty in [0, 1].

    ``0`` for the smoothest candidate; others penalized by their fractional excess badness over
    the best, clipped at 1.0. Unlike min-max normalization this reflects the *actual* magnitude of
    the difference, so near-equal candidates get near-equal penalties (letting the storyboard's
    arc/bias decide) while a genuinely much rougher candidate is still penalized.
    """
    vals = {k: float(v) for k, v in values.items()}
    bmin = min(vals.values())
    return {k: float(np.clip((v - bmin) / (bmin + 1e-6), 0.0, 1.0)) for k, v in vals.items()}


def _bias_bonus(source: str, plan: SectionPlan) -> float:
    bonus = 0.0
    gen = "reuse" if source.startswith("reuse") else source
    if plan.generator_bias in {"lodge", "edge"} and gen == plan.generator_bias:
        bonus -= _W_BIAS
    if source.startswith("reuse") and plan.reuse_of is not None:
        bonus -= _W_REUSE
    return bonus


def _fit_reuse_clip(source: np.ndarray, target_frames: int,
                    requested_time_scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """Fit reusable material to a section without extreme temporal compression.

    ``requested_time_scale`` is output duration divided by selected source duration. A value of
    ``1.0`` preserves the source tempo, values below one speed it up, and values above one slow it
    down. When the source phrase is longer than needed, a centered excerpt is selected before the
    bounded retime so the section stays coherent instead of squeezing the entire phrase.
    """
    clip = np.ascontiguousarray(source, dtype=np.float32)
    source_frames = int(clip.shape[0])
    target = int(target_frames)
    if source_frames < 2:
        raise ValueError(f"reuse source must contain at least 2 frames, got {source_frames}")
    if target < 2:
        raise ValueError(f"reuse target must contain at least 2 frames, got {target}")

    try:
        requested = float(requested_time_scale)
    except (TypeError, ValueError):
        requested = 1.0
    if not np.isfinite(requested) or requested <= 0.0:
        requested = 1.0
    bounded = float(np.clip(requested, MIN_REUSE_TIME_SCALE, MAX_REUSE_TIME_SCALE))

    desired_source_frames = target / bounded
    selected_frames = int(
        np.floor(desired_source_frames)
        if bounded <= 1.0
        else np.ceil(desired_source_frames)
    )
    selected_frames = int(np.clip(selected_frames, 2, source_frames))
    source_start = max(0, (source_frames - selected_frames) // 2)
    source_end = source_start + selected_frames
    selected = clip[source_start:source_end]
    fitted = selected.copy() if selected_frames == target else retime(selected, target)
    actual_time_scale = target / selected_frames

    return fitted, {
        "source_frames": source_frames,
        "source_start": source_start,
        "source_end": source_end,
        "selected_frames": selected_frames,
        "target_frames": target,
        "requested_time_scale": round(requested, 4),
        "bounded_time_scale": round(bounded, 4),
        "actual_time_scale": round(actual_time_scale, 4),
        "playback_speed": round(1.0 / actual_time_scale, 4),
        "cropped": selected_frames < source_frames,
        "capped": abs(bounded - requested) > 1e-6,
        "source_limited": selected_frames == source_frames
        and target / source_frames > MAX_REUSE_TIME_SCALE,
    }


def _remap_reuse_cues(cues: list[CommonMotionCue], fit: dict,
                      *, mirrored: bool = False,
                      retrograded: bool = False) -> list[CommonMotionCue]:
    """Keep only cues inside a selected reuse excerpt and remap their relative positions."""
    source_frames = int(fit["source_frames"])
    source_start = int(fit["source_start"])
    selected_frames = int(fit["selected_frames"])
    source_end = source_start + selected_frames
    remapped: list[CommonMotionCue] = []
    for cue in cues:
        source_position = float(cue.position) * max(1, source_frames - 1)
        if source_position < source_start or source_position > source_end - 1:
            continue
        position = (source_position - source_start) / max(1, selected_frames - 1)
        if retrograded:
            position = 1.0 - position
        direction = cue.direction
        if mirrored and direction in {"left", "right"}:
            direction = "right" if direction == "left" else "left"
        remapped.append(replace(
            cue,
            position=float(np.clip(position, 0.0, 1.0)),
            direction=direction,
            mirror=not cue.mirror if mirrored else cue.mirror,
        ))
    return remapped


def _motion_window(spec: MotionSpec, repeats: int, n_frames: int,
                   position: float) -> tuple[int, int]:
    minimum = minimum_window_frames(spec.minimum_frames * repeats)
    preferred = spec.frames * repeats + 16
    width = min(n_frames, max(minimum, preferred))
    center = int(round(float(np.clip(position, 0.0, 1.0)) * max(0, n_frames - 1)))
    start = int(np.clip(center - width // 2, 0, max(0, n_frames - width)))
    return start, start + width


def _overlaps(interval: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    a, b = interval
    return any(a < y and x < b for x, y in occupied)


def _cue_key(cue: CommonMotionCue) -> tuple[str, str, int]:
    return cue.motion_id, cue.motif, int(round(cue.position * 1000))


def _merge_cues(base: list[CommonMotionCue],
                extra: list[CommonMotionCue]) -> list[CommonMotionCue]:
    merged = list(base)
    seen = {_cue_key(cue) for cue in merged}
    for cue in extra:
        if _cue_key(cue) not in seen:
            merged.append(cue)
            seen.add(_cue_key(cue))
    return merged


def apply_planned_common_motions(
    clip: np.ndarray,
    cues: list[CommonMotionCue],
    *,
    section_start: int = 0,
    music_beat_frames: np.ndarray | None = None,
    motion_bank: MotionBank | None = None,
    occupied_cues: list[CommonMotionCue] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Compose validated storyboard cues into a section without changing its duration."""
    out = np.ascontiguousarray(clip, dtype=np.float32).copy()
    if not cues:
        return out, []
    bank = motion_bank or default_motion_bank()
    beats = (
        np.asarray(music_beat_frames, dtype=np.int64).reshape(-1)
        if music_beat_frames is not None
        else np.zeros(0, dtype=np.int64)
    )
    occupied: list[tuple[int, int]] = []
    for cue in occupied_cues or []:
        try:
            spec = bank.resolve(cue.motion_id)
        except KeyError:
            continue
        occupied.append(_motion_window(spec, cue.repeats, out.shape[0], cue.position))
    reports: list[dict] = []

    for cue in sorted(cues, key=lambda item: item.position):
        try:
            spec = bank.resolve(cue.motion_id)
            minimum = minimum_window_frames(spec.minimum_frames * cue.repeats)
            if out.shape[0] < minimum:
                raise ValueError(
                    f"section has {out.shape[0]} frames; {spec.name} needs at least {minimum}"
                )
            start, end = _motion_window(spec, cue.repeats, out.shape[0], cue.position)
            if _overlaps((start, end), occupied):
                reports.append({
                    **cue.to_dict(),
                    "name": spec.name,
                    "status": "skipped",
                    "detail": "overlaps an earlier or inherited common motion",
                })
                continue
            global_start = int(section_start + start)
            local_beats = beats[
                (beats >= global_start) & (beats < section_start + end)
            ] - global_start

            def apply_once(intensity: float, *, use_beats: bool) -> tuple[np.ndarray, dict]:
                return bank.apply(
                    out[start:end],
                    cue.motion_id,
                    beats=local_beats if use_beats and local_beats.size else None,
                    mode="replace",
                    anchor=cue.anchor,
                    mirror=cue.mirror,
                    direction=cue.direction,
                    intensity=intensity,
                    repeats=cue.repeats,
                    blend_frames=8,
                )

            applied_intensity = cue.intensity
            beat_lock_relaxed = False
            try:
                edited, raw_report = apply_once(applied_intensity, use_beats=True)
            except ValueError as exc:
                if "failed semantic validation" not in str(exc):
                    raise
                if applied_intensity < 1.0:
                    applied_intensity = 1.0
                    try:
                        edited, raw_report = apply_once(applied_intensity, use_beats=True)
                    except ValueError as retry_exc:
                        if "failed semantic validation" not in str(retry_exc):
                            raise
                        beat_lock_relaxed = True
                        edited, raw_report = apply_once(applied_intensity, use_beats=False)
                else:
                    beat_lock_relaxed = True
                    edited, raw_report = apply_once(applied_intensity, use_beats=False)
            action_start, action_end = (int(value) for value in raw_report["action_range"])
            section_range = (start + action_start, start + action_end)
            if _overlaps(section_range, occupied):
                reports.append({
                    **cue.to_dict(),
                    "name": spec.name,
                    "status": "skipped",
                    "detail": "overlaps an earlier planned common motion",
                })
                continue
            out[start:end] = edited
            occupied.append(section_range)
            reports.append({
                **raw_report,
                "status": "applied",
                "position": round(cue.position, 3),
                "planned_intensity": round(cue.intensity, 3),
                "intensity_adjusted": applied_intensity > cue.intensity + 1e-6,
                "beat_lock_relaxed": beat_lock_relaxed,
                "motif": cue.motif,
                "rationale": cue.rationale,
                "section_action_range": list(section_range),
                "global_action_range": [
                    int(section_start + section_range[0]),
                    int(section_start + section_range[1]),
                ],
            })
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning(
                "Could not realize common motion %s in section starting at frame %d: %s",
                cue.motion_id, section_start, exc,
            )
            reports.append({
                **cue.to_dict(),
                "status": "skipped",
                "detail": str(exc),
            })
    return np.ascontiguousarray(out, dtype=np.float32), reports


# --------------------------------------------------------------------------- section selection
def _clip_sections(structure: MusicStructure, n: int) -> list[Section]:
    """Clip/trim sections to the available frame count ``n`` (drop empties)."""
    from agentlodge.config import FPS

    out: list[Section] = []
    for s in structure.sections:
        a, b = int(s.start_frame), min(int(s.end_frame), n)
        if b - a < 2:
            continue
        out.append(Section(start_frame=a, end_frame=b, start_sec=a / FPS, end_sec=b / FPS,
                           label=s.label, role=s.role, energy=s.energy))
    if out:
        last = out[-1]
        out[-1] = Section(last.start_frame, n, last.start_frame / FPS, n / FPS,
                          last.label, last.role, last.energy)
    return out


def select_sources(lodge_z: np.ndarray, edge_z: np.ndarray, structure: MusicStructure,
                   storyboard: Storyboard, *, motif_reuse: bool = True,
                   energy_shaping: bool = False, recapitulate: bool = False,
                   post_variations: dict | None = None,
                   music_beat_frames: np.ndarray | None = None,
                   motion_bank: MotionBank | None = None) -> list[dict]:
    """Choose, per section, the material realizing the storyboard (pure numpy, no blending).

    Returns an ordered list of dicts: ``{a, b, source, role, clip, costs}``. ``source`` is
    ``"lodge"``, ``"edge"`` or ``"reuse:<i>"``. ``clip`` is the raw (pre-blend) chosen slice.
    When ``recapitulate`` is set, the final section also gets a mirrored+retrograded reuse of the
    OPENING section as a strong candidate, imposing an ABA ("reuse the intro, mirror it at the
    end") close even when the music does not strictly repeat. ``post_variations`` maps a section
    index to length-preserving edits ``{mirror, retrograde, amplitude}`` applied to that section's
    chosen clip (used by the editing agent). Planned common motions are composed after source
    selection and cached in ``chosen_raw`` so reused sections inherit them exactly once.
    """
    n = min(lodge_z.shape[0], edge_z.shape[0])
    sections = _clip_sections(structure, n)
    plans_by_idx = {p.section_index: p for p in storyboard.plans}
    chosen_raw: dict[int, np.ndarray] = {}   # section_index -> selected raw clip (for reuse)
    effective_cues: dict[int, list[CommonMotionCue]] = {}
    decisions: list[dict] = []

    for i, sec in enumerate(sections):
        a, b = sec.start_frame, sec.end_frame
        plan = plans_by_idx.get(i, SectionPlan(section_index=i, role=sec.role,
                                               target_intensity=sec.energy,
                                               vocabulary="", generator_bias="auto"))
        cands: dict[str, np.ndarray] = {"lodge": lodge_z[a:b], "edge": edge_z[a:b]}
        reuse_fits: dict[str, dict] = {}

        if (motif_reuse and plan.reuse_of is not None
                and plan.reuse_of in chosen_raw):
            reuse_clip, reuse_fit = _fit_reuse_clip(
                chosen_raw[plan.reuse_of],
                b - a,
                plan.variation.get("retime", 1.0),
            )
            mirrored = bool(plan.variation.get("mirror"))
            retrograded = bool(plan.variation.get("retrograde"))
            if mirrored:
                reuse_clip = mirror(reuse_clip)
            if retrograded:
                reuse_clip = retrograde(reuse_clip)
            if energy_shaping and abs(float(plan.variation.get("amplitude", 1.0)) - 1.0) > 1e-3:
                reuse_clip = amplitude_scale(reuse_clip, float(plan.variation["amplitude"]))
            reuse_key = f"reuse:{plan.reuse_of}"
            cands[reuse_key] = reuse_clip
            reuse_fits[reuse_key] = {
                **reuse_fit,
                "mirror": mirrored,
                "retrograde": retrograded,
            }

        # Recapitulation (ABA close): reuse the opening section at the last section, mirrored +
        # retrograded, even when the music doesn't strictly repeat. Given a decisive bonus below.
        recap_key: str | None = None
        if recapitulate and i == len(sections) - 1 and i > 0 and 0 in chosen_raw:
            recap_key = "reuse:0"
            if recap_key not in cands:
                recap_clip, recap_fit = _fit_reuse_clip(
                    chosen_raw[0],
                    b - a,
                    (b - a) / max(1, len(chosen_raw[0])),
                )
                cands[recap_key] = retrograde(mirror(recap_clip))
                reuse_fits[recap_key] = {
                    **recap_fit,
                    "mirror": True,
                    "retrograde": True,
                }

        # energy match: normalize candidate energies to [0,1] within the section.
        energies = {k: _energy_mean(v) for k, v in cands.items()}
        emin, emax = min(energies.values()), max(energies.values())
        erel = {k: (0.0 if emax <= emin else (v - emin) / (emax - emin))
                for k, v in energies.items()}
        coh_pen = _coh_penalty({k: _coh_badness(v) for k, v in cands.items()})

        costs = {}
        for k in cands:
            arc_pen = abs(erel[k] - float(plan.target_intensity))
            costs[k] = _W_COH * coh_pen[k] + _W_ARC * arc_pen + _bias_bonus(k, plan)
        if recap_key is not None:
            costs[recap_key] -= _W_RECAP
        source = min(costs, key=costs.get)

        gen = "reuse" if source.startswith("reuse") else source
        matched_bias = plan.generator_bias in {"lodge", "edge"} and gen == plan.generator_bias
        inherited_common_motion_ids: list[str] = []
        recalled_common_motion_ids: list[str] = []
        inherited_cues: list[CommonMotionCue] = []
        cues_to_apply = list(plan.common_motions)
        if source.startswith("reuse:"):
            reused_idx = int(source.split(":", 1)[1])
            inherited_cues = _remap_reuse_cues(
                effective_cues.get(reused_idx, []),
                reuse_fits[source],
                mirrored=bool(reuse_fits[source].get("mirror")),
                retrograded=bool(reuse_fits[source].get("retrograde")),
            )
            inherited_common_motion_ids = list(dict.fromkeys(
                cue.motion_id for cue in inherited_cues
            ))
        elif plan.reuse_of is not None and plan.reuse_of in effective_cues:
            # The source selector may prefer fresh LODGE/EDGE material despite a structural reuse
            # plan. Reapply the earlier named-motion motif so the chorus/hook identity still recurs.
            recalled_cues = list(effective_cues[plan.reuse_of])
            recalled_common_motion_ids = list(dict.fromkeys(
                cue.motion_id for cue in recalled_cues
            ))
            cues_to_apply = _merge_cues(recalled_cues, cues_to_apply)

        out_clip, common_motion_reports = apply_planned_common_motions(
            cands[source],
            cues_to_apply,
            section_start=a,
            music_beat_frames=music_beat_frames,
            motion_bank=motion_bank,
            occupied_cues=inherited_cues,
        )
        applied_keys = {
            (
                str(report["id"]),
                str(report.get("motif", "")),
                int(round(float(report.get("position", 0.5)) * 1000)),
            )
            for report in common_motion_reports
            if report.get("status") == "applied"
        }
        applied_cues = [cue for cue in cues_to_apply if _cue_key(cue) in applied_keys]
        effective_cues[i] = _merge_cues(inherited_cues, applied_cues)
        common_motion_ids = list(dict.fromkeys(
            cue.motion_id for cue in effective_cues[i]
        ))
        chosen_raw[i] = out_clip

        # Length-preserving per-section edits from the editing agent are applied to the output
        # only, leaving chosen_raw available as the stable motif for later reuse.
        pv = (post_variations or {}).get(i) or {}
        applied_pv: list[str] = []
        if pv.get("mirror"):
            out_clip = mirror(out_clip)
            applied_pv.append("mirror")
        if pv.get("retrograde"):
            out_clip = retrograde(out_clip)
            applied_pv.append("retrograde")
        if pv.get("amplitude") and abs(float(pv["amplitude"]) - 1.0) > 1e-3:
            out_clip = amplitude_scale(out_clip, float(pv["amplitude"]))
            applied_pv.append(f"amp={pv['amplitude']}")

        decisions.append({
            "a": a, "b": b, "source": source, "role": sec.role,
            "clip": out_clip,
            "costs": {k: round(v, 4) for k, v in costs.items()},
            "target_intensity": float(plan.target_intensity),
            "plan_bias": plan.generator_bias,
            "matched_bias": bool(matched_bias),
            "vocabulary": plan.vocabulary,
            "energies": {k: round(float(energies[k]), 4) for k in cands},
            "chosen_cost": round(float(costs[source]), 4),
            "caption": caption_segment(out_clip, energy_norm=erel[source]),
            "plan_alignment": round(plan_realization_alignment(plan, erel[source]), 4),
            "post_variation": applied_pv,
            "planned_common_motions": [cue.to_dict() for cue in plan.common_motions],
            "effective_common_motions": [cue.to_dict() for cue in effective_cues[i]],
            "common_motions": common_motion_reports,
            "common_motion_ids": common_motion_ids,
            "inherited_common_motion_ids": inherited_common_motion_ids,
            "recalled_common_motion_ids": recalled_common_motion_ids,
            "reuse_fit": reuse_fits.get(source),
        })
    return decisions


# --------------------------------------------------------------------------- assembly
def _continuous(prev: dict | None, cur: dict) -> bool:
    """True if ``cur`` continues ``prev`` with no discontinuity (same generator, contiguous)."""
    if prev is None:
        return False
    if prev["source"].startswith("reuse") or cur["source"].startswith("reuse"):
        return False
    return prev["source"] == cur["source"] and prev["b"] == cur["a"]


def assemble_story(decisions: list[dict], *, blend_frames: int = 15) -> np.ndarray:
    """Concatenate chosen section clips, inertially blending at each source discontinuity."""
    committed: np.ndarray | None = None
    prev: dict | None = None
    for cur in decisions:
        seg = cur["clip"]
        if seg.shape[0] == 0:
            continue
        if committed is None:
            committed = seg.copy()
        elif _continuous(prev, cur):
            committed = np.concatenate([committed, seg], axis=0)
        else:
            blended = blend_onto(committed[-2:], seg, blend_frames,
                                 canonical_yaw=None, align_facing=False)
            committed = np.concatenate([committed, blended], axis=0)
        prev = cur
    return committed if committed is not None else np.zeros((0, 139), dtype=np.float32)


def build_story_dance(lodge_motion: np.ndarray, edge_motion: np.ndarray,
                      structure: MusicStructure, storyboard: Storyboard,
                      metadata: "SongMetadata", *, blend_frames: int = 15,
                      motif_reuse: bool = True, energy_shaping: bool = False,
                      recapitulate: bool = False, post_variations: dict | None = None) -> StoryResult:
    """Assemble a structure-aware dance from independent LODGE and EDGE motions + a storyboard."""
    lodge = to_zup(to_agentlodge139(ensure_lodge139(lodge_motion)))  # native -> AgentLODGE, Y->Z up
    edge = to_agentlodge139(ensure_lodge139(edge_motion))            # EDGE already Z-up
    n = min(lodge.shape[0], edge.shape[0], structure.total_frames)
    if n < 30:
        raise ValueError(f"Motions too short for story assembly ({n} frames)")
    lodge, edge = lodge[:n], edge[:n]

    decisions = select_sources(lodge, edge, structure, storyboard,
                               motif_reuse=motif_reuse, energy_shaping=energy_shaping,
                               recapitulate=recapitulate, post_variations=post_variations,
                               music_beat_frames=getattr(metadata, "beat_frames", None))
    if not decisions:
        raise ValueError("no usable sections for story assembly")

    motion = assemble_story(decisions, blend_frames=blend_frames)
    schedule = [(d["a"], d["b"], d["source"], d["role"]) for d in decisions]
    _score_keys = ("a", "b", "source", "role", "costs", "target_intensity",
                   "plan_bias", "matched_bias", "vocabulary", "energies", "chosen_cost",
                   "caption", "plan_alignment", "post_variation", "planned_common_motions",
                   "effective_common_motions", "common_motions", "common_motion_ids",
                   "inherited_common_motion_ids", "recalled_common_motion_ids", "reuse_fit")
    section_scores = [{k: d[k] for k in _score_keys} for d in decisions]
    n_reuse = sum(1 for d in decisions if d["source"].startswith("reuse"))
    n_lodge = sum(1 for d in decisions if d["source"] == "lodge")
    n_edge = sum(1 for d in decisions if d["source"] == "edge")
    n_honored = sum(1 for d in decisions if d["matched_bias"])
    n_biased = sum(1 for d in decisions if d["plan_bias"] in {"lodge", "edge"})
    n_common_applied = sum(
        1
        for d in decisions
        for report in d["common_motions"]
        if report.get("status") == "applied"
    )
    n_common_sections = sum(1 for d in decisions if d["common_motion_ids"])

    from agentlodge.config import FPS
    arc = storyboard.arc if storyboard is not None else "?"
    logger.info("Story: realizing %d-section plan (arc: %s)", len(decisions), arc)
    for d in decisions:
        a, b = d["a"], d["b"]
        mark = "=bias" if d["matched_bias"] else ("~auto" if d["plan_bias"] == "auto" else "!=bias")
        cost_str = " ".join(f"{k}:{v:.3f}" for k, v in d["costs"].items())
        logger.info(
            "  %5.1f-%5.1fs %-8s -> %-8s (%s) plan[bias=%-5s tgtE=%.2f] "
            "motions=%s chose_cost=%.3f | costs %s",
            a / FPS, b / FPS, d["role"], d["source"], mark,
            d["plan_bias"], d["target_intensity"],
            ",".join(d["common_motion_ids"]) or "-", d["chosen_cost"], cost_str,
        )

    schedule_summary = ", ".join(
        f"{a / FPS:.1f}-{b / FPS:.1f}s:{src}[{role}]" for a, b, src, role in schedule
    )
    reasoning = (
        f"story assembly: {len(decisions)} sections "
        f"({n_lodge} LODGE, {n_edge} EDGE, {n_reuse} motif-reuse); "
        f"storyboard bias honored in {n_honored}/{n_biased} explicitly-biased sections; "
        f"{n_common_applied} common-motion cues applied across {n_common_sections} sections; "
        f"schedule: {schedule_summary}"
    )
    logger.info("Story schedule: %s", schedule_summary)
    return StoryResult(motion=motion, schedule=schedule, storyboard=storyboard,
                       structure=structure, reasoning=reasoning, section_scores=section_scores)
