"""LLM choreography storyboard agent.

Given a song's detected musical form (see ``agentlodge.audio.structure``), this agent authors a
high-level *choreographic plan* -- one :class:`SectionPlan` per musical section -- describing how
the dance should be composed across the whole song: an overall energy/narrative arc, a movement
vocabulary and preferred generator per section, sparse cues from the curated common-motion bank,
and (optionally) which sections should reuse a recurring motif. The structure-aware assembler
(``agentlodge.dance.story``) then realizes this plan training-free by arranging/retiming LODGE and
EDGE material, composing common motions, and joining sections with inertialized transitions.

This mirrors ``agentlodge.agent.selector``: an OpenAI chat model produces a strict-JSON plan, and
a deterministic rule-based fallback is used when no API key is configured or the call fails -- so
the feature is fully functional offline.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from agentlodge.audio.structure import MusicStructure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agentlodge.audio.preprocess import AudioDescriptor, SongMetadata

logger = logging.getLogger(__name__)

VOCABULARY = (
    "grounded_minimal",     # low energy, small/still
    "sustained_lyrical",    # slow, flowing, expressive
    "flowing_smooth",       # mid energy, continuous
    "expansive_traveling",  # mid-high, covering space
    "percussive_sharp",     # high, accented/staccato
    "explosive_fast",       # peak, big dynamic movement
)
GENERATORS = ("lodge", "edge", "auto")
COMMON_MOTION_ANCHORS = ("start", "early", "center", "beat", "late", "end")
MAX_COMMON_MOTIONS_PER_SECTION = 3


def _bounded_float(value: object, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return float(np.clip(parsed, low, high))


def _bounded_int(value: object, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return int(np.clip(parsed, low, high))


@dataclass(frozen=True)
class CommonMotionCue:
    """One validated, duration-preserving named-motion cue within a section."""

    motion_id: str
    position: float = 0.5
    anchor: str = "beat"
    intensity: float = 0.65
    direction: str | None = "auto"
    mirror: bool = False
    repeats: int = 1
    motif: str = ""
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "motion_id": self.motion_id,
            "position": round(self.position, 3),
            "anchor": self.anchor,
            "intensity": round(self.intensity, 3),
            "direction": self.direction,
            "mirror": self.mirror,
            "repeats": self.repeats,
            "motif": self.motif,
            "rationale": self.rationale,
        }


@dataclass
class SectionPlan:
    section_index: int
    role: str
    target_intensity: float          # [0, 1]
    vocabulary: str
    generator_bias: str              # lodge | edge | auto
    reuse_of: int | None = None      # earlier same-label section index, or None
    variation: dict = field(default_factory=lambda: {"mirror": False, "retime": 1.0, "amplitude": 1.0})
    common_motions: list[CommonMotionCue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "section_index": self.section_index,
            "role": self.role,
            "target_intensity": round(self.target_intensity, 3),
            "vocabulary": self.vocabulary,
            "generator_bias": self.generator_bias,
            "reuse_of": self.reuse_of,
            "variation": self.variation,
            "common_motions": [cue.to_dict() for cue in self.common_motions],
        }


@dataclass
class Storyboard:
    arc: str
    plans: list[SectionPlan]
    reasoning: str = ""
    used_fallback: bool = False

    def to_dict(self) -> dict:
        return {
            "arc": self.arc,
            "reasoning": self.reasoning,
            "used_fallback": self.used_fallback,
            "plans": [p.to_dict() for p in self.plans],
        }

    def describe(self) -> str:
        """Human-readable multi-line summary of the plan (for INFO logging / inspection)."""
        source = "rule-based fallback" if self.used_fallback else "LLM"
        lines = [
            f"arc      : {self.arc}",
            f"source   : {source}",
            f"reasoning: {self.reasoning or '(none)'}",
            "sections :",
        ]
        for p in self.plans:
            reuse = f" reuse<-{p.reuse_of}" if p.reuse_of is not None else ""
            var = ""
            if p.reuse_of is not None:
                v = p.variation
                flags = []
                if v.get("mirror"):
                    flags.append("mirror")
                if v.get("retrograde"):
                    flags.append("retrograde")
                if abs(float(v.get("retime", 1.0)) - 1.0) > 1e-3:
                    flags.append(f"retime={v['retime']}")
                if abs(float(v.get("amplitude", 1.0)) - 1.0) > 1e-3:
                    flags.append(f"amp={v['amplitude']}")
                if flags:
                    var = " [" + ",".join(flags) + "]"
            motions = ""
            if p.common_motions:
                cues = ", ".join(
                    f"{cue.motion_id}@{cue.position:.2f}"
                    + (f":{cue.motif}" if cue.motif else "")
                    for cue in p.common_motions
                )
                motions = f" motions=[{cues}]"
            elif p.reuse_of is not None:
                motions = " motions=[inherited]"
            lines.append(
                f"  [{p.section_index}] {p.role:<8} intensity={p.target_intensity:.2f} "
                f"bias={p.generator_bias:<5} vocab={p.vocabulary}{reuse}{var}{motions}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- deterministic fallback
def _vocab_for_energy(e: float) -> str:
    if e >= 0.78:
        return "explosive_fast"
    if e >= 0.58:
        return "percussive_sharp"
    if e >= 0.42:
        return "expansive_traveling"
    if e >= 0.25:
        return "flowing_smooth"
    return "grounded_minimal"


def _first_same_label(structure: MusicStructure, i: int) -> int | None:
    """Earliest earlier section with the same repetition label as section ``i``."""
    label = structure.sections[i].label
    for j in range(i):
        if structure.sections[j].label == label:
            return j
    return None


_ROLE_MOTION_PALETTES = {
    "intro": ("rise_reach", "bounce_in_place", "wave"),
    "verse": ("step_touch", "side_step", "step_forward", "point_side", "body_roll"),
    "chorus": ("clap_repeat", "jump_arms_up", "side_step", "turn_quarter"),
    "bridge": ("body_roll", "turn_quarter", "crouch_drop", "rise_reach", "wave"),
    "drop": ("jump_two_foot", "jump_arms_up", "arm_punch", "chest_pop", "turn_half"),
    "outro": ("turn_half", "crouch_drop", "clap_single", "rise_reach", "step_backward"),
}
_ROLE_CUE_POSITIONS = {
    "intro": 0.68,
    "verse": 0.48,
    "chorus": 0.50,
    "bridge": 0.52,
    "drop": 0.46,
    "outro": 0.66,
}


def _fallback_common_motions(structure: MusicStructure, idx: int,
                             reuse_of: int | None) -> list[CommonMotionCue]:
    """Choose one sparse, role-aware signature cue for a new section motif."""
    if reuse_of is not None:
        return []

    from agentlodge.editor.motion_bank import default_motion_bank, minimum_window_frames

    section = structure.sections[idx]
    bank = default_motion_bank()
    palette = _ROLE_MOTION_PALETTES.get(section.role, _ROLE_MOTION_PALETTES["verse"])
    offset = (int(section.label) + idx) % len(palette)
    candidates = palette[offset:] + palette[:offset]
    spec = None
    for motion_id in candidates:
        candidate = bank.resolve(motion_id)
        if section.n_frames >= minimum_window_frames(candidate.minimum_frames):
            spec = candidate
            break
    if spec is None:
        return []

    intensity = float(np.clip(0.42 + 0.50 * float(section.energy), 0.35, 0.95))
    motif = f"label_{section.label}_{spec.id}"
    rationale = (
        f"{section.role} signature selected for section label {section.label}; "
        "repeat-label returns inherit the same composed motif"
    )
    return [CommonMotionCue(
        motion_id=spec.id,
        position=_ROLE_CUE_POSITIONS.get(section.role, 0.5),
        anchor=spec.default_anchor,
        intensity=intensity,
        direction="auto" if spec.directions else None,
        motif=motif,
        rationale=rationale,
    )]


def _rule_based_storyboard(structure: MusicStructure, *, motif_reuse: bool) -> Storyboard:
    """Derive a storyboard directly from the numeric structure (no LLM)."""
    plans: list[SectionPlan] = []
    sections = structure.sections
    for i, sec in enumerate(sections):
        e = float(sec.energy)
        bias = "edge" if e >= 0.55 else "lodge"
        reuse = _first_same_label(structure, i) if motif_reuse else None
        variation = {"mirror": False, "retime": 1.0, "amplitude": 1.0}
        if reuse is not None:
            src = sections[reuse]
            # even/odd recurrence -> mirror alternate occurrences for variety
            variation = {
                "mirror": (i - reuse) % 2 == 1,
                "retime": round(sec.n_frames / max(src.n_frames, 1), 4),
                "amplitude": round(float(np.clip((e + 1e-3) / (src.energy + 1e-3), 0.7, 1.4)), 3),
            }
        common_motions = _fallback_common_motions(structure, i, reuse)
        plans.append(SectionPlan(
            section_index=i, role=sec.role, target_intensity=e,
            vocabulary=_vocab_for_energy(e), generator_bias=bias,
            reuse_of=reuse, variation=variation, common_motions=common_motions,
        ))
    arc = _describe_arc(structure)
    return Storyboard(arc=arc, plans=plans,
                      reasoning="rule-based storyboard from detected structure", used_fallback=True)


def _describe_arc(structure: MusicStructure) -> str:
    roles = [s.role for s in structure.sections]
    peak = structure.climax_index
    return (f"{len(roles)} sections building toward the {roles[peak] if roles else 'peak'} "
            f"at section {peak}, then resolving: " + " -> ".join(roles))


# --------------------------------------------------------------------------- LLM path
def _build_prompt(structure: MusicStructure, metadata: "SongMetadata",
                  descriptor: "AudioDescriptor | None") -> str:
    from agentlodge.editor.motion_bank import default_motion_bank

    lines = []
    for i, s in enumerate(structure.sections):
        same = _first_same_label(structure, i)
        lines.append(
            f"  [{i}] role~{s.role}  {s.start_sec:.1f}-{s.end_sec:.1f}s  "
            f"energy={s.energy:.2f}  repeat_label={s.label}"
            + (f"  (repeats section {same})" if same is not None else "")
        )
    desc = ""
    if descriptor is not None:
        desc = (f"\nAcoustic feel: tempo={getattr(descriptor, 'tempo_feel', '?')}, "
                f"energy={getattr(descriptor, 'energy_level', '?')}, "
                f"brightness={getattr(descriptor, 'brightness', '?')}, "
                f"key={getattr(descriptor, 'key', '?')} {getattr(descriptor, 'mode', '')}, "
                f"mood={getattr(descriptor, 'mood', '?')}.")
    motion_lines = []
    for spec in default_motion_bank().specs:
        directions = "/".join(spec.directions) if spec.directions else "none"
        motion_lines.append(
            f"  - {spec.id}: {spec.name}; category={spec.category}; "
            f"beats~{spec.recommended_beats:g}; stationary={str(spec.stationary).lower()}; "
            f"repeatable={str(spec.repeatable).lower()}; directions={directions}"
        )
    return f"""You are a choreographer authoring a high-level STORYBOARD for a music-driven dance.
The dance is assembled from two generators: LODGE (smooth, flowing, graceful, sustained) and EDGE
(sharp, percussive, energetic). Design a coherent whole-song composition with an energy/narrative
arc (build -> climax -> resolution), sectional contrast, and recurring movement motifs. You may
accent the generated dance with the exact curated common motions listed below. Use them sparsely
and intentionally: usually 0-2 cues per section, never more than {MAX_COMMON_MOTIONS_PER_SECTION}.
Establish motifs economically in intros/verses, reinforce a recognizable signature at choruses or
hooks, reserve jumps/punches/large accents for peaks and drops, use turns at transitions, and use
recall/deceleration/resolution in outros. Avoid crowding, adjacent overlapping cues, or unrelated
novelty in every section. For a normal multi-section song, include at least one common-motion cue;
when a chorus/hook repeat exists, establish at least one signature cue that its reuse can carry.

Favor RECAPITULATION for structure: when a section repeats an earlier one (same repeat_label), set
reuse_of to recall its full composed motif, including its common-motion cues. Leave common_motions
empty on a reused section unless you deliberately want one additional variation accent. For a
satisfying ABA close, when the final repeat_label matches the opening, consider reusing it with
variation {{"mirror": true, "retrograde": true}}. Give matching chorus/hook labels a stable motif
identity.

Song: duration={getattr(metadata, 'duration_seconds', 0.0):.1f}s, bpm={getattr(metadata, 'bpm', 0.0):.0f},
climax at section {structure.climax_index}.{desc}

Sections (already detected from the audio; DO NOT invent new ones):
{chr(10).join(lines)}

Available common motions (use ONLY these exact IDs):
{chr(10).join(motion_lines)}

For EACH section (same count and order), decide:
- target_intensity: float 0..1 following the overall arc (peak near section {structure.climax_index}).
- vocabulary: one of {list(VOCABULARY)}.
- generator_bias: "lodge" (flowing/graceful), "edge" (sharp/energetic), or "auto".
- reuse_of: an EARLIER section index with the SAME repeat_label to recur its motif, else null.
- variation: {{"mirror": bool, "retrograde": bool, "retime": 1.0, "amplitude": 1.0}} (only
  meaningful when reuse_of set; "retrograde" plays the reused motif backward in time -- good for a
  B->A' return or mirroring the intro at the end).
- common_motions: a JSON list of 0-{MAX_COMMON_MOTIONS_PER_SECTION} cues. Each cue is
  {{"motion_id": exact catalog ID, "position": normalized float 0..1 within the section,
    "anchor": one of {list(COMMON_MOTION_ANCHORS)}, "intensity": 0..1,
    "direction": "auto" or a supported direction or null, "mirror": bool, "repeats": 1..4,
    "motif": short recurring motif name, "rationale": brief musical/choreographic reason}}.
  Put strong action events on beats or phrase accents. Keep the first/last 5% clear for joins.

Respond with JSON ONLY, exactly:
{{"arc": "one-sentence description of the energy/narrative arc",
  "reasoning": "brief justification",
  "plans": [
    {{"section_index": 0, "role": "intro", "target_intensity": 0.1, "vocabulary": "grounded_minimal",
      "generator_bias": "lodge", "reuse_of": null,
      "variation": {{"mirror": false, "retrograde": false, "retime": 1.0, "amplitude": 1.0}},
      "common_motions": [
        {{"motion_id": "rise_reach", "position": 0.7, "anchor": "beat", "intensity": 0.45,
          "direction": null, "mirror": false, "repeats": 1, "motif": "opening_reach",
          "rationale": "establish a restrained upward motif before the build"}}
      ]}}
  ]}}
"""


def _coerce_common_motions(raw: object, idx: int,
                           structure: MusicStructure) -> list[CommonMotionCue]:
    from agentlodge.editor.motion_bank import default_motion_bank, minimum_window_frames

    if not isinstance(raw, list):
        return []
    bank = default_motion_bank()
    section = structure.sections[idx]
    cues: list[CommonMotionCue] = []
    occupied_keys: set[tuple[str, int]] = set()

    for item in raw[:MAX_COMMON_MOTIONS_PER_SECTION]:
        if not isinstance(item, dict):
            continue
        value = str(item.get("motion_id") or item.get("id") or "").strip()
        try:
            spec = bank.resolve(value)
        except KeyError:
            logger.warning("Ignoring invented common motion %r in section %d", value, idx)
            continue

        repeats = _bounded_int(item.get("repeats", 1), 1, 1, 4)
        if not spec.repeatable:
            repeats = 1
        if section.n_frames < minimum_window_frames(spec.minimum_frames * repeats):
            logger.warning(
                "Ignoring %s in section %d: %d frames cannot fit its natural minimum",
                spec.id, idx, section.n_frames,
            )
            continue
        position = _bounded_float(item.get("position", 0.5), 0.5, 0.05, 0.95)
        duplicate_key = (spec.id, int(round(position * 20)))
        if duplicate_key in occupied_keys:
            continue
        occupied_keys.add(duplicate_key)

        anchor = str(item.get("anchor", spec.default_anchor)).strip().lower()
        if anchor not in COMMON_MOTION_ANCHORS:
            anchor = spec.default_anchor
        direction_raw = item.get("direction", "auto" if spec.directions else None)
        direction = None if direction_raw in (None, "", "none") else str(direction_raw).strip().lower()
        if spec.directions:
            if direction != "auto" and direction not in spec.directions:
                direction = "auto"
        else:
            direction = None
        mirror = bool(item.get("mirror", False)) and spec.mirrorable
        if mirror:
            direction = None
        elif direction not in (None, "auto"):
            mirror = False
        cues.append(CommonMotionCue(
            motion_id=spec.id,
            position=position,
            anchor=anchor,
            intensity=_bounded_float(item.get("intensity", 0.65), 0.65, 0.0, 1.0),
            direction=direction,
            mirror=mirror,
            repeats=repeats,
            motif=str(item.get("motif", "")).strip()[:64],
            rationale=str(item.get("rationale", "")).strip()[:240],
        ))
    return cues


def _coerce_plan(raw: dict, idx: int, structure: MusicStructure) -> SectionPlan:
    role = str(raw.get("role") or structure.sections[idx].role)
    intensity = _bounded_float(
        raw.get("target_intensity", structure.sections[idx].energy),
        float(structure.sections[idx].energy),
        0.0,
        1.0,
    )
    vocab = str(raw.get("vocabulary", "")).strip()
    if vocab not in VOCABULARY:
        vocab = _vocab_for_energy(intensity)
    bias = str(raw.get("generator_bias", "auto")).lower().strip()
    if bias not in GENERATORS:
        bias = "auto"
    reuse = raw.get("reuse_of")
    reuse_of = None
    if isinstance(reuse, int) and 0 <= reuse < idx:
        # only accept reuse of an earlier section that shares the repeat label
        if structure.sections[reuse].label == structure.sections[idx].label:
            reuse_of = reuse
    var = raw.get("variation") or {}
    if not isinstance(var, dict):
        var = {}
    variation = {
        "mirror": bool(var.get("mirror", False)) if reuse_of is not None else False,
        "retrograde": bool(var.get("retrograde", False)) if reuse_of is not None else False,
        "retime": _bounded_float(var.get("retime", 1.0), 1.0, 0.25, 4.0)
        if reuse_of is not None else 1.0,
        "amplitude": _bounded_float(var.get("amplitude", 1.0), 1.0, 0.7, 1.4)
        if reuse_of is not None else 1.0,
    }
    common_motions = _coerce_common_motions(raw.get("common_motions"), idx, structure)
    return SectionPlan(section_index=idx, role=role, target_intensity=intensity,
                       vocabulary=vocab, generator_bias=bias, reuse_of=reuse_of,
                       variation=variation, common_motions=common_motions)


def _parse_response(text: str, structure: MusicStructure) -> Storyboard:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("storyboard response contained no JSON")
    payload = json.loads(match.group(), strict=False)
    raw_plans = payload.get("plans")
    if not isinstance(raw_plans, list) or len(raw_plans) != len(structure.sections):
        raise ValueError(
            f"storyboard has {len(raw_plans) if isinstance(raw_plans, list) else 0} plans, "
            f"expected {len(structure.sections)}"
        )
    plans = [_coerce_plan(dict(raw_plans[i]), i, structure) for i in range(len(structure.sections))]
    return Storyboard(
        arc=str(payload.get("arc", "")).strip() or _describe_arc(structure),
        plans=plans,
        reasoning=str(payload.get("reasoning", "")).strip(),
        used_fallback=False,
    )


def _ensure_common_motion_coverage(board: Storyboard,
                                   structure: MusicStructure) -> Storyboard:
    """Add one validated signature only when an LLM omitted the catalog entirely."""
    if any(plan.common_motions for plan in board.plans):
        return board
    reused_sources = [
        plan.reuse_of for plan in board.plans
        if plan.reuse_of is not None
    ]
    order = list(dict.fromkeys(
        reused_sources
        + [structure.climax_index]
        + list(range(len(board.plans)))
    ))
    for idx in order:
        if not 0 <= idx < len(board.plans):
            continue
        plan = board.plans[idx]
        cues = _fallback_common_motions(structure, idx, plan.reuse_of)
        if cues:
            plan.common_motions = cues
            suffix = "Added one validated signature because the LLM returned no common motions."
            board.reasoning = f"{board.reasoning} {suffix}".strip()
            break
    return board


def author_storyboard(structure: MusicStructure, metadata: "SongMetadata",
                      descriptor: "AudioDescriptor | None", api_key: str | None,
                      *, motif_reuse: bool = True, chat_model: str | None = None) -> Storyboard:
    """Author a :class:`Storyboard` for ``structure`` (LLM if ``api_key`` else rule-based)."""
    if not structure.sections:
        return Storyboard(arc="(empty)", plans=[], reasoning="no sections", used_fallback=True)
    if not api_key:
        board = _rule_based_storyboard(structure, motif_reuse=motif_reuse)
    else:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            model = chat_model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model,
                max_tokens=2400,
                messages=[{"role": "user", "content": _build_prompt(structure, metadata, descriptor)}],
            )
            text = response.choices[0].message.content or ""
            board = _ensure_common_motion_coverage(
                _parse_response(text, structure),
                structure,
            )
        except Exception as exc:  # noqa: BLE001 - robust fallback on any failure
            logger.warning("Storyboard agent failed (%s); using rule-based fallback", exc)
            board = _rule_based_storyboard(structure, motif_reuse=motif_reuse)
    logger.info("Storyboard authored (%s):\n%s",
                "rule-based fallback" if board.used_fallback else "LLM", board.describe())
    return board
