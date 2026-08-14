"""Natural-language dance editing agent with a bounded, verified refine loop (AgentBanana-style).

A user describes a change ("make the drop more energetic", "reverse the intro", "reuse the opening
and mirror it at the end", "tighten the beat"). The agent parses it to ONE bounded operation over
the pipeline's existing controls, applies it, RE-ASSEMBLES, VERIFIES the change achieved the intent
(objective metric deltas), and refines up to ``max_iters`` times before stopping -- returning the
best attempt with an explanation if it can't fully satisfy the request (Self-Refine / Reflexion;
Madaan 2023, Shinn 2023; the nano-banana / AgentBanana propose→apply→verify→refine loop).

Design for safety + testability:
* Edits map only onto EXISTING bounded controls -- per-section ``target_intensity`` /
  ``generator_bias`` / post-variation (mirror·retrograde·amplitude) and the whole-piece
  recapitulation toggle -- so the agent can never synthesize arbitrary/unsafe motion.
* The inner parse→apply→verify loop runs on the pure-numpy ``select_sources`` decisions (no torch),
  so it is fully unit-testable; the final rendered motion is produced by ``build_story_dance`` once
  the edit is accepted (call ``assemble()``).
* Parsing uses an LLM when an API key is given, else a deterministic keyword parser (offline-safe).
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from agentlodge.agent.storyboard import Storyboard
from agentlodge.dance.beat_metrics import beat_alignment_score
from agentlodge.dance.story import select_sources

logger = logging.getLogger(__name__)

_ROLE_WORDS = {
    "intro": "intro", "beginning": "intro", "opening": "intro", "start": "intro",
    "outro": "outro", "ending": "outro", "close": "outro", "finale": "outro",
    "drop": "drop", "chorus": "chorus", "verse": "verse", "bridge": "bridge",
}


@dataclass
class EditOp:
    action: str                      # recapitulate|set_intensity|set_bias|mirror|retrograde|amplitude|beat
    section: int | None = None
    params: dict = field(default_factory=dict)
    raw: str = ""

    def to_dict(self) -> dict:
        return {"action": self.action, "section": self.section, "params": self.params}


@dataclass
class EditResult:
    ok: bool
    op: EditOp | None
    iterations: int
    feedback: str
    decisions: list
    storyboard: Storyboard
    recapitulate: bool
    post_variations: dict


def resolve_section(ref, structure) -> int | None:
    """Resolve a section reference (index, role word, 'M:SS' / 'Ns' time, first/last) to an index."""
    secs = list(getattr(structure, "sections", []))
    if not secs or ref is None:
        return None
    if isinstance(ref, (int, np.integer)):
        i = int(ref)
        return i if 0 <= i < len(secs) else None
    s = str(ref).strip().lower()
    m = re.search(r"(\d+):(\d+)", s) or re.search(r"\b(\d+)\s*s(?:ec)?\b", s)
    if m:
        t = (int(m.group(1)) * 60 + int(m.group(2))) if ":" in m.group(0) else int(m.group(1))
        for i, sec in enumerate(secs):
            if sec.start_sec <= t < sec.end_sec:
                return i
        return None
    if any(w in s for w in ("climax", "the drop", "peak", "highest")):
        return int(getattr(structure, "climax_index", 0))
    for word, role in _ROLE_WORDS.items():
        if word in s:
            for i, sec in enumerate(secs):
                if sec.role == role:
                    return i
    if any(w in s for w in ("last", "final", "the end")):
        return len(secs) - 1
    if "first" in s:
        return 0
    return None


def _rule_parse(instruction: str, structure) -> EditOp:
    s = instruction.lower()
    sec = resolve_section(instruction, structure)
    if any(w in s for w in ("recapitulat", "reuse the intro", "reuse the opening", "aba",
                            "mirror it at the end", "opening at the end", "mirror the intro at",
                            "bookend")):
        return EditOp("recapitulate", None, {"on": True}, instruction)
    if any(w in s for w in ("reverse", "retrograde", "backward", "in reverse")):
        return EditOp("retrograde", sec, {}, instruction)
    if any(w in s for w in ("mirror", "flip", "reflect")):
        return EditOp("mirror", sec, {}, instruction)
    if any(w in s for w in ("more energetic", "more energy", "bigger", "stronger", "livelier",
                            "more intense", "more powerful", "punchier", "amp up", "hype")):
        return EditOp("set_intensity", sec, {"direction": 1}, instruction)
    if any(w in s for w in ("calmer", "less energetic", "softer", "gentler", "smaller", "subdued",
                            "tone it down", "less intense", "mellow")):
        return EditOp("set_intensity", sec, {"direction": -1}, instruction)
    if any(w in s for w in ("beat", "sync", "on time", "on beat", "tighten", "rhythm")):
        return EditOp("beat", sec, {}, instruction)
    if any(w in s for w in ("exaggerate", "amplify", "bigger movement", "larger")):
        return EditOp("amplitude", sec, {"factor": 1.3}, instruction)
    if any(w in s for w in ("smoother", "graceful", "flowing", "lodge", "lyrical")):
        return EditOp("set_bias", sec, {"gen": "lodge"}, instruction)
    if any(w in s for w in ("sharper", "percussive", "edge", "staccato", "snappy")):
        return EditOp("set_bias", sec, {"gen": "edge"}, instruction)
    return EditOp("beat", sec, {}, instruction)  # safe default: try to improve beat sync


def _llm_parse(instruction: str, structure, api_key: str) -> EditOp:
    from openai import OpenAI

    roles = ", ".join(f"[{i}] {s.role} {s.start_sec:.0f}-{s.end_sec:.0f}s"
                      for i, s in enumerate(structure.sections))
    prompt = (
        "Map the user's dance-edit request to ONE JSON operation. Sections: " + roles + ".\n"
        "Allowed actions: recapitulate (reuse opening mirrored+retrograded at the end; section=null),"
        " set_intensity (params.direction 1 or -1), set_bias (params.gen 'lodge'|'edge'),"
        " mirror, retrograde, amplitude (params.factor ~1.3), beat (improve beat sync).\n"
        "Respond with JSON only: {\"action\":..., \"section\": <index or null>, \"params\": {...}}.\n"
        f"Request: {instruction}"
    )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model="gpt-4o-mini", max_tokens=200,
                                          messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON in edit-parse response")
    raw = json.loads(m.group())
    action = str(raw.get("action", "")).strip()
    if action not in {"recapitulate", "set_intensity", "set_bias", "mirror", "retrograde",
                      "amplitude", "beat"}:
        raise ValueError(f"unknown action {action!r}")
    sec = raw.get("section")
    sec = int(sec) if isinstance(sec, (int, float)) and 0 <= int(sec) < len(structure.sections) else None
    return EditOp(action, sec, dict(raw.get("params") or {}), instruction)


def parse_instruction(instruction: str, structure, *, api_key: str | None = None) -> EditOp:
    """Parse an NL edit request into a bounded :class:`EditOp` (LLM if ``api_key`` else keywords)."""
    if api_key:
        try:
            return _llm_parse(instruction, structure, api_key)
        except Exception as exc:  # noqa: BLE001 - robust fallback
            logger.warning("Edit parse via LLM failed (%s); using keyword parser", exc)
    return _rule_parse(instruction, structure)


class EditAgent:
    """Stateful NL editor over a story assembly. Operates on already-processed Z-up 139 motions."""

    def __init__(self, structure, storyboard: Storyboard, lodge_z: np.ndarray, edge_z: np.ndarray,
                 *, music_beat_frames=None, recapitulate: bool = False,
                 post_variations: dict | None = None, api_key: str | None = None,
                 max_iters: int = 3):
        self.structure = structure
        self.storyboard = storyboard
        self.lodge = lodge_z
        self.edge = edge_z
        self.beats = np.asarray(music_beat_frames) if music_beat_frames is not None else None
        self.recapitulate = recapitulate
        self.post_variations = copy.deepcopy(post_variations) or {}
        self.api_key = api_key
        self.max_iters = int(max_iters)

    def _decisions(self, sb, recap, pv):
        return select_sources(self.lodge, self.edge, self.structure, sb,
                              recapitulate=recap, post_variations=pv,
                              music_beat_frames=self.beats)

    def _section_bas(self, clip, a, b) -> float:
        if self.beats is None:
            return 0.0
        local = self.beats[(self.beats >= a) & (self.beats < b)] - a
        return beat_alignment_score(clip, local)

    def _apply(self, op, sb, recap, pv, *, escalate):
        sb = copy.deepcopy(sb)
        recap = recap
        pv = copy.deepcopy(pv)
        plans = {p.section_index: p for p in sb.plans}
        if op.action == "recapitulate":
            recap = bool(op.params.get("on", True))
            return sb, recap, pv
        sec = op.section
        if sec is None or sec not in plans:
            return sb, recap, pv
        p = plans[sec]
        if op.action == "set_intensity":
            step = 0.25 * escalate * float(op.params.get("direction", 1))
            p.target_intensity = float(np.clip(p.target_intensity + step, 0.0, 1.0))
        elif op.action == "set_bias":
            p.generator_bias = op.params.get("gen", "auto")
        elif op.action in ("mirror", "retrograde"):
            pv.setdefault(sec, {})[op.action] = True
        elif op.action == "amplitude":
            base = float(op.params.get("factor", 1.3))
            pv.setdefault(sec, {})["amplitude"] = float(np.clip(1.0 + (base - 1.0) * escalate, 0.7, 1.4))
        elif op.action == "beat":
            a, b = self.structure.sections[sec].start_frame, self.structure.sections[sec].end_frame
            bl = self._section_bas(self.lodge[a:b], a, b)
            be = self._section_bas(self.edge[a:b], a, b)
            p.generator_bias = "edge" if be >= bl else "lodge"
        return sb, recap, pv

    def _verify(self, op, before, after) -> tuple[bool, str]:
        sec = op.section
        if op.action == "recapitulate":
            ok = bool(after) and after[-1]["source"] == "reuse:0"
            return ok, "recapitulation close applied" if ok else "could not add ABA close"
        if sec is None or sec >= len(after):
            return False, "section could not be resolved"
        d = after[sec]
        if op.action == "set_intensity":
            le = {k: v for k, v in d["energies"].items() if k in ("lodge", "edge")}
            want = (max(le, key=le.get) if op.params.get("direction", 1) > 0 else min(le, key=le.get))
            ok = d["source"] == want
            return ok, (f"section now uses {d['source']}" if ok else "energy did not shift enough")
        if op.action == "set_bias" or op.action == "beat":
            gen = op.params.get("gen") or {"lodge": "lodge", "edge": "edge"}.get(
                after[sec]["plan_bias"], after[sec]["plan_bias"])
            ok = d["source"] == gen
            return ok, (f"section now uses {gen}" if ok else f"section still uses {d['source']}")
        if op.action in ("mirror", "retrograde"):
            ok = op.action in d.get("post_variation", [])
            return ok, (f"{op.action} applied" if ok else f"{op.action} not applied")
        if op.action == "amplitude":
            ok = any(str(x).startswith("amp") for x in d.get("post_variation", []))
            return ok, ("amplitude applied" if ok else "amplitude not applied")
        return False, "unrecognized operation"

    def edit(self, instruction: str) -> EditResult:
        """Parse → apply → verify → refine (bounded). Commits to agent state on success."""
        op = parse_instruction(instruction, self.structure, api_key=self.api_key)
        before = self._decisions(self.storyboard, self.recapitulate, self.post_variations)
        best = None
        for it in range(1, self.max_iters + 1):
            sb, recap, pv = self._apply(op, self.storyboard, self.recapitulate,
                                        self.post_variations, escalate=it)
            after = self._decisions(sb, recap, pv)
            ok, feedback = self._verify(op, before, after)
            best = (sb, recap, pv, after, feedback, it)
            if ok:
                self.storyboard, self.recapitulate, self.post_variations = sb, recap, pv
                logger.info("Edit satisfied in %d iter(s): %s -> %s", it, op.to_dict(), feedback)
                return EditResult(True, op, it, feedback, after, sb, recap, pv)
        sb, recap, pv, after, feedback, it = best
        # commit the best attempt even if not fully satisfied, so the user sees progress
        self.storyboard, self.recapitulate, self.post_variations = sb, recap, pv
        msg = f"Could not fully satisfy '{instruction}' after {self.max_iters} tries: {feedback}"
        logger.info(msg)
        return EditResult(False, op, it, msg, after, sb, recap, pv)
