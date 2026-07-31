"""Agent-driven windowed dance editing: interpret ANY instruction, compose tools, log the walk.

The earlier editor mapped an instruction to ONE of a fixed set of tags (more_energetic / calmer /
more_on_beat / ...). This module makes editing genuinely agentic: an LLM reads the free-form request
plus the window's current metrics and **plans an ordered sequence of window tools** (which it can
compose -- e.g. "calmer but keep it tight" -> energy(down) then beat_align), executes them, and
records a short, human-readable **log of how it walked through the change** (rationale + tool + the
metric it moved). Without an API key it falls back to a deterministic keyword plan, so it still runs
offline and is unit-testable.

Every tool is a pure function on the window clip (139-dim), so the agent can never synthesize unsafe
motion; the only tool that calls a diffusion backbone is ``regenerate`` (windowed), and it degrades
gracefully when no generator is available.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from agentlodge.dance.transition import (
    amplitude_scale,
    beat_align_warp,
    mirror,
    retrograde,
    splice_window,
    temporal_sharpen,
    temporal_smooth,
)
from agentlodge.editor.window_edit import (
    EditGoal,
    WindowEditResult,
    _window_beats,
    parse_window_instruction,
    window_metrics,
)

logger = logging.getLogger(__name__)


# ============================================================================ toolbox
def _tool_beat_align(clip, ctx, *, strength: float = 1.0):
    strength = float(np.clip(strength, 0.0, 1.0))
    out = beat_align_warp(clip, ctx["wbeats"], strength=strength, passes=3)
    return out, f"snapped the window's accents onto the music beats (strength {strength:.1f})"


def _tool_energy(clip, ctx, *, direction: str = "up", amount: float = 0.5):
    amount = float(np.clip(amount, 0.0, 1.0))
    if str(direction).lower() == "down":
        factor = 1.0 - 0.5 * amount
        verb = "reduced"
    else:
        factor = 1.0 + 0.6 * amount
        verb = "raised"
    out = amplitude_scale(clip, factor)
    return out, f"{verb} movement amplitude x{factor:.2f}"


def _tool_smooth(clip, ctx, *, amount: float = 0.6):
    out = temporal_smooth(clip, float(np.clip(amount, 0.0, 1.0)))
    return out, f"low-pass smoothed the motion (amount {float(amount):.1f}) to reduce jerk"


def _tool_sharpen(clip, ctx, *, amount: float = 0.6):
    out = temporal_sharpen(clip, float(np.clip(amount, 0.0, 1.2)))
    return out, f"accentuated hits (unsharp, amount {float(amount):.1f}) for snappier motion"


def _tool_mirror(clip, ctx):
    return mirror(clip), "mirrored the window left<->right"


def _tool_reverse(clip, ctx):
    return retrograde(clip), "reversed the window in time"


def _tool_regenerate(clip, ctx, *, backbone: str = "auto", energy: float = 0.5, k: int = 3):
    """Resample a fresh dance for JUST this window from a diffusion backbone (best of a few seeds)."""
    gen = ctx.get("generator")
    if gen is None:
        return clip, "no backbone available - kept the current motion"
    a, b, wbeats = ctx["a"], ctx["b"], ctx["wbeats"]
    bbs = ["edge", "lodge"] if backbone == "auto" else [backbone]
    best, best_score, chosen = None, None, None
    for bb in bbs:
        for s in range(max(1, int(k))):
            cand = gen.generate(bb, a, b, s, energy=float(np.clip(energy, 0.0, 1.0)),
                                beats=wbeats, context=ctx.get("context"))
            if cand is None or np.asarray(cand).shape[0] < 2:
                continue
            cand = np.asarray(cand, dtype=np.float32)
            m = window_metrics(cand, wbeats)
            # prefer the requested intensity: score by closeness of energy to target
            score = -abs(m["energy"] - (0.2 + 0.8 * float(energy)))
            if best_score is None or score > best_score:
                best, best_score, chosen = cand, score, bb
    if best is None:
        return clip, "backbone returned nothing - kept the current motion"
    if best.shape[0] != clip.shape[0]:                       # fit to the window length
        from agentlodge.dance.transition import retime
        best = retime(best, clip.shape[0])
    return best, f"regenerated the window with {chosen} (best of {k} seeds)"


@dataclass
class ToolSpec:
    fn: object
    doc: str
    params: str = ""


TOOLS: dict[str, ToolSpec] = {
    "beat_align": ToolSpec(_tool_beat_align,
        "time-warp so the motion accents land on the music beats; raises beat alignment (BAS)",
        'params: {"strength": 0..1}'),
    "energy": ToolSpec(_tool_energy,
        "scale movement size; up = bigger/livelier/more energetic, down = calmer/smaller",
        'params: {"direction": "up"|"down", "amount": 0..1}'),
    "smooth": ToolSpec(_tool_smooth,
        "low-pass the motion for flowing/graceful movement; lowers jerk",
        'params: {"amount": 0..1}'),
    "sharpen": ToolSpec(_tool_sharpen,
        "accentuate hits for crisp/snappy/staccato/percussive movement; raises jerk",
        'params: {"amount": 0..1}'),
    "mirror": ToolSpec(_tool_mirror, "flip the window left<->right", "params: {}"),
    "reverse": ToolSpec(_tool_reverse, "play the window backward (retrograde)", "params: {}"),
    "regenerate": ToolSpec(_tool_regenerate,
        "resample a fresh dance for THIS window from a diffusion backbone (lodge=smooth, "
        "edge=sharp); use when the user wants different choreography, not just a tweak",
        'params: {"backbone": "lodge"|"edge"|"auto", "energy": 0..1}'),
}


# ============================================================================ plan
@dataclass
class PlanStep:
    tool: str
    params: dict = field(default_factory=dict)
    why: str = ""


@dataclass
class AgentPlan:
    summary: str
    steps: list
    expect_metric: str | None = None      # bas|energy|jerk|foot
    expect_dir: str | None = None         # up|down


# objective (keyword parser) -> a single-tool plan, used as the offline fallback.
_OBJ_PLAN = {
    "more_on_beat": ("beat_align", {"strength": 1.0}, "bas", "up", "tighten timing onto the beats"),
    "calmer":       ("energy", {"direction": "down"}, "energy", "down", "reduce intensity"),
    "more_energetic": ("energy", {"direction": "up"}, "energy", "up", "boost intensity"),
    "smoother":     ("smooth", {}, "jerk", "down", "smooth out the motion"),
    "sharper":      ("sharpen", {}, "jerk", "up", "make the hits crisper"),
    "reverse":      ("reverse", {}, None, None, "play it backward"),
    "mirror":       ("mirror", {}, None, None, "flip left-right"),
    "exaggerate":   ("energy", {"direction": "up", "amount": 0.9}, "energy", "up", "make it bigger"),
}


def _keyword_plan(instruction: str) -> AgentPlan:
    goal = parse_window_instruction(instruction, api_key=None)
    tool, params, metric, direction, why = _OBJ_PLAN.get(
        goal.objective, ("beat_align", {"strength": 1.0}, "bas", "up", "improve musical timing"))
    p = dict(params)
    if "amount" not in p and tool in ("energy", "smooth", "sharpen"):
        p["amount"] = 0.8 if goal.magnitude >= 0.7 else 0.5
    return AgentPlan(summary=f"{goal.objective.replace('_', ' ')} the selected window",
                     steps=[PlanStep(tool, p, why)], expect_metric=metric, expect_dir=direction)


def _llm_plan(instruction: str, ctx_metrics: dict, a_sec: float, b_sec: float,
              api_key: str) -> AgentPlan:
    from openai import OpenAI

    tool_lines = "\n".join(f"- {name}({spec.params}): {spec.doc}" for name, spec in TOOLS.items())
    prompt = (
        "You are a dance-motion editing agent. The user selected the window "
        f"[{a_sec:.1f}s..{b_sec:.1f}s] of a dance and asked: \"{instruction}\".\n"
        f"Current window metrics -> energy(intensity): {ctx_metrics.get('energy')}, "
        f"beat_alignment(BAS 0-1, higher=tighter): {ctx_metrics.get('bas')}, "
        f"jerk(lower=smoother): {ctx_metrics.get('jerk')}, "
        f"foot_contact(0-1, higher=less sliding): {ctx_metrics.get('foot')}.\n\n"
        "Compose 1-3 of these tools, in order, to satisfy the request:\n" + tool_lines + "\n\n"
        "Return JSON ONLY:\n"
        '{"summary": "<one plain-English line describing your plan>",\n'
        ' "steps": [{"tool": "<name>", "params": {...}, "why": "<short reason>"}],\n'
        ' "expect": {"metric": "bas"|"energy"|"jerk"|"foot"|null, "direction": "up"|"down"|null}}\n'
        "Pick tools by meaning, not keywords; you may combine (e.g. energy down + beat_align)."
    )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model="gpt-4o-mini", max_tokens=400, temperature=0.2,
                                          messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON in agent plan response")
    raw = json.loads(m.group())
    steps = []
    for s in (raw.get("steps") or []):
        name = str(s.get("tool", "")).strip()
        if name in TOOLS:
            steps.append(PlanStep(name, dict(s.get("params") or {}), str(s.get("why", ""))))
    if not steps:
        raise ValueError("agent plan had no valid tool steps")
    exp = raw.get("expect") or {}
    metric = exp.get("metric") if exp.get("metric") in ("bas", "energy", "jerk", "foot") else None
    direction = exp.get("direction") if exp.get("direction") in ("up", "down") else None
    return AgentPlan(str(raw.get("summary", "")).strip() or "edit the selected window",
                     steps, metric, direction)


def plan_edit(instruction: str, ctx_metrics: dict, a_sec: float, b_sec: float,
              *, api_key: str | None = None) -> AgentPlan:
    """Plan an edit as an ordered tool sequence (LLM agent if ``api_key`` else keyword fallback)."""
    if api_key:
        try:
            return _llm_plan(instruction, ctx_metrics, a_sec, b_sec, api_key)
        except Exception as exc:  # noqa: BLE001 - robust offline fallback
            logger.warning("agent plan via LLM failed (%s); using keyword plan", exc)
    return _keyword_plan(instruction)


# ============================================================================ execute
def _verify(plan: AgentPlan, before: dict, after: dict) -> tuple[bool, str]:
    metric, direction = plan.expect_metric, plan.expect_dir
    if not metric or not direction:
        return True, "applied the planned edit"
    lo, hi = before.get(metric, 0.0), after.get(metric, 0.0)
    if direction == "up":
        ok = hi > lo + 1e-3
    else:
        ok = hi < lo - 1e-3
    arrow = "->"
    return ok, f"{metric} {lo:.3f} {arrow} {hi:.3f}"


def run_agent_edit(motion: np.ndarray, a: int, b: int, instruction: str,
                   generator=None, *, beats=None, api_key: str | None = None,
                   blend_frames: int = 15, k: int = 3, max_cycles: int = 1,
                   progress_cb=None, context=None) -> WindowEditResult:
    """Plan (agentically) → apply tools in order (logging each) → splice → verify.

    Returns a :class:`WindowEditResult` whose ``log`` is the step-by-step agent walk and
    ``agent_summary`` the one-line plan, for the UI to display.
    """
    def _emit(ev: dict) -> None:
        if progress_cb is not None:
            try:
                progress_cb(ev)
            except Exception:  # noqa: BLE001
                pass

    motion = np.ascontiguousarray(motion, dtype=np.float32)
    L = int(motion.shape[0])
    a, b = int(a), int(b)
    if not (0 <= a < b <= L):
        raise ValueError(f"invalid window [{a}, {b}) for motion of length {L}")
    wbeats = _window_beats(beats, a, b)
    before = window_metrics(motion[a:b], wbeats)
    a_sec, b_sec = a / 30.0, b / 30.0

    plan = plan_edit(instruction, before, a_sec, b_sec, api_key=api_key)
    _emit({"phase": "plan", "summary": plan.summary,
           "steps": [{"tool": s.tool, "why": s.why} for s in plan.steps],
           "metrics_before": before})

    ctx = {"wbeats": wbeats, "a": a, "b": b, "generator": generator, "context": context,
           "blend_frames": blend_frames, "k": k}
    cur = np.ascontiguousarray(motion[a:b], dtype=np.float32)
    log: list = []
    for i, step in enumerate(plan.steps, 1):
        spec = TOOLS.get(step.tool)
        m_before = window_metrics(cur, wbeats)
        if spec is None:
            log.append({"step": i, "tool": step.tool, "why": step.why,
                        "note": "unknown tool - skipped", "metrics": m_before})
            continue
        try:
            cur2, note = spec.fn(cur, ctx, **(step.params or {}))
        except Exception as exc:  # noqa: BLE001 - a bad tool step must not abort the edit
            logger.warning("agent tool %s failed: %s", step.tool, exc)
            log.append({"step": i, "tool": step.tool, "why": step.why,
                        "note": f"failed ({exc}) - skipped", "metrics": m_before})
            continue
        cur2 = np.ascontiguousarray(cur2, dtype=np.float32)
        m_after = window_metrics(cur2, wbeats)
        entry = {"step": i, "tool": step.tool, "params": step.params, "why": step.why,
                 "note": note, "metrics_before": m_before, "metrics_after": m_after}
        log.append(entry)
        cur = cur2
        _emit({"phase": "step", "step": i, "n_steps": len(plan.steps), "tool": step.tool,
               "why": step.why, "note": note, "metrics": m_after})

    spliced = splice_window(motion, a, b, cur, blend_frames=blend_frames)
    after = window_metrics(spliced[a:b], wbeats)
    ok, verdict = _verify(plan, before, after)
    feedback = (verdict if ok else f"partially applied: {verdict}")
    _emit({"phase": "done", "ok": ok, "metrics_after": after, "feedback": feedback,
           "summary": plan.summary})

    goal = EditGoal(objective="agent", backbone="agent", magnitude=0.5, raw=instruction)
    return WindowEditResult(ok, goal, (a, b), spliced, before, after, backbone="agent",
                            chosen_seed=None, cycles=log, feedback=feedback,
                            log=log, agent_summary=plan.summary)
