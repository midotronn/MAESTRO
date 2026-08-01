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
    crossfade_edit,
    mirror,
    retrograde,
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
def _tool_beat_align(clip, ctx, *, strength: float = 1.0, passes: int = 3):
    strength = float(np.clip(strength, 0.0, 1.0))
    passes = int(np.clip(passes, 1, 8))
    out = beat_align_warp(clip, ctx["wbeats"], strength=strength, passes=passes)
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
    """Resample a fresh dance for JUST this window from a diffusion backbone (best of a few seeds).

    Candidates are ranked by how well they advance the edit's DECLARED goals (``ctx['goals']``) --
    e.g. "regenerate this more on beat" keeps the highest-BAS seed, "bigger" keeps the most energetic
    -- rather than a fixed energy target. Only when there is no measurable goal (e.g. "give me
    different moves") does it fall back to matching the requested intensity.
    """
    gen = ctx.get("generator")
    if gen is None:
        return clip, "no backbone available - kept the current motion"
    a, b, wbeats = ctx["a"], ctx["b"], ctx["wbeats"]
    goals = ctx.get("goals") or []
    base_m = window_metrics(clip, wbeats)                    # rank candidates relative to the current window
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
            if goals:                                        # advance the goals the agent declared
                score = _reward_goals(goals, base_m, m)
            else:                                            # goalless: match the requested intensity
                score = -abs(m["energy"] - (0.2 + 0.8 * float(energy)))
            if best_score is None or score > best_score:
                best, best_score, chosen = cand, score, bb
    if best is None:
        return clip, "backbone returned nothing - kept the current motion"
    if best.shape[0] != clip.shape[0]:                       # fit to the window length
        from agentlodge.dance.transition import retime
        best = retime(best, clip.shape[0])
    goal_txt = (" toward " + ", ".join(lbl for _m, _d, lbl in goals)) if goals else ""
    return best, f"regenerated the window with {chosen} (best of {k} seeds{goal_txt})"


@dataclass
class ToolSpec:
    fn: object
    doc: str
    params: str = ""


TOOLS: dict[str, ToolSpec] = {
    "beat_align": ToolSpec(_tool_beat_align,
        "time-warp so the motion accents land on the music beats; raises beat alignment (BAS)",
        'params: {"strength": 0..1, "passes": 1..8 (more = tighter)}'),
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


# ============================================================================ metric goals
# Each metric a tool is *supposed* to improve. The executor uses this to reject a step that moves
# its own target the wrong way (e.g. beat_align must never LOWER beat alignment). Tools with no
# monotone contract (mirror/reverse/regenerate) return None and are never rejected on a metric.
METRIC_LABEL = {"bas": "beat alignment", "energy": "energy", "jerk": "smoothness", "foot": "foot contact"}


def _tool_target(tool: str, params: dict | None) -> tuple[str, str] | None:
    params = params or {}
    if tool == "beat_align":
        return ("bas", "up")
    if tool == "energy":
        return ("energy", "down" if str(params.get("direction", "up")).lower() == "down" else "up")
    if tool == "smooth":
        return ("jerk", "down")
    if tool == "sharpen":
        return ("jerk", "up")
    return None


def _tol(metric: str, before: float) -> float:
    """Movement below this is 'held' (noise), not a real change. Absolute floor per metric scale."""
    floors = {"bas": 2e-3, "foot": 2e-3, "energy": 3e-3, "jerk": 1e-4}
    return max(floors.get(metric, 1e-3), 0.01 * abs(before))


def _classify(metric: str, before: float, after: float, direction: str) -> str:
    """'improved' | 'held' | 'regressed' for a metric moved toward ``direction``."""
    d = after - before
    tol = _tol(metric, before)
    if abs(d) <= tol:
        return "held"
    good = (direction == "up" and d > 0) or (direction == "down" and d < 0)
    return "improved" if good else "regressed"


# instruction phrase -> (metric, direction). Multiple can fire ("calmer but keep it tight" ->
# energy-down AND beat-up), which is exactly the multi-goal case the single-metric verifier missed.
_METRIC_INTENTS: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("bas", "up", "beat alignment",
     ("on beat", "on-beat", "on the beat", "beat align", "beat-align", "beat aligned", "beataligned",
      "to the beat", "tighten", "tighter", "tight", "in time", "in sync", "sync", "synced",
      "rhythm", "rhythmic", "timing", "musical", "lock to the beat", "locked", "on the grid", "beat")),
    ("energy", "down", "energy",
     ("calmer", "calm", "softer", "gentler", "gentle", "mellow", "subdued", "less energetic",
      "less energy", "tone it down", "tone down", "smaller", "chill", "relax", "slower feel",
      "decrease energy", "lower energy", "reduce energy", "less power")),
    ("energy", "up", "energy",
     ("more energetic", "more energy", "energetic", "bigger", "stronger", "livelier", "lively",
      "more intense", "intense", "powerful", "hype", "amp up", "amp it up", "pump", "explosive",
      "exaggerate", "amplify", "over the top", "go bigger", "increase energy", "increase the energy",
      "raise energy", "boost energy", "more power", "energize", "up the energy", "add energy",
      "more dynamic", "high energy")),
    ("jerk", "down", "smoothness",
     ("smoother", "smooth", "flowing", "graceful", "fluid", "lyrical", "glide", "flowy", "elegant",
      "less jitter", "less jittery")),
    ("jerk", "up", "smoothness",
     ("sharper", "snappy", "snappier", "staccato", "percussive", "crisp", "punchier", "punchy",
      "sharp", "edgy", "hit harder", "hits", "crisper")),
]


def _requested_metrics(instruction: str) -> list[tuple[str, str, str]]:
    """Extract every metric the user asked to move -> [(metric, direction, label)].

    Deterministic and independent of the LLM plan, so the verifier always checks what the USER
    asked for (e.g. beat alignment) even if the planner's own ``expect`` says something else.
    Conflicting up/down for the same metric cancels out (ambiguous -> don't enforce).
    """
    s = " " + instruction.lower().strip() + " "
    hits: dict[str, tuple[str, str]] = {}     # metric -> (direction, label)
    conflict: set[str] = set()
    for metric, direction, label, phrases in _METRIC_INTENTS:
        if any(p in s for p in phrases):
            if metric in hits and hits[metric][0] != direction:
                conflict.add(metric)
            else:
                hits.setdefault(metric, (direction, label))
    return [(m, d, lbl) for m, (d, lbl) in hits.items() if m not in conflict]


def _verify_goals(goals, before: dict, after: dict) -> tuple[bool, list, str]:
    """Check every requested metric. ok only if NONE regressed and each improved-or-acceptably-held.

    'Acceptably held' = already strong (bas/foot >= 0.9) so there is no headroom to push. Returns
    (ok, per-metric checks, one-line verdict).
    """
    checks, verdicts, ok = [], [], True
    for metric, direction, label in goals:
        b, a = float(before.get(metric, 0.0)), float(after.get(metric, 0.0))
        status = _classify(metric, b, a, direction)
        ceiling = metric in ("bas", "foot") and a >= 0.9
        met = status == "improved" or (status == "held" and ceiling)
        ok = ok and met
        checks.append({"metric": metric, "label": label, "dir": direction,
                       "before": round(b, 4), "after": round(a, 4), "status": status, "met": met})
        tail = {"improved": "\u2191", "held": "\u2192", "regressed": "\u2193"}[status]
        verdicts.append(f"{label} {b:.3f}{tail}{a:.3f}")
    return ok, checks, "  ".join(verdicts)


def _reward_goals(goals, before: dict, after: dict) -> float:
    """Sum of signed improvement across all requested metrics (regressions hurt). Higher = better."""
    r = 0.0
    for metric, direction, _label in goals:
        d = float(after.get(metric, 0.0)) - float(before.get(metric, 0.0))
        sc = _tol(metric, before.get(metric, 0.0)) or 1.0
        r += (d if direction == "up" else -d) / sc
    return r


def _prefer(ok_a: bool, no_reg_a: bool, reward_a: float,
            ok_b: bool, no_reg_b: bool, reward_b: float) -> bool:
    """True if attempt A should replace the current best B. Ranking, most important first:
    (1) meets EVERY goal, (2) regresses NO declared goal (holding is fine), (3) higher summed reward.
    So a balanced attempt that satisfies all goals beats a big single-metric gain that regressed
    another goal, and -- crucially -- an edit that regresses a goal never beats simply keeping the
    original (which regresses nothing). The user asked for X up; we never ship X down."""
    return (int(ok_a), int(no_reg_a), reward_a) > (int(ok_b), int(no_reg_b), reward_b)


def _merge_goals(primary: list, secondary: list) -> list:
    """Union two goal lists as [(metric, dir, label)]. ``primary`` (the planning agent's declared
    goals) wins; ``secondary`` (the deterministic keyword safety net) only contributes metrics the
    primary didn't mention. A metric appears at most once."""
    out = list(primary)
    have = {m for m, _d, _lbl in out}
    for m, d, lbl in secondary:
        if m not in have:
            out.append((m, d, lbl))
            have.add(m)
    return out


def _smoothness_polish(win_cur, motion, a, b, goals, before, wbeats, blend_frames):
    """Quality guard: keep the edit from making the dance jittery. Amplitude scaling and the beat
    time-warp raise jerk as a side effect, so -- UNLESS the user asked for sharper (jerk up) -- apply
    the smallest light temporal smooth that pulls the window's jerk back toward baseline WITHOUT
    regressing any declared goal. Smoothing removes high-frequency jitter but leaves the lower-frequency
    energy/beat content, so the goals survive (measured). Returns (spliced, after, checks, note) or
    None when no polish is needed or none preserves the goals."""
    if any(m == "jerk" and d == "up" for m, d, _lbl in goals):
        return None
    spliced0 = crossfade_edit(motion, a, b, win_cur, blend_frames=blend_frames)
    j0 = window_metrics(spliced0[a:b], wbeats)["jerk"]
    base_j = float(before.get("jerk", 0.0))
    if base_j <= 0 or j0 <= base_j * 1.08:                 # already about as smooth as the original
        return None
    chosen = None
    for amt in (0.06, 0.14, 0.24, 0.34):
        cand = temporal_smooth(win_cur, amt)
        spl = crossfade_edit(motion, a, b, cand, blend_frames=blend_frames)
        m = window_metrics(spl[a:b], wbeats)
        ok, checks, _ = _verify_goals(goals, before, m) if goals else (True, [], "")
        if not ok:
            break                                          # this much smoothing broke a goal -> stop
        chosen = (spl, m, checks, amt)
        if m["jerk"] <= base_j * 1.05:
            break                                          # restored to baseline smoothness
    if chosen is None:
        return None
    spl, m, checks, amt = chosen
    return spl, m, checks, (f"smoothed (amount {amt:.2f}) to preserve quality \u2014 "
                            f"jerk {j0:.3f}\u2192{m['jerk']:.3f}")


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
    expect_metric: str | None = None      # bas|energy|jerk|foot (legacy single-target hint)
    expect_dir: str | None = None         # up|down
    goals: list = field(default_factory=list)   # [(metric, dir, label)] the plan will be graded on
    planner: str = "llm"                   # llm | keyword | keyword_fallback
    planner_note: str = ""                 # human-readable source / why-fallback (for the UI)


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

# (metric, direction) -> the tool that moves it, for composing a plan from the requested metrics.
_METRIC_TOOL = {
    ("bas", "up"): ("beat_align", {"strength": 1.0}),
    ("energy", "up"): ("energy", {"direction": "up", "amount": 0.6}),
    ("energy", "down"): ("energy", {"direction": "down", "amount": 0.6}),
    ("jerk", "down"): ("smooth", {"amount": 0.6}),
    ("jerk", "up"): ("sharpen", {"amount": 0.6}),
}


def _escalate_params(tool: str, params: dict, cycle: int) -> dict:
    """Push a tool harder on a refine cycle (offline escalation for the keyword planner)."""
    p = dict(params)
    if tool in ("energy", "smooth", "sharpen"):
        base = float(p.get("amount", 0.5))
        p["amount"] = float(min(1.0, base + 0.25 * cycle))
    elif tool == "beat_align":
        p["passes"] = int(min(8, int(p.get("passes", 3)) + 2 * cycle))
        p["strength"] = 1.0
    elif tool == "regenerate":
        p["k"] = int(p.get("k", 3)) + 2 * cycle          # search more seeds
    return p


def _keyword_plan(instruction: str, feedback: dict | None = None) -> AgentPlan:
    """Offline planner: compose one tool per requested metric (multi-goal aware).

    "calmer but keep it tight to the beat" -> energy(down) + beat_align, rather than a single tag.
    Beat-align is ordered LAST so it re-times after any amplitude change. Falls back to the
    single-objective mapping when no metric keyword is present (e.g. "flip and reverse").
    """
    reqs = _requested_metrics(instruction)
    cycle = int(feedback.get("cycle", 1)) if feedback else 0
    mag = 0.8 if any(w in (" " + instruction.lower() + " ")
                     for w in (" much ", " way ", " a lot ", " very ", " really ", " super ")) else 0.5
    if reqs:
        steps = []
        for metric, direction, label in reqs:
            tool, params = _METRIC_TOOL[(metric, direction)]
            p = dict(params)
            if "amount" in p:
                p["amount"] = mag
            if feedback:
                p = _escalate_params(tool, p, cycle)
            steps.append(PlanStep(tool, p, f"{'push harder to ' if feedback else ''}move {label} {direction}"))
        steps.sort(key=lambda st: 1 if st.tool == "beat_align" else 0)   # beat_align last
        labels = list(dict.fromkeys(lbl for _, _, lbl in reqs))
        primary = reqs[0]
        return AgentPlan(summary=("push harder: " if feedback else "") + "adjust " + ", ".join(labels),
                         steps=steps, expect_metric=primary[0], expect_dir=primary[1], goals=list(reqs))

    goal = parse_window_instruction(instruction, api_key=None)
    tool, params, metric, direction, why = _OBJ_PLAN.get(
        goal.objective, ("beat_align", {"strength": 1.0}, "bas", "up", "improve musical timing"))
    p = dict(params)
    if "amount" not in p and tool in ("energy", "smooth", "sharpen"):
        p["amount"] = 0.8 if goal.magnitude >= 0.7 else 0.5
    if feedback:                                            # refine: push the same tool harder
        p = _escalate_params(tool, p, cycle)
        why = f"push harder ({why})"
    g = [(metric, direction, METRIC_LABEL.get(metric, metric))] if metric and direction else []
    return AgentPlan(summary=f"{goal.objective.replace('_', ' ')} the selected window",
                     steps=[PlanStep(tool, p, why)], expect_metric=metric, expect_dir=direction, goals=g)


def _llm_plan(instruction: str, ctx_metrics: dict, a_sec: float, b_sec: float,
              api_key: str, feedback: dict | None = None, goals: list | None = None) -> AgentPlan:
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
        ' "goals": [{"metric": "bas"|"energy"|"jerk"|"foot", "direction": "up"|"down"}]}\n'
        "CRITICAL RULE for \"goals\": include ONLY the metric(s) the user EXPLICITLY asked to change, and "
        "NOTHING else -- extra goals conflict and make the edit do nothing. If the user says 'smooth it "
        "out' the ONLY goal is jerk-down; 'make it calmer' -> energy-down ONLY; 'reverse this'/'mirror "
        "it' -> [] (empty). Do NOT add energy, beat, or jerk goals the user did not mention. You are "
        "GRADED on exactly this list.\n"
        "Metric meanings: bas=beat alignment; energy=intensity; jerk=lower is smoother, so "
        "sharper/snappier/staccato/punchy/crisp = jerk UP and smoother/flowing/graceful = jerk DOWN; "
        "foot=foot-plant. List jerk ONLY when the user explicitly asked for smoother or sharper.\n"
        "Examples: \"more energetic and on beat\" -> "
        '[{"metric":"energy","direction":"up"},{"metric":"bas","direction":"up"}]; '
        '"smooth it out so it flows" -> [{"metric":"jerk","direction":"down"}]; '
        '"calmer but keep it tight" -> [{"metric":"energy","direction":"down"},{"metric":"bas","direction":"up"}]; '
        '"snappier staccato hits" -> [{"metric":"jerk","direction":"up"}]; '
        '"reverse this"/"mirror it" -> []. '
        "Pick tools by meaning, not keywords; you may combine (e.g. energy down + beat_align). "
        "Use 'regenerate' ONLY when the user wants genuinely DIFFERENT choreography/moves (it samples a "
        "slow diffusion backbone); for beat/energy/smoothness/sharpness use the FAST deterministic tools "
        "(beat_align/energy/smooth/sharpen). For a pure transform (reverse/mirror/flip) the plan is JUST "
        "that one tool and goals MUST be []. "
        "beat_align LOWERS energy and amplitude scaling can drift the beat, so when both energy-up and "
        "beat are goals, scale energy generously (amount >=0.7) so it stays up AFTER beat_align. "
        "Unless the user asked for sharper, you MAY end the plan with a light 'smooth' STEP (amount "
        "~0.1-0.2) as a finishing touch to shed jitter -- but that is a STEP, not a goal; do NOT put "
        "jerk in goals for it. If the user asked for sharper, do NOT smooth."
    )
    if goals:                                              # refine: remind of the graded constraints
        gl = ", ".join(f"{lbl} must go {d}" for _m, d, lbl in goals)
        prompt += ("\n\nThe user's request REQUIRES: " + gl + ". Every one of these will be checked; "
                   "a plan that lets any of them get WORSE is rejected. If a metric is already high "
                   "(e.g. beat alignment > 0.9) at minimum do not reduce it. When you include a tool "
                   "whose side effect could hurt one of these (e.g. scaling energy can drift timing), "
                   "add the corrective tool too (e.g. beat_align LAST).")
    if feedback:                                            # Self-Refine: revise from what went wrong
        misses = feedback.get("misses") or []
        miss_txt = "; ".join(
            f"{m['label']} went {m['before']:.3f}->{m['after']:.3f} ({m['status']}, needed {m['dir']})"
            for m in misses)
        prompt += (
            "\n\nYOUR PREVIOUS ATTEMPT under-delivered. Previous steps: "
            f"{json.dumps(feedback.get('prev_steps'))}. Unmet goals: " + (miss_txt or "the target metric")
            + ". A metric that REGRESSED (moved the WRONG way) is the worst outcome -- for it, increase "
            "its own tool's magnitude a lot AND order it so it survives the others (e.g. if beat_align "
            "dropped energy, scale energy higher and/or after beat_align). REVISE: push HARDER (larger "
            "amounts / more passes / more seeds), add or swap tools, and ensure the corrective tool for "
            "each unmet metric is present. Do not repeat the same weak plan."
        )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=400, temperature=(0.5 if feedback else 0.2),
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
    parsed_goals = []
    for g in (raw.get("goals") or []):
        gm = str(g.get("metric", "")).strip()
        gd = str(g.get("direction", "")).strip()
        if gm in ("bas", "energy", "jerk", "foot") and gd in ("up", "down"):
            parsed_goals.append((gm, gd, METRIC_LABEL.get(gm, gm)))
    exp = raw.get("expect") or {}                          # legacy single-target (optional)
    metric = exp.get("metric") if exp.get("metric") in ("bas", "energy", "jerk", "foot") else None
    direction = exp.get("direction") if exp.get("direction") in ("up", "down") else None
    if metric is None and parsed_goals:                    # keep the legacy hint populated
        metric, direction = parsed_goals[0][0], parsed_goals[0][1]
    return AgentPlan(str(raw.get("summary", "")).strip() or "edit the selected window",
                     steps, metric, direction, goals=parsed_goals)


def plan_edit(instruction: str, ctx_metrics: dict, a_sec: float, b_sec: float,
              *, api_key: str | None = None, feedback: dict | None = None,
              goals: list | None = None) -> AgentPlan:
    """Plan an edit as an ordered tool sequence AND declare the goal metrics (LLM agent if ``api_key``
    else keyword fallback). The returned ``AgentPlan.goals`` is what the request will be graded on --
    reasoned by the LLM, or derived from the keyword matcher offline.

    ``goals`` (input) is only used on a REFINE call, to remind the LLM of the graded constraints it
    must fix. ``feedback`` (from a failed prior attempt) drives Self-Refine: the LLM is told which
    goals it missed and by how much; the keyword planner escalates magnitude.
    """
    if api_key:
        try:
            p = _llm_plan(instruction, ctx_metrics, a_sec, b_sec, api_key,
                          feedback=feedback, goals=goals)
            p.planner, p.planner_note = "llm", "AI agent (LLM reasoning)"
            return p
        except Exception as exc:  # noqa: BLE001 - robust offline fallback
            logger.warning("agent plan via LLM failed (%s); using keyword plan", exc)
            p = _keyword_plan(instruction, feedback=feedback)
            p.planner = "keyword_fallback"
            p.planner_note = f"offline keyword planner (LLM call failed: {str(exc)[:120]})"
            return p
    p = _keyword_plan(instruction, feedback=feedback)
    p.planner, p.planner_note = "keyword", "offline keyword planner (no API key configured)"
    return p


# ============================================================================ execute
def _execute_plan(base_clip: np.ndarray, plan: AgentPlan, ctx: dict, wbeats, *,
                  cycle: int, emit) -> tuple[np.ndarray, list]:
    """Apply a plan's tools in order to a FRESH copy of the base window; return (clip, step log).

    **Executor guardrail:** after each tool runs, if it moved its OWN target metric the wrong way
    (e.g. ``beat_align`` that *lowers* beat alignment), the step is REJECTED -- the pre-step motion
    is kept and the step is logged as rejected with a reason. This is what catches "I asked to raise
    beat alignment but it went down". Tools with no monotone contract (mirror/reverse/regenerate)
    are never rejected on a metric.
    """
    cur = np.ascontiguousarray(base_clip, dtype=np.float32)
    log: list = []
    n = len(plan.steps)

    def _emit_step(i, tool, why, note, status, metrics):
        emit({"phase": "step", "cycle": cycle, "step": i, "n_steps": n, "tool": tool,
              "why": why, "note": note, "status": status, "metrics": metrics})

    for i, step in enumerate(plan.steps, 1):
        spec = TOOLS.get(step.tool)
        m_before = window_metrics(cur, wbeats)
        entry = {"cycle": cycle, "step": i, "tool": step.tool, "params": step.params,
                 "why": step.why, "metrics_before": m_before, "metrics_after": m_before}
        if spec is None:
            entry.update(status="skipped", note="unknown tool")
            log.append(entry); _emit_step(i, step.tool, step.why, "unknown tool", "skipped", m_before)
            continue
        try:
            cur2, note = spec.fn(cur, ctx, **(step.params or {}))
        except Exception as exc:  # noqa: BLE001 - a bad tool step must not abort the edit
            logger.warning("agent tool %s failed: %s", step.tool, exc)
            entry.update(status="failed", note=f"failed ({exc})")
            log.append(entry); _emit_step(i, step.tool, step.why, f"failed ({exc})", "failed", m_before)
            continue
        cur2 = np.ascontiguousarray(cur2, dtype=np.float32)
        m_after = window_metrics(cur2, wbeats)
        target = _tool_target(step.tool, step.params)
        entry["target"] = list(target) if target else None
        if target is not None and _classify(target[0], m_before[target[0]],
                                             m_after[target[0]], target[1]) == "regressed":
            metric = target[0]
            reason = (f"rejected: would move {METRIC_LABEL[metric]} the wrong way "
                      f"({m_before[metric]:.3f}\u2192{m_after[metric]:.3f}) \u2014 kept the previous motion")
            entry.update(status="rejected", note=note, reject_reason=reason)
            log.append(entry); _emit_step(i, step.tool, step.why, reason, "rejected", m_before)
            continue                                        # cur unchanged -> regression discarded
        entry.update(status="applied", note=note, metrics_after=m_after)
        log.append(entry)
        cur = cur2
        _emit_step(i, step.tool, step.why, note, "applied", m_after)
    return cur, log


def run_agent_edit(motion: np.ndarray, a: int, b: int, instruction: str,
                   generator=None, *, beats=None, api_key: str | None = None,
                   blend_frames: int = 15, k: int = 3, max_cycles: int = 1, max_refine: int = 2,
                   progress_cb=None, context=None) -> WindowEditResult:
    """Plan → execute (with per-step guardrail) → verify EVERY requested metric → **refine**.

    The metrics the user asked to move are extracted deterministically from the instruction
    (``_requested_metrics``) so verification checks what the USER wanted -- not just whatever single
    metric the planner happened to pick. An attempt succeeds only if no requested metric regressed
    and each improved (or is already at its ceiling). On a miss, every unmet metric is fed back to
    the planner, which revises; we keep the best attempt by summed improvement. The returned
    :class:`WindowEditResult` carries a structured ``trace`` (planner plan + executor steps incl.
    rejections + verify checks, per attempt) for the expandable UI.
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
    base_clip = np.ascontiguousarray(motion[a:b], dtype=np.float32)
    before = window_metrics(base_clip, wbeats)
    a_sec, b_sec = a / 30.0, b / 30.0
    ctx = {"wbeats": wbeats, "a": a, "b": b, "generator": generator, "context": context,
           "blend_frames": blend_frames, "k": k}

    # The planning agent DECLARES the goals it will be graded on (semantic reasoning over the request).
    # The deterministic keyword matcher is only a safety net: it contributes any obvious metric the
    # agent dropped, and is the sole source offline (no api key). Neither has to be exhaustive alone.
    plan = plan_edit(instruction, before, a_sec, b_sec, api_key=api_key)
    goals = _merge_goals(list(plan.goals), _requested_metrics(instruction))
    ctx["goals"] = goals                               # let regenerate rank seeds by the declared goals
    goals_json = [{"metric": m, "dir": d, "label": lbl} for m, d, lbl in goals]

    _emit({"phase": "plan", "summary": plan.summary,
           "steps": [{"tool": s.tool, "why": s.why, "params": s.params} for s in plan.steps],
           "goals": goals_json, "planner": plan.planner, "planner_note": plan.planner_note,
           "metrics_before": before})

    trace: dict = {"instruction": instruction, "goals": goals_json,
                   "planner": plan.planner, "planner_note": plan.planner_note, "attempts": []}
    full_log: list = []
    plan_summary0 = plan.summary                       # what the agent proposed (kept for the UI header)
    total_attempts = (max(0, int(max_refine)) + 1) if goals else 1

    # Baseline = keep the window untouched. An attempt is adopted only if it BEATS this on the goal
    # metrics, so the agent can never ship an edit that is worse than the original -- e.g. when beat
    # alignment is already tight and any change (plus the edge-blend) would only lower it.
    if goals:
        ok0, checks0, _ = _verify_goals(goals, before, before)
        best = (0.0, motion, before, ok0, True, checks0, 0, base_clip)   # (reward, spliced, after, ok, no_reg, checks, cycle, win_cur)
    else:
        best = None                                     # a goalless op (reverse/mirror) always applies

    for cycle in range(total_attempts):
        cur, step_log = _execute_plan(base_clip, plan, ctx, wbeats, cycle=cycle + 1, emit=_emit)
        full_log.extend(step_log)
        # Verify on the SPLICED window -- exactly what the user ends up with (edge cross-fade included),
        # not the raw edited clip. crossfade_edit blends the window's edges back toward its OWN original
        # boundary, so an unchanged edit is a true no-op (no phantom energy/BAS loss) and a real edit's
        # effect survives the splice -- unlike neighbour-snap splicing, which hid the earlier regression.
        spliced_try = crossfade_edit(motion, a, b, cur, blend_frames=blend_frames)
        after = window_metrics(spliced_try[a:b], wbeats)
        if goals:
            ok, checks, verdict = _verify_goals(goals, before, after)
            reward = _reward_goals(goals, before, after)
            no_reg = not any(c.get("status") == "regressed" for c in checks)
        else:
            ok, checks, verdict, reward, no_reg = True, [], "applied the planned edit", 0.0, True
        _emit({"phase": "verify", "cycle": cycle + 1, "ok": ok, "feedback": verdict,
               "checks": checks, "metrics_after": after})
        trace["attempts"].append({
            "n": cycle + 1,
            "plan": {"summary": plan.summary, "planner": plan.planner,
                     "planner_note": plan.planner_note,
                     "steps": [{"tool": s.tool, "params": s.params, "why": s.why} for s in plan.steps]},
            "steps": step_log,
            "verify": {"ok": ok, "checks": checks, "verdict": verdict},
        })
        if best is None or _prefer(ok, no_reg, reward, best[3], best[4], best[0]):
            best = (reward, spliced_try, after, ok, no_reg, checks, cycle + 1, cur)
        if ok or cycle + 1 >= total_attempts:
            break
        # ---- refine: feed EVERY unmet metric back to the planner ----
        misses = [c for c in checks if not c["met"]]
        fb = {"prev_summary": plan.summary,
              "prev_steps": [{"tool": s.tool, "params": s.params} for s in plan.steps],
              "misses": misses, "cycle": cycle + 1}
        plan = plan_edit(instruction, after, a_sec, b_sec, api_key=api_key, feedback=fb, goals=goals)
        _emit({"phase": "refine", "cycle": cycle + 2, "summary": plan.summary,
               "planner": plan.planner, "planner_note": plan.planner_note,
               "steps": [{"tool": s.tool, "why": s.why} for s in plan.steps]})

    reward, spliced, after, ok, no_reg, checks, win_cycle, win_cur = best
    n_attempts = len(trace["attempts"])
    kept_original = win_cycle == 0

    # Quality guard: unless the user asked for sharper, don't ship a jitterier dance than we started
    # with. Amplitude/beat-align edits raise jerk; pull it back toward baseline with a light smooth
    # that keeps every declared goal met. (No-op edits and goalless ops are skipped.)
    polished = _smoothness_polish(win_cur, motion, a, b, goals, before, wbeats, blend_frames) \
        if (goals and not kept_original) else None
    if polished is not None:
        spliced, after, checks, pol_note = polished
        ok = True                                          # the polish only applies if goals stay met
        polish_step = {"cycle": win_cycle, "step": len(trace["attempts"][win_cycle - 1]["steps"]) + 1,
                       "tool": "smooth", "params": {}, "why": "quality guard: keep it smooth",
                       "note": pol_note, "status": "applied", "target": ["jerk", "down"],
                       "metrics_before": trace["attempts"][win_cycle - 1]["steps"][-1].get("metrics_after", before)
                       if trace["attempts"][win_cycle - 1]["steps"] else before,
                       "metrics_after": after, "polish": True}
        full_log.append(polish_step)
        trace["attempts"][win_cycle - 1]["steps"].append(polish_step)
        trace["attempts"][win_cycle - 1]["verify"]["checks"] = checks
        _emit({"phase": "step", "cycle": win_cycle, "step": polish_step["step"],
               "n_steps": polish_step["step"], "tool": "smooth", "why": polish_step["why"],
               "note": pol_note, "status": "applied", "metrics": after})

    refined = f" (refined {n_attempts - 1}x)" if n_attempts > 1 else ""
    if kept_original:
        detail = ", ".join(f"{lbl} {before.get(m, 0.0):.3f}" for m, _d, lbl in goals)
        feedback = f"left unchanged \u2014 no edit beat the current {detail} without hurting it" + refined
    else:
        parts = "  ".join(f"{c['label']} {c['before']:.3f}\u2192{c['after']:.3f}" for c in checks) or "applied"
        feedback = (parts if ok else f"couldn't fully satisfy: {parts}") + refined
    trace["final"] = {"ok": ok, "verdict": feedback, "attempts": n_attempts, "checks": checks,
                      "kept_original": kept_original, "metrics_before": before, "metrics_after": after}
    _emit({"phase": "done", "ok": ok, "metrics_after": after, "feedback": feedback,
           "summary": plan_summary0})

    goal = EditGoal(objective="agent", backbone="agent", magnitude=0.5, raw=instruction)
    return WindowEditResult(ok, goal, (a, b), spliced, before, after, backbone="agent",
                            chosen_seed=None, cycles=full_log, feedback=feedback,
                            log=full_log, agent_summary=plan_summary0, trace=trace)
