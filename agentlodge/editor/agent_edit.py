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
import os
import re
from dataclasses import dataclass, field

import numpy as np

from agentlodge.dance.transition import (
    accentuate,
    beat_align_warp,
    crossfade_edit,
    mirror,
    retrograde,
    splice_window,
    temporal_smooth,
)
from agentlodge.editor.motion_bank import default_motion_bank, normalize_name, verify_applied_motion
from agentlodge.editor.window_edit import (
    EditGoal,
    WindowEditResult,
    _window_beats,
    parse_window_instruction,
    window_metrics,
)

logger = logging.getLogger(__name__)


# ============================================================================ toolbox
# One dispatch maps each metric with a monotone lever to its (apply, strengths). Both the planner
# (_METRIC_TOOL) and the post-regenerate guarantee (_reach_goals_after_regen) use it, so there is no
# per-metric special-casing beyond this single table.
def _lever_for(metric: str, direction: str, ctx: dict):
    """(apply(clip, strength) -> clip, ascending strengths) for the deterministic monotone lever that
    moves (metric, direction), or (None, ()) when the metric has no lever (e.g. foot contact)."""
    up = str(direction).lower() != "down"

    def _taper(clip):
        return int(min(6, max(1, clip.shape[0] // 6)))

    if metric == "energy":
        strengths = (1.15, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4, 2.7) if up else (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
        return (lambda clip, s: accentuate(clip, s, baseline_win=13, taper_frames=_taper(clip), trans_gain=0.6)), strengths
    if metric == "jerk" and up:                              # sharper: amplify the fast band
        return (lambda clip, s: accentuate(clip, s, baseline_win=5, taper_frames=_taper(clip), trans_gain=0.6)), (1.2, 1.4, 1.6, 1.8, 2.0, 2.4)
    if metric == "jerk":                                     # smoother: low-pass
        return (lambda clip, s: temporal_smooth(clip, s)), (0.2, 0.35, 0.5, 0.7, 0.9)
    if metric == "bas":                                      # tighter to the beat
        return (lambda clip, s: beat_align_warp(clip, ctx["wbeats"], strength=1.0, passes=int(s))), (3, 4, 5, 6, 8)
    return None, ()


def _tool_beat_align(clip, ctx, *, strength: float = 1.0, passes: int = 3):
    strength = float(np.clip(strength, 0.0, 1.0))
    passes = int(np.clip(passes, 1, 8))
    out = beat_align_warp(clip, ctx["wbeats"], strength=strength, passes=passes)
    return out, f"snapped the window's accents onto the music beats (strength {strength:.1f})"


def _tool_energy(clip, ctx, *, direction: str = "up", amount: float = 0.5):
    """Deterministic, GUARANTEED energy lever: scale the motion's dynamics about its own smooth
    baseline (see transition.accentuate). Monotone in the energy metric, so 'more/less energetic' is
    always satisfiable without regenerating -- and it keeps the SAME choreography and tempo (no
    sped-up-copy look), just bigger or smaller."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if str(direction).lower() == "down":
        gain = 1.0 - 0.45 * amount
        verb = "calmed"
    else:
        gain = 1.0 + 0.85 * amount
        verb = "energized"
    taper = int(min(6, max(1, clip.shape[0] // 6)))
    out = accentuate(clip, gain, baseline_win=13, taper_frames=taper, trans_gain=0.6)
    return out, f"{verb} the motion x{gain:.2f} (accentuated the dance dynamics about its smooth baseline)"


def _tool_smooth(clip, ctx, *, amount: float = 0.6):
    out = temporal_smooth(clip, float(np.clip(amount, 0.0, 1.0)))
    return out, f"low-pass smoothed the motion (amount {float(amount):.1f}) to reduce jerk"


def _tool_sharpen(clip, ctx, *, amount: float = 0.6):
    """Amplify the FAST movement band (narrow baseline) for snappier, more staccato attack -- raises
    jerk by construction, so 'sharper/punchier' is always satisfiable without regenerating."""
    amount = float(np.clip(amount, 0.0, 1.2))
    gain = 1.0 + 0.7 * amount
    taper = int(min(6, max(1, clip.shape[0] // 6)))
    out = accentuate(clip, gain, baseline_win=5, taper_frames=taper, trans_gain=0.6)
    return out, f"sharpened the attack x{gain:.2f} (amplified the fast band for snappier hits)"


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


def _tool_motion_bank(clip, ctx, *, motion_id: str, mode: str = "replace",
                      anchor: str | None = None, mirror: bool = False,
                      intensity: float = 0.5, repeats: int = 1):
    bank = default_motion_bank()
    spec = bank.resolve(motion_id)
    dropped = []
    # The bank refuses repetition or mirroring it cannot do, which would sink the whole edit.
    # Doing the nearest valid thing and saying so beats handing the user back an unchanged window.
    if int(repeats) > 1 and not spec.repeatable:
        repeats = 1
        dropped.append("repetition")
    if bool(mirror) and not spec.mirrorable:
        mirror = False
        dropped.append("mirroring")
    out, report = bank.apply(
        clip, motion_id, beats=ctx.get("bank_beats", ctx.get("wbeats")),
        beat_strengths=ctx.get("bank_beat_strengths"),
        mode=mode, anchor=anchor,
        mirror=bool(mirror), intensity=float(intensity), repeats=int(repeats),
        blend_frames=int(ctx.get("blend_frames", 8)),
    )
    report["dropped"] = dropped
    ctx["_motion_bank_report"] = report
    # The bank now layers only the channels the named action owns onto this exact host window.
    # It is therefore an edit of the original choreography, not a foreign generated segment that
    # needs to be chained a second time through ``splice_window``. The old double-splice changed
    # root motion and contacts again after composition and was the path the static bank tests missed.
    ctx["_foreign_motion"] = False
    note = (
        f"placed {report['name']} as a {report['mode']} edit, aligned at frame "
        f"{report['event_frame']} ({report['source']}, {report['license']})"
    )
    if dropped:
        note += f"; {spec.name} does not support {' or '.join(dropped)}, so it plays once as authored"
    return out, note


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
        "scale the motion's DYNAMICS up ('more energetic/bigger/stronger/intense') or down "
        "('calmer/softer/smaller') about its OWN smooth baseline -- same choreography and tempo, just "
        "bigger or smaller. Moves the energy metric MONOTONICALLY, so it ALWAYS satisfies a more/less "
        "energetic request without regenerating (fast, deterministic)",
        'params: {"direction": "up"|"down", "amount": 0..1}'),
    "smooth": ToolSpec(_tool_smooth,
        "low-pass the motion for flowing/graceful movement; lowers jerk",
        'params: {"amount": 0..1}'),
    "sharpen": ToolSpec(_tool_sharpen,
        "amplify the fast movement band for snappier/punchier/staccato attack; raises jerk "
        "(guaranteed), no regeneration",
        'params: {"amount": 0..1}'),
    "mirror": ToolSpec(_tool_mirror, "flip the window left<->right", "params: {}"),
    "reverse": ToolSpec(_tool_reverse, "play the window backward (retrograde)", "params: {}"),
    "regenerate": ToolSpec(_tool_regenerate,
        "sample genuinely NEW choreography for THIS window from a dance backbone (edge=sharp/punchy, "
        "lodge=smooth/flowing); the VARIETY tool -- use for 'different/new moves', 'freestyle', 'mix "
        "it up', 'surprise me', i.e. when the user wants DIFFERENT movement, not a metric tweak of the "
        "current dance",
        'params: {"backbone": "lodge"|"edge"|"auto", "energy": 0..1}'),
    "motion_bank": ToolSpec(_tool_motion_bank,
        "retrieve a specific common action from the curated named-motion bank, then fit it to this "
        "window while preserving the song duration. Use for clap, jump, wave, point, punch, steps, "
        "turns, body roll, crouch, rise, and other listed named actions",
        'params: {"motion_id": "<bank id>", "mode": "replace"|"insert", '
        '"anchor": optional "early"|"center"|"late"|"beat" (omit for the motion default), "mirror": bool, '
        '"intensity": 0..1, "repeats": 1..8}'),
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
    if tool == "smooth":
        return ("jerk", "down")
    if tool == "energy":
        return ("energy", "down" if str(params.get("direction", "up")).lower() == "down" else "up")
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
      "more dynamic", "high energy",
      # "maintain / match / keep the current energy" is an energy FLOOR -> treat as energy-up so the
      # agent dials a fresh regenerated take up to at least the original level.
      "maintain the energy", "maintains the energy", "maintain the current energy",
      "maintains the current energy", "match or exceed", "matches or exceed", "match the energy",
      "matches the energy", "keep the energy", "keep the current energy", "same energy", "same level")),
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


def _explicit_window_beat_request(instruction: str) -> bool:
    """Distinguish a whole-window timing request from merely placing an action on a beat."""
    s = " " + normalize_name(instruction) + " "
    if not any(word in s for word in (" beat ", " timing ", " rhythm ", " sync ")):
        return False
    scoped = (
        " rest ",
        " everything ",
        " whole window ",
        " entire window ",
        " whole dance ",
        " entire dance ",
        " overall ",
        " all the motion ",
        " all of it ",
    )
    timing_action = (
        " tighten ",
        " tighter ",
        " align ",
        " lock ",
        " sync ",
        " put everything ",
    )
    compound = any(joiner in s for joiner in (" and ", " then ", " also "))
    return any(phrase in s for phrase in scoped) or (
        (compound or " window " in s)
        and any(phrase in s for phrase in timing_action)
    )


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


def _prefer(ok_a: bool, guard_a: bool, no_reg_a: bool, reward_a: float,
            ok_b: bool, guard_b: bool, no_reg_b: bool, reward_b: float) -> bool:
    """True if attempt A should replace the current best B. Ranking, most important first:
    (1) meets EVERY declared goal, (2) introduces NO artifact (jerk/foot guard stays clean),
    (3) regresses NO declared goal, (4) higher summed reward. So among goal-meeting attempts we
    prefer the one that does not jitter or foot-skate, and an edit that misses the goals AND jitters
    never beats simply keeping the original (which regresses nothing)."""
    return ((int(ok_a), int(guard_a), int(no_reg_a), reward_a)
            > (int(ok_b), int(guard_b), int(no_reg_b), reward_b))


# Implicit quality guards: smoothness (jerk) and foot-contact must never be made WORSE by an edit,
# even when the user did not mention them -- this is what stops a beat/energy edit from shipping a
# jittery or foot-skating window. A guard is skipped for a metric the user explicitly asked to move
# (e.g. "sharper" = jerk up), which is then a goal, not a guard.
_ARTIFACT_GUARDS: list[tuple[str, str, str]] = [("jerk", "down", "smoothness"), ("foot", "up", "foot contact")]


def _active_guards(goals) -> list[tuple[str, str, str]]:
    requested = {m for m, _d, _l in goals}
    return [(m, d, l) for (m, d, l) in _ARTIFACT_GUARDS if m not in requested]


def _jerk_ceiling(before: dict, after: dict, goals) -> float:
    """Absolute jerk value the artifact guard tolerates before calling the edit 'jittery'.

    Bigger, more energetic dancing is legitimately jerkier (larger, faster, snappier moves have a
    higher third derivative), so when the user asked to RAISE energy the jerk budget scales with the
    energy actually delivered -- otherwise the quality guard would smooth the very energy the user
    requested straight back out (the "more energetic did almost nothing" bug). When energy is NOT a
    goal the budget is tight (near baseline), so an incidental jerk rise from beat_align is still
    polished away.
    """
    base_j = float(before.get("jerk", 0.0))
    up_energy = any(m == "energy" and d == "up" for m, d, _l in goals)
    eb = float(before.get("energy", 0.0))
    if up_energy and eb > 1e-6:
        ratio = float(after.get("energy", 0.0)) / eb           # energy delivered (>= 1 when raised)
        return base_j * min(2.6, 1.2 + 1.6 * max(0.0, ratio - 1.0))
    return base_j * 1.08


def _guard_report(before: dict, after: dict, goals) -> tuple[bool, float, list]:
    """(clean, penalty, checks) for the artifact guards. ``clean`` is False if the edit regressed
    smoothness or foot-contact; ``penalty`` is the summed regression (in tolerance units) to subtract
    from the attempt's reward; ``checks`` are UI rows flagged ``guard=True``. The smoothness (jerk)
    guard uses an energy-proportional ceiling (:func:`_jerk_ceiling`) so a requested energy increase
    is not treated as an artifact, while incidental jitter still is."""
    clean, penalty, checks = True, 0.0, []
    jceil = _jerk_ceiling(before, after, goals)
    for metric, good_dir, label in _active_guards(goals):
        b, a = float(before.get(metric, 0.0)), float(after.get(metric, 0.0))
        if metric == "jerk":
            over = a > jceil + _tol("jerk", b)                 # only jerk beyond the budget is an artifact
            if over:
                clean = False
                penalty += (a - jceil) / (_tol(metric, b) or 1.0)
                status = "regressed"
            elif a <= b:
                status = _classify(metric, b, a, good_dir)     # genuinely smoother (or unchanged)
            else:
                status = "held"                                # rose but within the energy-proportional budget
        else:
            status = _classify(metric, b, a, good_dir)
            if status == "regressed":
                clean = False
                penalty += abs(a - b) / (_tol(metric, b) or 1.0)
        checks.append({"metric": metric, "label": label, "dir": good_dir,
                       "before": round(b, 4), "after": round(a, 4), "status": status,
                       "met": status != "regressed", "guard": True})
    return clean, penalty, checks


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


def _bank_blend_frames(blend_frames: int, n: int, report: dict | None):
    """Keep the outer editor crossfade outside the named action's semantic range."""
    if report is None:
        return int(blend_frames)
    start, end = (int(x) for x in report.get("action_range", (0, n)))
    start = int(np.clip(start, 0, n))
    end = int(np.clip(end, start, n))
    requested = int(max(0, blend_frames))
    return min(requested, start), min(requested, n - end)


def _crossfade_result(motion, a, b, window, blend_frames, bank_report=None):
    return crossfade_edit(
        motion,
        a,
        b,
        window,
        blend_frames=_bank_blend_frames(blend_frames, b - a, bank_report),
    )


def _smoothness_polish(
    win_cur,
    motion,
    a,
    b,
    goals,
    before,
    wbeats,
    blend_frames,
    *,
    bank_report=None,
):
    """Quality guard: keep the edit from making the dance JITTERY without smoothing away the effect
    the user asked for. The beat time-warp and (for non-energy edits) the accentuate lever raise jerk
    as a side effect, so -- UNLESS the user asked for sharper (jerk up) -- apply the smallest light
    temporal smooth that pulls the window's jerk back under an energy-proportional ceiling
    (:func:`_jerk_ceiling`) WITHOUT regressing any declared goal. When the user asked for MORE energy,
    that ceiling is generous (bigger moves are legitimately jerkier), so the energy survives; when they
    did not, it is near baseline, so incidental jitter is removed. Returns (spliced, after, checks,
    note) or None when no polish is needed or none preserves the goals."""
    # Named actions carry an exact event pose. Generic whole-window smoothing can open a clap or
    # lower a jump while still satisfying an unrelated metric goal, so leave their quality warning
    # visible rather than silently smoothing away the requested action.
    if bank_report is not None:
        return None
    if any(m == "jerk" and d == "up" for m, d, _lbl in goals):
        return None
    spliced0 = _crossfade_result(motion, a, b, win_cur, blend_frames)
    m0 = window_metrics(spliced0[a:b], wbeats)
    j0 = m0["jerk"]
    base_j = float(before.get("jerk", 0.0))
    jceil = _jerk_ceiling(before, m0, goals)               # energy-proportional budget for this edit
    if base_j <= 0 or j0 <= jceil:                         # already within the jitter budget
        return None
    chosen = None
    for amt in (0.06, 0.14, 0.24, 0.34):
        cand = temporal_smooth(win_cur, amt)
        spl = _crossfade_result(motion, a, b, cand, blend_frames)
        m = window_metrics(spl[a:b], wbeats)
        ok, checks, _ = _verify_goals(goals, before, m) if goals else (True, [], "")
        if not ok:
            break                                          # this much smoothing broke a goal -> stop
        chosen = (spl, m, checks, amt)
        if m["jerk"] <= max(jceil, base_j * 1.05):
            break                                          # within budget -> stop (keep the energy)
    if chosen is None:
        return None
    spl, m, checks, amt = chosen
    return spl, m, checks, (f"smoothed (amount {amt:.2f}) to preserve quality, "
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


# objective (keyword parser) -> a single-tool plan, used as the offline fallback. Metric intents map
# to the DETERMINISTIC, monotone levers (energy/sharpen/beat_align/smooth) which are guaranteed and
# instant; only pure variety ("different / freestyle") uses the generative regenerate tool.
_OBJ_PLAN = {
    "more_on_beat": ("beat_align", {"strength": 1.0}, "bas", "up", "tighten timing onto the beats"),
    "calmer":       ("energy", {"direction": "down", "amount": 0.6}, "energy", "down", "calm the motion down"),
    "more_energetic": ("energy", {"direction": "up", "amount": 0.7}, "energy", "up", "energize the motion"),
    "smoother":     ("smooth", {}, "jerk", "down", "smooth out the motion"),
    "sharper":      ("sharpen", {"amount": 0.7}, "jerk", "up", "sharpen the attack"),
    "reverse":      ("reverse", {}, None, None, "play it backward"),
    "mirror":       ("mirror", {}, None, None, "flip left-right"),
    "exaggerate":   ("energy", {"direction": "up", "amount": 0.9}, "energy", "up", "make the moves bigger"),
}

# (metric, direction) -> the tool that moves it, for composing a plan from the requested metrics.
# Every metric now has a DETERMINISTIC, monotone lever, so a metric-directed request never depends on
# what the diffusion backbone happens to sample (the old regenerate route was music-bound and could
# silently "hold"). Variety requests carry no metric goal and use regenerate separately.
_METRIC_TOOL = {
    ("bas", "up"): ("beat_align", {"strength": 1.0}),
    ("energy", "up"): ("energy", {"direction": "up", "amount": 0.7}),
    ("energy", "down"): ("energy", {"direction": "down", "amount": 0.6}),
    ("jerk", "down"): ("smooth", {"amount": 0.6}),
    ("jerk", "up"): ("sharpen", {"amount": 0.7}),
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


def _requested_motion_anchor(instruction: str) -> str | None:
    """Return only a placement the user explicitly requested."""
    text = f" {normalize_name(instruction)} "
    if any(phrase in text for phrase in (" before ", " ahead of ")):
        return "early"
    if any(phrase in text for phrase in (" after ", " following ")):
        return "late"
    if any(phrase in text for phrase in (
        " at the start ", " at start ", " at the beginning ", " at beginning ",
    )):
        return "start"
    if any(phrase in text for phrase in (" at the end ", " at end ")):
        return "end"
    if any(phrase in text for phrase in (" in the middle ", " at the center ", " in the center ")):
        return "center"
    if any(phrase in text for phrase in (
        " on beat ", " on the beat ", " to the beat ", " nearest beat ", " hit the beat ",
    )):
        return "beat"
    return None


def _apply_motion_anchor_defaults(plan: AgentPlan, instruction: str) -> AgentPlan:
    """Make named-action timing deterministic instead of accepting an LLM's arbitrary anchor."""
    requested = _requested_motion_anchor(instruction)
    bank = default_motion_bank()
    for step in plan.steps:
        if step.tool != "motion_bank":
            continue
        motion_id = step.params.get("motion_id")
        try:
            spec = bank.resolve(motion_id)
        except (TypeError, ValueError):
            continue
        step.params = dict(step.params)
        step.params["anchor"] = requested or spec.default_anchor
    return plan


def _keyword_plan(instruction: str, feedback: dict | None = None) -> AgentPlan:
    """Offline planner: compose one tool per requested metric (multi-goal aware).

    "calmer but keep it tight to the beat" -> energy(down) + beat_align, rather than a single tag.
    Beat-align is ordered LAST so it re-times after any amplitude change. Falls back to the
    single-objective mapping when no metric keyword is present (e.g. "flip and reverse").
    """
    bank_spec = default_motion_bank().match_instruction(instruction)
    if bank_spec is not None:
        text = f" {normalize_name(instruction)} "
        explicit_insert = any(phrase in text for phrase in (
            " insert ", " before ", " after ", " between ", " ahead of ", " following ",
        ))
        anchor = _requested_motion_anchor(instruction) or bank_spec.default_anchor
        repeats = 3 if any(x in text for x in (" three times ", " thrice ")) else (
            2 if any(x in text for x in (" twice ", " two times ")) else 1
        )
        mirror_requested = bool(
            bank_spec.mirrorable
            and any(x in text for x in (" left ", " left side ", " left hand ", " to the left "))
        )
        intensity = 0.8 if any(x in text for x in (
            " big ", " strong ", " explosive ", " dramatic ", " high ", " deep ",
        )) else 0.5
        mode = "insert" if explicit_insert else "replace"
        params = {
            "motion_id": bank_spec.id, "mode": mode, "anchor": anchor,
            "mirror": mirror_requested, "intensity": intensity, "repeats": repeats,
        }
        steps = []
        goals = []
        if _explicit_window_beat_request(instruction):
            steps.append(PlanStep(
                "beat_align",
                {"strength": 1.0},
                "tighten the rest of the selected window onto the beats",
            ))
            goals.append(("bas", "up", METRIC_LABEL["bas"]))
        steps.append(PlanStep(
            "motion_bank",
            params,
            f"retrieve the named {bank_spec.name} action",
        ))
        return AgentPlan(
            summary=f"{mode} {bank_spec.name.lower()} in the selected window",
            steps=steps,
            goals=goals,
        )

    reqs = _requested_metrics(instruction)
    cycle = int(feedback.get("cycle", 1)) if feedback else 0
    s = " " + instruction.lower().strip() + " "
    mag = 0.8 if any(w in s for w in (" much ", " way ", " a lot ", " very ", " really ", " super ")) else 0.5
    # Does the user want genuinely NEW motion (not just a reshape of the current moves)?
    wants_new = any(w in s for w in (
        "different", "freestyle", "fresh", "new move", "new choreo", "new motion", "mix it up",
        "mix things up", "switch it up", "surprise me", "something else", "vary it", "more interesting",
        "change it up", "remix", "reinvent", "create a new", "generate", "choreograph", "come up with",
        "make a new", "make new", "brand new"))
    if wants_new:
        # regenerate FIRST (the only tool that creates new motion), then dial any requested metric
        # onto the fresh clip. Pure variety (no metric) -> empty goals; compound -> the metric goals.
        rp = {"backbone": "auto", "energy": 0.7 if any(m == "energy" and d == "up" for m, d, _l in reqs) else 0.5}
        if feedback:
            rp = _escalate_params("regenerate", rp, cycle)
        steps = [PlanStep("regenerate", rp, "sample genuinely new choreography for this window")]
        for metric, direction, label in reqs:
            tool, params = _METRIC_TOOL[(metric, direction)]
            p = dict(params)
            if "amount" in p:
                p["amount"] = mag
            steps.append(PlanStep(tool, p, f"dial the fresh motion's {label} {direction}"))
        summary = "regenerate new choreography" + (
            " and " + ", ".join(dict.fromkeys(f"{d} {lbl}" for _m, d, lbl in reqs)) if reqs else "")
        return AgentPlan(summary=("push harder: " if feedback else "") + summary, steps=steps,
                         expect_metric=(reqs[0][0] if reqs else None),
                         expect_dir=(reqs[0][1] if reqs else None), goals=list(reqs))
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
    _specs = default_motion_bank().specs
    bank_lines = ", ".join(f"{s.id} ({s.name}; aliases: {', '.join(s.aliases)})" for s in _specs)
    repeatable = ", ".join(s.id for s in _specs if s.repeatable) or "none"
    mirrorable = ", ".join(s.id for s in _specs if s.mirrorable) or "none"
    beat_default = ", ".join(s.id for s in _specs if s.default_anchor == "beat") or "none"
    prompt = (
        "You are a dance-motion editing agent. The user selected the window "
        f"[{a_sec:.1f}s..{b_sec:.1f}s] of a dance and asked: \"{instruction}\".\n"
        f"Current window metrics -> energy(intensity): {ctx_metrics.get('energy')}, "
        f"beat_alignment(BAS 0-1, higher=tighter): {ctx_metrics.get('bas')}, "
        f"jerk(lower=smoother): {ctx_metrics.get('jerk')}, "
        f"foot_contact(0-1, higher=less sliding): {ctx_metrics.get('foot')}.\n\n"
        "Compose 1-3 of these tools, in order, to satisfy the request:\n" + tool_lines + "\n\n"
        "Named-motion vocabulary for motion_bank:\n" + bank_lines + "\n"
        "For a recognized named action, use motion_bank instead of regenerate. Use mode=replace for "
        "ordinary requests such as 'add a clap here'. Use mode=insert only for explicit relational "
        "wording such as insert/before/after/between. Insert still preserves the selected window and "
        "song duration. Never invent a motion_id outside this vocabulary.\n"
        f"Only these accept repeats>1: {repeatable}. Only these accept mirror=true: {mirrorable}. "
        "Asking for either outside those lists is dropped and the action simply plays once, so "
        "prefer a motion that supports repetition when the user asks for something twice.\n"
        f"These named actions are beat-hit motions and default to anchor=beat: {beat_default}. "
        "Their clap, impact, accent, or arrival pose must land on the strongest feasible musical beat "
        "unless the user explicitly asks for early, late, before, after, start, center, or end. "
        "For every named action, omit anchor when the user gives no placement; MAESTRO applies the "
        "motion's manifest default deterministically.\n\n"
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
        '"tighten it to the beat" -> [{"metric":"bas","direction":"up"}]; '
        '"create a new motion that matches or exceeds the energy" -> [{"metric":"energy","direction":"up"}] '
        "(regenerate for the NEW motion, energy-up to hit the target; NOTE energy ONLY, NOT beat); "
        '"different / fresh / freestyle / surprise me / mix it up" -> []; '
        '"reverse this"/"mirror it" -> []. '
        "Pick the FEWEST tools that match the user's intent, by meaning not keywords. Every metric has "
        "a DEDICATED, deterministic lever that is guaranteed to move it the right way -- prefer these "
        "over regenerate for any metric-directed request:\n"
        "1) TIMING (tighter / on the beat / in time / lock to the grid) -> 'beat_align' ALONE, "
        "goal [bas up]. Do NOT add an energy or jerk goal.\n"
        "2) SMOOTHNESS (smoother / flowing / less jitter) -> 'smooth' ALONE, goal [jerk down].\n"
        "3) ENERGY / INTENSITY (more energetic / bigger / stronger / intense -> 'energy' direction=up; "
        "calmer / softer / smaller / mellower -> 'energy' direction=down). The 'energy' lever scales the "
        "SAME choreography's dynamics up or down about its own smooth baseline (no tempo change, no "
        "sped-up-copy look) and moves the energy metric monotonically, so it ALWAYS satisfies the "
        "request. goal [energy up] or [energy down]. Amount 0.6-0.9 for 'much/way', ~0.5 otherwise.\n"
        "4) SNAP (snappier / punchier / staccato / crisper / sharper) -> 'sharpen' ALONE, goal [jerk up].\n"
        "5) EXACT TRANSFORM (reverse / mirror / flip) -> that ONE tool, goals MUST be [].\n"
        "6) VARIETY / NEW MOTION -- any request to have DIFFERENT movement or to CREATE / GENERATE / "
        "CHOREOGRAPH / COME UP WITH / MAKE a NEW motion / new choreography / new moves (also: different, "
        "fresh, freestyle, mix it up, surprise me, more interesting, replace/redo the moves) -> "
        "'regenerate' is REQUIRED. It is the ONLY tool that changes WHAT the body does: it samples "
        "genuinely NEW choreography from the backbones (edge = punchy, lodge = smooth; auto = agent "
        "picks). The energy/smooth/sharpen levers only RESHAPE the existing moves -- they never create "
        "new motion, so a 'create a new motion' request is WRONG without regenerate. For PURE VARIETY "
        "(no metric mentioned) the goal list is EMPTY [] (any fresh sample is a success).\n"
        "COMBINING (very important): when the user wants a NEW motion AND a metric target (e.g. 'create "
        "a new motion that matches or exceeds the energy', 'give me different, more energetic moves', "
        "'fresh choreography but calmer', 'new moves, snappier') you MUST use TWO steps: 'regenerate' "
        "FIRST (to create the new motion) THEN the metric lever ('energy' up/down, or 'sharpen') to dial "
        "the fresh clip onto the target. goal = the metric ONLY (e.g. [energy up]) and NOTHING the user "
        "did not mention (do NOT add beat/bas unless they said beat/timing/sync). NEVER satisfy a "
        "'create / generate a new motion' request with the energy lever alone -- that only resizes the "
        "OLD moves and creates nothing new. A light 'smooth' STEP (amount ~0.1-0.2) after a regenerate "
        "is a STEP, not a goal -- do NOT put jerk in goals for it. NEVER add a goal the user did not "
        "explicitly ask for: an extra goal conflicts with the real one and gets the whole edit rejected "
        "(e.g. adding beat to 'more energetic' makes it do nothing)."
    )
    if goals:                                              # refine: remind of the graded constraints
        gl = ", ".join(f"{lbl} must go {d}" for _m, d, lbl in goals)
        prompt += ("\n\nThe user's request REQUIRES: " + gl + ". Every one of these will be checked; "
                   "a plan that lets any of them get WORSE is rejected. If a metric is already high "
                   "(e.g. beat alignment > 0.9) at minimum do not reduce it. When a tool's side effect "
                   "could hurt one of these (e.g. regenerating can shift timing), add the corrective "
                   "tool too (e.g. beat_align LAST).")
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
    model = os.environ.get("AGENTLODGE_PLANNER_MODEL", "gpt-4o")   # stronger reasoning by default
    resp = client.chat.completions.create(
        model=model, max_tokens=400, temperature=(0.5 if feedback else 0.2),
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
            p = _apply_motion_anchor_defaults(p, instruction)
            p.planner, p.planner_note = "llm", "AI agent (LLM reasoning)"
            return p
        except Exception as exc:  # noqa: BLE001 - robust offline fallback
            logger.warning("agent plan via LLM failed (%s); using keyword plan", exc)
            p = _keyword_plan(instruction, feedback=feedback)
            p = _apply_motion_anchor_defaults(p, instruction)
            p.planner = "keyword_fallback"
            p.planner_note = f"offline keyword planner (LLM call failed: {str(exc)[:120]})"
            return p
    p = _apply_motion_anchor_defaults(_keyword_plan(instruction, feedback=feedback), instruction)
    p.planner, p.planner_note = "keyword", "offline keyword planner (no API key configured)"
    return p


_LEVER_TOOLS = {"energy", "beat_align", "smooth", "sharpen"}


def _plan_for_regen(plan: AgentPlan) -> AgentPlan:
    """When the plan REGENERATES, drop the metric-lever steps (energy/beat_align/smooth/sharpen): the
    post-regenerate guarantee (:func:`_reach_goals_after_regen`) dials every declared metric onto the
    target from the FRESH clip, so an LLM-added lever step would only double-amplify. Non-lever steps
    (mirror/reverse) are kept. Plans without a regenerate are returned unchanged."""
    if (
        not any(s.tool == "regenerate" for s in plan.steps)
        or any(s.tool == "motion_bank" for s in plan.steps)
    ):
        return plan
    kept = [s for s in plan.steps if s.tool not in _LEVER_TOOLS]
    if kept != plan.steps:
        plan.steps = kept
    return plan


def _normalize_plan(plan: AgentPlan) -> AgentPlan:
    """Keep the named action last so its event metadata describes the final edited window."""
    plan = _plan_for_regen(plan)
    if any(step.tool == "motion_bank" for step in plan.steps):
        plan.steps = (
            [step for step in plan.steps if step.tool != "motion_bank"]
            + [step for step in plan.steps if step.tool == "motion_bank"]
        )
    return plan


def _reach_goals_after_regen(cur, motion, a, b, goals, before, wbeats, blend_frames, ctx):
    """After a ``regenerate``, DIAL each declared metric's lever on the fresh clip until the SPLICED
    window MEETS the goal versus the original (matches/exceeds for up, matches/undercuts for down),
    escalating the lever's strength until it is met or the lever's range is exhausted.

    This is the agent-level guarantee that a fresh (diffusion) take is put ON the user's target instead
    of being rejected for merely holding the metric ("none of the regenerated takes improved <metric>
    -> kept the original"). It is measured on the SPLICED window -- what the user actually gets -- so
    the crossfade's edge dilution is accounted for (a raw-clip match is not enough). Generalized over
    every metric via :func:`_lever_for`. Returns (cur, log_steps)."""
    steps = []

    def _spliced(clip):
        return window_metrics(crossfade_edit(motion, a, b, clip, blend_frames=blend_frames)[a:b], wbeats)

    for metric, direction, label in goals:
        apply, strengths = _lever_for(metric, direction, ctx)
        if apply is None:
            continue
        m_before = _spliced(cur)
        if _classify(metric, before.get(metric, 0.0), m_before[metric], direction) == "improved":
            continue                                         # this metric already meets the goal
        best_clip, best_val = cur, m_before[metric]
        for s in strengths:
            cand = apply(cur, s)                             # dial from the fresh clip (not compounding)
            v = _spliced(cand)[metric]
            improved_toward = (v > best_val) if direction == "up" else (v < best_val)
            if improved_toward:
                best_clip, best_val = cand, v
            if _classify(metric, before.get(metric, 0.0), v, direction) == "improved":
                best_clip, best_val = cand, v                # reached the goal on the spliced window
                break
        m_after = _spliced(best_clip)
        tool = (_METRIC_TOOL.get((metric, direction)) or (metric,))[0]
        steps.append({"cycle": ctx.get("_cycle", 1), "tool": tool, "params": {"match": True},
                      "why": f"dial {label} onto the target after regenerating",
                      "note": f"dialed the new motion's {label} to {best_val:.3f} (target {before.get(metric, 0.0):.3f})",
                      "status": "applied", "target": [metric, direction],
                      "metrics_before": m_before, "metrics_after": m_after})
        cur = best_clip
    return cur, steps


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
                      f"({m_before[metric]:.3f}\u2192{m_after[metric]:.3f}), kept the previous motion")
            entry.update(status="rejected", note=note, reject_reason=reason)
            log.append(entry); _emit_step(i, step.tool, step.why, reason, "rejected", m_before)
            continue                                        # cur unchanged -> regression discarded
        entry.update(status="applied", note=note, metrics_after=m_after)
        if step.tool == "motion_bank" and ctx.get("_motion_bank_report"):
            entry["motion_bank"] = dict(ctx["_motion_bank_report"])
        log.append(entry)
        cur = cur2
        _emit_step(i, step.tool, step.why, note, "applied", m_after)
    return cur, log


def run_agent_edit(motion: np.ndarray, a: int, b: int, instruction: str,
                   generator=None, *, beats=None, beat_strengths=None, api_key: str | None = None,
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

    from agentlodge.dance.format import to_editor139

    motion = to_editor139(motion)
    L = int(motion.shape[0])
    a, b = int(a), int(b)
    if not (0 <= a < b <= L):
        raise ValueError(f"invalid window [{a}, {b}) for motion of length {L}")
    wbeats = _window_beats(beats, a, b)
    beat_grid = None if beats is None else np.asarray(beats, dtype=float)
    strength_grid = (
        None if beat_strengths is None else np.asarray(beat_strengths, dtype=float).reshape(-1)
    )
    if strength_grid is not None:
        if beat_grid is None or strength_grid.size != beat_grid.size:
            raise ValueError(
                "beat_strengths must contain one value for every beat "
                f"({strength_grid.size} != {0 if beat_grid is None else beat_grid.size})"
            )
    bank_beats = None if beat_grid is None or beat_grid.size == 0 else beat_grid - a
    bank_beat_strengths = (
        None if bank_beats is None or strength_grid is None else strength_grid
    )
    base_clip = np.ascontiguousarray(motion[a:b], dtype=np.float32)
    before = window_metrics(base_clip, wbeats)
    a_sec, b_sec = a / 30.0, b / 30.0
    ctx = {"wbeats": wbeats, "bank_beats": bank_beats,
           "bank_beat_strengths": bank_beat_strengths,
           "a": a, "b": b, "generator": generator, "context": context,
           "blend_frames": blend_frames, "k": k, "base_metrics": before}

    # The planning agent DECLARES the goals it will be graded on (semantic reasoning over the request).
    # The deterministic keyword matcher is only a safety net: it contributes any obvious metric the
    # agent dropped, and is the sole source offline (no api key). Neither has to be exhaustive alone.
    plan = plan_edit(instruction, before, a_sec, b_sec, api_key=api_key)
    planner_goals = list(plan.goals)
    goals = _merge_goals(planner_goals, _requested_metrics(instruction))
    # A named action with anchor=beat has one explicit semantic hit to grade. Whole-window BAS asks
    # whether *all* velocity accents in the selection resemble the beat grid; adding one correctly
    # timed clap can lower that aggregate score by changing unrelated peaks, which made a valid clap
    # report failure (0.528 -> 0.449). The bank report carries the actual event-to-beat error instead.
    planner_declared_bas = any(goal[0] == "bas" for goal in planner_goals)
    if (
        any(s.tool == "motion_bank" and s.params.get("anchor") == "beat" for s in plan.steps)
        and not planner_declared_bas
        and not _explicit_window_beat_request(instruction)
    ):
        goals = [g for g in goals if g[0] != "bas"]
    plan = _normalize_plan(plan)
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

    # A request that REGENERATES wants a genuinely new motion; the untouched original is NOT an
    # acceptable output for it, so it does not seed the baseline (best=None -> the new motion is always
    # adopted, and the post-regen guarantee dials it onto the target). A pure metric tweak (no
    # regenerate) DOES seed the untouched baseline, so the agent can honestly keep the original when no
    # deterministic edit can beat it. Goalless ops (reverse/mirror) always apply.
    wants_new = any(s.tool in ("regenerate", "motion_bank") for s in plan.steps)
    if goals and not wants_new:
        ok0, checks0, _ = _verify_goals(goals, before, before)
        best = (
            0.0, motion, before, ok0, True, True, checks0, 0, base_clip, None,
        )  # reward, spliced, after, ok, guard_ok, no_reg, checks, cycle, win_cur, bank report
    else:
        best = None

    for cycle in range(total_attempts):
        ctx["_cycle"] = cycle + 1
        ctx.pop("_motion_bank_report", None)
        ctx.pop("_foreign_motion", None)
        cur, step_log = _execute_plan(base_clip, plan, ctx, wbeats, cycle=cycle + 1, emit=_emit)
        # Post-regenerate guarantee: dial each goal metric onto the target on the SPLICED window so a
        # fresh take that merely held the metric is put on the user's target instead of being rejected.
        if (
            goals
            and ctx.get("_motion_bank_report") is None
            and any(
                s.get("tool") == "regenerate" and s.get("status") == "applied"
                for s in step_log
            )
        ):
            cur, reach_steps = _reach_goals_after_regen(cur, motion, a, b, goals, before, wbeats,
                                                        blend_frames, ctx)
            for rs in reach_steps:
                _emit({"phase": "step", "cycle": cycle + 1, "step": len(step_log) + 1,
                       "n_steps": len(step_log) + 1, "tool": rs["tool"], "why": rs["why"],
                       "note": rs["note"], "status": "applied", "metrics": rs["metrics_after"]})
                step_log.append(rs)
        full_log.extend(step_log)
        # Verify on the SPLICED window -- exactly what the user ends up with (edge cross-fade included),
        # not the raw edited clip. crossfade_edit blends the window's edges back toward its OWN original
        # boundary, so an unchanged edit is a true no-op (no phantom energy/BAS loss) and a real edit's
        # effect survives the splice -- unlike neighbour-snap splicing, which hid the earlier regression.
        bank_report = ctx.get("_motion_bank_report")
        spliced_try = (
            splice_window(motion, a, b, cur, blend_frames=blend_frames)
            if ctx.get("_foreign_motion")
            else _crossfade_result(
                motion,
                a,
                b,
                cur,
                blend_frames,
                bank_report=bank_report,
            )
        )
        after = window_metrics(spliced_try[a:b], wbeats)
        if goals:
            ok, checks, verdict = _verify_goals(goals, before, after)
            reward = _reward_goals(goals, before, after)
            no_reg = not any(c.get("status") == "regressed" for c in checks)
        else:
            ok, checks, verdict, reward, no_reg = True, [], "applied the planned edit", 0.0, True
        bank_steps = [s for s in step_log if s.get("tool") == "motion_bank"]
        if bank_steps and bank_report is None:
            failure = bank_steps[-1].get("note", "named motion was not applied")
            checks.append({
                "metric": "semantic", "label": "named motion", "dir": "match",
                "before": 0.0, "after": 0.0, "status": "regressed", "met": False,
                "detail": failure,
            })
            ok, no_reg = False, False
            verdict = failure
        elif bank_report is not None:
            semantic = verify_applied_motion(
                spliced_try[a:b],
                bank_report,
                reference=base_clip,
            )
            beat_error = bank_report.get("beat_error_frames")
            beat_ok = beat_error is None or float(beat_error) <= 0.51
            semantic_ok = bool(semantic["ok"] and beat_ok)
            detail = semantic["detail"]
            if beat_error is not None:
                detail += f", event {float(beat_error):.2f} frames from beat"
            semantic_check = {
                "metric": "semantic",
                "label": bank_report["name"],
                "dir": "match",
                "before": 0.0,
                "after": 1.0 if semantic_ok else 0.0,
                "status": "improved" if semantic_ok else "regressed",
                "met": semantic_ok,
                "detail": detail,
            }
            checks.append(semantic_check)
            ok = ok and semantic_ok
            no_reg = no_reg and semantic_ok
            verdict = (
                f"verified {bank_report['name']}: {detail}"
                if semantic_ok else
                f"{bank_report['name']} failed verification: {detail}"
            )
        # artifact guardrail: never make the window jittery / foot-skating, even off-goal. A
        # violation penalises the reward, keeps the loop refining toward a clean solution, and is
        # dispreferred vs an edit (or the untouched baseline) that stays clean.
        guard_ok, guard_pen, guard_checks = _guard_report(before, after, goals)
        reward -= guard_pen
        ok_full = ok and guard_ok
        all_checks = checks + guard_checks
        _emit({"phase": "verify", "cycle": cycle + 1, "ok": ok, "feedback": verdict,
               "checks": all_checks, "metrics_after": after})
        trace["attempts"].append({
            "n": cycle + 1,
            "plan": {"summary": plan.summary, "planner": plan.planner,
                     "planner_note": plan.planner_note,
                     "steps": [{"tool": s.tool, "params": s.params, "why": s.why} for s in plan.steps]},
            "steps": step_log,
            "verify": {"ok": ok_full, "checks": all_checks, "verdict": verdict},
        })
        if best is None or _prefer(ok, guard_ok, no_reg, reward, best[3], best[4], best[5], best[0]):
            best = (
                reward,
                spliced_try,
                after,
                ok,
                guard_ok,
                no_reg,
                all_checks,
                cycle + 1,
                cur,
                dict(bank_report) if bank_report is not None else None,
            )
        if ok_full or cycle + 1 >= total_attempts:
            break
        # ---- refine: feed EVERY unmet metric back to the planner ----
        misses = [c for c in checks if not c["met"]]
        fb = {"prev_summary": plan.summary,
              "prev_steps": [{"tool": s.tool, "params": s.params} for s in plan.steps],
              "misses": misses, "cycle": cycle + 1}
        plan = plan_edit(instruction, after, a_sec, b_sec, api_key=api_key, feedback=fb, goals=goals)
        plan = _normalize_plan(plan)
        _emit({"phase": "refine", "cycle": cycle + 2, "summary": plan.summary,
               "planner": plan.planner, "planner_note": plan.planner_note,
               "steps": [{"tool": s.tool, "why": s.why} for s in plan.steps]})

    (
        reward,
        spliced,
        after,
        semantic_ok,
        guard_ok,
        no_reg,
        checks,
        win_cycle,
        win_cur,
        best_bank_report,
    ) = best
    ok = semantic_ok
    n_attempts = len(trace["attempts"])
    kept_original = win_cycle == 0

    # Quality guard: unless the user asked for sharper, don't ship a jitterier dance than we started
    # with. Amplitude/beat-align edits raise jerk; pull it back toward baseline with a light smooth
    # that keeps every declared goal met. (No-op edits and goalless ops are skipped.)
    polished = _smoothness_polish(
        win_cur,
        motion,
        a,
        b,
        goals,
        before,
        wbeats,
        blend_frames,
        bank_report=best_bank_report,
    ) \
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
        if any(s.tool == "regenerate" for s in plan.steps):
            goal_txt = ", ".join(lbl for _m, _d, lbl in goals) or "the goal"
            feedback = f"kept the original: none of the regenerated takes improved {goal_txt} on this window" + refined
        else:
            detail = ", ".join(f"{lbl} {before.get(m, 0.0):.3f}" for m, _d, lbl in goals)
            feedback = f"left unchanged: no edit beat the current {detail} without hurting it" + refined
    else:
        parts = "  ".join(f"{c['label']} {c['before']:.3f}\u2192{c['after']:.3f}"
                          for c in checks if not c.get("guard")) or "applied"
        feedback = (parts if ok else f"couldn't fully satisfy: {parts}") + refined
    trace["final"] = {"ok": ok, "verdict": feedback, "attempts": n_attempts, "checks": checks,
                      "kept_original": kept_original, "metrics_before": before, "metrics_after": after}
    _emit({"phase": "done", "ok": ok, "metrics_after": after, "feedback": feedback,
           "summary": plan_summary0})

    goal = EditGoal(objective="agent", backbone="agent", magnitude=0.5, raw=instruction)
    return WindowEditResult(ok, goal, (a, b), spliced, before, after, backbone="agent",
                            chosen_seed=None, cycles=full_log, feedback=feedback,
                            log=full_log, agent_summary=plan_summary0, trace=trace)
