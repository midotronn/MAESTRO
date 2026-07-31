"""Windowed natural-language dance editing: the AgentBanana-style regen loop (Phase 1 core).

The user selects a time window ``[a, b)`` of the assembled dance and gives a natural-language
instruction ("more energetic", "more on beat", "calmer", "sharper", "smoother", "reverse this",
"mirror this", "exaggerate"). We:

1. THINK  -- :func:`parse_window_instruction` maps the text to a bounded :class:`EditGoal`
   (objective + preferred backbone + magnitude).
2. TOOL   -- objectives that are pure geometry (reverse / mirror / exaggerate) are applied
   deterministically to the window; the rest RE-QUERY a backbone.
3. GEN K  -- for a re-query objective we ask the :class:`WindowGenerator` for K seeded candidates
   over the window (optionally from both backbones when the hint is "auto"), passing an energy
   target derived from the objective.
4. SPLICE -- the best candidate is inertially spliced back in with
   :func:`agentlodge.dance.transition.splice_window`, preserving every frame outside ``[a, b)``.
5. VERIFY -- objective metric deltas on the window (BAS up for on-beat; energy up/down for
   energetic/calm; jerk down for smoother) decide success.
6. REFINE -- if unmet, escalate (fresh seeds + a stronger energy target) up to ``max_cycles``,
   keeping the best attempt (Self-Refine / Reflexion; AgentBanana propose->apply->verify->refine).

Backbones are reached ONLY through the :class:`WindowGenerator` protocol, so this whole module runs
and is unit-tested offline via :class:`MockWindowGenerator` (no torch, no GPU). A real pod-backed
generator implementing the same protocol lands in Phase 2.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from agentlodge.dance.beat_metrics import (
    _kinematic_speed,
    beat_alignment_score,
    foot_contact_consistency,
)
from agentlodge.dance.best_of_k import objective_weights
from agentlodge.dance.transition import (
    NUM_JOINTS,
    amplitude_scale,
    beat_align_warp,
    mirror,
    retrograde,
    splice_window,
)

logger = logging.getLogger(__name__)

_KIN = 135  # trans(3) + rot(132); excludes the 4 contact labels
_FROZEN_ENERGY = 0.01

# Objectives that RE-QUERY a backbone over the window, vs. deterministic transforms of the window.
# ``more_on_beat`` is deterministic: best-of-K sampling barely moves beat alignment (the backbones
# don't target specific beats), so we TIME-WARP the window's accents onto the music beats instead.
REGEN_OBJECTIVES = ("more_energetic", "calmer", "sharper", "smoother")
DETERMINISTIC_OBJECTIVES = ("reverse", "mirror", "exaggerate", "more_on_beat")
OBJECTIVES = REGEN_OBJECTIVES + DETERMINISTIC_OBJECTIVES

# keyword -> objective (first hit wins, order matters: check specific phrases first)
_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("reverse", "retrograde", "backward", "backwards", "in reverse"), "reverse"),
    (("mirror", "flip", "reflect", "left-right", "left right"), "mirror"),
    (("exaggerate", "amplify", "exaggerated", "bigger movement", "larger movement",
      "bigger movements", "over the top"), "exaggerate"),
    (("more on beat", "on-beat", "on the beat", "on beat", "tighten", "tighter", "sync",
      "in time", "to the beat", "rhythm", "musical timing"), "more_on_beat"),
    (("calmer", "calm", "softer", "gentler", "mellow", "subdued", "less energetic",
      "less energy", "tone it down", "tone down", "smaller", "chill", "relax"), "calmer"),
    (("sharper", "snappy", "snappier", "staccato", "percussive", "crisp", "punchier",
      "sharp", "edgy", "hit harder"), "sharper"),
    (("smoother", "smooth", "flowing", "graceful", "fluid", "lyrical", "glide",
      "flowy", "elegant"), "smoother"),
    (("more energetic", "more energy", "energetic", "bigger", "stronger", "livelier",
      "more intense", "intense", "powerful", "hype", "amp up", "amp it up", "pump"),
     "more_energetic"),
]

# preferred backbone per objective (LODGE = smooth, EDGE = sharp/energetic; auto = try both).
_BACKBONE_HINT = {
    "more_energetic": "edge", "sharper": "edge", "exaggerate": "edge",
    "calmer": "lodge", "smoother": "lodge",
    "more_on_beat": "auto", "reverse": "auto", "mirror": "auto",
}


@dataclass
class EditGoal:
    """A parsed, bounded editing intent over a window."""

    objective: str                       # one of OBJECTIVES
    backbone: str = "auto"               # lodge | edge | auto
    magnitude: float = 0.5               # 0..1 strength hint
    raw: str = ""

    @property
    def is_regen(self) -> bool:
        return self.objective in REGEN_OBJECTIVES

    def to_dict(self) -> dict:
        return {"objective": self.objective, "backbone": self.backbone,
                "magnitude": round(float(self.magnitude), 3), "raw": self.raw}


@dataclass
class WindowEditResult:
    """Outcome of one windowed edit."""

    ok: bool
    goal: EditGoal
    window: tuple[int, int]
    motion: np.ndarray                    # full assembled dance after the edit (L, 139)
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    backbone: str = ""
    chosen_seed: int | None = None
    cycles: list = field(default_factory=list)
    feedback: str = ""

    def summary(self) -> dict:
        return {
            "ok": self.ok, "goal": self.goal.to_dict(), "window": list(self.window),
            "backbone": self.backbone, "chosen_seed": self.chosen_seed,
            "metrics_before": self.metrics_before, "metrics_after": self.metrics_after,
            "n_cycles": len(self.cycles), "feedback": self.feedback,
        }


# ============================================================================ parsing
def _rule_parse(instruction: str) -> EditGoal:
    s = " " + instruction.lower().strip() + " "
    for phrases, obj in _KEYWORDS:
        if any(p in s for p in phrases):
            mag = 0.8 if any(w in s for w in ("much", "way", "a lot", "very", "really")) else 0.5
            return EditGoal(obj, _BACKBONE_HINT[obj], mag, instruction)
    # safe default: try to improve musical timing (never invents unsafe motion)
    return EditGoal("more_on_beat", "auto", 0.5, instruction)


def _llm_parse(instruction: str, api_key: str) -> EditGoal:
    from openai import OpenAI

    prompt = (
        "Map the user's dance-edit request for a selected time window to ONE JSON object.\n"
        f"Allowed \"objective\": {list(OBJECTIVES)}.\n"
        "\"backbone\": \"lodge\" (smooth/flowing), \"edge\" (sharp/energetic), or \"auto\".\n"
        "\"magnitude\": 0..1 strength.\n"
        "Respond with JSON only: {\"objective\":..., \"backbone\":..., \"magnitude\":...}.\n"
        f"Request: {instruction}"
    )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(model="gpt-4o-mini", max_tokens=120,
                                          messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON in window-edit parse response")
    raw = json.loads(m.group())
    obj = str(raw.get("objective", "")).strip()
    if obj not in OBJECTIVES:
        raise ValueError(f"unknown objective {obj!r}")
    bb = str(raw.get("backbone", "auto")).strip().lower()
    if bb not in ("lodge", "edge", "auto"):
        bb = _BACKBONE_HINT.get(obj, "auto")
    mag = float(np.clip(float(raw.get("magnitude", 0.5)), 0.0, 1.0))
    return EditGoal(obj, bb, mag, instruction)


def parse_window_instruction(instruction: str, *, api_key: str | None = None) -> EditGoal:
    """Parse an NL window-edit request into a bounded :class:`EditGoal` (LLM if key, else keywords)."""
    if api_key:
        try:
            return _llm_parse(instruction, api_key)
        except Exception as exc:  # noqa: BLE001 - robust offline fallback
            logger.warning("window-edit parse via LLM failed (%s); using keyword parser", exc)
    return _rule_parse(instruction)


# ============================================================================ metrics + reward
def window_metrics(clip: np.ndarray, beats=None) -> dict:
    """Objective signals for a window clip: energy, BAS, jerk, foot-contact consistency."""
    n = int(clip.shape[0])
    energy = float(np.mean(_kinematic_speed(clip))) if n >= 2 else 0.0
    bas = (beat_alignment_score(clip, np.asarray(beats))
           if beats is not None and len(beats) > 0 else 0.0)
    jerk = (float(np.mean(np.linalg.norm(np.diff(clip[:, :_KIN], n=3, axis=0), axis=1)))
            if n >= 4 else 0.0)
    foot = foot_contact_consistency(clip)
    return {"energy": round(energy, 5), "bas": round(float(bas), 5),
            "jerk": round(jerk, 5), "foot": round(foot, 5)}


def reward_weights_for(objective: str) -> dict:
    """Composite reward weights + energy target for an objective (see best_of_k.objective_weights)."""
    return objective_weights(objective)


def goal_reward(clip: np.ndarray, beats, goal: EditGoal) -> tuple[float, dict]:
    """Scalar reward for a candidate window under ``goal`` (higher = better realizes the intent)."""
    m = window_metrics(clip, beats)
    o = goal.objective
    if o in ("more_energetic", "sharper", "exaggerate"):
        r = m["energy"]
    elif o == "calmer":
        r = -m["energy"]
    elif o == "smoother":
        r = -m["jerk"]
    else:  # more_on_beat (and any fallback)
        r = m["bas"]
    if m["energy"] < _FROZEN_ENERGY:      # never reward a frozen/dead window
        r -= 10.0
    return float(r), m


def _verify(goal: EditGoal, before: dict, after: dict) -> tuple[bool, str]:
    o = goal.objective
    if o in ("more_energetic", "sharper", "exaggerate"):
        ok = after["energy"] > before["energy"] * 1.02
        return ok, (f"energy {before['energy']:.3f} -> {after['energy']:.3f}")
    if o == "calmer":
        ok = after["energy"] < before["energy"] * 0.98
        return ok, (f"energy {before['energy']:.3f} -> {after['energy']:.3f}")
    if o == "smoother":
        ok = after["jerk"] < before["jerk"] * 0.98
        return ok, (f"jerk {before['jerk']:.4f} -> {after['jerk']:.4f}")
    if o == "reverse":
        return True, "window reversed in time (retrograde)"
    if o == "mirror":
        return True, "window mirrored left<->right"
    # more_on_beat
    ok = after["bas"] > before["bas"] + 1e-3
    return ok, (f"BAS {before['bas']:.3f} -> {after['bas']:.3f}")


# ============================================================================ generator seam
@runtime_checkable
class WindowGenerator(Protocol):
    """Produces a fresh ``(b - a, 139)`` Z-up motion for a window from a backbone + seed.

    ``energy`` in [0, 1] is the requested intensity (maps to generator guidance), ``beats`` are the
    window-local music-beat frames (0-based within the window) the motion should hit. Implementations
    may ignore ``context`` (surrounding frames) or use them for continuity.
    """

    def generate(self, backbone: str, a: int, b: int, seed: int, *,
                 energy: float = 0.5, beats=None, context=None) -> np.ndarray:
        ...


class MockWindowGenerator:
    """Deterministic offline stand-in for the real backbones (no torch / GPU).

    Produces valid 139-dim motion whose mean speed scales with the requested ``energy`` and whose
    kinematic-speed troughs land on ``beats`` with a depth that grows with ``seed`` -- so best-of-K
    by BAS meaningfully improves timing and the energy target meaningfully raises/lowers intensity,
    letting the whole edit loop be unit-tested. It mirrors how a real EDGE/LODGE call would respond
    to guidance, without pretending to be musically real.
    """

    def __init__(self, *, step_scale: float = 0.12, seed_offset: int = 0):
        self.step_scale = float(step_scale)
        self.seed_offset = int(seed_offset)

    def generate(self, backbone: str, a: int, b: int, seed: int, *,
                 energy: float = 0.5, beats=None, context=None) -> np.ndarray:
        from agentlodge.dance.transition import _orthonormalize6d

        n = int(b - a)
        if n <= 0:
            return np.zeros((0, 139), dtype=np.float32)
        key = (hash((str(backbone), int(seed) + self.seed_offset, n)) & 0xFFFFFFFF)
        rng = np.random.default_rng(key)
        amp = 0.15 + 0.9 * float(np.clip(energy, 0.0, 1.0))

        env = np.ones(n, dtype=np.float32)
        if beats is not None and len(beats) > 0:
            depth = min(0.75, 0.14 * (int(seed) + 1))     # deeper beat-troughs at higher seeds
            sig = 2.5
            t = np.arange(n)
            for bt in np.asarray(beats, dtype=float):
                if 0 <= bt < n:
                    env = env * (1.0 - depth * np.exp(-((t - bt) ** 2) / (2 * sig ** 2)))
        scale = (amp * env)[:, None]

        base = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), NUM_JOINTS)   # identity 6D
        walk = np.cumsum(rng.standard_normal((n, NUM_JOINTS * 6)).astype(np.float32) * scale, axis=0)
        rot6 = base[None, :] + self.step_scale * walk
        rot6 = _orthonormalize6d(rot6.reshape(n, NUM_JOINTS, 6)).reshape(n, NUM_JOINTS * 6)
        trans = np.cumsum(rng.standard_normal((n, 3)).astype(np.float32) * scale * 0.05, axis=0)
        contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
        return np.concatenate([trans, rot6, contact], axis=1).astype(np.float32)


# ============================================================================ the edit loop
def _energy_target(objective: str, cycle: int, magnitude: float) -> float:
    """Energy guidance for a regen objective at a given refine ``cycle`` (0-based)."""
    if objective in ("more_energetic", "sharper"):
        return float(np.clip(0.7 + 0.15 * cycle + 0.1 * (magnitude - 0.5), 0.0, 1.0))
    if objective in ("calmer", "smoother"):
        return float(np.clip(0.3 - 0.15 * cycle - 0.1 * (magnitude - 0.5), 0.0, 1.0))
    return 0.5  # more_on_beat: keep intensity roughly neutral, vary seeds for timing


def _window_beats(beats, a: int, b: int) -> np.ndarray:
    """Music beats falling inside ``[a, b)``, re-expressed 0-based within the window."""
    if beats is None:
        return np.zeros(0, dtype=float)
    mb = np.asarray(beats, dtype=float)
    return mb[(mb >= a) & (mb < b)] - a


def _deterministic_window(objective: str, clip: np.ndarray, magnitude: float,
                          wbeats=None) -> np.ndarray:
    if objective == "reverse":
        return retrograde(clip)
    if objective == "mirror":
        return mirror(clip)
    if objective == "more_on_beat":
        # Snap the window's motion beats onto the music beats (see transition.beat_align_warp).
        passes = 3 + (1 if magnitude >= 0.7 else 0)
        return beat_align_warp(clip, wbeats if wbeats is not None else np.zeros(0), passes=passes)
    return amplitude_scale(clip, 1.0 + 0.4 * (0.5 + magnitude))   # exaggerate


def apply_window_edit(motion: np.ndarray, a: int, b: int, instruction: str,
                      generator: WindowGenerator | None = None, *, beats=None,
                      k: int = 6, max_cycles: int = 3, blend_frames: int = 15,
                      api_key: str | None = None,
                      goal: EditGoal | None = None,
                      progress_cb=None) -> WindowEditResult:
    """Run one windowed NL edit over ``motion[a:b]`` and return the edited full motion + report.

    ``progress_cb`` (optional) is called with a small dict for each generated candidate and at each
    verify step, so a UI can stream live "cycle n/K" progress.
    """
    def _emit(event: dict) -> None:
        if progress_cb is not None:
            try:
                progress_cb(event)
            except Exception:  # noqa: BLE001 - never let UI streaming break the edit
                pass

    motion = np.ascontiguousarray(motion, dtype=np.float32)
    L = int(motion.shape[0])
    a, b = int(a), int(b)
    if not (0 <= a < b <= L):
        raise ValueError(f"invalid window [{a}, {b}) for motion of length {L}")
    goal = goal or parse_window_instruction(instruction, api_key=api_key)
    wbeats = _window_beats(beats, a, b)
    before = window_metrics(motion[a:b], wbeats)
    _emit({"phase": "parsed", "goal": goal.to_dict(), "metrics_before": before})

    # ---- deterministic transforms of the window: no regeneration ----
    if goal.objective in DETERMINISTIC_OBJECTIVES:
        new_win = _deterministic_window(goal.objective, motion[a:b], goal.magnitude, wbeats)
        spliced = splice_window(motion, a, b, new_win, blend_frames=blend_frames)
        after = window_metrics(spliced[a:b], wbeats)
        ok, fb = _verify(goal, before, after)
        cyc = [{"cycle": 1, "op": goal.objective, "reward": None,
                "metrics": after}]
        _emit({"phase": "done", "ok": ok, "metrics_after": after, "feedback": fb})
        return WindowEditResult(ok, goal, (a, b), spliced, before, after,
                                backbone="none", chosen_seed=None, cycles=cyc, feedback=fb)

    if generator is None:
        raise ValueError("a WindowGenerator is required for a regeneration objective")

    backbones = ([goal.backbone] if goal.backbone in ("lodge", "edge")
                 else ["edge", "lodge"])
    ctx = {"prefix": motion[max(0, a - blend_frames):a], "suffix": motion[b:b + blend_frames]}
    total = max_cycles * len(backbones) * k

    best = None  # (reward, spliced, after, seed, backbone)
    cycles: list = []
    done = 0
    for cycle in range(max_cycles):
        etgt = _energy_target(goal.objective, cycle, goal.magnitude)
        seed0 = cycle * k
        for bb in backbones:
            for s in range(seed0, seed0 + k):
                cand = generator.generate(bb, a, b, s, energy=etgt, beats=wbeats, context=ctx)
                done += 1
                if cand is None or np.asarray(cand).shape[0] < 2:
                    continue
                spliced = splice_window(motion, a, b, np.asarray(cand), blend_frames=blend_frames)
                reward, mets = goal_reward(spliced[a:b], wbeats, goal)
                cycles.append({"cycle": cycle + 1, "backbone": bb, "seed": s,
                               "energy_target": round(etgt, 3), "reward": round(reward, 5),
                               "metrics": mets})
                if best is None or reward > best[0]:
                    best = (reward, spliced, mets, s, bb)
                _emit({"phase": "candidate", "cycle": cycle + 1, "backbone": bb, "seed": s,
                       "done": done, "total": total, "reward": round(reward, 5),
                       "metrics": mets, "best_reward": round(best[0], 5)})
        # verify the best-so-far against the goal; stop early if satisfied
        if best is not None:
            ok, fb = _verify(goal, before, best[2])
            _emit({"phase": "verify", "cycle": cycle + 1, "ok": ok, "feedback": fb,
                   "metrics_after": best[2]})
            if ok:
                _emit({"phase": "done", "ok": True, "metrics_after": best[2], "feedback": fb})
                return WindowEditResult(True, goal, (a, b), best[1], before, best[2],
                                        backbone=best[4], chosen_seed=best[3],
                                        cycles=cycles, feedback=fb)

    if best is None:
        raise RuntimeError("window generator produced no valid candidates")
    ok, fb = _verify(goal, before, best[2])
    msg = fb if ok else f"could not fully satisfy '{instruction}' after {max_cycles} cycle(s): {fb}"
    _emit({"phase": "done", "ok": ok, "metrics_after": best[2], "feedback": msg})
    return WindowEditResult(ok, goal, (a, b), best[1], before, best[2],
                            backbone=best[4], chosen_seed=best[3], cycles=cycles, feedback=msg)
