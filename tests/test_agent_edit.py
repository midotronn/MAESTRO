"""Unit tests for the agent-driven windowed editor (offline keyword planner + tool executor + log)."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor import agent_edit as AE
from agentlodge.editor.window_edit import MockWindowGenerator


def _base(n: int = 300, energy: float = 0.6, seed: int = 1) -> np.ndarray:
    return MockWindowGenerator().generate("edge", 0, n, seed, energy=energy, beats=None)


def _beats(n: int = 300, step: int = 15) -> np.ndarray:
    return np.arange(0, n, step).astype(float)


# --------------------------------------------------------------------------- planning (offline)
def test_keyword_plan_maps_intents_to_tools():
    assert AE.plan_edit("make this much more on beat", {}, 3, 7).steps[0].tool == "beat_align"
    assert AE.plan_edit("calm it down a lot", {}, 3, 7).steps[0].tool == "energy"
    assert AE.plan_edit("make it much more energetic", {}, 3, 7).steps[0].tool == "energy"
    assert AE.plan_edit("smoother and flowing", {}, 3, 7).steps[0].tool == "smooth"
    assert AE.plan_edit("snappier staccato hits", {}, 3, 7).steps[0].tool == "sharpen"
    assert AE.plan_edit("reverse this part", {}, 3, 7).steps[0].tool == "reverse"
    assert AE.plan_edit("mirror it", {}, 3, 7).steps[0].tool == "mirror"


def test_plan_carries_expected_metric():
    p = AE.plan_edit("tighten to the beat", {}, 3, 7)
    assert p.expect_metric == "bas" and p.expect_dir == "up"
    p2 = AE.plan_edit("make it calmer", {}, 3, 7)
    assert p2.expect_metric == "energy" and p2.expect_dir == "down"


# --------------------------------------------------------------------------- execution + log
def test_on_beat_edit_logs_and_raises_bas():
    m = _base(300)
    r = AE.run_agent_edit(m, 90, 210, "make this much more on beat", beats=_beats())
    assert r.ok and r.backbone == "agent"
    assert r.metrics_after["bas"] > r.metrics_before["bas"]
    assert r.log and r.log[0]["tool"] == "beat_align"
    assert "note" in r.log[0] and r.agent_summary
    # everything outside the window is untouched
    assert np.array_equal(r.motion[:90], m[:90]) and np.array_equal(r.motion[210:], m[210:])


def test_calmer_edit_lowers_energy():
    m = _base(300, energy=0.9)
    r = AE.run_agent_edit(m, 90, 210, "make this a lot calmer", beats=_beats())
    assert r.metrics_after["energy"] < r.metrics_before["energy"]
    assert r.log[0]["tool"] == "energy"


def test_smoother_edit_lowers_jerk():
    m = _base(300, energy=0.7)
    r = AE.run_agent_edit(m, 60, 200, "make it smoother and more graceful", beats=_beats())
    assert r.metrics_after["jerk"] < r.metrics_before["jerk"]


def test_composed_plan_runs_all_steps(monkeypatch):
    # simulate the LLM composing two tools: calm it down AND tighten to the beat
    plan = AE.AgentPlan(
        summary="calm it down and lock it to the beat",
        steps=[AE.PlanStep("energy", {"direction": "down", "amount": 0.6}, "less intensity"),
               AE.PlanStep("beat_align", {"strength": 1.0}, "tighten timing")],
        expect_metric="bas", expect_dir="up")
    monkeypatch.setattr(AE, "plan_edit", lambda *a, **k: plan)
    m = _base(300, energy=0.9)
    r = AE.run_agent_edit(m, 90, 210, "calmer but keep it on beat", beats=_beats())
    tools = [e["tool"] for e in r.log if e.get("cycle") == 1]     # first attempt runs both, in order
    assert tools == ["energy", "beat_align"]
    # step 1 (energy) lowers intensity in its own log entry
    e1 = r.log[0]
    assert e1["metrics_after"]["energy"] < e1["metrics_before"]["energy"]
    # step 2 (beat_align) raises BAS relative to its input (the calmed motion)
    e2 = r.log[1]
    assert e2["metrics_after"]["bas"] > e2["metrics_before"]["bas"]
    assert r.agent_summary == "calm it down and lock it to the beat"


def test_regenerate_tool_uses_generator():
    m = _base(300, energy=0.3)
    plan = AE.AgentPlan("resample bigger moves",
                        steps=[AE.PlanStep("regenerate", {"backbone": "edge", "energy": 0.9}, "new moves")],
                        expect_metric="energy", expect_dir="up")
    import agentlodge.editor.agent_edit as AEmod
    orig = AEmod.plan_edit
    AEmod.plan_edit = lambda *a, **k: plan
    try:
        r = AE.run_agent_edit(m, 90, 210, "give me totally different big moves",
                              MockWindowGenerator(), beats=_beats())
    finally:
        AEmod.plan_edit = orig
    assert r.log[0]["tool"] == "regenerate"
    assert r.motion.shape == m.shape
    assert np.array_equal(r.motion[:90], m[:90]) and np.array_equal(r.motion[210:], m[210:])


def test_regenerate_ranks_seeds_by_declared_goals(monkeypatch):
    # regenerate must pick the seed that best ADVANCES the declared goals, not one matching a fixed
    # energy target. 3 seeds: seed0=high-energy/low-BAS, seed1=low-energy/HIGH-BAS, seed2=middling.
    class FakeGen:
        def generate(self, bb, a, b, s, energy=0.5, beats=None, context=None):
            c = np.zeros((120, 139), np.float32); c[0, 0] = float(s); return c
    by_tag = {0: {"energy": 0.9, "bas": 0.5, "jerk": 0.1, "foot": 1.0},
              1: {"energy": 0.3, "bas": 0.9, "jerk": 0.1, "foot": 1.0},
              2: {"energy": 0.5, "bas": 0.6, "jerk": 0.1, "foot": 1.0}}
    monkeypatch.setattr(AE, "window_metrics", lambda clip, beats=None: by_tag[int(round(float(clip[0, 0])))])
    base = np.zeros((120, 139), np.float32); base[0, 0] = 2.0            # current window ~ seed2 (BAS 0.6)
    ctx = {"a": 0, "b": 120, "wbeats": np.array([10.0, 20.0]), "generator": FakeGen(),
           "goals": [("bas", "up", "beat alignment")], "context": None}
    out, _note = AE._tool_regenerate(base, ctx, backbone="edge", energy=0.9, k=3)
    assert int(round(float(out[0, 0]))) == 1               # highest-BAS seed, NOT the energy-target one (0)


def test_regenerate_without_generator_degrades_gracefully():
    m = _base(200)
    plan = AE.AgentPlan("resample", steps=[AE.PlanStep("regenerate", {}, "x")],
                        expect_metric=None, expect_dir=None)
    import agentlodge.editor.agent_edit as AEmod
    orig = AEmod.plan_edit
    AEmod.plan_edit = lambda *a, **k: plan
    try:
        r = AE.run_agent_edit(m, 60, 150, "regenerate", None, beats=_beats(200))
    finally:
        AEmod.plan_edit = orig
    assert r.motion.shape == m.shape                     # no crash, kept motion
    assert "kept the current motion" in r.log[0]["note"]


# --------------------------------------------------------------------------- refine loop
def test_refine_escalates_when_first_attempt_misses(monkeypatch):
    # attempt 1 is a no-op (energy amount 0 -> no change -> verify fails); the refine feedback then
    # yields a stronger plan that actually lowers energy. Best (2nd) attempt is kept.
    def fake_plan(instruction, metrics, a_sec, b_sec, *, api_key=None, feedback=None, goals=None):
        if feedback is None:
            return AE.AgentPlan("weak", [AE.PlanStep("energy", {"direction": "down", "amount": 0.0}, "x")],
                                expect_metric="energy", expect_dir="down")
        return AE.AgentPlan("stronger", [AE.PlanStep("energy", {"direction": "down", "amount": 0.9}, "harder")],
                            expect_metric="energy", expect_dir="down")
    monkeypatch.setattr(AE, "plan_edit", fake_plan)
    m = _base(300, energy=0.9)
    r = AE.run_agent_edit(m, 90, 210, "make it much calmer", beats=_beats(), max_refine=2)
    assert r.ok                                          # the refined attempt satisfies the goal
    assert r.metrics_after["energy"] < r.metrics_before["energy"]
    cycles = {e["cycle"] for e in r.log}
    assert cycles == {1, 2}                              # it took a refine cycle
    assert "refined 1x" in r.feedback


def test_keyword_refine_escalates_amount():
    p0 = AE.plan_edit("make it calmer", {}, 3, 7)                      # first attempt
    p1 = AE.plan_edit("make it calmer", {}, 3, 7, feedback={"cycle": 1})  # refine
    assert p1.steps[0].params["amount"] > p0.steps[0].params["amount"]


def test_goalless_op_skips_refine(monkeypatch):
    # reverse has no measurable goal -> exactly one attempt, no refine calls
    calls = {"n": 0}
    def fake_plan(*a, **k):
        calls["n"] += 1
        return AE.AgentPlan("reverse", [AE.PlanStep("reverse", {}, "backward")],
                            expect_metric=None, expect_dir=None)
    monkeypatch.setattr(AE, "plan_edit", fake_plan)
    m = _base(200)
    r = AE.run_agent_edit(m, 60, 150, "reverse this", beats=_beats(200), max_refine=3)
    assert calls["n"] == 1                               # planned once, no refine
    assert {e["cycle"] for e in r.log} == {1}


# --------------------------------------------------------------------------- executor guardrail (regression catch)
def test_requested_metrics_extracts_multiple():
    reqs = {m: d for m, d, _ in AE._requested_metrics("calmer but keep it tight to the beat")}
    assert reqs.get("energy") == "down" and reqs.get("bas") == "up"
    # a contradictory instruction cancels the ambiguous metric
    assert not any(m == "energy" for m, _, _ in AE._requested_metrics("calmer but more energetic"))


def test_executor_rejects_step_that_regresses_its_metric(monkeypatch):
    # A beat_align that would LOWER beat alignment must be REJECTED and the motion kept (this is the
    # exact bug: user asked to raise beat alignment but BAS dropped -- the executor now catches it).
    def fake_metrics(clip, beats=None):
        v = float(np.mean(clip))
        return {"energy": 0.3, "bas": round(1.0 - v, 5), "jerk": 0.1, "foot": 1.0}   # bigger clip -> lower BAS
    monkeypatch.setattr(AE, "window_metrics", fake_metrics)
    monkeypatch.setitem(AE.TOOLS, "beat_align",
                        AE.ToolSpec(lambda clip, ctx, **kw: (clip + 0.5, "fake align"), "x", ""))
    base = np.zeros((120, 139), np.float32)
    plan = AE.AgentPlan("tighten", [AE.PlanStep("beat_align", {}, "tighten")], "bas", "up")
    cur, log = AE._execute_plan(base, plan, {"wbeats": None}, None, cycle=1, emit=lambda e: None)
    assert log[0]["status"] == "rejected"
    assert "wrong way" in log[0]["reject_reason"]
    assert np.array_equal(cur, base)                     # the regressing edit was discarded


def test_executor_applies_step_that_improves_its_metric(monkeypatch):
    def fake_metrics(clip, beats=None):
        v = float(np.mean(clip))
        return {"energy": 0.3, "bas": round(v, 5), "jerk": 0.1, "foot": 1.0}          # bigger clip -> higher BAS
    monkeypatch.setattr(AE, "window_metrics", fake_metrics)
    monkeypatch.setitem(AE.TOOLS, "beat_align",
                        AE.ToolSpec(lambda clip, ctx, **kw: (clip + 0.5, "fake align"), "x", ""))
    base = np.zeros((120, 139), np.float32)
    plan = AE.AgentPlan("tighten", [AE.PlanStep("beat_align", {}, "tighten")], "bas", "up")
    cur, log = AE._execute_plan(base, plan, {"wbeats": None}, None, cycle=1, emit=lambda e: None)
    assert log[0]["status"] == "applied" and float(np.mean(cur)) > 0


def test_trace_carries_plan_executor_and_verify():
    m = _base(300)
    r = AE.run_agent_edit(m, 90, 210, "make this much more on beat", beats=_beats())
    t = r.trace
    assert t.get("goals") and t.get("attempts")
    at = t["attempts"][0]
    assert set(("plan", "steps", "verify")).issubset(at)
    assert "checks" in at["verify"] and at["plan"]["steps"]
    assert all("status" in s for s in at["steps"])


def test_prefer_ranks_all_goals_met_over_higher_reward():
    # the bug: an attempt with a huge single-metric gain (but a regressed goal, so NOT ok) must not
    # beat a balanced attempt that meets EVERY goal, even though its summed reward is larger.
    assert AE._prefer(True, 51.6, False, 67.6)          # ok beats not-ok regardless of reward
    assert not AE._prefer(False, 67.6, True, 51.6)
    assert AE._prefer(True, 60.0, True, 51.6)           # ties on ok -> higher reward wins
    assert AE._prefer(False, 70.0, False, 67.6)
    assert not AE._prefer(False, 10.0, False, 67.6)


def test_merge_goals_planner_wins_keyword_fills_gaps():
    primary = [("energy", "up", "energy")]                       # planner-declared
    secondary = [("energy", "down", "energy"), ("bas", "up", "beat alignment")]  # keyword net
    merged = {m: d for m, d, _ in AE._merge_goals(primary, secondary)}
    assert merged["energy"] == "up"                              # planner wins the conflict
    assert merged["bas"] == "up"                                 # keyword fills the metric it missed


def test_smoothness_polish_picks_smooth_that_restores_jerk(monkeypatch):
    # deterministic logic test (the random-walk mock entangles energy+jerk, so use controlled metrics):
    # tag the clip with the smooth amount; map amount -> jerk. The polish should pick the smallest
    # amount that restores jerk to ~baseline while the energy goal stays met.
    def fake_smooth(clip, amt):
        c = clip.copy(); c[0, 1] = amt; return c
    JERK = {0.0: 0.30, 0.12: 0.18, 0.22: 0.12, 0.34: 0.09}
    def fake_metrics(clip, beats=None):
        return {"energy": 0.6, "bas": 0.7, "jerk": JERK.get(round(float(clip[0, 1]), 2), 0.30), "foot": 1.0}
    monkeypatch.setattr(AE, "temporal_smooth", fake_smooth)
    monkeypatch.setattr(AE, "crossfade_edit", lambda motion, a, b, cur, blend_frames=12: cur)
    monkeypatch.setattr(AE, "window_metrics", fake_metrics)
    win_cur = np.zeros((120, 139), np.float32)                  # tag 0 -> jerk 0.30 (jittery edit)
    before = {"energy": 0.4, "bas": 0.7, "jerk": 0.12, "foot": 1.0}   # baseline jerk 0.12
    pol = AE._smoothness_polish(win_cur, win_cur, 0, 120,
                                [("energy", "up", "energy")], before, np.array([10.0, 20.0]), 12)
    assert pol is not None
    _spl, after, checks, note = pol
    assert after["jerk"] <= before["jerk"] * 1.05              # restored to ~baseline smoothness
    assert after["energy"] > before["energy"]                 # energy goal kept
    assert all(c["met"] for c in checks) and "smoothed" in note


def test_smoothness_polish_skipped_when_sharper_requested():
    from agentlodge.editor.window_edit import window_metrics, _window_beats
    m = _base(300, energy=0.5)
    wb = _window_beats(_beats(), 90, 210)
    before = window_metrics(m[90:210], wb)
    # jerk-up is a declared goal -> the guard must NOT smooth (short-circuits before any work)
    assert AE._smoothness_polish(m[90:210].copy(), m, 90, 210,
                                 [("jerk", "up", "smoothness")], before, wb, 12) is None


def test_planner_declared_goals_drive_verification(monkeypatch):
    # goals come from the planning agent's reasoning, NOT a keyword match on the instruction
    plan = AE.AgentPlan("boost intensity",
                        [AE.PlanStep("energy", {"direction": "up", "amount": 0.7}, "livelier")],
                        expect_metric="energy", expect_dir="up", goals=[("energy", "up", "energy")])
    monkeypatch.setattr(AE, "plan_edit", lambda *a, **k: plan)
    m = _base(300, energy=0.3)
    r = AE.run_agent_edit(m, 90, 210, "zhoozh this bit up", beats=_beats())   # no keyword would match
    assert any(g["metric"] == "energy" and g["dir"] == "up" for g in r.trace["goals"])
    assert r.metrics_after["energy"] > r.metrics_before["energy"]
