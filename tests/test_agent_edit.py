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
    tools = [e["tool"] for e in r.log]
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
