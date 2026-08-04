"""Unit tests for the agent-driven windowed editor (offline keyword planner + tool executor + log)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor import agent_edit as AE
from agentlodge.editor.window_edit import MockWindowGenerator


def _base(n: int = 300, energy: float = 0.6, seed: int = 1) -> np.ndarray:
    return MockWindowGenerator().generate("edge", 0, n, seed, energy=energy, beats=None)


def _beats(n: int = 300, step: int = 15) -> np.ndarray:
    return np.arange(0, n, step).astype(float)


def _smooth(n: int = 300, seed: int = 0, n_joints: int = 22) -> np.ndarray:
    """A smooth, dance-tempo valid 139-dim motion (so the energy lever behaves like it does on real
    dances, unlike the random-walk mock whose energy/jerk are entangled)."""
    from agentlodge.dance import transition as T
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0 * np.pi, n)
    axes = rng.standard_normal((n_joints, 3)); axes /= np.linalg.norm(axes, axis=-1, keepdims=True) + 1e-9
    ang = (0.15 + 0.25 * rng.random((n_joints, 1))) * np.sin(
        (1.2 + 1.6 * rng.random((n_joints, 1))) * t[None, :] + 2 * np.pi * rng.random((n_joints, 1)))
    aa = np.transpose(axes[:, None, :] * ang[:, :, None], (1, 0, 2))
    r6 = T._matrix_to_sixd(T._axis_angle_to_matrix(aa)).reshape(n, n_joints * 6)
    trans = 0.02 * np.cumsum(np.sin(t)[:, None] * np.ones((1, 3)), axis=0)
    contact = (rng.random((n, 4)) > 0.5).astype(np.float32)
    return np.concatenate([trans, r6, contact], axis=1).astype(np.float32)


# --------------------------------------------------------------- LLM planner path (mocked, no API)
def _fake_openai(monkeypatch, response_json: str, captured: dict):
    """Patch openai.OpenAI so _llm_plan runs offline; capture the model + prompt it would send."""
    openai = pytest.importorskip("openai")

    class _Msg:
        def __init__(self, c): self.content = c

    class _Choice:
        def __init__(self, c): self.message = _Msg(c)

    class _Resp:
        def __init__(self, c): self.choices = [_Choice(c)]

    class _Completions:
        def create(self, *, model, messages, **kw):
            captured["model"] = model
            captured["prompt"] = messages[0]["content"]
            return _Resp(response_json)

    class _Client:
        def __init__(self, api_key=None): self.chat = type("C", (), {"completions": _Completions()})()

    monkeypatch.setattr(openai, "OpenAI", _Client)


def test_llm_plan_parses_compound_response_and_uses_configurable_model(monkeypatch):
    captured: dict = {}
    _fake_openai(monkeypatch,
                 '{"summary":"new + energetic","steps":['
                 '{"tool":"regenerate","params":{},"why":"new motion"},'
                 '{"tool":"energy","params":{"direction":"up"},"why":"hit the target"}],'
                 '"goals":[{"metric":"energy","direction":"up"}]}', captured)
    monkeypatch.setenv("AGENTLODGE_PLANNER_MODEL", "gpt-unit-test")
    p = AE.plan_edit("create a new motion that matches or exceeds the energy",
                     {"energy": 0.3, "bas": 0.5, "jerk": 0.1, "foot": 1.0}, 46.0, 52.0, api_key="sk-x")
    assert p.planner == "llm"
    assert captured["model"] == "gpt-unit-test"              # model is configurable via env
    assert [s.tool for s in p.steps] == ["regenerate", "energy"]
    assert ("energy", "up", "energy") in p.goals and len(p.goals) == 1   # no hallucinated extra goal


def test_llm_prompt_carries_the_critical_routing_rules(monkeypatch):
    # guard the prompt content: a future edit must not silently drop the rules that fix the reported
    # bugs (create-new -> regenerate; do not invent goals).
    captured: dict = {}
    _fake_openai(monkeypatch, '{"summary":"x","steps":[{"tool":"smooth","params":{}}],'
                              '"goals":[{"metric":"jerk","direction":"down"}]}', captured)
    AE.plan_edit("smooth it", {"energy": 0.3, "bas": 0.5, "jerk": 0.1, "foot": 1.0},
                 46.0, 52.0, api_key="sk-x")
    prompt = captured["prompt"].lower()
    assert "regenerate" in prompt
    assert "new motion" in prompt and "choreograph" in prompt          # create-new routing present
    assert "only" in prompt and "goal" in prompt                       # goal-discipline present
    assert "reshape" in prompt or "resize" in prompt                   # lever-vs-regenerate distinction


def test_llm_plan_falls_back_to_keyword_when_the_api_errors(monkeypatch):
    openai = pytest.importorskip("openai")

    class _Boom:
        def __init__(self, api_key=None): raise RuntimeError("no network")

    monkeypatch.setattr(openai, "OpenAI", _Boom)
    p = AE.plan_edit("make it more energetic", {}, 46.0, 52.0, api_key="sk-x")
    assert p.planner == "keyword_fallback" and p.steps[0].tool == "energy"


# --------------------------------------------------------------------------- behavioural guarantees
def test_more_energetic_delivers_a_substantial_gain_not_a_token_one():
    # regression guard for the polish clawing a requested energy increase back to ~tolerance. Compare
    # the loop's delivered gain against the BARE lever's gain (same gain the 'much' planner uses): the
    # quality polish must keep most of it, not smooth it away to a token amount (the +0.007 bug).
    from agentlodge.dance.transition import accentuate
    from agentlodge.editor.window_edit import window_metrics
    m = _smooth(300, seed=5)
    a, b = 60, 240
    before = window_metrics(m[a:b])["energy"]
    raw = window_metrics(accentuate(m[a:b], 1.68))["energy"]  # bare lever at the 'much' gain, no polish
    r = AE.run_agent_edit(m, a, b, "make this much more energetic", generator=None, beats=_beats())
    assert r.ok and not r.trace["final"]["kept_original"]
    after = r.metrics_after["energy"]
    assert raw > before * 1.05                                # the bare lever really does move it
    assert after >= before + 0.5 * (raw - before), (before, raw, after)   # polish kept >=50% of it


@pytest.mark.parametrize("instruction, metric, direction", [
    ("make this much more energetic", "energy", "up"),
    ("make it a lot calmer", "energy", "down"),
    ("smooth it out so it flows", "jerk", "down"),
    ("snappier punchier staccato hits", "jerk", "up"),
    ("tighten it to the beat", "bas", "up"),
])
def test_metric_battery_every_query_is_satisfied(instruction, metric, direction):
    # each metric-directed request must be MET (ok) by a real change (not kept-original), moving its
    # metric the right way -- the "100% satisfied by the loop" guarantee, deterministic + no backbone.
    m = _smooth(300, seed=2)
    r = AE.run_agent_edit(m, 60, 240, instruction, generator=None, beats=_beats(), max_refine=2)
    assert r.ok and not r.trace["final"]["kept_original"], f"{instruction!r} not satisfied"
    b, a = r.metrics_before[metric], r.metrics_after[metric]
    assert (a > b) if direction == "up" else (a < b), f"{instruction!r}: {metric} {b}->{a}"


def test_create_new_motion_actually_regenerates_not_just_amplifies(monkeypatch):
    # THE reported bug: "create a new motion ..." must run a REGENERATE step (new choreography), not
    # merely reshape the current window with the energy lever. Drive the offline planner so the test
    # is deterministic (no API), then assert a regenerate step actually executed.
    m = _base(300, energy=0.4)
    r = AE.run_agent_edit(m, 90, 210,
                          "create a new motion for this window that matches or exceeds the energy",
                          generator=MockWindowGenerator(), beats=_beats())
    assert any(s["tool"] == "regenerate" and s.get("status") == "applied" for s in r.log), \
        f"expected a regenerate step, got {[s['tool'] for s in r.log]}"


def test_compound_new_and_energetic_regenerates_then_raises_energy(monkeypatch):
    m = _base(300, energy=0.3)
    r = AE.run_agent_edit(m, 90, 210, "give me different, more energetic moves",
                          generator=MockWindowGenerator(), beats=_beats())
    tools = [s["tool"] for s in r.log if s.get("status") == "applied"]
    assert "regenerate" in tools                              # new motion was created
    # the window actually changed (fresh choreography spliced in), outside is untouched
    assert not np.array_equal(r.motion[90:210], m[90:210])
    assert np.array_equal(r.motion[:90], m[:90]) and np.array_equal(r.motion[210:], m[210:])


@pytest.mark.parametrize("instruction", ["make it more energetic", "smooth it out", "tighten to the beat"])
def test_edits_never_touch_frames_outside_the_window(instruction):
    m = _smooth(300, seed=7)
    r = AE.run_agent_edit(m, 96, 204, instruction, generator=None, beats=_beats())
    assert np.array_equal(r.motion[:96], m[:96]) and np.array_equal(r.motion[204:], m[204:])


# --------------------------------------------------------------------------- planning (offline)
def test_keyword_plan_maps_intents_to_tools():
    # every metric now maps to its dedicated deterministic lever; only variety regenerates
    assert AE.plan_edit("make this much more on beat", {}, 3, 7).steps[0].tool == "beat_align"
    assert AE.plan_edit("smoother and flowing", {}, 3, 7).steps[0].tool == "smooth"
    assert AE.plan_edit("reverse this part", {}, 3, 7).steps[0].tool == "reverse"
    assert AE.plan_edit("mirror it", {}, 3, 7).steps[0].tool == "mirror"
    assert AE.plan_edit("calm it down a lot", {}, 3, 7).steps[0].tool == "energy"
    assert AE.plan_edit("make it much more energetic", {}, 3, 7).steps[0].tool == "energy"
    assert AE.plan_edit("snappier staccato hits", {}, 3, 7).steps[0].tool == "sharpen"
    assert AE.plan_edit("give me different freestyle moves", {}, 3, 7).steps[0].tool == "regenerate"


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
    r = AE.run_agent_edit(m, 90, 210, "make this a lot calmer", MockWindowGenerator(), beats=_beats())
    assert r.log[0]["tool"] == "energy"                           # deterministic monotone lever
    assert r.metrics_after["energy"] < r.metrics_before["energy"]


def test_more_energetic_edit_raises_energy_without_generator():
    # the guarantee: 'more energetic' is satisfied by the deterministic lever even with NO backbone
    m = _base(300, energy=0.4)
    r = AE.run_agent_edit(m, 90, 210, "make this much more energetic", generator=None, beats=_beats())
    assert r.ok and r.log[0]["tool"] == "energy"
    assert r.metrics_after["energy"] > r.metrics_before["energy"]
    # frames outside the window are byte-identical (tapered seams)
    assert np.array_equal(r.motion[:90], m[:90]) and np.array_equal(r.motion[210:], m[210:])


def test_sharper_edit_raises_jerk_without_generator():
    m = _base(300, energy=0.5)
    r = AE.run_agent_edit(m, 90, 210, "snappier punchier staccato hits", generator=None, beats=_beats())
    assert r.ok and r.log[0]["tool"] == "sharpen"
    assert r.metrics_after["jerk"] > r.metrics_before["jerk"]


def test_smoother_edit_lowers_jerk():
    m = _base(300, energy=0.7)
    r = AE.run_agent_edit(m, 60, 200, "make it smoother and more graceful", beats=_beats())
    assert r.metrics_after["jerk"] < r.metrics_before["jerk"]


def test_composed_plan_runs_all_steps(monkeypatch):
    # simulate the LLM composing two deterministic tools: smooth it AND tighten to the beat
    plan = AE.AgentPlan(
        summary="smooth it and lock it to the beat",
        steps=[AE.PlanStep("smooth", {"amount": 0.6}, "less jitter"),
               AE.PlanStep("beat_align", {"strength": 1.0}, "tighten timing")],
        expect_metric="bas", expect_dir="up", goals=[("bas", "up", "beat alignment")])
    monkeypatch.setattr(AE, "plan_edit", lambda *a, **k: plan)
    m = _base(300, energy=0.9)
    r = AE.run_agent_edit(m, 90, 210, "zhoozh this", beats=_beats())    # no keyword adds goals
    tools = [e["tool"] for e in r.log if e.get("cycle") == 1]     # first attempt runs both, in order
    assert tools == ["smooth", "beat_align"]
    # step 1 (smooth) lowers jerk in its own log entry
    e1 = r.log[0]
    assert e1["metrics_after"]["jerk"] < e1["metrics_before"]["jerk"]
    # step 2 (beat_align) raises BAS relative to its input (the smoothed motion)
    e2 = r.log[1]
    assert e2["metrics_after"]["bas"] > e2["metrics_before"]["bas"]
    assert r.agent_summary == "smooth it and lock it to the beat"


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
    # attempt 1 is a no-op (smooth amount 0 -> no change -> verify fails); the refine feedback then
    # yields a stronger plan that actually lowers jerk. Best (2nd) attempt is kept.
    def fake_plan(instruction, metrics, a_sec, b_sec, *, api_key=None, feedback=None, goals=None):
        if feedback is None:
            return AE.AgentPlan("weak", [AE.PlanStep("smooth", {"amount": 0.0}, "x")],
                                expect_metric="jerk", expect_dir="down", goals=[("jerk", "down", "smoothness")])
        return AE.AgentPlan("stronger", [AE.PlanStep("smooth", {"amount": 0.9}, "harder")],
                            expect_metric="jerk", expect_dir="down", goals=[("jerk", "down", "smoothness")])
    monkeypatch.setattr(AE, "plan_edit", fake_plan)
    m = _base(300, energy=0.9)
    r = AE.run_agent_edit(m, 90, 210, "make it much smoother", beats=_beats(), max_refine=2)
    assert r.ok                                          # the refined attempt satisfies the goal
    assert r.metrics_after["jerk"] < r.metrics_before["jerk"]
    cycles = {e["cycle"] for e in r.log}
    assert cycles == {1, 2}                              # it took a refine cycle
    assert "refined 1x" in r.feedback


def test_keyword_refine_escalates_amount():
    p0 = AE.plan_edit("make it smoother", {}, 3, 7)                      # first attempt
    p1 = AE.plan_edit("make it smoother", {}, 3, 7, feedback={"cycle": 1})  # refine
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
    # ranking tiers, most important first: (1) meets all goals, (2) artifact-guard clean (no
    # jitter/foot-skate), (3) no declared goal regressed, (4) higher reward.
    assert AE._prefer(True, True, True, 51.6, False, True, True, 67.6)      # ok beats not-ok
    assert not AE._prefer(False, True, True, 67.6, True, True, True, 51.6)
    assert AE._prefer(True, True, True, 5.0, True, False, True, 99.0)       # clean beats guard-violating
    assert AE._prefer(False, True, True, 5.0, False, True, False, 99.0)     # no-regression beats regressing
    assert not AE._prefer(False, True, False, 99.0, False, True, True, 5.0)
    assert AE._prefer(True, True, True, 60.0, True, True, True, 51.6)       # ties -> higher reward


def test_merge_goals_planner_wins_keyword_fills_gaps():
    primary = [("energy", "up", "energy")]                       # planner-declared
    secondary = [("energy", "down", "energy"), ("bas", "up", "beat alignment")]  # keyword net
    merged = {m: d for m, d, _ in AE._merge_goals(primary, secondary)}
    assert merged["energy"] == "up"                              # planner wins the conflict
    assert merged["bas"] == "up"                                 # keyword fills the metric it missed


def test_smoothness_polish_keeps_energy_jerk_within_proportional_budget(monkeypatch):
    # deterministic logic test (the random-walk mock entangles energy+jerk, so use controlled metrics):
    # tag the clip with the smooth amount; map amount -> jerk. For an ENERGY-UP goal the polish must
    # trim jitter only down to the energy-proportional budget (not all the way to baseline), so the
    # requested energy survives -- it should pick the smallest amount that lands within budget.
    def fake_smooth(clip, amt):
        c = clip.copy(); c[0, 1] = amt; return c
    JERK = {0.0: 0.30, 0.06: 0.20, 0.14: 0.12, 0.24: 0.10, 0.34: 0.09}
    def fake_metrics(clip, beats=None):
        return {"energy": 0.6, "bas": 0.7, "jerk": JERK.get(round(float(clip[0, 1]), 2), 0.30), "foot": 1.0}
    monkeypatch.setattr(AE, "temporal_smooth", fake_smooth)
    monkeypatch.setattr(AE, "crossfade_edit", lambda motion, a, b, cur, blend_frames=12: cur)
    monkeypatch.setattr(AE, "window_metrics", fake_metrics)
    win_cur = np.zeros((120, 139), np.float32)                  # tag 0 -> jerk 0.30 (jittery edit)
    before = {"energy": 0.4, "bas": 0.7, "jerk": 0.12, "foot": 1.0}   # baseline jerk 0.12
    goals = [("energy", "up", "energy")]
    pol = AE._smoothness_polish(win_cur, win_cur, 0, 120, goals, before, np.array([10.0, 20.0]), 12)
    assert pol is not None                                     # 0.30 exceeds the budget -> polish runs
    _spl, after, checks, note = pol
    budget = AE._jerk_ceiling(before, after, goals)           # energy 0.6/0.4 -> generous ceiling
    assert after["jerk"] < 0.30 and after["jerk"] <= budget   # trimmed to within budget
    assert after["jerk"] > before["jerk"]                     # but NOT smoothed back to baseline
    assert after["energy"] > before["energy"]                 # energy goal kept
    assert all(c["met"] for c in checks) and "0.06" in note   # smallest amount that lands in budget


def test_jerk_ceiling_strict_unless_energy_requested():
    before = {"energy": 0.4, "jerk": 0.10}
    # energy NOT a goal -> near-baseline budget (incidental jitter is polished away)
    strict = AE._jerk_ceiling(before, {"energy": 0.4, "jerk": 0.20}, [("bas", "up", "beat alignment")])
    assert abs(strict - 0.10 * 1.08) < 1e-9
    # energy UP a goal -> budget grows with the energy actually delivered (bigger moves are jerkier)
    gen = AE._jerk_ceiling(before, {"energy": 0.56, "jerk": 0.20}, [("energy", "up", "energy")])
    assert gen > strict and gen > 0.10
    # capped so it can never run away
    huge = AE._jerk_ceiling(before, {"energy": 4.0, "jerk": 9.0}, [("energy", "up", "energy")])
    assert huge <= 0.10 * 2.6 + 1e-9


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
                        [AE.PlanStep("regenerate", {"backbone": "edge", "energy": 0.9}, "livelier")],
                        expect_metric="energy", expect_dir="up", goals=[("energy", "up", "energy")])
    monkeypatch.setattr(AE, "plan_edit", lambda *a, **k: plan)
    m = _base(300, energy=0.3)
    r = AE.run_agent_edit(m, 90, 210, "zhoozh this bit up", MockWindowGenerator(), beats=_beats())
    assert any(g["metric"] == "energy" and g["dir"] == "up" for g in r.trace["goals"])
    assert r.metrics_after["energy"] > r.metrics_before["energy"]


def test_no_regression_guard_blocks_conflicting_edit(monkeypatch):
    # goals conflict: energy UP and jerk DOWN. energy scaling raises jerk -> would regress the jerk
    # goal. The no-regression guard must NOT ship an edit that regresses a declared goal; it keeps the
    # original (which regresses nothing) rather than trading one goal for another.
    plan = AE.AgentPlan("boost", [AE.PlanStep("energy", {"direction": "up", "amount": 0.8}, "x")],
                        goals=[("energy", "up", "energy"), ("jerk", "down", "smoothness")])
    monkeypatch.setattr(AE, "plan_edit", lambda *a, **k: plan)
    r = AE.run_agent_edit(_base(300, energy=0.5), 90, 210, "energetic and smooth",
                          beats=_beats(), max_refine=0)
    assert not any(c["status"] == "regressed" for c in r.trace["final"]["checks"])
    assert r.trace["final"]["kept_original"] is True


def test_offline_planner_declares_only_requested_goal():
    # the offline keyword planner must not over-declare: "more energetic" -> energy only (not jerk/bas)
    r = AE.run_agent_edit(_base(300, energy=0.4), 90, 210, "make it more energetic", beats=_beats())
    assert {g["metric"] for g in r.trace["goals"]} == {"energy"}
