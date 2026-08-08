"""Unit tests for the agent-driven windowed editor (offline keyword planner + tool executor + log)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor import agent_edit as AE
from agentlodge.editor.window_edit import MockWindowGenerator, window_metrics


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
    assert "beat-hit motions" in prompt and "clap_single" in prompt
    assert "direction=auto" in prompt and "dance" in prompt and "flow" in prompt


def test_llm_cannot_move_a_default_beat_hit_off_the_beat(monkeypatch):
    captured: dict = {}
    _fake_openai(
        monkeypatch,
        '{"summary":"clap","steps":[{"tool":"motion_bank","params":'
        '{"motion_id":"clap_single","anchor":"center"},"why":"add clap"}],"goals":[]}',
        captured,
    )
    plan = AE.plan_edit("add a clap here", {}, 1.0, 6.0, api_key="sk-x")
    assert plan.steps[0].params["anchor"] == "beat"
    assert plan.steps[0].params["intensity"] == AE.DEFAULT_MOTION_INTENSITY
    assert plan.steps[0].params["direction"] == "auto"
    assert plan.steps[0].params["mirror"] is False


def test_llm_cannot_invent_a_direction_but_explicit_direction_wins(monkeypatch):
    captured: dict = {}
    _fake_openai(
        monkeypatch,
        '{"summary":"clap","steps":[{"tool":"motion_bank","params":'
        '{"motion_id":"clap_single","direction":"right"},"why":"add clap"}],"goals":[]}',
        captured,
    )
    automatic = AE.plan_edit("add a clap here", {}, 1.0, 6.0, api_key="sk-x")
    assert automatic.steps[0].params["direction"] == "auto"

    explicit = AE.plan_edit("add a clap to the left", {}, 1.0, 6.0, api_key="sk-x")
    assert explicit.steps[0].params["direction"] == "left"


def test_explicit_named_motion_placement_overrides_its_default(monkeypatch):
    captured: dict = {}
    _fake_openai(
        monkeypatch,
        '{"summary":"late clap","steps":[{"tool":"motion_bank","params":'
        '{"motion_id":"clap_single","anchor":"center"},"why":"add clap"}],"goals":[]}',
        captured,
    )
    plan = AE.plan_edit("add a clap after the next move", {}, 1.0, 6.0, api_key="sk-x")
    assert plan.steps[0].params["anchor"] == "late"


def test_explicit_named_motion_intensity_overrides_the_default(monkeypatch):
    captured: dict = {}
    _fake_openai(
        monkeypatch,
        '{"summary":"neutral clap","steps":[{"tool":"motion_bank","params":'
        '{"motion_id":"clap_single","intensity":0.9},"why":"add clap"}],"goals":[]}',
        captured,
    )
    plan = AE.plan_edit("add a neutral clap here", {}, 1.0, 6.0, api_key="sk-x")
    assert plan.steps[0].params["intensity"] == 0.5


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("add a clap at intensity 0.72", 0.72),
        ("add a clap at 80% intensity", 0.8),
    ],
)
def test_numeric_named_motion_intensity_is_preserved(instruction, expected):
    plan = AE.plan_edit(instruction, {}, 1.0, 6.0)
    assert plan.steps[0].params["intensity"] == pytest.approx(expected)


def test_right_now_does_not_get_misread_as_a_spatial_direction():
    plan = AE.plan_edit("add a clap right now", {}, 1.0, 6.0)
    assert plan.steps[0].params["direction"] == "auto"


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


# ---------------------- lever dispatch: the single source both the planner and the guarantee share
def test_lever_for_covers_every_metric_tool():
    # completeness: every (metric, direction) the planner can target via _METRIC_TOOL MUST have a lever
    # in _lever_for -- otherwise the post-regen guarantee could not dial it. Catches registry/lever drift.
    ctx = {"wbeats": _beats(120)}
    for (metric, direction), (tool, _p) in AE._METRIC_TOOL.items():
        apply, strengths = AE._lever_for(metric, direction, ctx)
        assert apply is not None and len(strengths) > 0, f"{metric},{direction} ({tool}) has no lever"
        out = apply(_smooth(120, seed=1), strengths[0])
        assert isinstance(out, np.ndarray) and out.shape == (120, 139)


def test_lever_for_returns_none_for_metric_without_a_lever():
    assert AE._lever_for("foot", "up", {"wbeats": None}) == (None, ())


@pytest.mark.parametrize("metric, direction, push", [
    ("energy", "up", lambda T, m: T.accentuate(m, 0.55)),    # calmer take -> energy lever must raise it
    ("energy", "down", lambda T, m: T.accentuate(m, 1.6)),   # louder take -> energy lever must lower it
    ("jerk", "down", lambda T, m: T.accentuate(m, 1.8)),     # jerkier take -> smooth lever must lower jerk
    ("jerk", "up", lambda T, m: T.temporal_smooth(m, 0.5)),  # over-smoothed -> sharpen lever must raise jerk
])
def test_lever_for_moves_metric_in_the_goal_direction(metric, direction, push):
    from agentlodge.dance import transition as T
    m = _smooth(150, seed=6)
    off = push(T, m)
    apply, strengths = AE._lever_for(metric, direction, {"wbeats": None})
    strong = apply(off, strengths[-1])                       # strongest setting
    v0, v1 = window_metrics(off)[metric], window_metrics(strong)[metric]
    assert (v1 > v0) if direction == "up" else (v1 < v0)     # monotone lever moves the right way


# ---------------------------- agent policy: regenerate strips levers; the guarantee dials on splice
def test_plan_for_regen_strips_lever_steps_only_when_regenerating():
    p = AE.AgentPlan("x", steps=[AE.PlanStep("regenerate", {}, ""), AE.PlanStep("energy", {"direction": "up"}, ""), AE.PlanStep("beat_align", {}, "")], goals=[])
    assert [s.tool for s in AE._plan_for_regen(p).steps] == ["regenerate"]          # levers dropped
    p2 = AE.AgentPlan("x", steps=[AE.PlanStep("regenerate", {}, ""), AE.PlanStep("mirror", {}, "")], goals=[])
    assert [s.tool for s in AE._plan_for_regen(p2).steps] == ["regenerate", "mirror"]  # non-levers kept
    p3 = AE.AgentPlan("x", steps=[AE.PlanStep("energy", {"direction": "up"}, "")], goals=[])
    assert [s.tool for s in AE._plan_for_regen(p3).steps] == ["energy"]              # no regenerate -> unchanged


@pytest.mark.parametrize("metric, direction, push", [
    ("energy", "up", lambda T, m: T.accentuate(m, 0.5)),     # low-energy fresh take -> dial up to floor
    ("jerk", "down", lambda T, m: T.accentuate(m, 1.6)),     # jerky fresh take -> dial smoother
])
def test_reach_goals_after_regen_moves_the_spliced_window_toward_target(metric, direction, push):
    from agentlodge.dance import transition as T
    from agentlodge.dance.transition import crossfade_edit
    m = _smooth(300, seed=2)
    a, b = 90, 210
    before = window_metrics(m[a:b])
    off = push(T, m[a:b])                                    # a fresh take that misses the goal
    spl_off = window_metrics(crossfade_edit(m, a, b, off, blend_frames=15)[a:b])[metric]
    ctx = {"wbeats": None, "base_metrics": before}
    cur, steps = AE._reach_goals_after_regen(off, m, a, b, [(metric, direction, metric)], before, None, 15, ctx)
    spl_cur = window_metrics(crossfade_edit(m, a, b, cur, blend_frames=15)[a:b])[metric]
    assert steps and steps[0]["status"] == "applied"
    # the guarantee moved the SPLICED metric toward the goal (and reaches the original when the lever
    # range allows -- verified end-to-end for energy below)
    assert (spl_cur > spl_off) if direction == "up" else (spl_cur < spl_off)


# ============ THE reported user cases: a music-bound regenerate must ship a NEW motion on target ====
def _fixed_take_gen(clip):
    """A generator that always returns the SAME fixed clip (a 'music-bound' fresh take)."""
    class _G:
        def generate(self, bb, a, b, s, energy=0.5, beats=None, context=None):
            return np.asarray(clip, dtype=np.float32)
    return _G()


def test_regen_matches_energy_ships_new_motion_not_kept_original(monkeypatch):
    # EXACTLY the reported bug: "new motion that matches or exceeds the energy" where the diffusion
    # takes are LOWER energy. Old behaviour: kept the original. New: ship a new motion dialed up to
    # match/exceed on the spliced window. Also simulates the LLM adding a FIXED energy step (which
    # _plan_for_regen strips so the guarantee owns the dialing).
    from agentlodge.dance import transition as T
    m = _smooth(300, seed=1)
    base_win = m[90:210].copy()
    low_take = T.accentuate(T.mirror(base_win), 0.55)        # ~half energy, different motion
    plan = AE.AgentPlan("Create a new motion that matches or exceeds the energy level.",
                        steps=[AE.PlanStep("regenerate", {"backbone": "edge"}, "new"),
                               AE.PlanStep("energy", {"direction": "up", "amount": 0.9}, "match")],
                        goals=[("energy", "up", "energy")])
    monkeypatch.setattr(AE, "plan_edit", lambda *a, **k: AE.AgentPlan(
        plan.summary, [AE.PlanStep(s.tool, dict(s.params), s.why) for s in plan.steps], goals=list(plan.goals)))
    r = AE.run_agent_edit(m, 90, 210, "give me a new motion for this window that matches or exceeds the energy level",
                          generator=_fixed_take_gen(low_take), beats=_beats(), max_refine=2)
    fin = r.trace["final"]
    assert not fin["kept_original"]                          # NEVER keeps the original for a new-motion request
    assert fin["metrics_after"]["energy"] >= fin["metrics_before"]["energy"]   # matched or exceeded
    assert not np.array_equal(r.motion[90:210], base_win)    # genuinely new motion (not the original)
    assert any(s["tool"] == "regenerate" and s.get("status") == "applied" for s in r.log)


def test_regen_never_keeps_original_even_if_target_unreachable(monkeypatch):
    # if even max dialing can't fully reach the target, a NEW motion is still shipped (best effort) --
    # keeping the original defeats the whole point of a "give me a new motion" request.
    from agentlodge.dance import transition as T
    m = _smooth(300, seed=3)
    base_win = m[90:210].copy()
    tiny = T.accentuate(base_win, 0.2)                       # extremely low energy, hard to fully restore
    plan = AE.AgentPlan("new + match energy", steps=[AE.PlanStep("regenerate", {}, "new")],
                        goals=[("energy", "up", "energy")])
    monkeypatch.setattr(AE, "plan_edit", lambda *a, **k: AE.AgentPlan(
        plan.summary, [AE.PlanStep(s.tool, dict(s.params), s.why) for s in plan.steps], goals=list(plan.goals)))
    r = AE.run_agent_edit(m, 90, 210, "give me a totally new motion, at least as energetic",
                          generator=_fixed_take_gen(tiny), beats=_beats(), max_refine=1)
    assert not r.trace["final"]["kept_original"]             # ships the new motion regardless
    assert not np.array_equal(r.motion[90:210], base_win)


def test_pure_variety_regen_ships_new_motion_no_metric(monkeypatch):
    # "give me different moves" (no metric goal) must also ship a fresh take, never keep the original.
    from agentlodge.dance import transition as T
    m = _smooth(300, seed=4)
    base_win = m[90:210].copy()
    plan = AE.AgentPlan("regenerate", steps=[AE.PlanStep("regenerate", {}, "new")], goals=[])
    monkeypatch.setattr(AE, "plan_edit", lambda *a, **k: AE.AgentPlan(
        plan.summary, [AE.PlanStep(s.tool, dict(s.params), s.why) for s in plan.steps], goals=[]))
    r = AE.run_agent_edit(m, 90, 210, "give me completely different moves",
                          generator=_fixed_take_gen(T.mirror(base_win)), beats=_beats())
    assert not r.trace["final"]["kept_original"]
    assert not np.array_equal(r.motion[90:210], base_win)


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
