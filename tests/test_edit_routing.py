"""Executable SPEC for how instructions must route to tools + goals (offline keyword planner).

The LLM planner is prompted to make exactly these decisions; the offline keyword planner is the
deterministic reference implementation AND the no-API-key fallback, so testing it here both (a)
locks the intended routing as a spec and (b) guards the fallback. These tests are the regression
net for the classes of mistakes seen in practice:

  * "create a new motion ..." got AMPLIFIED instead of REGENERATED (missing the regenerate step),
  * a metric-only request ("more energetic") HALLUCINATED an extra goal (beat alignment),
  * a timing request ("tighten to the beat") picked up a spurious energy/jerk goal.

Every case asserts BOTH the tool sequence and the EXACT goal set, so an extra/missing goal fails.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor import agent_edit as AE


def _route(instruction: str):
    """(ordered tool names, set of (metric, direction) goals) from the offline planner."""
    p = AE.plan_edit(instruction, {}, 46.0, 52.0)         # no api_key -> deterministic keyword planner
    return [s.tool for s in p.steps], {(m, d) for m, d, _lbl in p.goals}


# ------------------------------------------------------------- single deterministic metric levers
@pytest.mark.parametrize("instruction, tool, goals", [
    ("make it more energetic", "energy", {("energy", "up")}),
    ("bigger, stronger, more intense", "energy", {("energy", "up")}),
    ("way more energetic please", "energy", {("energy", "up")}),
    ("calmer", "energy", {("energy", "down")}),
    ("softer and gentler, tone it down", "energy", {("energy", "down")}),
    ("smoother", "smooth", {("jerk", "down")}),
    ("make it flow, more graceful", "smooth", {("jerk", "down")}),
    ("snappier", "sharpen", {("jerk", "up")}),
    ("punchier staccato hits", "sharpen", {("jerk", "up")}),
    ("tighten it to the beat", "beat_align", {("bas", "up")}),
    ("lock it to the grid, in time", "beat_align", {("bas", "up")}),
])
def test_single_metric_routes_to_its_lever_with_exact_goal(instruction, tool, goals):
    tools, g = _route(instruction)
    assert tools[0] == tool, f"{instruction!r} -> {tools}, expected {tool} first"
    assert g == goals, f"{instruction!r} goals {g} != {goals}"


# ------------------------------------------------------------------------ exact transforms (no goal)
@pytest.mark.parametrize("instruction, tool", [
    ("reverse this part", "reverse"),
    ("play it backward", "reverse"),
    ("mirror it left to right", "mirror"),
    ("flip it", "mirror"),
])
def test_exact_transforms_have_no_goals(instruction, tool):
    tools, g = _route(instruction)
    assert tools == [tool] and g == set()


# ------------------------------------------------------------------ VARIETY / NEW MOTION -> regenerate
# THE reported bug: "create a new motion ..." must REGENERATE (create new choreography), never just
# reshape the current window with a metric lever.
@pytest.mark.parametrize("instruction", [
    "give me something different",
    "freestyle here",
    "generate new choreography",
    "create a new motion for this window",
    "create a new motion for this window that matches or exceeds the energy of the past edit",
    "choreograph something new",
    "come up with new moves",
    "surprise me",
    "mix it up",
    "make new moves here",
    "brand new choreography",
])
def test_new_motion_requests_regenerate(instruction):
    tools, _g = _route(instruction)
    assert "regenerate" in tools, f"{instruction!r} must include a regenerate step, got {tools}"
    assert tools[0] == "regenerate", f"{instruction!r} must regenerate FIRST, got {tools}"


# --------------------------------------------------------- COMPOUND: new motion AND a metric target
# regenerate FIRST (creates the motion) then the metric lever (dials it onto the target); goal = the
# metric ONLY -- nothing the user did not ask for.
@pytest.mark.parametrize("instruction, second, goals", [
    ("give me different, more energetic moves", "energy", {("energy", "up")}),
    ("choreograph something new but calmer", "energy", {("energy", "down")}),
    ("new moves, snappier", "sharpen", {("jerk", "up")}),
    ("fresh choreography, smoother", "smooth", {("jerk", "down")}),
])
def test_compound_new_plus_metric_regenerates_then_dials(instruction, second, goals):
    tools, g = _route(instruction)
    assert tools[0] == "regenerate", f"{instruction!r} must regenerate first, got {tools}"
    assert second in tools, f"{instruction!r} must also dial with {second}, got {tools}"
    assert g == goals, f"{instruction!r} goals {g} != {goals}"


# ------------------------------------------------------------------------- GOAL DISCIPLINE (no extras)
# The core failure mode: adding a goal the user never mentioned. Each of these must carry EXACTLY the
# one goal it names -- an extra goal here would have failed the real edit.
@pytest.mark.parametrize("instruction, goals", [
    ("make it more energetic", {("energy", "up")}),          # NOT beat, NOT jerk
    ("tighten it to the beat", {("bas", "up")}),             # NOT energy, NOT jerk
    ("smoother", {("jerk", "down")}),                        # NOT energy, NOT beat
    ("calmer", {("energy", "down")}),
    ("snappier", {("jerk", "up")}),
])
def test_no_hallucinated_goals(instruction, goals):
    _tools, g = _route(instruction)
    assert g == goals, f"{instruction!r} added goals it should not have: {g - goals}"


# ------------------------------------------------------------------------ multi-metric (explicit only)
def test_multi_metric_when_user_asks_for_both():
    tools, g = _route("more energetic and on beat")
    assert g == {("energy", "up"), ("bas", "up")}
    assert "energy" in tools and "beat_align" in tools
    assert tools[-1] == "beat_align"                          # timing re-applied LAST


def test_calmer_but_tight_is_two_goals():
    tools, g = _route("calmer but keep it tight to the beat")
    assert g == {("energy", "down"), ("bas", "up")}
    assert "energy" in tools and "beat_align" in tools


# --------------------------------------------------------------- goal extraction (deterministic core)
@pytest.mark.parametrize("instruction, expected", [
    ("more energetic and on beat", {("energy", "up"), ("bas", "up")}),
    ("tighten to the beat", {("bas", "up")}),
    ("smoother and flowing", {("jerk", "down")}),
    ("snappier staccato", {("jerk", "up")}),
    ("just calmer", {("energy", "down")}),
])
def test_requested_metrics_extracts_only_named(instruction, expected):
    got = {(m, d) for m, d, _lbl in AE._requested_metrics(instruction)}
    assert got == expected


def test_conflicting_directions_cancel():
    # "more energetic" and "calmer" both fire for energy -> ambiguous -> energy is dropped, not guessed
    got = {m for m, _d, _lbl in AE._requested_metrics("make it more energetic but also calmer")}
    assert "energy" not in got


# ----------------------------------------------------------------------------- tool-target contracts
@pytest.mark.parametrize("tool, params, target", [
    ("beat_align", {}, ("bas", "up")),
    ("smooth", {}, ("jerk", "down")),
    ("sharpen", {}, ("jerk", "up")),
    ("energy", {"direction": "up"}, ("energy", "up")),
    ("energy", {"direction": "down"}, ("energy", "down")),
    ("mirror", {}, None),
    ("reverse", {}, None),
    ("regenerate", {}, None),
])
def test_tool_target_contracts(tool, params, target):
    assert AE._tool_target(tool, params) == target
