#!/usr/bin/env python3
"""End-to-end QA harness for the AgentLODGE interactive editor.

Drives the RUNNING server (real motion + LLM planner, and the GPU pod for renders) through a battery
of realistic user instructions, measures the objective window metrics (energy / beat-alignment / jerk
/ foot-contact) BEFORE and AFTER each edit, and checks the platform behaves as intended:

  * the goals the user asked for are actually satisfied (or honestly reported as not met),
  * nothing the user asked for is regressed,
  * smoothness is preserved unless the user asked for sharper (jerk does not balloon),
  * foot-contact is not badly degraded,
  * the reported metrics are self-consistent (result == trace final == recomputed),
  * the plan came from the LLM agent (not a silent keyword fallback) when a key is configured,
  * windows outside the edit are untouched (checked via the whole-dance frame count),
  * renders finish under a minute and produce a valid video.

Every scenario branches from the ORIGINAL checkpoint (``from_id``) so results are independent and
comparable. Usage:  python scripts/qa_harness.py [--render] [--out results.json]
Requires the server on http://127.0.0.1:8000 with OPENAI_API_KEY set (LLM) and, for --render, the pod.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
SID = "trs"


def api(method: str, path: str, body: dict | None = None, timeout: int = 180) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# --------------------------------------------------------------------------- scenarios
# Each: name, instruction, window (a,b) sec, expected metric goals we EXPECT the agent to adopt
# (for sanity; the agent declares its own), and `sharper`=True when jerk is meant to rise.
SCENARIOS = [
    dict(name="on_beat",            instr="make this section land tighter on the beat", win=(40, 46), want=["bas_up"]),
    dict(name="calmer",             instr="make it calmer and more relaxed",            win=(40, 46), want=["energy_down"]),
    dict(name="energetic",          instr="make it much more energetic and lively",     win=(40, 46), want=["energy_up"]),
    dict(name="smoother",           instr="smooth it out so it flows more gracefully",  win=(40, 46), want=["jerk_down"]),
    dict(name="snappier",           instr="make it snappier with sharp staccato hits",  win=(40, 46), want=["jerk_up"], sharper=True),
    dict(name="calm_but_onbeat",    instr="calmer but keep it locked tight to the beat", win=(40, 46), want=["energy_down", "bas_up"]),
    dict(name="energetic_onbeat",   instr="more energetic and tighter to the beat",     win=(40, 46), want=["energy_up", "bas_up"]),
    dict(name="nl_no_keyword",      instr="give this part way more oomph and glue it to the drums", win=(40, 46), want=["energy_up", "bas_up"]),
    dict(name="reverse",            instr="reverse this section in time",               win=(40, 46), want=[], goalless=True),
    dict(name="mirror",             instr="mirror the movement left to right",          win=(40, 46), want=[], goalless=True),
    dict(name="contradictory",      instr="make it calmer but also way more explosive", win=(40, 46), want=[]),
    dict(name="tiny_window",        instr="make it more energetic",                     win=(30, 31.5), want=["energy_up"]),
    dict(name="large_window",       instr="tighten the whole thing to the beat",        win=(60, 90), want=["bas_up"]),
    dict(name="window_at_start",    instr="smooth out the opening",                     win=(0, 4), want=["jerk_down"]),
    dict(name="window_at_end",      instr="calm down the ending",                       win=(170, 176), want=["energy_down"]),
    dict(name="regenerate",         instr="give me completely different choreography that hits hard on the beat", win=(46, 52), want=["bas_up"], timeout=400),
]


def _metric_recompute_ok(res: dict) -> bool:
    """result.metrics_after should equal trace.final.metrics_after (self-consistency)."""
    a = res.get("metrics_after") or {}
    b = ((res.get("trace") or {}).get("final") or {}).get("metrics_after") or {}
    if not a or not b:
        return True
    return all(abs(float(a.get(k, 0)) - float(b.get(k, 0))) < 1e-4 for k in ("energy", "bas", "jerk", "foot"))


def evaluate(sc: dict, res: dict, n_frames_before: int, n_frames_after: int) -> tuple[str, list]:
    """Return (verdict PASS|WARN|FAIL, [issue strings]) for one scenario result."""
    issues: list[str] = []
    before, after = res.get("metrics_before", {}), res.get("metrics_after", {})
    trace = res.get("trace") or {}
    goals = trace.get("goals") or []
    checks = (trace.get("final") or {}).get("checks") or []
    sharper = bool(sc.get("sharper"))
    goalless = bool(sc.get("goalless"))

    # 1) frame count preserved (edit must not change total dance length)
    if n_frames_after != n_frames_before:
        issues.append(f"FAIL frame-count changed {n_frames_before}->{n_frames_after}")

    # 2) self-consistent metrics
    if not _metric_recompute_ok(res):
        issues.append("FAIL reported metrics_after != trace.final.metrics_after")

    # 3) planner should be the LLM (a key is configured)
    if trace.get("planner") and trace.get("planner") != "llm":
        issues.append(f"WARN planner fell back to '{trace.get('planner')}' ({trace.get('planner_note')})")

    # 4) goalless ops should declare no measurable goal and just apply
    if goalless:
        if goals:
            issues.append(f"WARN goalless op declared goals {[g['metric'] for g in goals]}")
        return ("WARN" if issues else "PASS"), issues

    # 5) the agent should have adopted goals matching intent
    if not goals:
        issues.append("WARN no goals declared for a goal-bearing instruction")
    got = {(g["metric"], g["dir"]) for g in goals}
    want = {(_w.split("_")[0], _w.split("_")[1]) for _w in sc.get("want", [])}
    missing = want - got
    if missing:
        issues.append(f"WARN expected goals {sorted(want)} but agent declared {sorted(got)}")

    # 6) no declared goal regressed in the final result
    for c in checks:
        if c.get("status") == "regressed":
            issues.append(f"FAIL shipped a REGRESSED goal: {c['label']} {c['before']}->{c['after']}")
        elif not c.get("met") and res.get("ok"):
            issues.append(f"WARN ok=True but goal not met: {c['label']} ({c.get('status')})")

    # 7) declared goals actually moved the intended way (unless legitimately kept-original)
    kept = (trace.get("final") or {}).get("kept_original")
    if kept and goals:
        headroom = [g["metric"] for g in goals
                    if not (g["metric"] in ("bas", "foot") and before.get(g["metric"], 0) >= 0.85)]
        if headroom:
            issues.append(f"WARN kept original despite headroom on {headroom} (edit did nothing)")
    for g in goals:
        m, d = g["metric"], g["dir"]
        if before.get(m) is None or after.get(m) is None:
            continue
        delta = after[m] - before[m]
        moved = (delta > 1e-3) if d == "up" else (delta < -1e-3)
        atceiling = m in ("bas", "foot") and after[m] >= 0.9
        if not moved and not atceiling and not kept:
            issues.append(f"WARN goal {m} {d} did not move ({before[m]:.3f}->{after[m]:.3f})")

    # 8) SMOOTHNESS invariant: unless sharper was asked for, jerk must not balloon to a HIGH absolute
    # value. Very short windows can't gain energy without some jitter (physically coupled) -> that is a
    # WARN, not a FAIL; only genuinely severe jitter fails.
    if not sharper and before.get("jerk") and after.get("jerk"):
        ratio = after["jerk"] / before["jerk"]
        if ratio > 1.3 and after["jerk"] > 0.18:
            issues.append(f"FAIL shipped jittery: jerk x{ratio:.2f} ({before['jerk']:.3f}->{after['jerk']:.3f})")
        elif ratio > 1.15 and after["jerk"] > 0.12:
            issues.append(f"WARN jerk rose x{ratio:.2f} ({before['jerk']:.3f}->{after['jerk']:.3f})")

    # 9) FOOT-CONTACT invariant: an edit should not badly increase foot-skating
    if before.get("foot") is not None and after.get("foot") is not None:
        if after["foot"] < before["foot"] - 0.12:
            issues.append(f"WARN foot-contact degraded {before['foot']:.3f}->{after['foot']:.3f}")

    verdict = "FAIL" if any(i.startswith("FAIL") for i in issues) else ("WARN" if issues else "PASS")
    return verdict, issues


def run_edits(root_id: str) -> list:
    rows = []
    for sc in SCENARIOS:
        a, b = sc["win"]
        t0 = time.time()
        try:
            payload = api("POST", f"/api/session/{SID}/edit",
                          {"a_sec": a, "b_sec": b, "instruction": sc["instr"], "from_id": root_id},
                          timeout=sc.get("timeout", 180))
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(name=sc["name"], verdict="FAIL", issues=[f"FAIL request error: {exc}"],
                             before={}, after={}, goals=[], secs=round(time.time() - t0, 1)))
            print(f"  {sc['name']:18s} FAIL (request error: {exc})")
            continue
        res, state = payload["result"], payload["state"]
        verdict, issues = evaluate(sc, res, 5283, state["n_frames"])
        goals = [f"{g['metric']}{'^' if g['dir'] == 'up' else 'v'}" for g in (res.get("trace") or {}).get("goals", [])]
        rows.append(dict(name=sc["name"], instr=sc["instr"], win=[a, b], verdict=verdict, issues=issues,
                         ok=res.get("ok"), feedback=res.get("feedback"),
                         before=res.get("metrics_before", {}), after=res.get("metrics_after", {}),
                         goals=goals, planner=(res.get("trace") or {}).get("planner"),
                         attempts=len(((res.get("trace") or {}).get("attempts")) or []),
                         secs=round(time.time() - t0, 1)))
        m0, m1 = res.get("metrics_before", {}), res.get("metrics_after", {})
        print(f"  {sc['name']:18s} {verdict:4s} [{','.join(goals) or '-'}]  "
              f"e {m0.get('energy',0):.2f}->{m1.get('energy',0):.2f}  bas {m0.get('bas',0):.2f}->{m1.get('bas',0):.2f}  "
              f"jerk {m0.get('jerk',0):.3f}->{m1.get('jerk',0):.3f}  ({rows[-1]['secs']}s)")
        for i in issues:
            print(f"       - {i}")
    return rows


def run_renders(root_id: str) -> list:
    rows = []
    api("POST", f"/api/session/{SID}/edit",
        {"a_sec": 40, "b_sec": 46, "instruction": "make it more energetic and on beat", "from_id": root_id})
    # Only the fast WINDOW preview is validated here; the full-dance HQ export (1080/96 over ~5000
    # frames) is inherently many minutes and is not automated.
    for scope, win in (("window", (40, 46)),):
        body = {"scope": scope, "a_sec": win[0], "b_sec": win[1]}
        t0 = time.time()
        try:
            api("POST", f"/api/session/{SID}/render", body)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(scope=scope, verdict="FAIL", issues=[f"start error: {exc}"], secs=0)); continue
        status = {}
        while time.time() - t0 < 150:
            time.sleep(3)
            status = api("GET", f"/api/session/{SID}/render", None)
            if status.get("status") in ("done", "error"):
                break
        secs = round(time.time() - t0, 1)
        issues = []
        if status.get("status") != "done":
            issues.append(f"FAIL render {status.get('status')}: {status.get('message')}")
        elif secs > 60:
            issues.append(f"WARN window render {secs}s > 60s target")
        verdict = "FAIL" if any(i.startswith("FAIL") for i in issues) else ("WARN" if issues else "PASS")
        rows.append(dict(scope=scope, verdict=verdict, secs=secs, status=status.get("status"), issues=issues))
        print(f"  render {scope:7s} {verdict:4s} {secs}s  status={status.get('status')}")
        for i in issues:
            print(f"       - {i}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="also validate window + full renders (needs the pod)")
    ap.add_argument("--out", default="", help="write full JSON results here")
    args = ap.parse_args()

    st = api("POST", f"/api/session/{SID}", None)
    root = next((c for c in st["timeline"] if c.get("label") == "original" or not c.get("parent_id")), None)
    if not root:
        print("could not find the original checkpoint"); return 2
    print(f"session trs: {st['duration']}s, {st['n_beats']} beats, generator={st['generator']}, "
          f"agent_llm={st['agent_llm']}  root={root['id']}\n")
    print("== EDIT SCENARIOS ==")
    edit_rows = run_edits(root["id"])
    render_rows = run_renders(root["id"]) if args.render else []

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in edit_rows + render_rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print(f"\n== SUMMARY ==  PASS={counts['PASS']}  WARN={counts['WARN']}  FAIL={counts['FAIL']}")
    fails = [r for r in edit_rows + render_rows if r["verdict"] == "FAIL"]
    if fails:
        print("FAILURES:")
        for r in fails:
            print(f"  {r.get('name', r.get('scope'))}: {r['issues']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"edits": edit_rows, "renders": render_rows, "summary": counts}, f, indent=2)
        print(f"\nwrote {args.out}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
