"""Build a blind, paired real-host visual audit for every named motion.

Each unlabeled edit is paired with the exact source choreography so the reviewer identifies what
the named edit added instead of accidentally scoring a jump, step, or turn already present in the
host. The review page locks and persists a guess before it can load the separate answer key.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import secrets
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.dance.format import to_editor139  # noqa: E402
from agentlodge.editor.motion_bank import (  # noqa: E402
    MotionBank,
    _root_yaw_series,
    _yaw_rotate,
)
from scripts.build_motion_bank import build_motion  # noqa: E402
from server.fk import save_poses_npz  # noqa: E402


CONTRACTS = ROOT / "assets" / "motion_bank" / "visual_contracts.json"


def _new_audit_id(output: Path, seed: int) -> str:
    """Return a fresh browser-storage namespace for one generated audit."""
    return f"{output.name}-seed-{int(seed)}-{secrets.token_hex(8)}"


def _load_vector(path: Path | None, fallback: np.ndarray) -> np.ndarray:
    if path is None:
        return fallback
    values = np.asarray(np.load(path)).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite values")
    return values


def _paired_views(
    edited: np.ndarray,
    reference: np.ndarray,
    action_start: int,
    *,
    normalize_facing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return matched front/side edit and reference views.

    A faceless audit rig makes forward and backward impossible to score reliably when every host
    window arrives at an arbitrary world heading. Semantic audits therefore rotate both clips by
    the same amount so root yaw is zero when the action starts. Production-camera audits can opt
    out and retain the song's original heading.
    """
    action_start = int(np.clip(action_start, 0, len(edited) - 1))
    front = np.ascontiguousarray(edited, dtype=np.float32)
    control = np.ascontiguousarray(reference, dtype=np.float32)
    heading = float(_root_yaw_series(front[action_start:action_start + 1])[0])
    if normalize_facing:
        front = _yaw_rotate(front.copy(), -heading, front[action_start, :3].copy())
        control = _yaw_rotate(
            control.copy(),
            -heading,
            control[action_start, :3].copy(),
        )
    side = _yaw_rotate(front.copy(), np.pi / 2.0, front[action_start, :3].copy())
    control_side = _yaw_rotate(
        control.copy(),
        np.pi / 2.0,
        control[action_start, :3].copy(),
    )
    return front, side, control, control_side, heading


def _review_html(
    takes: list[dict],
    rule: str,
    *,
    audit_id: str,
    normalized_facing: bool,
) -> str:
    cards = []
    for take in takes:
        key = take["take"]
        control = take["control"]
        cards.append(
            f"""
            <article class="take" data-take="{html.escape(key)}">
              <h2>{html.escape(key)}</h2>
              <div class="pair">
                <section>
                  <h3>Source choreography: front left, side right</h3>
                  <video class="control" controls loop muted playsinline preload="metadata"
                         src="videos/{html.escape(control)}.mp4"></video>
                </section>
                <section>
                  <h3>Unlabeled edit: front left, side right</h3>
                  <video class="edited" controls loop muted playsinline preload="metadata"
                         src="videos/{html.escape(key)}.mp4"></video>
                </section>
              </div>
              <button class="play-pair" type="button">Play both from the start</button>
              <p><a href="phase_sheets/{html.escape(key)}_review.html" target="_blank"
                    rel="noopener">Open synchronized views and edit-minus-source strip</a></p>
              <label>What action did you see?
                <input class="guess" type="text" autocomplete="off" spellcheck="false">
              </label>
              <div class="actions">
                <button class="lock" type="button">Lock guess</button>
              </div>
              <p class="status" aria-live="polite"></p>
              <section class="answer" hidden>
                <p><strong class="answer-name"></strong> (<code class="answer-id"></code>)</p>
                <p>Must read as: <span class="recognizable"></span></p>
                <ul class="phases"></ul>
              </section>
            </article>
            """
        )
    facing_note = (
        "<p>The views are normalized to the dancer's heading at action start. "
        "In the side view, the dancer faces screen right.</p>"
        if normalized_facing
        else "<p>The views preserve the production heading from the source choreography.</p>"
    )
    audit_json = json.dumps(str(audit_id))
    take_ids_json = json.dumps([take["take"] for take in takes])
    normalized_json = "true" if normalized_facing else "false"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MAESTRO blind motion audit</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: system-ui, sans-serif;
            color: #eee; background: #17171b; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(640px, 1fr));
            gap: 20px; }}
    .instructions {{ max-width: 1000px; margin: 0 auto 24px; line-height: 1.5; }}
    .take {{ background: #26262d; border: 1px solid #444; border-radius: 12px;
             padding: 16px; }}
    .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    h3 {{ margin: 0 0 8px; font-size: 0.95rem; color: #ccc; }}
    video {{ width: 100%; background: #000; }}
    label {{ display: block; margin: 12px 0; }}
    input {{ box-sizing: border-box; display: block; width: 100%; margin-top: 6px;
             padding: 10px; color: #fff; background: #111; border: 1px solid #666; }}
    button {{ margin-right: 8px; padding: 8px 12px; cursor: pointer; }}
    button:disabled {{ cursor: not-allowed; opacity: 0.55; }}
    .status {{ min-height: 1.5em; color: #d8ccff; }}
    .answer {{ margin-top: 12px; padding: 12px; border: 1px solid #6555a5;
               border-radius: 8px; background: #1c1928; }}
    code {{ color: #d8ccff; }}
    @media (max-width: 760px) {{
      main {{ grid-template-columns: 1fr; }}
      .pair {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <section class="instructions">
    <h1>MAESTRO blind motion audit</h1>
    <p><strong>Pass rule:</strong> {html.escape(rule)}</p>
    <p>Compare the source choreography with the edited take, then name only the action that was
       added. The answer key is not loaded until every take has a non-empty locked guess.</p>
    {facing_note}
    <ol>
      <li>Play the paired source and edit at normal speed before pausing or scrubbing.</li>
      <li>Use the synchronized review and difference strip when the host motion could be mistaken
          for the edit.</li>
      <li>Record and lock the action added by the edit.</li>
      <li>If the action is unclear or guessed incorrectly, mark it failed.</li>
      <li>After revealing, inspect every required phase from both front and side.</li>
    </ol>
    <button id="export" type="button">Export locked audit results</button>
    <button id="reveal-all" type="button" disabled>Lock every guess before reveal</button>
    <span id="reveal-status" class="status" aria-live="polite"></span>
  </section>
  <main>
    {''.join(cards)}
  </main>
  <script>
    const auditId = {audit_json};
    const takeIds = {take_ids_json};
    const normalizedFacing = {normalized_json};
    const storageKey = `maestro-motion-audit:${{auditId}}`;
    let state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    let answersPromise = null;

    function save() {{
      localStorage.setItem(storageKey, JSON.stringify(state));
    }}

    function normalize(value) {{
      return String(value || "").toLowerCase().replace(/[_-]+/g, " ")
        .replace(/[^a-z0-9 ]+/g, "").replace(/\\s+/g, " ").trim();
    }}

    function loadAnswers() {{
      if (!answersPromise) {{
        answersPromise = fetch("answer_key.json", {{cache: "no-store"}}).then(response => {{
          if (!response.ok) throw new Error(`answer key request failed: ${{response.status}}`);
          return response.json();
        }});
      }}
      return answersPromise;
    }}

    function allGuessesLocked() {{
      return takeIds.length > 0 && takeIds.every(take =>
        state[take] && String(state[take].guess || "").trim() && state[take].lockedAt
      );
    }}

    function allAnswersRevealed() {{
      return takeIds.length > 0 && takeIds.every(take =>
        state[take] && state[take].revealedAt
      );
    }}

    function refreshRevealAvailability() {{
      const revealAll = document.querySelector("#reveal-all");
      const remaining = takeIds.filter(take =>
        !(state[take] && String(state[take].guess || "").trim() && state[take].lockedAt)
      ).length;
      revealAll.disabled = !allGuessesLocked() || allAnswersRevealed();
      if (allAnswersRevealed()) {{
        revealAll.textContent = "Answers revealed";
      }} else if (remaining === 0) {{
        revealAll.textContent = "Reveal all answers";
      }} else {{
        revealAll.textContent =
          `Lock ${{remaining}} remaining guess${{remaining === 1 ? "" : "es"}} before reveal`;
      }}
    }}

    function restore(card) {{
      const take = card.dataset.take;
      const saved = state[take];
      if (!saved) return;
      const input = card.querySelector(".guess");
      input.value = saved.guess;
      input.disabled = true;
      card.querySelector(".lock").disabled = true;
      card.querySelector(".status").textContent = saved.revealedAt
        ? `Locked: "${{saved.guess}}" - answer already revealed`
        : `Locked: "${{saved.guess}}"`;
    }}

    function renderAnswer(card, answer) {{
      const take = card.dataset.take;
      if (!answer) throw new Error(`missing answer for ${{take}}`);
      const accepted = [answer.id, answer.name, ...(answer.aliases || [])].map(normalize);
      const recognized = accepted.includes(normalize(state[take].guess));
      state[take] = {{
        ...state[take],
        actual: answer.id,
        recognized,
        revealedAt: state[take].revealedAt || new Date().toISOString(),
      }};
      card.querySelector(".answer-name").textContent = answer.name;
      card.querySelector(".answer-id").textContent = answer.id;
      card.querySelector(".recognizable").textContent =
        answer.visual_contract.recognizable_as;
      const phases = card.querySelector(".phases");
      phases.replaceChildren();
      answer.visual_contract.required_phases.forEach(phase => {{
        const item = document.createElement("li");
        item.textContent = phase;
        phases.appendChild(item);
      }});
      card.querySelector(".answer").hidden = false;
      card.querySelector(".status").textContent = recognized
        ? `PASS - locked guess matched ${{answer.id}}`
        : `FAIL - locked guess was "${{state[take].guess}}", answer is ${{answer.id}}`;
    }}

    document.querySelectorAll(".take").forEach(card => {{
      restore(card);
      const take = card.dataset.take;
      const input = card.querySelector(".guess");
      const lock = card.querySelector(".lock");
      const status = card.querySelector(".status");

      card.querySelector(".play-pair").addEventListener("click", () => {{
        const videos = [...card.querySelectorAll("video")];
        videos.forEach(video => {{ video.pause(); video.currentTime = 0; }});
        Promise.all(videos.map(video => video.play())).catch(() => {{}});
      }});

      lock.addEventListener("click", () => {{
        const guess = input.value.trim();
        if (!guess) {{
          status.textContent = "Enter a guess before locking.";
          input.focus();
          return;
        }}
        state[take] = {{guess, lockedAt: new Date().toISOString()}};
        save();
        input.disabled = true;
        lock.disabled = true;
        status.textContent = `Locked: "${{guess}}"`;
        refreshRevealAvailability();
      }});
    }});

    const revealAll = document.querySelector("#reveal-all");
    const revealStatus = document.querySelector("#reveal-status");
    revealAll.addEventListener("click", async () => {{
      if (!allGuessesLocked()) {{
        revealStatus.textContent = "Lock every guess before loading the answer key.";
        refreshRevealAvailability();
        return;
      }}
      revealAll.disabled = true;
      revealStatus.textContent = "Loading answer key...";
      try {{
        const answers = await loadAnswers();
        document.querySelectorAll(".take").forEach(card => {{
          renderAnswer(card, answers[card.dataset.take]);
        }});
        save();
        revealStatus.textContent = "All answers revealed. Inspect every required phase.";
        refreshRevealAvailability();
      }} catch (error) {{
        revealStatus.textContent = `Could not reveal answers: ${{error.message}}`;
        refreshRevealAvailability();
      }}
    }});

    refreshRevealAvailability();
    if (allAnswersRevealed()) {{
      loadAnswers().then(answers => {{
        document.querySelectorAll(".take").forEach(card => {{
          renderAnswer(card, answers[card.dataset.take]);
        }});
        revealStatus.textContent = "Previously revealed answers restored.";
      }}).catch(error => {{
        revealStatus.textContent = `Could not restore answers: ${{error.message}}`;
      }});
    }}

    document.querySelector("#export").addEventListener("click", () => {{
      const payload = {{
        audit: auditId,
        normalized_facing: normalizedFacing,
        exported_at: new Date().toISOString(),
        guesses: state,
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${{auditId}}-locked-results.json`;
      link.click();
      URL.revokeObjectURL(link.href);
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True, help="real 139-channel host dance")
    parser.add_argument("--beats", type=Path, help="optional beat-frame .npy")
    parser.add_argument("--beat-strengths", type=Path, help="optional beat-strength .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--preserve-heading",
        action="store_true",
        help="keep the source world heading instead of normalizing the semantic review views",
    )
    parser.add_argument(
        "--motions",
        nargs="+",
        help="optional motion ids to re-audit instead of rendering the full bank",
    )
    args = parser.parse_args()

    payload = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    contracts = payload["motions"]
    bank = MotionBank()
    bank_ids = {spec.id for spec in bank.specs}
    if set(contracts) != bank_ids:
        missing = sorted(bank_ids - set(contracts))
        extra = sorted(set(contracts) - bank_ids)
        raise ValueError(f"visual contracts do not match the bank; missing={missing}, extra={extra}")

    host = to_editor139(np.load(args.host).astype(np.float32))
    frame_count = min(max(24, int(args.frames)), host.shape[0])
    host = np.ascontiguousarray(host[:frame_count], dtype=np.float32)
    default_beats = np.arange(0, frame_count, 15, dtype=np.int64)
    beats = _load_vector(args.beats, default_beats).astype(np.int64)
    valid = (beats >= 0) & (beats < frame_count)
    beats = beats[valid]
    if beats.size < 2:
        raise ValueError("the audit needs at least two in-range beats to preserve action tempo")
    strengths = None
    if args.beat_strengths is not None:
        strengths = _load_vector(args.beat_strengths, np.ones_like(beats, dtype=np.float32))
        if strengths.shape != valid.shape:
            raise ValueError("beat strengths must contain one value per unfiltered beat")
        strengths = strengths[valid].astype(np.float32)

    specs = list(bank.specs)
    if args.motions:
        requested = {motion.strip() for value in args.motions for motion in value.split(",")}
        unknown = sorted(requested - bank_ids)
        if unknown:
            raise ValueError(f"unknown motion ids: {unknown}")
        specs = [spec for spec in specs if spec.id in requested]
    for spec in specs:
        if not np.allclose(
            bank.load_clip(spec),
            build_motion(spec.id, spec.frames),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                f"{spec.id}: committed clip is stale; run scripts/build_motion_bank.py first"
            )
    random.Random(args.seed).shuffle(specs)
    args.output.mkdir(parents=True, exist_ok=True)
    audit_id = _new_audit_id(args.output, args.seed)
    takes: list[dict] = []
    controls: list[dict] = []
    control_keys: dict[tuple[int, int, int], str] = {}
    answers: dict[str, dict] = {}
    context = max(0, int(args.context))
    normalize_facing = not bool(args.preserve_heading)
    for index, spec in enumerate(specs, start=1):
        take = f"take_{index:02d}"
        motion, report = bank.apply(
            host,
            spec.id,
            beats=beats,
            beat_strengths=strengths,
        )
        start, end = map(int, report["action_range"])
        lo = max(0, start - context)
        hi = min(len(motion), end + context)
        action_start = start - lo
        front, side, control_front, control_side, heading = _paired_views(
            motion[lo:hi],
            host[lo:hi],
            action_start,
            normalize_facing=normalize_facing,
        )
        control_key = (lo, hi, int(round(heading * 1_000_000)))
        control = control_keys.get(control_key)
        if control is None:
            control = f"control_{len(control_keys) + 1:02d}"
            control_keys[control_key] = control
            np.save(args.output / f"{control}.npy", control_front)
            save_poses_npz(control_front, args.output / f"{control}_front.npz")
            save_poses_npz(control_side, args.output / f"{control}_side.npz")
            controls.append({"control": control, "frames": int(len(control_front))})
        np.save(args.output / f"{take}.npy", front)
        save_poses_npz(front, args.output / f"{take}_front.npz")
        save_poses_npz(side, args.output / f"{take}_side.npz")

        takes.append({
            "take": take,
            "control": control,
            "frames": int(len(front)),
            "action_range": [start - lo, end - lo],
            "event_frame": int(report["event_frame"]) - lo,
        })
        answers[take] = {
            "id": spec.id,
            "name": spec.name,
            "aliases": list(spec.aliases),
            "visual_contract": contracts[spec.id],
            "report": report,
        }

    review = {
        "audit_id": audit_id,
        "fps": 30,
        "fixed_camera": True,
        "normalized_facing": normalize_facing,
        "review_protocol_version": 2,
        "seed": int(args.seed),
        "acceptance_rule": payload["acceptance_rule"],
        "controls": controls,
        "takes": takes,
    }
    (args.output / "review.json").write_text(
        json.dumps(review, indent=2), encoding="utf-8"
    )
    (args.output / "answer_key.json").write_text(
        json.dumps(answers, indent=2), encoding="utf-8"
    )
    (args.output / "review.html").write_text(
        _review_html(
            takes,
            payload["acceptance_rule"],
            audit_id=audit_id,
            normalized_facing=normalize_facing,
        ),
        encoding="utf-8",
    )
    print(f"built {len(takes)} blind audit takes in {args.output}")


if __name__ == "__main__":
    main()
