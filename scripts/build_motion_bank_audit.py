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
import shutil
import sys
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentlodge.dance.format import to_editor139  # noqa: E402
from agentlodge.dance.transition import _sixd_to_matrix  # noqa: E402
from agentlodge.editor.motion_audit import (  # noqa: E402
    REVIEW_PROTOCOL_VERSION,
    audit_case_id,
    audit_variants,
    motion_fingerprint,
)
from agentlodge.editor.motion_bank import (  # noqa: E402
    MotionBank,
    _COUNTERFLOW_CLAP_JOINTS,
    _COUNTERFLOW_CLAP_TURN_DEGREES,
    _MIRROR_JOINTS,
    _fit_event_frame,
    _root_yaw_series,
    _yaw_rotate,
)
from scripts.build_motion_bank import build_motion  # noqa: E402
from server.fk import BODY_PARENTS, compute_poses, save_poses_npz  # noqa: E402


CONTRACTS = ROOT / "assets" / "motion_bank" / "visual_contracts.json"


def _new_audit_id(output: Path, seed: int) -> str:
    """Return a fresh browser-storage namespace for one generated audit."""
    return f"{output.name}-seed-{int(seed)}-{secrets.token_hex(8)}"


def _prepare_output(output: Path) -> None:
    """Remove only generated audit artifacts before reusing an output directory."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("videos", "phase_sheets"):
        path = output / name
        if path.is_dir():
            shutil.rmtree(path)
    for path in output.glob("*_frames"):
        if path.is_dir():
            shutil.rmtree(path)
    for pattern in (
        "take_*.npy",
        "take_*_front.npz",
        "take_*_side.npz",
        "take_*_front.log",
        "take_*_side.log",
        "control_*.npy",
        "control_*_front.npz",
        "control_*_side.npz",
        "control_*_front.log",
        "control_*_side.log",
    ):
        for path in output.glob(pattern):
            if path.is_file():
                path.unlink()
    for name in (
        "answer_key.json",
        "render_receipt.json",
        "review.html",
        "review.json",
    ):
        path = output / name
        if path.is_file():
            path.unlink()


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


def _global_joint_rotations(motion: np.ndarray) -> np.ndarray:
    local = _sixd_to_matrix(np.asarray(motion)[:, 3:135].reshape(-1, 22, 6))
    global_r = np.empty_like(local)
    global_r[:, 0] = local[:, 0]
    for joint in range(1, 22):
        global_r[:, joint] = global_r[:, BODY_PARENTS[joint]] @ local[:, joint]
    return global_r


def _clap_contact_lateral(edited: np.ndarray, event: int) -> float:
    joints = compute_poses(edited)["fk_joints"]
    root_yaw = float(_root_yaw_series(edited[event:event + 1])[0])
    local_left = np.array([np.cos(root_yaw), np.sin(root_yaw)], dtype=np.float32)
    hand_center = 0.5 * (joints[event, 20] + joints[event, 21]) - joints[event, 0]
    return float(hand_center[:2] @ local_left)


def _audit_auto_direction(base: np.ndarray, spec) -> str:
    """Independently derive the direction that an automatic edit should follow."""
    clip = np.asarray(base, dtype=np.float32)
    n = int(clip.shape[0])
    flow = "forward"
    if n >= 4:
        width = int(np.clip(n // 5, 2, 18))
        delta = np.mean(clip[-width:, :2], axis=0) - np.mean(clip[:width, :2], axis=0)
        yaw_series = _root_yaw_series(clip)
        center = yaw_series[
            max(0, n // 2 - width // 2):min(n, n // 2 + width // 2 + 1)
        ]
        yaw = float(np.median(center))
        local_left = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        local_forward = np.array([-np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        lateral = float(delta @ local_left)
        forward = float(delta @ local_forward)
        if abs(lateral) >= 0.04 and abs(lateral) >= 0.45 * abs(forward):
            flow = "left" if lateral > 0.0 else "right"
        else:
            yaw_change = float(yaw_series[-1] - yaw_series[0])
            if abs(yaw_change) >= 0.25:
                flow = "left" if yaw_change > 0.0 else "right"
    if flow in spec.directions:
        return flow
    if "forward" in spec.directions:
        return "forward"
    return spec.canonical_direction


def _declared_clap_contacts(spec, report: dict, gaps: np.ndarray) -> tuple[int, ...]:
    start, end = map(int, report["action_range"])
    local_event = int(report["event_frame"]) - start
    runs: list[list[int]] = []
    for frame in spec.intensity_lock_frames:
        if not runs or frame != runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    contacts = []
    for run in runs:
        mapped = {
            start + _fit_event_frame(
                frame,
                spec.frames,
                spec.event_frame,
                end - start,
                local_event,
            )
            for frame in run
        }
        contacts.append(min(mapped, key=lambda frame: float(gaps[frame])))
    return tuple(contacts)


def _machine_checks(
    host: np.ndarray,
    edited: np.ndarray,
    spec,
    report: dict,
) -> tuple[list[dict], str]:
    """Evaluate visual invariants that previously passed generic motion metrics."""
    requested_direction = report.get("direction_request")
    resolved_direction = report.get("direction")
    natural_direction = (
        _audit_auto_direction(host, spec)
        if spec.directions
        else None
    )
    expected_counterflow_turn = 0.0
    expected_counterflow_joints: tuple[int, ...] = ()
    if (
        spec.id.startswith("clap_")
        and requested_direction in {"left", "right"}
        and resolved_direction == requested_direction
        and requested_direction != natural_direction
    ):
        expected_counterflow_turn = (
            _COUNTERFLOW_CLAP_TURN_DEGREES
            if requested_direction == "left"
            else -_COUNTERFLOW_CLAP_TURN_DEGREES
        )
        expected_counterflow_joints = _COUNTERFLOW_CLAP_JOINTS

    checks = [{
        "name": "semantic_validator",
        "passed": bool(report["validation"]["ok"]),
        "detail": report["validation"]["detail"],
    }]
    owned = set(spec.absolute_joints) | set(spec.additive_joints)
    owned.update(expected_counterflow_joints)
    if (
        spec.direction_mode == "mirror"
        and resolved_direction != spec.canonical_direction
    ):
        owned = {_MIRROR_JOINTS[joint] for joint in owned}
    unchanged = True
    max_unowned_drift = 0.0
    for joint in set(range(22)) - owned:
        channels = slice(3 + 6 * joint, 3 + 6 * (joint + 1))
        drift = float(np.max(np.abs(edited[:, channels] - host[:, channels])))
        max_unowned_drift = max(max_unowned_drift, drift)
        unchanged = unchanged and drift <= 5e-7
    owned_axes = set(spec.translation_axes)
    if owned_axes.intersection({0, 1}):
        # Authored horizontal travel is dancer-local and rotates into both world-plane axes.
        owned_axes.update({0, 1})
    for axis in set(range(3)) - owned_axes:
        drift = float(np.max(np.abs(edited[:, axis] - host[:, axis])))
        max_unowned_drift = max(max_unowned_drift, drift)
        unchanged = unchanged and drift <= 5e-7
    if not spec.replace_contacts:
        drift = float(np.max(np.abs(edited[:, 135:139] - host[:, 135:139])))
        max_unowned_drift = max(max_unowned_drift, drift)
        unchanged = unchanged and drift <= 5e-7
    checks.append({
        "name": "declared_channel_ownership",
        "passed": bool(unchanged),
        "detail": (
            "all unowned pose, root, and contact channels remain source-identical "
            f"within float32 round-trip tolerance; max drift {max_unowned_drift:.2e}"
        ),
    })

    if spec.directions:
        expected = (
            natural_direction
            if requested_direction == "auto"
            else requested_direction
        )
        valid = resolved_direction in spec.directions and resolved_direction == expected
        checks.append({
            "name": "direction_resolution",
            "passed": bool(valid),
            "detail": (
                f"requested {requested_direction!r}, independently expected {expected!r}, "
                f"resolved {resolved_direction!r}"
            ),
        })

    if spec.id.startswith("clap_"):
        try:
            reported_counterflow_turn = float(
                report.get("counterflow_turn_degrees", 0.0)
            )
            reported_counterflow_joints = tuple(
                int(joint)
                for joint in report.get("counterflow_turn_joints", ())
            )
        except (TypeError, ValueError):
            reported_counterflow_turn = float("nan")
            reported_counterflow_joints = ()
        declaration_passed = (
            report.get("natural_direction") == natural_direction
            and abs(reported_counterflow_turn - expected_counterflow_turn) < 1e-6
            and reported_counterflow_joints == expected_counterflow_joints
        )
        checks.append({
            "name": "counterflow_declaration",
            "passed": bool(declaration_passed),
            "detail": (
                f"independently expected natural direction {natural_direction!r}, "
                f"turn {expected_counterflow_turn:.2f} degrees on joints "
                f"{list(expected_counterflow_joints)}; report declared "
                f"{report.get('natural_direction')!r}, "
                f"{reported_counterflow_turn:.2f} degrees on joints "
                f"{list(reported_counterflow_joints)}"
            ),
        })
        start, end = map(int, report["action_range"])
        joints = compute_poses(edited)["fk_joints"]
        global_r = _global_joint_rotations(edited)
        host_global = _global_joint_rotations(host)
        gaps = np.linalg.norm(joints[:, 20] - joints[:, 21], axis=-1)
        contact_frames = _declared_clap_contacts(spec, report, gaps)
        palm = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        left_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        contact_gaps = []
        palm_dots = []
        finger_dots = []
        finger_ups = []
        for frame in contact_frames:
            left_palm = global_r[frame, 20] @ palm
            right_palm = global_r[frame, 21] @ palm
            left_finger = global_r[frame, 20] @ left_axis
            right_finger = global_r[frame, 21] @ right_axis
            contact_gaps.append(float(gaps[frame]))
            palm_dots.append(float(left_palm @ right_palm))
            finger_dots.append(float(left_finger @ right_finger))
            finger_ups.append(float(0.5 * (left_finger[2] + right_finger[2])))
        host_yaw = np.arctan2(host_global[:, 9, 1, 0], host_global[:, 9, 0, 0])
        edit_yaw = np.arctan2(global_r[:, 9, 1, 0], global_r[:, 9, 0, 0])
        yaw_delta = np.arctan2(
            np.sin(edit_yaw - host_yaw),
            np.cos(edit_yaw - host_yaw),
        )
        peak_yaw = float(np.rad2deg(np.max(np.abs(yaw_delta[start:end]))))
        event_yaw = float(np.rad2deg(yaw_delta[int(report["event_frame"])]))
        if abs(expected_counterflow_turn) > 1e-6:
            facing_passed = (
                abs(event_yaw - expected_counterflow_turn) < 3.0
                and peak_yaw < abs(expected_counterflow_turn) + 3.0
            )
            facing_detail = (
                f"explicit counter-flow turn {event_yaw:.2f} degrees at contact "
                f"versus expected {expected_counterflow_turn:.2f}; "
                f"peak {peak_yaw:.2f} degrees"
            )
        else:
            facing_passed = peak_yaw < 3.0
            facing_detail = (
                f"peak chest-heading change versus source {peak_yaw:.2f} degrees"
            )
        checks.extend([
            {
                "name": "visible_palm_contact",
                "passed": bool(contact_frames) and all(gap < 0.01 for gap in contact_gaps),
                "detail": (
                    f"contact frames {list(contact_frames)}, wrist gaps "
                    f"{[round(gap, 4) for gap in contact_gaps]} m (all must be < 0.0100 m)"
                ),
            },
            {
                "name": "opposed_palm_planes",
                "passed": bool(contact_frames) and all(
                    palm_dot < -0.98 and finger_dot > 0.98
                    for palm_dot, finger_dot in zip(palm_dots, finger_dots)
                ),
                "detail": (
                    f"palm dots {[round(value, 4) for value in palm_dots]}, "
                    f"finger dots {[round(value, 4) for value in finger_dots]}"
                ),
            },
            {
                "name": "relaxed_hand_angle",
                "passed": bool(contact_frames) and all(
                    0.20 < finger_up < 0.75 for finger_up in finger_ups
                ),
                "detail": (
                    "average world-up finger components "
                    f"{[round(value, 4) for value in finger_ups]}"
                ),
            },
            {
                "name": "host_facing_continuity",
                "passed": facing_passed,
                "detail": facing_detail,
            },
        ])
    return checks, ("pass" if all(check["passed"] for check in checks) else "fail")


def _append_clap_direction_matrix_checks(records: dict[str, dict[str, dict]]) -> None:
    """Require visible left/right separation from the same motion's forward contact."""
    for motion_id, directions in records.items():
        if not {"forward", "left", "right"}.issubset(directions):
            continue
        forward = float(directions["forward"]["lateral"])
        left = float(directions["left"]["lateral"])
        right = float(directions["right"]["lateral"])
        minimum_separation = 0.14
        passed = (
            left > forward + minimum_separation
            and right < forward - minimum_separation
        )
        detail = (
            f"local-left offsets: left={left:.4f}, forward={forward:.4f}, "
            f"right={right:.4f} m; each side must separate by "
            f"> {minimum_separation:.4f} m"
        )
        for direction in ("forward", "left", "right"):
            directions[direction]["checks"].append({
                "name": "directional_contact_matrix",
                "passed": bool(passed),
                "detail": f"{motion_id}: {detail}",
            })
        automatic = directions.get("auto")
        if automatic is not None:
            resolved = automatic["resolved"]
            reference = directions.get(resolved)
            auto_passed = (
                passed
                and reference is not None
                and abs(float(automatic["lateral"]) - float(reference["lateral"])) < 0.01
            )
            automatic["checks"].append({
                "name": "automatic_contact_direction",
                "passed": bool(auto_passed),
                "detail": (
                    f"{motion_id}: auto resolved {resolved!r} at "
                    f"{float(automatic['lateral']):.4f} m local-left; "
                    f"explicit {resolved!r} is "
                    f"{float(reference['lateral']):.4f} m"
                    if reference is not None
                    else f"{motion_id}: auto resolved unsupported direction {resolved!r}"
                ),
            })


def _review_html(
    takes: list[dict],
    rule: str,
    *,
    audit_id: str,
    normalized_facing: bool,
    motion_fingerprint_value: str = "",
) -> str:
    vocabulary = [(spec.id, spec.name) for spec in MotionBank().specs]
    vocabulary_options = "".join(
        f'<option value="{html.escape(motion_id)}">{html.escape(name)}</option>'
        for motion_id, name in vocabulary
    )
    vocabulary_items = "".join(
        f"<li><code>{html.escape(motion_id)}</code>: {html.escape(name)}</li>"
        for motion_id, name in vocabulary
    )
    cards = []
    for take in takes:
        key = take["take"]
        control = take["control"]
        cache_query = html.escape(
            urlencode({
                "audit_id": str(audit_id),
                "motion_fingerprint": str(motion_fingerprint_value),
                "take_id": str(key),
            }),
            quote=True,
        )
        cards.append(
            f"""
            <article class="take" data-take="{html.escape(key)}">
              <h2>{html.escape(key)}</h2>
              <div class="pair">
                <section>
                  <h3>Source choreography: front left, side right</h3>
                  <video class="control" controls muted playsinline preload="metadata"
                         src="videos/{html.escape(control)}.mp4?{cache_query}"></video>
                </section>
                <section>
                  <h3>Unlabeled edit: front left, side right</h3>
                  <video class="edited" controls muted playsinline preload="metadata"
                         src="videos/{html.escape(key)}.mp4?{cache_query}"></video>
                </section>
              </div>
              <button class="play-pair" type="button">Play both from the start</button>
              <p class="playback-status" aria-live="polite"></p>
              <p><a class="review-link"
                    href="phase_sheets/{html.escape(key)}_review.html?{cache_query}"
                    target="_blank">Open synchronized views and edit-minus-source strip</a></p>
              <label>What action did you see?
                <input class="guess" type="text" list="motion-vocabulary"
                       autocomplete="off" spellcheck="false">
              </label>
              <label>What dancer-relative direction did the added action face or travel?
                <select class="direction-guess">
                  <option value="">Choose a direction</option>
                  <option value="none">No distinct direction</option>
                  <option value="forward">Forward</option>
                  <option value="left">Left (dancer's left)</option>
                  <option value="right">Right (dancer's right)</option>
                </select>
              </label>
              <div class="actions">
                <button class="lock" type="button" disabled>Lock guess</button>
              </div>
              <p class="status" aria-live="polite"></p>
              <section class="answer" hidden>
                <p><strong class="answer-name"></strong> (<code class="answer-id"></code>)</p>
                <p class="direction-detail"></p>
                <p>Must read as: <span class="recognizable"></span></p>
                <ul class="phases"></ul>
                <p><strong>Machine visual invariants</strong></p>
                <ul class="machine-checks"></ul>
                <section class="phase-review">
                  <label>Final visual verdict
                    <select class="visual-verdict">
                      <option value="">Choose pass or fail</option>
                      <option value="pass">Pass</option>
                      <option value="fail">Fail</option>
                    </select>
                  </label>
                  <label>Required evidence note
                    <textarea class="visual-evidence" rows="3"
                              placeholder="Describe contact, hand angle, host continuity, and required phases."></textarea>
                  </label>
                </section>
              </section>
            </article>
            """
        )
    facing_note = (
        "<p>The views are normalized to the dancer's heading at action start. "
        "In the front view, dancer-left appears on screen right and dancer-right appears on "
        "screen left. In the side view, the dancer faces screen right, so forward is screen "
        "right and backward is screen left.</p>"
        if normalized_facing
        else (
            "<p>The views preserve the production heading from the source choreography. "
            "All left/right choices remain relative to the dancer, never the viewer.</p>"
        )
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
    input, select, textarea {{ box-sizing: border-box; display: block; width: 100%; margin-top: 6px;
             padding: 10px; color: #fff; background: #111; border: 1px solid #666; }}
    input[type="checkbox"] {{ display: inline; width: auto; margin-right: 8px; }}
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
    <details>
      <summary>Supported motion vocabulary ({len(vocabulary)})</summary>
      <ul>{vocabulary_items}</ul>
    </details>
    <datalist id="motion-vocabulary">{vocabulary_options}</datalist>
    <ol>
      <li>Use <strong>Play both from the start</strong>. The lock remains disabled until both
          videos finish together at 1.0x without pausing, seeking, or changing speed.</li>
      <li>Open the synchronized source/edit/difference review for every take before locking it.</li>
      <li>Record and lock both the action and its observed direction.</li>
      <li>If the action is unclear or guessed incorrectly, mark it failed.</li>
      <li>After revealing, inspect every required phase from both front and side.</li>
    </ol>
    <p id="playback-progress" class="status" aria-live="polite"></p>
    <button id="export" type="button" disabled>Export blocking review result</button>
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
    const motionFingerprint = {json.dumps(str(motion_fingerprint_value))};
    const storageKey = `maestro-motion-audit:${{auditId}}`;
    let state = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    let answersPromise = null;
    const comparisonWindows = new Map();

    function save() {{
      localStorage.setItem(storageKey, JSON.stringify(state));
    }}

    function normalize(value) {{
      return String(value || "").toLowerCase().replace(/[_-]+/g, " ")
        .replace(/[^a-z0-9 ]+/g, "").replace(/\\s+/g, " ").trim();
    }}

    function normalizeDirection(value) {{
      const normalized = normalize(value);
      if (["none", "no direction", "not directional", "non directional"].includes(normalized)) {{
        return "none";
      }}
      return normalized;
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

    function playbackComplete(take) {{
      return state[take]
        && state[take].normalSpeedPlayback
        && state[take].normalSpeedPlayback.completed === true;
    }}

    function comparisonOpened(take) {{
      const saved = state[take] || {{}};
      const acknowledgment = saved.comparisonAcknowledgment || {{}};
      return Boolean(
        saved.comparisonOpenedAt
        && acknowledgment.auditId === auditId
        && acknowledgment.motionFingerprint === motionFingerprint
        && acknowledgment.takeId === take
      );
    }}

    function validComparisonAcknowledgment(event, take) {{
      const acknowledgment = event.data || {{}};
      return (
        event.origin === window.location.origin
        && event.source === comparisonWindows.get(take)
        && acknowledgment.type === "maestro-motion-audit-comparison-ready"
        && acknowledgment.auditId === auditId
        && acknowledgment.motionFingerprint === motionFingerprint
        && acknowledgment.takeId === take
      );
    }}

    function takeReadyToLock(take) {{
      return playbackComplete(take) && comparisonOpened(take);
    }}

    function allTakePlaybackComplete() {{
      return takeIds.length > 0 && takeIds.every(playbackComplete);
    }}

    function allTakeComparisonsOpened() {{
      return takeIds.length > 0 && takeIds.every(comparisonOpened);
    }}

    function allGuessesLocked() {{
      return takeIds.length > 0 && takeIds.every(take =>
        takeReadyToLock(take) && state[take] && String(state[take].guess || "").trim()
        && String(state[take].directionGuess || "").trim() && state[take].lockedAt
      );
    }}

    function allAnswersRevealed() {{
      return takeIds.length > 0 && takeIds.every(take =>
        state[take] && state[take].revealedAt
      );
    }}

    function allVisualReviewsComplete() {{
      return takeIds.length > 0 && takeIds.every(take =>
        state[take] && ["pass", "fail"].includes(state[take].visualStatus)
        && String(state[take].visualEvidence || "").trim()
        && Array.isArray(state[take].requiredPhases)
        && state[take].requiredPhases.length > 0
        && state[take].requiredPhases.every(phase =>
          (state[take].verifiedPhases || []).includes(phase)
        )
      );
    }}

    function refreshExportAvailability() {{
      document.querySelector("#export").disabled = !(
        allAnswersRevealed()
        && allVisualReviewsComplete()
        && allTakePlaybackComplete()
        && allTakeComparisonsOpened()
      );
    }}

    function refreshPlaybackProgress() {{
      const played = takeIds.filter(playbackComplete).length;
      const compared = takeIds.filter(comparisonOpened).length;
      document.querySelector("#playback-progress").textContent =
        `Verified playback: ${{played}}/${{takeIds.length}}; `
        + `synchronized comparison opened: ${{compared}}/${{takeIds.length}}.`;
    }}

    function refreshCardReadiness(card) {{
      const take = card.dataset.take;
      const saved = state[take] || {{}};
      const lock = card.querySelector(".lock");
      const playbackStatus = card.querySelector(".playback-status");
      const played = playbackComplete(take);
      const compared = comparisonOpened(take);
      if (saved.lockedAt) {{
        lock.disabled = true;
      }} else {{
        lock.disabled = !(played && compared);
      }}
      if (played && compared) {{
        playbackStatus.textContent =
          "Ready to lock: full synchronized playback and comparison review completed.";
      }} else if (played) {{
        playbackStatus.textContent =
          "Playback complete. Open the synchronized comparison before locking.";
      }} else if (saved.normalSpeedPlayback && saved.normalSpeedPlayback.reason) {{
        playbackStatus.textContent =
          `Playback incomplete: ${{saved.normalSpeedPlayback.reason}}`;
      }} else if (compared) {{
        playbackStatus.textContent =
          "Comparison opened. Complete uninterrupted synchronized playback before locking.";
      }} else {{
        playbackStatus.textContent =
          "Lock blocked until uninterrupted 1.0x playback and comparison review are complete.";
      }}
    }}

    function refreshRevealAvailability() {{
      const revealAll = document.querySelector("#reveal-all");
      const remaining = takeIds.filter(take =>
        !(takeReadyToLock(take) && state[take] && String(state[take].guess || "").trim()
          && String(state[take].directionGuess || "").trim() && state[take].lockedAt)
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
      const input = card.querySelector(".guess");
      const direction = card.querySelector(".direction-guess");
      if (saved) {{
        input.value = saved.guess || "";
        direction.value = saved.directionGuess || "";
        if (saved.lockedAt) {{
          input.disabled = true;
          direction.disabled = true;
          card.querySelector(".status").textContent = saved.revealedAt
            ? `Locked: "${{saved.guess}}" - answer already revealed`
            : `Locked: "${{saved.guess}}"`;
        }}
      }}
      refreshCardReadiness(card);
    }}

    function renderAnswer(card, answer) {{
      const take = card.dataset.take;
      if (!answer) throw new Error(`missing answer for ${{take}}`);
      const accepted = [answer.id, answer.name, ...(answer.aliases || [])].map(normalize);
      const recognized = accepted.includes(normalize(state[take].guess));
      const expectedDirection = normalizeDirection(answer.resolved_direction || "none");
      const directionRecognized =
        normalizeDirection(state[take].directionGuess) === expectedDirection;
      state[take] = {{
        ...state[take],
        actual: answer.id,
        recognized,
        expectedDirection,
        directionRecognized,
        requiredPhases: [...answer.visual_contract.required_phases],
        verifiedPhases: state[take].verifiedPhases || [],
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
        const label = document.createElement("label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "phase-reviewed";
        checkbox.checked = state[take].verifiedPhases.includes(phase);
        checkbox.addEventListener("change", () => {{
          const reviewed = new Set(state[take].verifiedPhases || []);
          if (checkbox.checked) reviewed.add(phase);
          else reviewed.delete(phase);
          state[take] = {{...state[take], verifiedPhases: [...reviewed]}};
          save();
          refreshExportAvailability();
        }});
        label.append(checkbox, ` Reviewed: ${{phase}}`);
        item.appendChild(label);
        phases.appendChild(item);
      }});
      const direction = answer.requested_direction || "not directional";
      card.querySelector(".direction-detail").textContent =
        `Requested direction: ${{direction}}; resolved direction: ${{answer.resolved_direction || "none"}}`;
      const machineChecks = card.querySelector(".machine-checks");
      machineChecks.replaceChildren();
      (answer.machine_checks || []).forEach(check => {{
        const item = document.createElement("li");
        item.textContent = `${{check.passed ? "PASS" : "FAIL"}} - ${{check.name}}: ${{check.detail}}`;
        machineChecks.appendChild(item);
      }});
      card.querySelector(".answer").hidden = false;
      card.querySelector(".visual-verdict").value = state[take].visualStatus || "";
      card.querySelector(".visual-evidence").value = state[take].visualEvidence || "";
      card.querySelector(".status").textContent =
        recognized && directionRecognized
          ? `PASS - action and direction matched ${{answer.id}} (${{expectedDirection}})`
          : `FAIL - guessed "${{state[take].guess}}" (${{state[take].directionGuess}}); `
            + `answer is ${{answer.id}} (${{expectedDirection}})`;
      save();
      refreshExportAvailability();
    }}

    window.addEventListener("message", event => {{
      const acknowledgment = event.data || {{}};
      const take = acknowledgment.takeId;
      if (!takeIds.includes(take) || !validComparisonAcknowledgment(event, take)) {{
        return;
      }}
      state[take] = {{
        ...(state[take] || {{}}),
        comparisonOpenedAt: new Date().toISOString(),
        comparisonAcknowledgment: {{
          auditId: acknowledgment.auditId,
          motionFingerprint: acknowledgment.motionFingerprint,
          takeId: acknowledgment.takeId,
        }},
      }};
      comparisonWindows.delete(take);
      save();
      const card = document.querySelector(`.take[data-take="${{take}}"]`);
      if (card) refreshCardReadiness(card);
      refreshPlaybackProgress();
      refreshRevealAvailability();
      refreshExportAvailability();
    }});

    document.querySelectorAll(".take").forEach(card => {{
      const take = card.dataset.take;
      const input = card.querySelector(".guess");
      const direction = card.querySelector(".direction-guess");
      const lock = card.querySelector(".lock");
      const status = card.querySelector(".status");
      const verdict = card.querySelector(".visual-verdict");
      const evidence = card.querySelector(".visual-evidence");
      const reviewLink = card.querySelector(".review-link");
      const videos = [...card.querySelectorAll("video")];
      const controlVideo = card.querySelector(".control");
      const editedVideo = card.querySelector(".edited");
      let playbackRun = null;

      function updateTake(patch) {{
        state[take] = {{...(state[take] || {{}}), ...patch}};
        save();
        refreshCardReadiness(card);
        refreshPlaybackProgress();
        refreshRevealAvailability();
        refreshExportAvailability();
      }}

      function invalidatePlayback(reason) {{
        if (!playbackRun || playbackRun.invalid) return;
        playbackRun.invalid = true;
        updateTake({{
          normalSpeedPlayback: {{
            completed: false,
            reason,
            started_at: playbackRun.startedAt,
            playback_rate: 1,
            seek_count: playbackRun.seekCount,
            pause_count: playbackRun.pauseCount,
            max_sync_drift: playbackRun.maxSyncDrift,
          }},
        }});
      }}

      function maybeCompletePlayback() {{
        if (!playbackRun || playbackRun.invalid || !videos.every(video => video.ended)) {{
          return;
        }}
        const run = playbackRun;
        if (run.maxSyncDrift > 0.12) {{
          invalidatePlayback(
            `Source/edit synchronization drifted by ${{run.maxSyncDrift.toFixed(3)}}s; `
              + "replay the pair.",
          );
          return;
        }}
        playbackRun = null;
        updateTake({{
          normalSpeedPlayback: {{
            completed: true,
            started_at: run.startedAt,
            completed_at: new Date().toISOString(),
            playback_rate: 1,
            seek_count: run.seekCount,
            pause_count: run.pauseCount,
            source_seconds: Number(controlVideo.currentTime.toFixed(3)),
            edit_seconds: Number(editedVideo.currentTime.toFixed(3)),
            elapsed_seconds: Number(((performance.now() - run.startedMs) / 1000).toFixed(3)),
            max_sync_drift: Number(run.maxSyncDrift.toFixed(3)),
          }},
        }});
      }}

      videos.forEach(video => {{
        video.addEventListener("timeupdate", () => {{
          if (!playbackRun || playbackRun.invalid) return;
          playbackRun.maxSyncDrift = Math.max(
            playbackRun.maxSyncDrift,
            Math.abs(controlVideo.currentTime - editedVideo.currentTime),
          );
        }});
        video.addEventListener("ratechange", () => {{
          if (playbackRun && Math.abs(video.playbackRate - 1) > 1e-6) {{
            invalidatePlayback("Playback speed changed; replay both videos at 1.0x.");
          }}
        }});
        video.addEventListener("seeking", () => {{
          if (playbackRun) {{
            playbackRun.seekCount += 1;
            invalidatePlayback("Seeking invalidated this run; replay both videos from the start.");
          }}
        }});
        video.addEventListener("pause", () => {{
          if (
            playbackRun
            && Number.isFinite(video.duration)
            && video.currentTime + 0.05 < video.duration
          ) {{
            playbackRun.pauseCount += 1;
            invalidatePlayback("Pausing invalidated this run; replay both videos uninterrupted.");
          }}
        }});
        video.addEventListener("ended", maybeCompletePlayback);
      }});

      function metadataReady(video) {{
        if (video.readyState >= 1) return Promise.resolve();
        return new Promise((resolve, reject) => {{
          video.addEventListener("loadedmetadata", resolve, {{once: true}});
          video.addEventListener(
            "error",
            () => reject(new Error("video metadata could not be loaded")),
            {{once: true}},
          );
        }});
      }}

      function rewind(video) {{
        video.pause();
        video.playbackRate = 1;
        if (video.currentTime <= 0.01) {{
          return Promise.resolve();
        }}
        return new Promise(resolve => {{
          video.addEventListener("seeked", resolve, {{once: true}});
          video.currentTime = 0;
        }});
      }}

      card.querySelector(".play-pair").addEventListener("click", async () => {{
        playbackRun = null;
        try {{
          await Promise.all(videos.map(metadataReady));
          await Promise.all(videos.map(rewind));
          const startedAt = new Date().toISOString();
          playbackRun = {{
            startedAt,
            startedMs: performance.now(),
            invalid: false,
            seekCount: 0,
            pauseCount: 0,
            maxSyncDrift: 0,
          }};
          updateTake({{
            normalSpeedPlayback: {{
              completed: false,
              reason: "Playback in progress.",
              started_at: startedAt,
              playback_rate: 1,
              seek_count: 0,
              pause_count: 0,
              max_sync_drift: 0,
            }},
          }});
          await Promise.all(videos.map(video => video.play()));
        }} catch (error) {{
          if (playbackRun) {{
            invalidatePlayback(`Playback could not start: ${{error.message}}`);
          }} else {{
            updateTake({{
              normalSpeedPlayback: {{
                completed: false,
                reason: `Playback could not start: ${{error.message}}`,
              }},
            }});
          }}
        }}
      }});

      reviewLink.addEventListener("click", event => {{
        event.preventDefault();
        const child = window.open(reviewLink.href, `maestro-motion-audit-${{take}}`);
        if (!child) {{
          status.textContent =
            "Comparison window was blocked. Allow pop-ups and open the comparison again.";
          return;
        }}
        comparisonWindows.set(take, child);
      }});

      lock.addEventListener("click", () => {{
        if (!takeReadyToLock(take)) {{
          status.textContent =
            "Complete uninterrupted paired playback and open the synchronized comparison first.";
          return;
        }}
        const guess = input.value.trim();
        if (!guess) {{
          status.textContent = "Enter a guess before locking.";
          input.focus();
          return;
        }}
        if (!direction.value) {{
          status.textContent = "Choose an observed direction before locking.";
          direction.focus();
          return;
        }}
        state[take] = {{
          ...(state[take] || {{}}),
          guess,
          directionGuess: direction.value,
          lockedAt: new Date().toISOString(),
        }};
        save();
        input.disabled = true;
        direction.disabled = true;
        lock.disabled = true;
        status.textContent = `Locked: "${{guess}}"`;
        refreshCardReadiness(card);
        refreshPlaybackProgress();
        refreshRevealAvailability();
        refreshExportAvailability();
      }});

      verdict.addEventListener("change", () => {{
        state[take] = {{...(state[take] || {{}}), visualStatus: verdict.value}};
        save();
        refreshExportAvailability();
      }});
      evidence.addEventListener("input", () => {{
        state[take] = {{...(state[take] || {{}}), visualEvidence: evidence.value}};
        save();
        refreshExportAvailability();
      }});
      restore(card);
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
        refreshExportAvailability();
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
        refreshExportAvailability();
      }}).catch(error => {{
        revealStatus.textContent = `Could not restore answers: ${{error.message}}`;
      }});
    }}

    document.querySelectorAll(".take").forEach(refreshCardReadiness);
    refreshPlaybackProgress();
    refreshExportAvailability();

    document.querySelector("#export").addEventListener("click", () => {{
      const payload = {{
        audit_id: auditId,
        motion_fingerprint: motionFingerprint,
        normalized_facing: normalizedFacing,
        normal_speed_reviewed: allTakePlaybackComplete(),
        source_edit_compared: allTakeComparisonsOpened(),
        exported_at: new Date().toISOString(),
        takes: Object.fromEntries(takeIds.map(take => [take, {{
          guess: state[take].guess,
          recognized: state[take].recognized === true,
          direction_guess: state[take].directionGuess,
          direction_recognized: state[take].directionRecognized === true,
          normal_speed_playback: state[take].normalSpeedPlayback,
          comparison_opened_at: state[take].comparisonOpenedAt,
          comparison_acknowledgment: state[take].comparisonAcknowledgment,
          locked_at: state[take].lockedAt,
          status: state[take].visualStatus,
          evidence: state[take].visualEvidence,
          verified_phases: state[take].verifiedPhases,
        }}])),
      }};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: "application/json"}});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${{auditId}}-review-result.json`;
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
        authored_directions = (
            spec.directions if spec.direction_mode == "clip" else (None,)
        )
        for direction in authored_directions:
            actual = bank.load_clip(spec, direction=direction)
            expected = build_motion(
                spec.id,
                spec.frames,
                direction=direction or "forward",
            )
            if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
                suffix = "" if direction is None else f"[{direction}]"
                raise ValueError(
                    f"{spec.id}{suffix}: committed clip is stale; "
                    "run scripts/build_motion_bank.py first"
                )
    cases = [
        (spec, direction)
        for spec in specs
        for direction in audit_variants(spec)
    ]
    random.Random(args.seed).shuffle(cases)
    _prepare_output(args.output)
    audit_id = _new_audit_id(args.output, args.seed)
    fingerprint = motion_fingerprint(ROOT)
    takes: list[dict] = []
    controls: list[dict] = []
    control_keys: dict[tuple[int, int, int], str] = {}
    answers: dict[str, dict] = {}
    clap_direction_records: dict[str, dict[str, dict]] = {}
    context = max(0, int(args.context))
    normalize_facing = not bool(args.preserve_heading)
    for index, (spec, direction) in enumerate(cases, start=1):
        take = f"take_{index:02d}"
        case_id = audit_case_id(spec, direction)
        motion, report = bank.apply(
            host,
            spec.id,
            beats=beats,
            beat_strengths=strengths,
            direction=direction,
        )
        machine_checks, machine_status = _machine_checks(host, motion, spec, report)
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
            "case_id": case_id,
            "id": spec.id,
            "name": spec.name,
            "aliases": list(spec.aliases),
            "requested_direction": direction,
            "resolved_direction": report.get("direction"),
            "visual_contract": contracts[spec.id],
            "machine_checks": machine_checks,
            "machine_status": machine_status,
            "report": report,
        }
        if spec.id.startswith("clap_") and direction in {"auto", "forward", "left", "right"}:
            clap_direction_records.setdefault(spec.id, {})[direction] = {
                "lateral": _clap_contact_lateral(motion, int(report["event_frame"])),
                "resolved": report.get("direction"),
                "checks": machine_checks,
                "answer": answers[take],
            }

    _append_clap_direction_matrix_checks(clap_direction_records)
    for answer in answers.values():
        answer["machine_status"] = (
            "pass"
            if all(check["passed"] for check in answer["machine_checks"])
            else "fail"
        )

    review = {
        "audit_id": audit_id,
        "fps": 30,
        "fixed_camera": True,
        "normalized_facing": normalize_facing,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "seed": int(args.seed),
        "bank_version": bank.version,
        "motion_fingerprint": fingerprint,
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
            motion_fingerprint_value=fingerprint,
        ),
        encoding="utf-8",
    )
    failures = [
        answer["case_id"]
        for answer in answers.values()
        if answer["machine_status"] != "pass"
    ]
    if failures:
        raise ValueError(f"machine visual invariants failed: {failures}")
    print(f"built {len(takes)} blind audit takes in {args.output}")


if __name__ == "__main__":
    main()
