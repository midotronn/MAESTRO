# AgentLODGE Interactive Dance Editor (web UI)

A clean web UI to **view a generated dance and edit it with natural language**. Select a time
window on the timeline, type an instruction ("make this more energetic", "more on beat", "calmer",
"smoother", "reverse this"), and an AgentBanana-style agent re-queries LODGE/EDGE over just that
window in a bounded *propose → generate(K) → splice → verify → refine* cycle, preserving the rest of
the dance exactly. Every accepted edit is a **checkpoint** you can undo / redo / roll back to.

## Run

```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

## A song folder: `server/media/<sid>/`

| file | meaning |
|------|---------|
| `base_motion.npy` | the assembled dance, Z-up AgentLODGE 139 layout `(L, 139)` |
| `beats.npy` | music beat frame indices at 30 FPS |
| `preview.mp4` | the rendered dance shown in the viewer |
| `bank/bank_<sid>_<lodge\|edge>_seed<n>.npy` | *optional* real backbone candidate bank |

If a `bank/` is present the editor selects window candidates from **real** LODGE/EDGE takes
(`BankWindowGenerator`, wrapped in a resilient fallback). Otherwise it uses the offline
`MockWindowGenerator`, so the UI is fully usable without a GPU (metric deltas + history are computed
on the real base motion; only the candidate window content is synthetic).

Build a bank on a GPU box once with `files/build_window_bank.py <sid>` (see the session notes), then
copy `bank_<sid>_*.npy` into `server/media/<sid>/bank/`.

## API

- `GET  /api/songs`
- `POST /api/session/{sid}` → duration, beats, timeline, metrics, preview URL
- `POST /api/session/{sid}/edit` `{a_sec,b_sec,instruction[,from_id,k,max_cycles]}`
- `WS   /api/session/{sid}/edit_ws` → streams live `cycle n/K` progress then the final result
- `POST /api/session/{sid}/undo|redo`, `POST /api/session/{sid}/restore {ckpt_id}`
- `GET  /api/session/{sid}/timeline`
- `GET  /api/jobs/{sid}` → upload/generation stage, overall progress, elapsed time, and deferred-bank progress

## Notes

- Re-rendering the *edited* motion back to video needs the Blender/GPU worker; the preview stays the
  base take, while metric deltas + checkpoint history reflect the real edited motion immediately.
- When the editor runs on the pod, persistent Blender daemons remove startup/import overhead while
  retaining the established quality settings: window renders use `448x448`, 8 EEVEE samples, and
  every frame; full renders use `1080x1080`, 96 samples, and every frame. Exact completed renders are
  cached under `server/media/<sid>/.render_cache/`. Full renders are split into six disjoint global
  frame ranges by default and use lossless raw intermediates before the established H.264 encoder.
- Upload readiness no longer waits for every editing candidate. The initial dance, seed-0 bank, and
  full preview complete first; additional LODGE/EDGE seeds populate `bank/` asynchronously.
- The editor exposes one compact live-progress surface for startup, uploads, generation, edits,
  history actions, comparisons, renders, media buffering, and deferred-bank generation. Upload bytes
  are measured in the browser, pipeline stages come from structured pod markers, and warm Blender
  jobs report completed frame counts without changing render settings or output quality.
- Uploaded-song jobs also persist `performance_trace.json` beside the media. The trace correlates a
  browser-generated request ID with browser upload/total latency, server stage intervals, remote
  LODGE/EDGE/Jukebox timings, render frames, and worker count. Summarize frozen warm runs with
  `python scripts/analyze_pipeline_traces.py <trace...>`.
- The LLM edit planner reads `OPENAI_API_KEY`, a private key file selected by `OAI_KEY_FILE`, or
  `~/.oai_key`. Without one of those sources it falls back to the offline keyword planner.
- Sessions persist under `server/sessions/<sid>/` (checkpoint tree + motion snapshots), so history
  survives a restart.
