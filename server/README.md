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
- The optional distributed path is disabled by default. Its backward-compatible default transport
  is the shared-volume protocol: set `AGENTLODGE_DISTRIBUTED=1`, point
  `AGENTLODGE_WORKER_REGISTRY` at a JSON registry based on
  `scripts/worker_registry.example.json`, and set `AGENTLODGE_SHARED_ROOT` to the common mount.
  Healthy workers advertise one or more explicit capabilities:
  `jukebox.extract`, `lodge.generate`, `edge.generate`, or `render.frames`.
- `AGENTLODGE_DISTRIBUTED_CAPABILITIES` can restrict a calibration to selected roles without
  changing the other stages, for example `jukebox.extract,lodge.generate,edge.generate`. When it is
  unset, distributed mode requires all roles reached by the request.
- Start one capability worker per resident model or Blender daemon with `scripts/runpod_worker.py`. Task
  requests and results live under each worker's configured `task_dir`; deterministic task IDs make
  retries idempotent, and heartbeat/version checks prevent dispatch to stale workers. Coordinators
  share `AGENTLODGE_DISTRIBUTED_STATE` (by default the worker directories' common
  `_coordinator` directory) to serialize submissions, reject conflicting task IDs, and choose the
  least-loaded healthy worker. A stale task is reassigned only while it is still unclaimed; claimed
  work is never duplicated automatically. If distributed mode is enabled but the required role is
  unavailable, the request fails explicitly rather than silently falling back to a slower or
  quality-changing path.
- Distributed render workers accept only the configured quality contract, render disjoint global
  frame ranges to worker-local lossless TGA/PNG files, hash every assigned source frame, and package
  each range as an FFV1 Matroska shard before crossing the shared volume. The coordinator verifies
  FFV1 codec, dimensions, frame rate, exact contiguous frame count/timestamps, artifact hash/size,
  and an indexed decoded-RGB digest before running the established H.264/audio encode once. Task
  identity includes the versioned render contract plus attested scene/renderer identity, preventing
  stale shard reuse after an upgrade. A one-visible-GPU container needs no selector shim. In a
  multi-GPU container,
  `AGENTLODGE_GPU_INDEX` names the CUDA/nvidia-smi index and the launcher validates the setup-built
  EGL selector shim before starting Blender.
- `render.frames` can instead use the no-shared-volume authenticated HTTP transport. Set
  `AGENTLODGE_DISTRIBUTED_TRANSPORT=http`, `AGENTLODGE_HTTP_COORDINATOR_URL`, an
  `AGENTLODGE_HTTP_TOKEN` or token file, and separate confined coordinator/worker scratch roots.
  Workers register and heartbeat over the coordinator API, lease deterministic tasks, download only
  coordinator-minted input artifact IDs, and stage FFV1 output artifacts under their active lease.
  The coordinator verifies size and SHA-256 on upload and download, atomically publishes an artifact
  with its idempotent completion, and reassigns expired leases. Failed tasks are retried only when
  the coordinator explicitly resubmits the identical canonical request; transport/5xx failures are
  retried within bounded lease/deadline windows while 4xx protocol, lease, payload, and artifact
  errors fail closed. Use HTTPS or a private authenticated network; the built-in service can
  terminate TLS with `--tls-cert` and `--tls-key`.
- The LLM edit planner reads `OPENAI_API_KEY`, a private key file selected by `OAI_KEY_FILE`, or
  `~/.oai_key`. Without one of those sources it falls back to the offline keyword planner.
- Sessions persist under `server/sessions/<sid>/` (checkpoint tree + motion snapshots), so history
  survives a restart.
- Full-song rendering can use the opt-in `AGENTLODGE_FULL_RENDER_BACKEND=filament` SLA candidate.
  It exports a GLB through one warm Blender daemon, renders contiguous ranges on the GPUs selected by
  `AGENTLODGE_FILAMENT_GPU_INDICES`, uses NVENC, validates exact frame coverage, and fails closed.
  Run `scripts/setup_filament_pod.sh` first. This does not change the default EEVEE quality contract;
  Filament still requires explicit visual approval. Disable its render cache during timing with
  `AGENTLODGE_FILAMENT_DISABLE_CACHE=1`.

Example worker commands on containers sharing `/workspace`:

```bash
python scripts/runpod_worker.py \
  --worker-id jukebox-0 --capability jukebox.extract \
  --task-dir /workspace/maestro-workers/jukebox-0 \
  --shared-root /workspace --edge-root /workspace/EDGE

python scripts/runpod_worker.py \
  --worker-id lodge-0 --capability lodge.generate \
  --task-dir /workspace/maestro-workers/lodge-0 \
  --shared-root /workspace \
  --lodge-root /workspace/LODGE \
  --lodge-weights /workspace/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt \
  --lodge-global-weights /workspace/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt

python scripts/runpod_worker.py \
  --worker-id edge-0 --capability edge.generate \
  --task-dir /workspace/maestro-workers/edge-0 \
  --shared-root /workspace --edge-root /workspace/EDGE \
  --edge-checkpoint /workspace/EDGE/checkpoint.pt

bash scripts/start_runpod_worker.sh render.frames render-0 \
  /workspace/maestro-workers/render-0
```

`setup_pod.sh` and `setup_gen_pod.sh` compile
`/workspace/.agentlodge/lib/libagentlodge_egl_cuda_device.so` from the audited C source; no binary is
committed. The guarded launcher keeps one-GPU behavior unchanged. In a multi-GPU container it
requires a valid `AGENTLODGE_GPU_INDEX` and selector, gives every worker a unique daemon root, and
defaults each worker to isolated `/tmp` scratch. `/dev/shm` is explicit opt-in through
`AGENTLODGE_RENDER_USE_SHM=1` and requires an aggregate per-worker reservation preflight.

```bash
# N resident Blender daemons per GPU in one two-GPU Pod.
DAEMONS_PER_GPU=16
mkdir -p /workspace/maestro-workers
for gpu in 0 1; do
  for slot in $(seq 0 $((DAEMONS_PER_GPU - 1))); do
    worker="render-g${gpu}-d${slot}"
    AGENTLODGE_GPU_INDEX="$gpu" \
      bash scripts/start_runpod_worker.sh \
        render.frames "$worker" "/workspace/maestro-workers/$worker" \
        >"/workspace/maestro-workers/$worker.launch.log" 2>&1 &
  done
done
```

Each process owns one warm daemon (`AGENTLODGE_WARM_POOL=1`), task directory, local frame/shard
scratch, and `AGENTLODGE_RENDER_DAEMON_ROOT`; multiple worker processes may intentionally share a
GPU. Register every filesystem worker ID, or let HTTP workers heartbeat dynamically.

The selector intercepts Blender/libepoxy EGL lookup only in the Blender daemon subprocess. It maps
the requested CUDA index through `EGL_CUDA_DEVICE_NV`, so EGL enumeration order is irrelevant;
`LD_PRELOAD` is not exported to the worker or generation processes. The shim atomically records the
selected CUDA/EGL device and build identity; warm-render daemon reuse additionally validates the
scene, renderer code, Blender version, protocol, quality, GPU UUID/PCI ID, and selector binary.

The render-plus-packaging saturation calibration on the 2x RTX PRO 4500 Pod was sublinear. The best one-GPU result used
20 daemons at 14.887 fps after external FFV1 packaging. The best two-GPU result used 32 daemons
(16/GPU): 24.132 raw fps and 22.462 end-to-end fps, a 1.509x speedup and 75.5% efficiency. Both GPUs
reached 100% utilization with 7,670 MiB maximum VRAM/GPU; 5,400 frames project to 240.4s before the
final merge/encode. Twelve full-quality TGA frames per physical GPU were byte-identical.

Direct Blender FFV1 was decoded-RGB exact and only 2.9% faster at saturation, but produced 45.4%
larger artifacts and removed the source-TGA digest contract, so it is not adopted. `/dev/shm`
improved a 12-daemon result only 17.951 to 18.373 fps (~2.4%), hence the safer `/tmp` default. See
`experiments/performance/runpod_calibration_20260819/eevee_multigpu_consolidated.json`. Use the
authenticated HTTP transport for independent Pods and scale beyond RunPod's four-GPU Pod limit.

A separate pre-optimization production-path filesystem run used four launcher workers (two per physical GPU) for
400 frames and completed in 45.283s (8.833 fps), including exact range, source/shard hash, and
decoded-RGB validation. It is an integrated transport result, not a replacement saturation point;
an integrated capacity sweep remains pending. See `integrated_render_validation.json` in the same
evidence directory.

For independent render containers, first run a coordinator with durable local state:

```bash
export AGENTLODGE_HTTP_TOKEN_FILE=/run/secrets/agentlodge-http-token
python scripts/run_http_coordinator.py \
  --bind 0.0.0.0 --port 8765 \
  --state-root /var/lib/agentlodge/coordinator \
  --artifact-root /var/lib/agentlodge/artifacts \
  --tls-cert /run/secrets/coordinator.crt \
  --tls-key /run/secrets/coordinator.key
```

Configure the render-owning process:

```bash
export AGENTLODGE_DISTRIBUTED=1
export AGENTLODGE_DISTRIBUTED_CAPABILITIES=render.frames
export AGENTLODGE_DISTRIBUTED_TRANSPORT=http
export AGENTLODGE_HTTP_COORDINATOR_URL=https://coordinator.example:8765
export AGENTLODGE_HTTP_TOKEN_FILE=/run/secrets/agentlodge-http-token
export AGENTLODGE_HTTP_COORDINATOR_SCRATCH=/var/lib/agentlodge/render-scratch
```

Then start the same guarded launcher in each worker container. No coordinator path is mounted into
the worker; one-GPU containers may omit `AGENTLODGE_GPU_INDEX`:

```bash
export AGENTLODGE_DISTRIBUTED_TRANSPORT=http
export AGENTLODGE_HTTP_COORDINATOR_URL=https://coordinator.example:8765
export AGENTLODGE_HTTP_TOKEN_FILE=/run/secrets/agentlodge-http-token
bash scripts/start_runpod_worker.sh render.frames render-0
```

The coordinator selects one canonical scene/renderer/selector/protocol/quality cohort for a render
and records its eligible worker IDs on every task; incompatible rolling-deployment workers cannot
claim or inherit those ranges. Input downloads and idempotent output uploads retry only network and
retryable 5xx failures while the lease remains valid. Permanent 4xx, hash, payload, and stale-lease
errors stop immediately. When shared-memory scratch is explicitly enabled, each assigned range is
preflighted from its frame count and resolution; insufficient `/dev/shm` falls back to that
worker's isolated `/tmp` root before Blender starts.

Render contract `render.frames-ffv1-v3` computes the canonical indexed RGB digest once from the
source TGA sequence. The worker packages FFV1 and performs a metadata/timing probe without a second
full shard decode. The coordinator still independently decodes every downloaded shard and requires
its indexed RGB digest to match before merge.

The same launcher accepts `jukebox.extract`, `lodge.generate`, and `edge.generate`. Jukebox uses
the isolated EDGE environment; the other roles use the MAESTRO environment. Worker preload loads
the Jukebox, LODGE, and EDGE model weights before the heartbeat changes to `ready`.

After workers are healthy, calibrate the exact pose sequence without changing quality:

```bash
python scripts/benchmark_render_scaling.py \
  --poses /workspace/calibration/poses.npz \
  --output-dir /workspace/calibration/results
```

The report records per-worker duration, observed daemon-attested CUDA index/UUID/PCI ID, workers per
physical GPU keyed by UUID (with container-local CUDA-index counts retained as diagnostics),
aggregate and median frame throughput, source/shard/decoded-RGB hashes, artifact IDs when HTTP is
selected, and the idealized worker count for the frozen 5,400-frame/23-second render budget.

Before scaling a target GPU, compare a small exact-scene range with the direct warm renderer:

```bash
python scripts/validate_render_equivalence.py \
  --poses /workspace/calibration/poses.npz \
  --output-dir /workspace/calibration/equivalence \
  --shared-root /workspace
```

The check requires both the direct and worker renders to have identical source-frame bytes and
requires the decoded FFV1 shard to have the same aggregate RGB hash.
