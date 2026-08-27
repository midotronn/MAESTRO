# Switching pods & installing deps

RunPod pods are ephemeral: on a **stop/restart** the apt-installed system libraries **and** anything
on `/root` (including a venv placed there) are wiped, while the `/workspace` network volume (repos,
checkpoints, generated motions) persists. Moving to a **brand-new pod** additionally needs the
`/workspace` data restored (clone + checkpoints).

Everything below is driven by a single connection config so switching pods is a one-liner.

## 1. Point at your pod

Copy the template and edit it (it is gitignored, so your host never gets committed):

```powershell
Copy-Item scripts\pod.config.example.ps1 scripts\pod.config.ps1
# edit host / port / key
```

Or set env vars directly:

```powershell
$env:AGENTLODGE_POD_HOST="213.173.107.238"; $env:AGENTLODGE_POD_PORT="20642"
$env:AGENTLODGE_POD_KEY="$HOME\.ssh\id_ed25519"
```

## 2. Provision the exact four-GPU service

`pod.ps1` automatically uses the current local Git branch. Override it only when intentionally
deploying another branch or immutable commit:

```powershell
$env:AGENTLODGE_GIT_REF = git branch --show-current
.\scripts\pod.ps1 setup4
```

`setup4` requires exactly four visible GPUs and is safe to rerun after a pod restart. On a new
volume it clones the requested ref, then:

1. installs CUDA PyTorch, Blender 4.2.3, LODGE, EDGE, all checkpoints, and the complete Jukebox
   runtime;
2. downloads and size-validates the approximately 10 GB Jukebox 5B prior during setup instead of
   deferring that cost to the first song;
3. builds the exact Y-Bot scene and the audited EGL CUDA-device selector;
4. builds checksum-pinned Filament v1.75.0, selects a driver-compatible NVENC FFmpeg, and retains a
   checksum-pinned FFmpeg fallback;
5. generates the static and animated Filament validation GLBs from the exact Y-Bot scene;
6. runs real Vulkan/Filament/NVENC smoke tests;
7. starts four resident Jukebox workers plus LODGE, EDGE, three audio workers, and the resident dance
   generation/cleanup worker;
8. starts the warm server on pod-local port `8011` with one Filament lane per GPU;
9. replaces RunPod port `8888` with the public interview editor in curated-song mode; and
10. verifies all ten fresh worker heartbeats, process IDs, CUDA/Blender quality attestation,
    `h264_nvenc`, both server endpoints, the blind study player, and planner status.

The retained quality contract is 1080x1080, 30 FPS, 96 samples, EEVEE scene export, every frame,
denoising enabled, TGA source frames, lossless FFV1 shards, and one final H.264/audio mux. The
four-GPU launcher enables all-contact-frame foot-mesh grounding after two distinct 5,400-frame
motions produced byte-identical animated GLBs versus the full-mesh path. Set
`AGENTLODGE_FILAMENT_FOOT_GROUNDING=0` to restore full-mesh scanning. The bounded/sampled fast mode
remains disabled because it changes the output.

Two exact browser-to-final validation runs with the all-frame foot-mesh path completed in 85.231s
and 88.302s, versus the prior 142.549s hot result. This provisionally clears the 90s p95 ceiling but
still misses the 60s p50 target; a multi-song 20-run study remains required for the final SLA claim.

The four-GPU worker launcher also keeps temporary EDGE WAV slices and per-slice Jukebox arrays under
`/tmp/maestro-jukebox-shared`, publishing only the final combined array to `/workspace`. The
resulting features were bit-identical to the network-volume path and reduced hot EDGE preprocessing
from 28.286s to 18.340s; the corresponding exact browser-to-final run completed in 76.265s.

An opt-in staged path can start single-candidate LODGE generation as soon as LODGE audio features
are ready, while EDGE/Jukebox preprocessing continues. Reuse requires matching SHA-256
fingerprints for both the source features and generated motion; stale, incomplete, or failed early
results automatically fall back to the normal generation path. Set
`AGENTLODGE_EARLY_LODGE_GENERATION=1` to benchmark this overlap. On the current four-GPU topology,
the exact hot result was 77.391s versus a contemporaneous disabled control at 76.916s. LODGE and
the final 5,400-frame story tensors remained byte-identical, but GPU-0 contention increased EDGE
preprocessing from 18.156s to 22.586s. The path therefore remains disabled by default.

To recheck a running pod without reinstalling anything:

```powershell
.\scripts\pod.ps1 ssh "cd /workspace/AgentLODGE && WORKSPACE=/workspace bash scripts/setup_four_gpu_pod.sh --verify-only"
```

The internal server binds to `127.0.0.1:8011`. `setup4` also binds the interview-facing editor to
`0.0.0.0:8888`, which is the port exposed by the RunPod proxy. RunPod's port `8001` is nginx, not
MAESTRO. Use an SSH tunnel when directly inspecting the internal service:

```powershell
ssh -p $env:AGENTLODGE_POD_PORT -i $env:AGENTLODGE_POD_KEY `
  -L 18001:127.0.0.1:8011 "root@$env:AGENTLODGE_POD_HOST"
```

### Secure planner provisioning

Never put an OpenAI credential in Git, chat, a process argument, or a setup script. Rotate any
credential exposed through those channels. Keep the replacement in a private local file outside the
repository and point the gitignored pod config at that file:

```powershell
$env:AGENTLODGE_OAI_KEY_FILE = "C:\secure\maestro-openai-key.txt"
.\scripts\pod.ps1 setup4
```

`setup4` uploads the credential through a transient staging file, installs it as
`/root/.oai_key` with mode `0600`, deletes the staging copy, and requires a successful live planner
probe before declaring the Pod ready. This avoids relying on shared-volume Unix permissions, which
some RunPod volumes do not preserve. `GET /api/planner/status` reports both `configured` and
`verified`; the four-GPU verifier rejects the configured-but-unverified state.

The full-song path exports one animated GLB through the warm Blender daemon, splits all source frames
into contiguous GPU ranges, renders and encodes each range with Filament plus NVENC, validates every
shard and the exact final frame count, concatenates locally, and atomically publishes one audio-muxed
MP4. Failures are fail-closed and never fall back to a lower-quality renderer. Render caching is
disabled in the four-GPU SLA server so a cache hit cannot be mistaken for throughput.

The legacy `.\scripts\pod.ps1 setup` command remains available for older single-pod generation and
render workflows.

For same-Pod multi-GPU EEVEE, run one `render.frames` worker process per resident daemon and set
`AGENTLODGE_GPU_INDEX` to the nvidia-smi index. The guarded launcher validates the selector and gives
each worker ID a unique daemon/tmp root. It defaults to isolated `/tmp`; shared memory is only used
when `AGENTLODGE_RENDER_USE_SHM=1` passes the aggregate reservation preflight:

```bash
DAEMONS_PER_GPU=16
mkdir -p /workspace/maestro-workers
for gpu in 0 1; do
  for slot in $(seq 0 $((DAEMONS_PER_GPU - 1))); do
    id="render-g${gpu}-d${slot}"
    AGENTLODGE_GPU_INDEX="$gpu" \
      bash scripts/start_runpod_worker.sh render.frames "$id" \
        "/workspace/maestro-workers/$id" &
  done
done
```

One-GPU containers may omit `AGENTLODGE_GPU_INDEX` and do not load the selector.
The selector is loaded only in Blender, atomically attests the actual CUDA index selected through
`EGL_CUDA_DEVICE_NV`, and is checked together with scene, renderer, protocol, quality, UUID, and PCI
identity before a warm daemon is reused.
Each render range also receives a runtime scratch estimate. An opted-in `/dev/shm` worker
automatically uses its worker-unique `/tmp` fallback when either live free space or its reservation
cannot cover the range; Blender is not started if neither filesystem can safely hold the source
frames and FFV1 staging.

The 2x RTX PRO 4500 render-plus-packaging saturation calibration peaked at 14.887 fps with 20 daemons on one GPU and
22.462 fps with 32 daemons (16/GPU) across two GPUs: 1.509x speedup, 75.5% efficiency, and
7,670 MiB maximum VRAM/GPU. `/dev/shm` improved the 12-daemon raw result only ~2.4%, so it is not the
default. Direct Blender FFV1 was rejected despite a 2.9% saturated throughput gain because artifacts
were 45.4% larger and it removed the source-TGA digest contract. See
`experiments/performance/runpod_calibration_20260819/eevee_multigpu_consolidated.json`.

The fully verified integrated filesystem path was measured separately before the worker decode
optimization: four launcher workers
(two/GPU) rendered 400 frames in 45.283s (8.833 fps) with range, source/shard hash, and decoded-RGB
validation enabled. This is not a saturation-capacity replacement; see
`experiments/performance/runpod_calibration_20260819/integrated_render_validation.json`.

In `render.frames-ffv1-v3`, the worker derives the indexed RGB digest from the source sequence,
packages and probes FFV1 without decoding the shard again, and leaves the independent full shard
decode to the coordinator. Rerun the integrated capacity sweep before replacing the recorded
8.833 fps baseline.

## 3. Build a real candidate bank for a song

```powershell
.\scripts\pod.ps1 bank trs 4      # 4 seeded LODGE + 4 seeded EDGE real takes
```

This is where the **real generation** happens: K seeded LODGE + K seeded EDGE diffusion runs
(minutes each). It converts them to the assembled Z-up space and pulls
`bank_trs_*_seed*.npy` into `server/media/trs/bank/`. The editor then selects window candidates
from these real takes at edit time (instant), so the heavy generation is paid once here, not per
edit. **More seeds = more variety = edits succeed more often.**

## 4. Run the editor

```powershell
uvicorn server.app:app --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000  (auto-loads the bank from server/media/<sid>/bank/)
```

When hosted on the pod, the editor keeps Blender and the exact Y-Bot scene resident. Rendering uses
the same resolution, sample count, frame cadence, grounding path, and final encoder as the established
cold quality path; the speedup comes from removing startup/SSH work, bulk-loading animation curves,
parallel before/after renders, six-way full-song frame sharding, lossless raw intermediates, and
reusing exact cached outputs. Override quality only through the existing
`AGENTLODGE_RENDER_WIN_*` and `AGENTLODGE_RENDER_FULL_*` variables. Full-render concurrency defaults
to six (`AGENTLODGE_FULL_RENDER_WORKERS` / `AGENTLODGE_WARM_POOL`).

Uploaded songs return after the initial LODGE/EDGE result and full-quality preview are ready. Seed 0
is staged immediately; seeds 1 through `AGENTLODGE_BANK_K-1` are generated afterward by a detached
bank job and copied into the editor automatically, so editing variety no longer blocks first view.
LODGE and EDGE preprocessing/generation overlap on CUDA, with an automatic sequential retry if a
parallel branch fails.

## Why edits are instant, and how to switch on **live pod mode**

By default the editor does **best-of-K selection over a pre-generated bank**, real LODGE/EDGE
material, but generated ahead of time (step 3), so editing is a fast select + splice that works even
with the pod switched off. Its only limit is variety: an edit can only pick from the seeds you baked.

**Live pod mode** removes that ceiling: instead of a fixed bank, every unseen seed runs a *fresh*
LODGE/EDGE diffusion sample **on the pod, on demand**, so the search space is unbounded. It plugs in
through the same `WindowGenerator` protocol and degrades gracefully, if the pod can't produce a take
(unreachable, or the song isn't preprocessed for generation) it falls back to the local bank, then to
the offline mock, so the UI never breaks.

### Requirements, provision a generation pod
Live mode needs a **generation-provisioned** pod, *not* a render-only one. One command does the whole
stack (verified on an RTX PRO 4500 Blackwell / sm_120, CUDA 13 driver):

```powershell
.\scripts\pod.ps1 ssh "cd /workspace/AgentLODGE && WORKSPACE=/workspace bash scripts/setup_gen_pod.sh"
```

`setup_gen_pod.sh` is idempotent and bakes in the hard-won fixes:
- **CUDA torch that works on Blackwell**, it *uninstalls* any pre-existing `torch==*+cpu` first (a
  leftover CPU wheel has a higher version string than every `cu128` wheel, so a plain install is a
  silent no-op), installs `cu128`, and **gates on a real GPU matmul** before continuing.
- **LODGE + EDGE weights** via `scripts/download_gdrive.py` (the pods' `gdown` is a broken 6.1.0 that
  fails on Google's large-file "virus scan" page; the helper parses the confirm form instead).
- **`pyrender` + OSMesa** (LODGE's `render.py` imports pyrender at module load, miss it and every
  LODGE gen dies with `ModuleNotFoundError: pyrender`).
- **EDGE venv + Jukebox** (`/workspace/EDGE/.venv` shares the CUDA venv via a `.pth`; jukebox +
  jukemirlib install `--no-deps` so they don't downgrade torch), verified importing under torch 2.11.
- **LODGE on GPU**, repoints `/workspace/LODGE/.venv` at the CUDA venv.

Then preprocess each song you want to live-edit (LODGE feats are fast; EDGE Jukebox slices need the
~10GB 5B prior, downloaded once):

```powershell
.\scripts\pod.ps1 ssh "cd /workspace/AgentLODGE && WORKSPACE=/workspace bash scripts/setup_gen_pod.sh --song <sid>"
# or LODGE-only (skip Jukebox):  ... setup_gen_pod.sh --song <sid> --lodge-only
```

The song's `LODGE/data/finedance/music_wav/<sid>.wav` must exist (a byproduct of an upload / step 3).
Seed 0 also reuses `<lodge|edge>_fd_<sid>_full.npy` when present.

### Turn it on
```powershell
# point at the pod (steps 1-2 above), then:
$env:AGENTLODGE_LIVE="1"                 # enable live pod mode
$env:AGENTLODGE_POD_PYTHON="/workspace/AgentLODGE/.venv/bin/python"   # the CUDA venv on the gen pod
# optional search budget (each new seed is minutes of GPU):
$env:AGENTLODGE_LIVE_K="2"; $env:AGENTLODGE_LIVE_CYCLES="2"
uvicorn server.app:app --host 127.0.0.1 --port 8000
```
The header badge shows **LIVE POD** when it is active. Each new seed calls `scripts/gen_take.py <sid>
<lodge|edge> <seed>` on the pod (shipped automatically), which generates one take, converts it to the
Z-up 139 space, and caches it as `bank_<sid>_<bb>_seed<n>.npy`; the client scps it back into
`server/media/<sid>/bank/`, so **live editing also grows your bank** for next time. Expect roughly
`AGENTLODGE_LIVE_K x AGENTLODGE_LIVE_CYCLES` new seeds per backbone per edit, a few minutes each.

> Prefer the bank for quick, offline iteration; switch on live mode when an edit plateaus and you want
> the pod to search harder. Both use the identical reward + splice path, so results are comparable.

### Performance note (network-volume import latency)
`setup_gen_pod.sh` puts the venvs on `/workspace` (a RunPod **network** volume), which survives pod
restarts but has slow cold imports: a fresh `python` that imports torch takes **~2 min** the first
time (LODGE and EDGE each spawn a subprocess, so a single new seed can pay this twice on top of the
3-6 min diffusion). Consequences:
- A live edit that generates several fresh seeds can take 10-40 min. Keep `AGENTLODGE_LIVE_K` /
  `AGENTLODGE_LIVE_CYCLES` small (default 2/1), and remember seeds already pulled are cached (instant).
- For much faster iteration, copy the CUDA venv to local SSD (`cp -a /workspace/AgentLODGE/.venv
  /root/al_gpu`), repoint `LODGE/.venv` + the EDGE `.pth` at it, and set
  `AGENTLODGE_POD_PYTHON=/root/al_gpu/bin/python`, imports drop to ~10s. This is **ephemeral** (lost
  on restart), so re-copy after each pod start.
