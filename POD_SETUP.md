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

## 2. Provision the pod (idempotent, re-run after every restart)

```powershell
.\scripts\pod.ps1 setup
```

For an existing workspace after a restart, `setup_pod.sh` installs, idempotently:
1. the system libs Blender/ffmpeg need, `libXrender/libXi/libEGL/libglvnd/...` + `libOSMesa`
   (which LODGE's PyOpenGL import needs, not Blender) + `ffmpeg`, `curl`, and `aria2`
   (these are wiped on **every** restart);
2. a Python venv (`/root/al_venv`, fast local disk) with `torch` + `pytorch3d` + the LODGE/EDGE
   generation deps (`pytorch-lightning`, `accelerate`, ...);
3. `LODGE/.venv` and `EDGE/.venv` links so the diffusion backends resolve.

The Jukebox 5B prior is stored under `/workspace/.cache/jukemirlib/` and linked back into
`/root/.cache/jukemirlib/`. The first EDGE preprocessing run still downloads the ~10GB file, but
subsequent pod restarts reuse it instead of downloading it again.

Set `AGENTLODGE_TORCH_INDEX=cpu` for a render-only box (no GPU generation), else the default
`cu128` installs CUDA torch for real backbone generation.

On a **new `/workspace` volume**, `pod.ps1 setup` detects that the repository is absent and runs
`setup_gen_pod.sh` as the full bootstrap. It clones MAESTRO, LODGE, and EDGE; downloads the model
weights; installs Blender; creates the persistent `/workspace/AgentLODGE/.venv`; builds the reusable
exact-Y-Bot scene; and verifies CUDA with a real GPU matrix multiplication. On later pod restarts,
the same command uses the lighter restart setup.

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
