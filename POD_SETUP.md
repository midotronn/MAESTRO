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

## 2. Provision the pod (idempotent — re-run after every restart)

```powershell
.\scripts\pod.ps1 setup
```

`setup_pod.sh` installs, idempotently:
1. the system libs Blender/ffmpeg need — `libXrender/libXi/libEGL/libglvnd/...` + `ffmpeg`
   (these are wiped on **every** restart);
2. a Python venv (`/root/al_venv`, fast local disk) with `torch` + `pytorch3d` + the LODGE/EDGE
   generation deps (`pytorch-lightning`, `accelerate`, ...);
3. `LODGE/.venv` and `EDGE/.venv` links so the diffusion backends resolve.

Set `AGENTLODGE_TORCH_INDEX=cpu` for a render-only box (no GPU generation), else the default
`cu128` installs CUDA torch for real backbone generation.

> First time on a **new /workspace volume** (no checkpoints yet), run
> `.\scripts\pod.ps1 ssh "cd /workspace/AgentLODGE && bash scripts/runpod_bootstrap.sh"` once to
> fetch the LODGE/EDGE checkpoints and Blender, then `pod.ps1 setup`.

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

## Why edits are instant — and how to switch on **live pod mode**

By default the editor does **best-of-K selection over a pre-generated bank** — real LODGE/EDGE
material, but generated ahead of time (step 3), so editing is a fast select + splice that works even
with the pod switched off. Its only limit is variety: an edit can only pick from the seeds you baked.

**Live pod mode** removes that ceiling: instead of a fixed bank, every unseen seed runs a *fresh*
LODGE/EDGE diffusion sample **on the pod, on demand**, so the search space is unbounded. It plugs in
through the same `WindowGenerator` protocol and degrades gracefully — if the pod can't produce a take
(unreachable, or the song isn't preprocessed for generation) it falls back to the local bank, then to
the offline mock, so the UI never breaks.

### Requirements
Live mode needs a **generation-provisioned** pod, *not* a render-only one:
- CUDA torch (`setup_pod.sh` with the default `AGENTLODGE_TORCH_INDEX=cu128`, **not** `cpu`),
- the LODGE + EDGE checkpoints (`runpod_bootstrap.sh`), and
- the song preprocessed for regen: `lodge_fd_<sid>_feats.npy`, `edge<sid>_slices.npy`, and
  `LODGE/data/finedance/music_wav/<sid>.wav` present in `/workspace` (a byproduct of step 3 / an
  upload). Seed 0 also reuses `<lodge|edge>_fd_<sid>_full.npy` when present.

### Turn it on
```powershell
# point at the pod (steps 1-2 above), then:
$env:AGENTLODGE_LIVE="1"                 # enable live pod mode
$env:AGENTLODGE_POD_PYTHON="/root/al_venv/bin/python"   # pod venv python (optional; default: python)
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

