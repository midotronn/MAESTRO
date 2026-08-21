# RunPod capacity proposal before paid benchmarking

## Target

- Browser upload start to final downloadable video.
- Three-minute song, approximately 5,400 frames at 30 FPS.
- p50 at or below 60 seconds and p95 at or below 90 seconds.
- Exact current full quality: 1080 x 1080, EEVEE, 96 samples, every frame, current Y-Bot scene,
  camera, color management, H.264, and audio.
- Warm workers. Cold image, model, and scene startup is reported separately.

## What the controlled baseline implies

Five accepted browser-to-final traces used the unchanged RTX PRO 4500 Blackwell Pod with six
resident Blender daemons. One additional attempt completed server-side but is excluded because its
browser runner terminated before persisting the final browser timing record. The completion rate is
therefore 5/6 (83.33%), and the excluded attempt remains documented in
`experiments/performance/baseline_failures.json`.

All five accepted runs used the same `warm_assets_cold_models` service state and rendered 5,120
full-quality frames. They establish this controlled baseline:

| Measurement | p50 | p95 |
|---|---:|---:|
| Browser upload | 7.926s | 9.811s |
| EDGE/Jukebox preprocessing | 227.041s | 234.471s |
| Concurrent LODGE/EDGE generation | 80.325s | 82.383s |
| Full-quality render and encode | 495.605s | 543.579s |
| Browser upload to downloadable video | **812.903s** | **884.710s** |

The observed end-to-end range is 790.448-897.966 seconds. These are controlled historical
measurements, not warm-model SLA measurements; target-GPU calibration must separately measure the
fully resident service state.

The median aggregate render rate is **10.331 frames/second**, and its low-tail p05 rate is
**9.437 frames/second**. The 23-second render budget for 5,400 frames requires
**234.783 frames/second**. Before coordination and encoding overhead, the capacity model therefore
requires:

- 23 measured RTX PRO 4500-equivalent render workers at ideal scaling;
- 26 equivalents at 90% scaling efficiency;
- 29 equivalents at 80% scaling efficiency.

Jukebox preprocessing requires a 28.37x p50 speedup and an 18.04x p95 speedup to fit its 8/13-second
budget. A single fixed-seed LODGE request requires 3.41x/2.32x acceleration against the
20/30-second concurrent-generation budget, while EDGE requires 3.58x/2.42x. LODGE and EDGE role
isolation removes contention but cannot parallelize one fixed-seed generation, so target-GPU
latency is a hard feasibility measurement rather than a worker-count estimate.

Therefore, a generic four-GPU proposal is not credible for the one-minute target unless the target
GPU and revised pipeline are several times faster than the measured RTX PRO 4500 path.

Reproduce these calculations with:

```bash
python scripts/model_pipeline_capacity.py
```

Once controlled traces are available, use the distribution-aware model instead:

```bash
python scripts/model_trace_capacity.py \
  experiments/performance/runs/*.json \
  --output experiments/performance/measured_capacity_report.json
```

It sizes the p50 render budget from median throughput and the p95 render budget from low-tail
throughput, while reporting the measured acceleration still required for Jukebox, LODGE, and EDGE.

## Required architecture changes before fleet sizing

The feature-flagged filesystem worker protocol is now implemented. It registers versioned workers
by capability, rejects stale heartbeats, uses deterministic task IDs for retries, and keeps the
single-pod path as the default. Implemented roles are:

- `jukebox.extract`: partitions ordered audio slices across persistent Jukebox workers;
- `lodge.generate`: loads both LODGE checkpoints before advertising ready, then retains them
  in-process for full-song requests;
- `edge.generate`: runs full-song EDGE with the checkpoint retained in-process;
- `render.frames`: renders a contiguous global frame range at the fixed quality contract to local
  images, hashes the source frames, and transfers one lossless FFV1 shard.

The original protocol still defaults to a shared mounted root configured with
`AGENTLODGE_WORKER_REGISTRY`, `AGENTLODGE_SHARED_ROOT`, and
`AGENTLODGE_DISTRIBUTED=1`. `render.frames` now also supports the opt-in authenticated HTTP
task/artifact transport for independent render containers. It uses deterministic task IDs,
expiring renewable leases, idempotent completions, coordinator-minted artifact IDs, confined
scratch roots, and SHA-256/size verification on both upload and download. Local/fake worker tests
validate authentication rejection, capability routing, deterministic result reuse and collision
rejection, lease reassignment, artifact tamper detection, path confinement, exact render settings,
and complete non-overlapping frame ranges. A two-shard synthetic RGB validation decoded all frames
in order with an exact aggregate pixel hash match; its immutable report is
`experiments/performance/lossless_transport_validation.json`.

The exact-scene validation over frames 600-611 also passed at the full quality contract. The
worker's recorded source-frame hash matched the retained worker files, and the FFV1 decoded RGB hash
matched those worker source pixels exactly. Separate EEVEE GPU renders are not byte-deterministic:
the direct reference and worker render differed in 4,345 of 41,990,400 color channels (0.0103%),
with mean absolute error 0.000105 on the 0-255 scale, maximum error 5/255, and PSNR 87.753 dB. This
is below the frozen strict thresholds of 0.05% changed channels, 0.001 mean error on the 0-255
scale, and 8/255 maximum error.
The immutable report is `experiments/performance/exact_render_equivalence.json`. Paid target-GPU
calibration is still gated on approval.

### Generation

- Keep LODGE, EDGE, Jukebox, and Blender processes resident.
- Place LODGE and EDGE on separate GPUs rather than contending on one device.
- Split independent best-of-K candidates across workers.
- Partition Jukebox's independent five-second audio slices across resident extraction workers, then
  restore their original order.
- Reuse the same audio decode, beat analysis, and feature artifacts instead of recomputing them.
- Overlap CPU structure analysis and staging with the remaining GPU work where dependencies allow.

### Rendering

- Give every render worker process an explicit CUDA/nvidia-smi index.
- Keep NVIDIA EGL explicit with
  `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
- Set `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` or `all`.
- In a multi-GPU container, load the setup-built EGL selector only in each Blender daemon process;
  it resolves the requested CUDA index through `EGL_CUDA_DEVICE_NV`.
- Render contiguous global frame ranges with the identical preloaded scene.
- Package each local shard losslessly before transfer; do not move thousands of uncompressed image
  files between pods.
- Verify FFV1 codec, dimensions, fps, exact frame count/timestamps, artifact hash/size, and indexed
  decoded-RGB digest before the one final H.264/audio encode.

`CUDA_VISIBLE_DEVICES` is not sufficient by itself because EEVEE renders through OpenGL/EGL rather
than the CUDA runtime. Native CUDA/EGL/DRI/PRIME environment routing all selected CUDA GPU 1. A
process-local `LD_PRELOAD` selector succeeded: it intercepts Blender/libepoxy EGL lookup, enumerates
EGL devices, maps each with `EGL_CUDA_DEVICE_NV`, and replaces the default display with
`eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, selected_device)`. The shim is scoped to Blender
daemon subprocesses; one-GPU containers do not load it.

## RunPod deployment constraints

Primary sources:

- GPU and pricing page: <https://www.runpod.io/pricing>
- Pod API: <https://docs.runpod.io/api-reference/pods/POST/pods>
- Pod pricing: <https://docs.runpod.io/pods/pricing>
- Network volumes: <https://docs.runpod.io/storage/network-volumes>
- S3-compatible uploads: <https://docs.runpod.io/storage/s3-api>
- Serverless endpoint configuration:
  <https://docs.runpod.io/serverless/endpoints/endpoint-configurations>

Time-sensitive observations from the August 2026 screening:

- Secure Cloud is preferable for a p95 latency target and supports network volumes.
- Community Cloud has weaker availability guarantees and is not suitable for the final SLA run.
- RunPod bills Pod compute by the second and states that ingress and egress have no separate fee.
- Network volumes provide a shared rendezvous, but workers should write separate shard directories.
- Standard network storage is $0.07/GB/month below 1 TB.
- The current setup has no shared network volume, so separate Pods cannot use the filesystem
  coordinator without an additional transfer layer.
- Async job submission and polling should remain in place rather than holding one request open.
- Multi-GPU pod availability is machine-dependent even though the API accepts `gpuCount`.

Official Secure Cloud on-demand Pod rates shown on the RunPod pricing page during the August 2026
screening were:

- RTX 4090: $0.74/GPU-hour;
- RTX 5090: $0.99/GPU-hour;
- L40S: $0.99/GPU-hour;
- RTX 6000 Ada: $0.84/GPU-hour;
- H100 PCIe: $2.89/GPU-hour.

Inventory still changes by region and must be confirmed in the deployment console immediately
before creation.

The live deployment console currently shows RTX PRO 4500 at $0.72/GPU-hour with high availability,
versus RTX 5090 at $0.99/GPU-hour with low availability. The first controlled scaling experiment
therefore uses RTX PRO 4500: matching the existing baseline removes GPU-model uplift as a confound
and directly measures isolation and two-GPU scaling. RTX 5090 remains a second-stage upper-bound
test only if serial generation latency still misses its budget.

## Paid calibration outcome: August 19, 2026

The user provisioned one Secure Cloud Pod with two RTX PRO 4500 Blackwell GPUs at
$0.72/GPU-hour each. Measurements ran for approximately 4,262 seconds, for an estimated compute
charge of **$1.70** through the end of measurement. This remained below the $2.88 compute ceiling
and $5 hard cap.

The immutable reports are under
`experiments/performance/runpod_calibration_20260819/`.

### Device isolation

CUDA workers isolated correctly:

- LODGE executed on physical GPU 0;
- EDGE executed on physical GPU 1;
- two Jukebox workers executed on distinct physical GPUs.

`scripts/start_runpod_worker.sh` accepts `AGENTLODGE_GPU_INDEX` for CUDA generation, Jukebox, and
render workers. Multi-GPU `render.frames` additionally requires the setup-built selector and rejects
missing/out-of-range indices or an invalid shim. Every worker ID receives an independent local
scratch and warm-daemon root, so one worker process can own one resident Blender daemon and multiple
workers can intentionally share a GPU.

Native routing attempts (`CUDA_VISIBLE_DEVICES`, `EGL_DEVICE_ID`, `EGL_VISIBLE_DEVICES`,
`DRI_PRIME`, runtime `NVIDIA_VISIBLE_DEVICES`, and PRIME variables) all failed and selected CUDA GPU
1. The selector's observed enumeration was EGL device 0 -> CUDA GPU 1 and EGL device 1 -> CUDA GPU
0; selection now uses the CUDA attribute rather than that fragile enumeration order. Both GPUs then
reached real utilization concurrently, and twelve TGA frames from each GPU were byte-identical.

### Same-Pod two-GPU render scaling

The exact full-quality render-plus-packaging saturation sweep measured:

| GPUs | Resident daemons | Daemons/GPU | Raw fps | External FFV1 end-to-end fps |
|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 14.080 | 13.526 |
| 1 | 16 | 16 | 15.034 | 14.476 |
| 1 | 20 | 20 | **15.496** | **14.887** |
| 2 | 20 | 10 | 21.619 | — |
| 2 | 24 | 12 | 22.780 | — |
| 2 | 28 | 14 | 23.788 | — |
| 2 | 32 | 16 | **24.132** | **22.462** |

The best full-path two-GPU speedup over the optimal one-GPU result is **1.509x**, or **75.5%**
scaling efficiency. Both GPUs reached 100% utilization; maximum observed VRAM was 7,670 MiB/GPU.
At 22.462 fps, 5,400 frames project to **240.4 seconds** before final merge/encode, so the 23-second
render budget remains far out of reach. Same-Pod multi-GPU rendering is valid, but RunPod's
four-GPU Pod limit still caps this topology and the authenticated HTTP transport remains required
for additional Pods.

The real-scene quality checks passed. Twelve full-quality TGA frames rendered on each physical GPU
were byte-identical, and the retained source-frame-to-FFV1 decoded RGB contract remained exact.

`/dev/shm` improved the 12-daemon two-GPU raw result only from 17.951 to 18.373 fps (~2.4%).
Workers therefore default to isolated `/tmp`; shared memory is explicit opt-in with an aggregate
reservation/preflight.

Direct Blender FFV1 was exact for 100 decoded RGB frames and improved a single daemon from 32.346s
to 23.486s. At saturation, however, it delivered 23.122 fps versus 22.462 fps for external
packaging (+2.9%) while producing 1.914 GB versus 1.316 GB for 3,200 frames (+45.4%) and removing
the source-TGA byte-digest contract. It is tested and rejected for now.

The consolidated immutable evidence is
`experiments/performance/runpod_calibration_20260819/eevee_multigpu_consolidated.json`.

A distinct fully verified pre-optimization production-path filesystem run used four launcher workers (two/GPU) for
400 frames and measured 45.283s, or 8.833 fps, with exact range, source/shard SHA-256, and decoded
RGB validation enabled. This lower integrated result includes transport/validation overhead and
does not replace the saturation sweep; an integrated capacity rerun is pending. Its immutable
summary is `experiments/performance/runpod_calibration_20260819/integrated_render_validation.json`.

The subsequent `render.frames-ffv1-v3` contract removes the redundant worker-side full FFV1 decode:
workers report the canonical indexed digest computed from source frames and only probe shard
structure/timing, while the coordinator remains responsible for an independent full decode and
digest match. The 8.833 fps result remains the immutable pre-optimization baseline.

### Resident generation

LODGE and EDGE ran concurrently on separate GPUs for three full-song seeds:

| Measurement | p50 | p95 |
|---|---:|---:|
| LODGE | 9.364s | 9.793s |
| EDGE | 5.104s | 5.726s |
| Concurrent generation wall time | **10.106s** | **10.522s** |

The 20/30-second generation budget is feasible on two resident RTX PRO 4500 workers. The previous
68-72-second role timings were dominated by repeated process/model startup rather than GPU
throughput.

### Jukebox preprocessing

All distributed outputs matched the original 69 x 150 x 4,800 baseline feature tensor bit-for-bit.

| Configuration | p50 | p95 |
|---|---:|---:|
| 1 warmed GPU | 101.761s | 102.630s |
| 2 warmed GPUs | **51.257s** | **52.789s** |

The measured two-GPU speedup is 1.985x, or 99.3% scaling efficiency. Four equivalent GPUs would
still require approximately 25.6 seconds, so the Pod's four-GPU maximum cannot meet the 8/13-second
preprocessing budget by scaling alone. The p50 budget requires 13 ideal equivalents, 15 at 90%
efficiency, or 16 at 80%.

Jukebox model preload alone was insufficient: each worker's first extraction paid a large
first-inference cost. The worker now runs representative inference before advertising ready.
Preloading both workers took 224.855 seconds, after which the first two-GPU request completed in
50.908 seconds and remained bit-identical.

## Decision after calibration

- Same-Pod multi-GPU rendering is now viable with the process-local EGL selector, but capacity
  planning must use the measured 1.509x best full-path two-GPU speedup rather than assume linear
  scaling.
- Do not switch to RTX 5090 yet: resident LODGE/EDGE already meet their budget, while Jukebox and
  cross-Pod rendering are architecture bottlenecks rather than simple serial-GPU bottlenecks.
- Use up to the four-GPU Pod limit where cost-effective, then calibrate the authenticated API-backed
  transport across additional render Pods.
- Optimize, batch, cache, or replace the first-round Jukebox preprocessing path before SLA fleet
  validation.
- Re-run a capped paid benchmark only after those changes can plausibly satisfy the 60/90-second
  target.

## Cost formulas

```text
job_compute_cost =
  sum(worker_seconds * worker_gpu_hourly_rate / 3600)

warm_idle_cost_per_day =
  sum(warm_worker_count * worker_gpu_hourly_rate * 24)

per_song_storage_and_transfer =
  measured shared-storage and publication cost
```

For context only, 20 Secure Cloud RTX PRO 4500 render workers at the live $0.72 rate would cost
$14.40/hour, approximately $0.24 for one fully utilized 60-second render interval, or $345.60/day
if held warm continuously. The 90%-efficiency planning count of 22 would cost $15.84/hour or
$380.16/day. These figures exclude generation workers, preprocessing workers, storage,
coordination, and idle capacity and are not a fleet recommendation.

## Further paid-resource gate

Before any paid resource is created, provide:

- the exact GPU type and count;
- live Secure Cloud price;
- maximum benchmark spend;
- expected benchmark duration;
- teardown procedure;
- the measurements that determine whether to continue or stop.

The completed two-GPU calibration did not authorize any additional GPU spending.
