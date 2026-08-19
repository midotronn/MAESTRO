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

The protocol requires a shared mounted root and is configured with
`AGENTLODGE_WORKER_REGISTRY`, `AGENTLODGE_SHARED_ROOT`, and
`AGENTLODGE_DISTRIBUTED=1`. Local/fake worker tests validate capability routing,
idempotent result reuse, stale-worker rejection, path confinement, exact render settings, and
complete non-overlapping frame ranges. A two-shard synthetic RGB validation decoded all frames in
order with an exact aggregate pixel hash match; its immutable report is
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

- Give every render worker exactly one visible GPU.
- Keep NVIDIA EGL explicit with
  `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
- Set `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` or `all`.
- Render contiguous global frame ranges with the identical preloaded scene.
- Package each local shard losslessly before transfer; do not move thousands of uncompressed image
  files between pods.
- Verify decoded source-frame hashes before the one final H.264/audio encode.

`CUDA_VISIBLE_DEVICES` is not sufficient by itself because EEVEE renders through OpenGL/EGL rather
than the CUDA runtime. NVIDIA EGL must be selected explicitly, and one-GPU containers remain the
lowest-risk isolation method until a multi-GPU pod proves device binding.

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

## Recommended paid experiment sequence

Do not provision the full fleet first.

### Calibration 1: one render GPU

Benchmark the exact warm Blender scene on one candidate from each useful price/performance class:

- RTX 4090 or RTX 5090;
- L40S or RTX 6000 Ada;
- optionally H100 only if EEVEE/OpenGL performance justifies its price.

Measure:

- frames/second for the exact 1080 x 1080, 96-sample workload;
- p50 and p95 shard startup;
- GPU utilization and memory;
- lossless shard packaging throughput;
- final encoding time.

### Calibration 2: generation split

Use two workers:

- LODGE resident on one GPU;
- EDGE/Jukebox resident on one GPU.

Then measure Jukebox slice sharding with one additional worker. This establishes which serial
component remains before buying a render fleet.

### Scale test

Choose between:

1. one 4- or 8-GPU Secure Cloud pod if EEVEE device isolation and availability are proven;
2. a Secure Cloud fleet of single-GPU workers using network storage; or
3. an 8-GPU render pod plus separate generation workers.

The likely worker count is:

```text
render_workers = ceil(5400 / (measured_frames_per_second_per_gpu * 23))
```

Add headroom for p95 only after measuring two-worker and four-worker scaling efficiency.

### Proposed first paid calibration

Pending explicit approval, use **one Secure Cloud Pod with two RTX 5090 GPUs** for no more than two
hours. The two workers share that Pod's container filesystem, avoiding a network-volume dependency:

- official screened rate: $0.99/GPU-hour;
- maximum compute charge: 2 GPUs x 2 hours x $0.99 = $3.96;
- total authorization cap: $5.00, including incidental storage;
- use the same base template and container-disk sizing as the current benchmark Pod;
- copy only the required MAESTRO repositories, checkpoints, Blender runtime, scene, and calibration
  inputs from the current Pod;
- first prove separate CUDA visibility and separate EEVEE/EGL GPU execution; do not report
  two-worker render scaling if both Blender processes resolve to the same physical GPU;
- run one-worker exact render throughput, two-worker scaling, LODGE/EDGE resident inference, and
  distributed Jukebox slice timing;
- terminate the Pod immediately after reports and artifacts are copied.

Stop without scaling further if worker source integrity fails, FFV1 decoded pixels differ from
their worker source, independent-render drift exceeds the frozen threshold, two-worker render
efficiency is below 80%, a fixed first-draw LODGE or EDGE request remains above its 30-second p95
budget, or the measured capacity/cost model cannot plausibly satisfy the 60/90-second service
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

For context only, if a Secure Cloud RTX 5090 delivered merely the measured RTX PRO 4500 render rate,
23 ideal render workers would cost $22.77/hour, approximately $0.38 for one fully utilized
60-second render interval, or $546.48/day if held warm continuously. The 90%-efficiency planning
count of 26 would cost $25.74/hour or $617.76/day. These figures exclude generation workers,
storage, coordination, and idle capacity and are not a fleet recommendation because RTX 5090
throughput has not yet been measured on this exact EEVEE workload.

## Approval gate

Before any paid resource is created, provide:

- the exact GPU type and count;
- live Secure Cloud price;
- maximum benchmark spend;
- expected benchmark duration;
- teardown procedure;
- the measurements that determine whether to continue or stop.

Plan approval did not authorize GPU spending.
