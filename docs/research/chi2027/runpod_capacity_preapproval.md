# RunPod capacity proposal before paid benchmarking

## Target

- Browser upload start to final downloadable video.
- Three-minute song, approximately 5,400 frames at 30 FPS.
- p50 at or below 60 seconds and p95 at or below 90 seconds.
- Exact current full quality: 1080 x 1080, EEVEE, 96 samples, every frame, current Y-Bot scene,
  camera, color management, H.264, and audio.
- Warm workers. Cold image, model, and scene startup is reported separately.

## What the current successful baseline implies

The first successful browser-to-final trace used one RTX PRO 4500 Blackwell with six resident
Blender daemons:

- 5,120 frames;
- browser upload: 10.168 seconds;
- EDGE/Jukebox preprocessing: 226.210 seconds;
- combined LODGE/EDGE generation: 77.744 seconds;
- full-quality render and encode: 460.743 seconds;
- browser upload to downloadable video: 790.448 seconds.

This is one completed baseline, not yet an SLA distribution. The frozen trace is
`experiments/performance/runs/trs_61887.json`; four additional unchanged-pod runs are required
before reporting baseline p50/p95.

At 5,120 / 460.743, the measured aggregate render rate is approximately
**11.11 frames/second**.
The 23-second render budget for 5,400 frames requires approximately **234.78 frames/second**. Under
unrealistic perfect linear scaling, this is **22 measured GPU equivalents for rendering alone**.
A planning fleet of 24 equivalent render workers provides only limited coordination and p95
headroom and remains provisional until target-GPU calibration.

Preprocessing plus generation currently consumes approximately 303 seconds. Jukebox's independent
five-second slices dominate preprocessing, while LODGE and EDGE each take about 68 seconds while
contending on the same GPU. These stages require resident, role-isolated workers rather than a
single aggregate throughput multiplier.

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
- `lodge.generate`: runs full-song LODGE with model caches retained in-process;
- `edge.generate`: runs full-song EDGE with the checkpoint retained in-process;
- `render.frames`: renders a contiguous global frame range at the fixed quality contract to local
  images, hashes the source frames, and transfers one lossless FFV1 shard.

The protocol requires a shared mounted root and is configured with
`AGENTLODGE_WORKER_REGISTRY`, `AGENTLODGE_SHARED_ROOT`, and
`AGENTLODGE_DISTRIBUTED=1`. Local/fake worker tests validate capability routing,
idempotent result reuse, stale-worker rejection, path confinement, exact render settings, and
complete non-overlapping frame ranges. A two-shard synthetic RGB validation decoded all frames in
order with an exact aggregate pixel hash match; its immutable report is
`experiments/performance/lossless_transport_validation.json`. An exact-scene source-frame hash
comparison remains required on the calibration worker before paid fleet scaling. Paid target-GPU
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

- GPU and pricing page: <https://www.runpod.io/gpu-models>
- Pod API: <https://docs.runpod.io/api-reference/pods/POST/pods>
- Pod pricing: <https://docs.runpod.io/pods/pricing>
- Network volumes: <https://docs.runpod.io/storage/network-volumes>
- S3-compatible uploads: <https://docs.runpod.io/storage/s3-api>
- Serverless endpoint configuration:
  <https://docs.runpod.io/serverless/endpoints/endpoint-configurations>

Time-sensitive observations from the August 2026 screening:

- Secure Cloud is preferable for a p95 latency target and supports network volumes.
- Community Cloud has weaker availability guarantees and is not suitable for the final SLA run.
- RunPod bills GPU time by the second.
- Network volumes provide a shared rendezvous, but workers should write separate shard directories.
- Async job submission and polling should remain in place rather than holding one request open.
- Multi-GPU pod availability is machine-dependent even though the API accepts `gpuCount`.

Public Community Cloud starting rates observed during screening included approximately:

- RTX 4090: $0.34/GPU-hour;
- RTX 5090: $0.69/GPU-hour;
- L40S: $0.79/GPU-hour;
- RTX A6000: $0.33/GPU-hour;
- A40: $0.35/GPU-hour;
- H100 PCIe: $1.99/GPU-hour.

These are not Secure Cloud quotes and must be rechecked immediately before approval.

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

## Cost formulas

```text
job_compute_cost =
  sum(worker_seconds * worker_gpu_hourly_rate / 3600)

warm_idle_cost_per_day =
  sum(warm_worker_count * worker_gpu_hourly_rate * 24)

per_song_storage_and_transfer =
  measured shared-storage and publication cost
```

For context only, 21 RTX 4090 equivalents at the screened Community rate would be $7.14/hour,
approximately $0.12 for one fully utilized 60-second job, and $171.36/day if kept warm continuously.
Secure Cloud would cost more. This is not a recommendation because RTX 4090 throughput relative to
the measured RTX PRO 4500 is unknown.

## Approval gate

Before any paid resource is created, provide:

- the exact GPU type and count;
- live Secure Cloud price;
- maximum benchmark spend;
- expected benchmark duration;
- teardown procedure;
- the measurements that determine whether to continue or stop.

Plan approval did not authorize GPU spending.
