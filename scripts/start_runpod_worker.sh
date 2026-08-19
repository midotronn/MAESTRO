#!/usr/bin/env bash
# Start one capability-scoped MAESTRO worker on an isolated GPU.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MAESTRO_ROOT="${MAESTRO_ROOT:-$WORKSPACE/AgentLODGE}"
CAPABILITY="${1:?usage: start_runpod_worker.sh <capability> [worker-id] [task-dir]}"
WORKER_ID="${2:-$(hostname)-${CAPABILITY//./-}}"
TASK_DIR="${3:-$WORKSPACE/maestro-workers/$WORKER_ID}"
SHARED_ROOT="${AGENTLODGE_SHARED_ROOT:-$WORKSPACE}"
MAIN_PY="${AGENTLODGE_WORKER_PYTHON:-$MAESTRO_ROOT/.venv/bin/python}"
EXTRA_ARGS=()

case "$CAPABILITY" in
  jukebox.extract)
    PY="${AGENTLODGE_JUKEBOX_PYTHON:-$WORKSPACE/EDGE/.venv/bin/python}"
    EXTRA_ARGS+=(--edge-root "${EDGE_CODE_PATH:-$WORKSPACE/EDGE}")
    ;;
  lodge.generate)
    PY="$MAIN_PY"
    EXTRA_ARGS+=(
      --lodge-root "${LODGE_CODE_PATH:-$WORKSPACE/LODGE}"
      --lodge-weights "${LODGE_WEIGHTS_PATH:-$WORKSPACE/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt}"
      --lodge-global-weights "${LODGE_GLOBAL_WEIGHTS_PATH:-$WORKSPACE/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt}"
      --lodge-genre "${LODGE_GENRE:-Hiphop}"
    )
    ;;
  edge.generate)
    PY="$MAIN_PY"
    EXTRA_ARGS+=(
      --edge-root "${EDGE_CODE_PATH:-$WORKSPACE/EDGE}"
      --edge-checkpoint "${EDGE_WEIGHTS_PATH:-$WORKSPACE/EDGE/checkpoint.pt}"
    )
    ;;
  render.frames)
    PY="$MAIN_PY"
    ;;
  *)
    echo "unsupported capability: $CAPABILITY" >&2
    exit 2
    ;;
esac

[ -x "$PY" ] || {
  echo "worker Python is not executable: $PY" >&2
  exit 1
}
[ -f "$MAESTRO_ROOT/scripts/runpod_worker.py" ] || {
  echo "MAESTRO worker entry point is missing under $MAESTRO_ROOT" >&2
  exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is required" >&2
  exit 1
}

GPU_COUNT="$(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null |
    sed '/^[[:space:]]*$/d' |
    wc -l |
    tr -d ' '
)"
GPU_INDEX="${AGENTLODGE_GPU_INDEX:-}"
if [ "$GPU_COUNT" = "1" ]; then
  if [ -n "$GPU_INDEX" ] && [ "$GPU_INDEX" != "0" ]; then
    echo "single-GPU container only exposes GPU index 0, not $GPU_INDEX" >&2
    exit 1
  fi
elif [ -z "$GPU_INDEX" ]; then
  echo "multi-GPU container requires AGENTLODGE_GPU_INDEX; detected $GPU_COUNT GPUs" >&2
  exit 1
elif ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null |
  sed 's/[[:space:]]//g' |
  grep -qx "$GPU_INDEX"; then
  echo "AGENTLODGE_GPU_INDEX=$GPU_INDEX is not exposed by this container" >&2
  exit 1
fi

if [ "$CAPABILITY" = "render.frames" ] && [ "$GPU_COUNT" != "1" ]; then
  echo "render.frames requires a container exposing exactly one GPU; NVIDIA EGL ignores process-level CUDA_VISIBLE_DEVICES in a multi-GPU container" >&2
  exit 1
fi

if [ -n "$GPU_INDEX" ]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi

mkdir -p "$TASK_DIR" "$SHARED_ROOT"
export AGENTLODGE_SHARED_ROOT="$SHARED_ROOT"
export PYTHONUNBUFFERED=1

if [ "$CAPABILITY" = "render.frames" ]; then
  export AGENTLODGE_WARM_POOL=1
  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
  export AGENTLODGE_WORKER_TMP="${AGENTLODGE_WORKER_TMP:-/tmp/maestro-render-$WORKER_ID}"
  mkdir -p "$AGENTLODGE_WORKER_TMP"
  EXTRA_ARGS+=(
    --worker-tmp "$AGENTLODGE_WORKER_TMP"
    --render-width "${AGENTLODGE_RENDER_FULL_W:-1080}"
    --render-height "${AGENTLODGE_RENDER_FULL_H:-1080}"
    --render-samples "${AGENTLODGE_RENDER_FULL_SAMPLES:-96}"
    --render-engine "${AGENTLODGE_RENDER_ENGINE:-eevee}"
    --render-denoise "${AGENTLODGE_RENDER_DENOISE:-1}"
    --render-frame-format "${AGENTLODGE_RENDER_FRAME_FORMAT:-tga}"
  )
fi

cd "$MAESTRO_ROOT"
exec "$PY" scripts/runpod_worker.py \
  --worker-id "$WORKER_ID" \
  --capability "$CAPABILITY" \
  --task-dir "$TASK_DIR" \
  --shared-root "$SHARED_ROOT" \
  --poll-interval "${AGENTLODGE_WORKER_POLL_INTERVAL:-0.1}" \
  --heartbeat-interval "${AGENTLODGE_WORKER_HEARTBEAT_INTERVAL:-2.0}" \
  "${EXTRA_ARGS[@]}"
