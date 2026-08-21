#!/usr/bin/env bash
# Start one capability-scoped MAESTRO worker on an explicitly selected GPU.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
MAESTRO_ROOT="${MAESTRO_ROOT:-$WORKSPACE/AgentLODGE}"
CAPABILITY="${1:?usage: start_runpod_worker.sh <capability> [worker-id] [task-dir]}"
WORKER_ID="${2:-$(hostname)-${CAPABILITY//./-}}"
TASK_DIR="${3:-$WORKSPACE/maestro-workers/$WORKER_ID}"
SHARED_ROOT="${AGENTLODGE_SHARED_ROOT:-$WORKSPACE}"
TRANSPORT="${AGENTLODGE_DISTRIBUTED_TRANSPORT:-filesystem}"
MAIN_PY="${AGENTLODGE_WORKER_PYTHON:-$MAESTRO_ROOT/.venv/bin/python}"
EXTRA_ARGS=()
ENV_HELPER="$MAESTRO_ROOT/scripts/render_worker_env.sh"

case "$CAPABILITY" in
  jukebox.extract)
    PY="${AGENTLODGE_JUKEBOX_PYTHON:-$WORKSPACE/EDGE/.venv/bin/python}"
    EXTRA_ARGS+=(--edge-root "${EDGE_CODE_PATH:-$WORKSPACE/EDGE}")
    ;;
  audio.lodge)
    PY="$MAIN_PY"
    EXTRA_ARGS+=(--lodge-root "${LODGE_CODE_PATH:-$WORKSPACE/LODGE}")
    ;;
  audio.edge)
    PY="$MAIN_PY"
    EXTRA_ARGS+=(--edge-root "${EDGE_CODE_PATH:-$WORKSPACE/EDGE}")
    ;;
  audio.beats)
    PY="$MAIN_PY"
    ;;
  dance.generate)
    PY="$MAIN_PY"
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
[ -f "$ENV_HELPER" ] || {
  echo "render worker environment helper is missing: $ENV_HELPER" >&2
  exit 1
}
# shellcheck disable=SC1090
source "$ENV_HELPER"
case "$CAPABILITY" in
  audio.*|dance.generate)
    export AGENTLODGE_RESOLVED_GPU_INDEX=cpu
    unset CUDA_VISIBLE_DEVICES
    unset NVIDIA_VISIBLE_DEVICES
    ;;
  *)
    agentlodge_configure_gpu "$CAPABILITY" "$WORKSPACE" || exit 1
    ;;
esac
if [ "$CAPABILITY" = "render.frames" ]; then
  agentlodge_configure_render_paths "$WORKER_ID" || exit 1
fi

export PYTHONUNBUFFERED=1

if [ "$TRANSPORT" = "http" ]; then
  [ "$CAPABILITY" = "render.frames" ] || {
    echo "HTTP transport is currently supported only for render.frames" >&2
    exit 1
  }
  [ -n "${AGENTLODGE_HTTP_COORDINATOR_URL:-}" ] || {
    echo "AGENTLODGE_HTTP_COORDINATOR_URL is required for HTTP transport" >&2
    exit 1
  }
  if [ -z "${AGENTLODGE_HTTP_TOKEN:-}" ] && [ -z "${AGENTLODGE_HTTP_TOKEN_FILE:-}" ]; then
    echo "AGENTLODGE_HTTP_TOKEN or AGENTLODGE_HTTP_TOKEN_FILE is required" >&2
    exit 1
  fi
  WORKER_SCRATCH="$AGENTLODGE_HTTP_WORKER_SCRATCH"
  mkdir -p "$WORKER_SCRATCH"
  EXTRA_ARGS+=(
    --transport http
    --coordinator-url "$AGENTLODGE_HTTP_COORDINATOR_URL"
    --worker-scratch "$WORKER_SCRATCH"
    --workspace-root "$WORKSPACE"
    --lease-seconds "${AGENTLODGE_HTTP_LEASE_SECONDS:-30}"
  )
else
  [ "$TRANSPORT" = "filesystem" ] || {
    echo "unsupported distributed transport: $TRANSPORT" >&2
    exit 1
  }
  mkdir -p "$TASK_DIR" "$SHARED_ROOT"
  export AGENTLODGE_SHARED_ROOT="$SHARED_ROOT"
  EXTRA_ARGS+=(
    --transport filesystem
    --task-dir "$TASK_DIR"
    --shared-root "$SHARED_ROOT"
  )
fi

if [ "$CAPABILITY" = "render.frames" ]; then
  export AGENTLODGE_WARM_POOL=1
  export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
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
  --poll-interval "${AGENTLODGE_WORKER_POLL_INTERVAL:-0.1}" \
  --heartbeat-interval "${AGENTLODGE_WORKER_HEARTBEAT_INTERVAL:-2.0}" \
  "${EXTRA_ARGS[@]}"
