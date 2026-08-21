#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT="$WORKSPACE/AgentLODGE"
LOG="$WORKSPACE/maestro-sla-server.log"
PID_FILE="$WORKSPACE/maestro-sla-server.pid"
PORT="${AGENTLODGE_SLA_PORT:-8011}"

health_ready() {
  local payload
  payload="$(curl -fsS "http://127.0.0.1:$PORT/api/songs" 2>/dev/null)" ||
    return 1
  printf '%s' "$payload" |
    "$ROOT/.venv/bin/python" -c \
      'import json, sys; value = json.load(sys.stdin); assert isinstance(value, (dict, list))' \
      >/dev/null 2>&1
}

if [ -f "$PID_FILE" ]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    if [ "${AGENTLODGE_SLA_FORCE_RESTART:-0}" != "1" ] && health_ready; then
      echo "server already running pid=$old_pid port=$PORT"
      exit 0
    fi
    kill "$old_pid"
    for _attempt in $(seq 1 30); do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$old_pid" 2>/dev/null; then
      kill -9 "$old_pid"
    fi
  fi
fi

export AGENTLODGE_LIVE=1
export AGENTLODGE_POD_HOST=127.0.0.1
export AGENTLODGE_POD_PORT=22
export AGENTLODGE_POD_WS="$WORKSPACE"
export AGENTLODGE_POD_USER=root
export AGENTLODGE_POD_PYTHON="$ROOT/.venv/bin/python"
export AGENTLODGE_DISTRIBUTED=1
export AGENTLODGE_DISTRIBUTED_CAPABILITIES="jukebox.extract,audio.lodge,audio.edge,audio.beats,dance.generate,lodge.generate,edge.generate"
export AGENTLODGE_DISTRIBUTED_TRANSPORT=filesystem
export AGENTLODGE_WORKER_REGISTRY="$WORKSPACE/maestro-workers/registry.json"
export AGENTLODGE_SHARED_ROOT="$WORKSPACE"
export AGENTLODGE_DISTRIBUTED_TMP="$WORKSPACE/maestro-distributed-tmp"
export AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE=30
export AGENTLODGE_FULL_RENDER_BACKEND=filament
export AGENTLODGE_FILAMENT_ROOT="$WORKSPACE/maestro-filament-poc"
export AGENTLODGE_NVIDIA_VK_ICD=/etc/vulkan/icd.d/nvidia_icd.json
export AGENTLODGE_FILAMENT_GPUS=4
export AGENTLODGE_FILAMENT_GPU_INDICES=0,1,2,3
export AGENTLODGE_FILAMENT_WORKERS_PER_GPU=1
export AGENTLODGE_FILAMENT_DISABLE_CACHE=1
export AGENTLODGE_WARM_POOL=1
export AGENTLODGE_RENDER_DAEMON_ROOT=/tmp/maestro-blend-daemon
export AGENTLODGE_GPU_INDEX=0
export AGENTLODGE_RENDER_MULTI_GPU=1
export AGENTLODGE_EGL_SELECTOR_SHIM="$WORKSPACE/.agentlodge/lib/libagentlodge_egl_cuda_device.so"
export AGENTLODGE_RENDER_FULL_W=1080
export AGENTLODGE_RENDER_FULL_H=1080
export AGENTLODGE_RENDER_FULL_SAMPLES=96
export AGENTLODGE_RENDER_ENGINE=eevee
export AGENTLODGE_RENDER_DENOISE=1
export AGENTLODGE_SERVICE_STATE=warm
export OAI_KEY_FILE="${OAI_KEY_FILE:-$WORKSPACE/.oai_key}"
if [ -n "${OPENAI_API_KEY:-}" ] || [ -s "$OAI_KEY_FILE" ]; then
  [ ! -s "$OAI_KEY_FILE" ] || chmod 600 "$OAI_KEY_FILE"
  export AGENTLODGE_VERIFY_LLM_PLANNER=1
  export AGENTLODGE_REQUIRE_LLM_PLANNER=1
elif [ "${AGENTLODGE_REQUIRE_LLM_PLANNER:-0}" = "1" ]; then
  echo "planner verification is required, but no secure OpenAI credential is configured" >&2
  exit 1
fi
export MAESTRO_ALLOW_UNAUDITED_RESEARCH=1
export NUMBA_CACHE_DIR=/tmp/maestro-numba-server-pipeline
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$WORKSPACE/maestro-distributed-tmp"
cd "$ROOT"
"$ROOT/.venv/bin/python" -c "import fastapi, uvicorn, multipart" 2>/dev/null ||
  "$ROOT/.venv/bin/pip" install -q \
    "fastapi>=0.110" "uvicorn[standard]>=0.29" "python-multipart>=0.0.9"

if [ "${AGENTLODGE_PREWARM_RENDER:-1}" = "1" ]; then
  "$ROOT/.venv/bin/python" - <<'PY'
from server import warm_render

ready = warm_render.ensure_pool(
    width=1080,
    height=1080,
    samples=96,
    engine="eevee",
    denoise=1,
    frame_format="tga",
    wait_ready=180,
)
if ready != 1:
    raise SystemExit(f"expected one warm Blender daemon, found {ready}")
print("MAESTRO_BLENDER_EXPORT_DAEMON_READY", flush=True)
PY
fi

rm -f "$LOG"
nohup "$ROOT/.venv/bin/python" -m uvicorn server.app:app \
  --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"

for _attempt in $(seq 1 90); do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    break
  fi
  if health_ready; then
    echo "MAESTRO_SLA_SERVER_READY pid=$(cat "$PID_FILE") port=$PORT"
    exit 0
  fi
  sleep 1
done
tail -n 100 "$LOG" >&2
exit 1
