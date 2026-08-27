#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT="$WORKSPACE/AgentLODGE"
WORKERS="$WORKSPACE/maestro-workers"
REGISTRY="$WORKERS/registry.json"
GENERATION_ENV_FILE="${MAESTRO_GENERATION_ENV_FILE:-$WORKSPACE/.maestro_generation.env}"
mkdir -p "$WORKERS"

if [ -f "$GENERATION_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$GENERATION_ENV_FILE"
  set +a
fi

WORKSPACE="$WORKSPACE" python3 - "$REGISTRY" <<'PY'
import json
import os
import sys

path = sys.argv[1]
workspace = os.environ["WORKSPACE"]
workers = []
for gpu in range(4):
    worker = f"jukebox-gpu{gpu}"
    workers.append({
        "id": worker,
        "capabilities": ["jukebox.extract"],
        "task_dir": f"{workspace}/maestro-workers/{worker}",
        "max_concurrency": 1,
        "metadata": {"gpu_index": str(gpu)},
    })
for worker, capability, gpu in (
    ("lodge-gpu0", "lodge.generate", 0),
    ("edge-gpu1", "edge.generate", 1),
    ("audio-lodge-cpu", "audio.lodge", 0),
    ("audio-edge-cpu", "audio.edge", 1),
    ("audio-beats-cpu", "audio.beats", 2),
    ("dance-generate-cpu", "dance.generate", 0),
):
    workers.append({
        "id": worker,
        "capabilities": [capability],
        "task_dir": f"{workspace}/maestro-workers/{worker}",
        "max_concurrency": 1,
        "metadata": {"gpu_index": str(gpu)},
    })
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"workers": workers}, handle, indent=2)
    handle.write("\n")
PY

start_worker() {
  local capability="$1"
  local worker="$2"
  local gpu="$3"
  local worker_root="$WORKERS/$worker"
  local pid_file="$worker_root/worker.pid"
  local cpu_threads=1
  local old_pid=""
  if [ "$capability" = "dance.generate" ]; then
    cpu_threads="${AGENTLODGE_DANCE_CPU_THREADS:-16}"
  elif [ "$capability" = "audio.beats" ]; then
    cpu_threads="${AGENTLODGE_BEAT_CPU_THREADS:-4}"
  fi
  mkdir -p "$worker_root"
  if [ -f "$pid_file" ]; then
    old_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
      if [ "${AGENTLODGE_FORCE_RESTART_WORKERS:-0}" != "1" ] \
          && "$ROOT/.venv/bin/python" - "$worker_root/heartbeat.json" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
assert payload.get("status") == "ready"
assert time.time() - path.stat().st_mtime <= 60
PY
      then
        echo "already running: $worker pid=$old_pid"
        return
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
  rm -f "$worker_root/heartbeat.json"
  WORKSPACE="$WORKSPACE" \
  AGENTLODGE_DISTRIBUTED=1 \
  AGENTLODGE_DISTRIBUTED_CAPABILITIES="jukebox.extract,audio.lodge,audio.edge,audio.beats,dance.generate,lodge.generate,edge.generate" \
  AGENTLODGE_SHARED_ROOT="$WORKSPACE" \
  AGENTLODGE_DISTRIBUTED_TRANSPORT=filesystem \
  AGENTLODGE_WORKER_REGISTRY="$REGISTRY" \
  AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE=60 \
  AGENTLODGE_GPU_INDEX="$gpu" \
  OAI_KEY_FILE="${OAI_KEY_FILE:-$HOME/.oai_key}" \
  OPENAI_CHAT_MODEL="${OPENAI_CHAT_MODEL:-gpt-4o-mini}" \
  AGENTLODGE_BEST_OF_K="${AGENTLODGE_BEST_OF_K:-1}" \
  AGENTLODGE_REQUIRE_FULL_BEST_OF_K="${AGENTLODGE_REQUIRE_FULL_BEST_OF_K:-0}" \
  AGENTLODGE_REQUIRE_LLM_STORYBOARD="${AGENTLODGE_REQUIRE_LLM_STORYBOARD:-0}" \
  AGENTLODGE_JUKEBOX_SHARED_SCRATCH="${AGENTLODGE_JUKEBOX_SHARED_SCRATCH:-/tmp/maestro-jukebox-shared}" \
  NUMBA_CACHE_DIR="/tmp/maestro-numba-$worker" \
  OMP_NUM_THREADS="$cpu_threads" \
  OPENBLAS_NUM_THREADS="$cpu_threads" \
  MKL_NUM_THREADS="$cpu_threads" \
  nohup bash "$ROOT/scripts/start_runpod_worker.sh" \
    "$capability" "$worker" "$worker_root" \
    >"$worker_root/worker.log" 2>&1 </dev/null &
  echo "$!" >"$pid_file"
  echo "started: $worker gpu=$gpu pid=$!"
}

for gpu in 0 1 2 3; do
  start_worker jukebox.extract "jukebox-gpu${gpu}" "$gpu"
done
start_worker lodge.generate lodge-gpu0 0
start_worker edge.generate edge-gpu1 1
start_worker audio.lodge audio-lodge-cpu 0
start_worker audio.edge audio-edge-cpu 1
start_worker audio.beats audio-beats-cpu 2
start_worker dance.generate dance-generate-cpu 0

if ! "$ROOT/.venv/bin/python" - "$REGISTRY" <<'PY'
import json
import sys
import time
from pathlib import Path

registry = json.loads(Path(sys.argv[1]).read_text())
deadline = time.time() + 900
pending = {worker["id"]: Path(worker["task_dir"]) for worker in registry["workers"]}
while pending and time.time() < deadline:
    for worker_id, root in list(pending.items()):
        heartbeat = root / "heartbeat.json"
        try:
            payload = json.loads(heartbeat.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("status") == "ready"
            and time.time() - heartbeat.stat().st_mtime <= 60
        ):
            print(f"WORKER_READY {worker_id} pid={payload.get('pid')}")
            pending.pop(worker_id)
        elif payload.get("status") in {"degraded", "stopping"}:
            raise SystemExit(f"worker {worker_id} entered {payload.get('status')}")
    if pending:
        time.sleep(2)
if pending:
    raise SystemExit(f"workers did not become ready: {sorted(pending)}")
print("FOUR_GPU_WORKERS_READY")
PY
then
  for log in "$WORKERS"/*/worker.log; do
    [ -f "$log" ] || continue
    echo "===== $log =====" >&2
    tail -n 80 "$log" >&2
  done
  exit 1
fi
