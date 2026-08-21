#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT="${AGENTLODGE_ROOT:-$WORKSPACE/AgentLODGE}"
PORT="${AGENTLODGE_SLA_PORT:-8011}"
MODE="setup"

usage() {
  cat <<'EOF'
Usage: setup_four_gpu_pod.sh [--verify-only]

Provision and verify the exact warm four-GPU MAESTRO service. Set
AGENTLODGE_GIT_REF before cloning the repository to select a branch or commit.
EOF
}

case "${1:-}" in
  "")
    ;;
  --verify-only)
    MODE="verify"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

gpu_count="$(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null |
    sed '/^[[:space:]]*$/d' |
    wc -l
)"
if [ "$gpu_count" -ne 4 ]; then
  echo "FOUR_GPU_SETUP_FAILED: expected exactly 4 visible GPUs, found $gpu_count" >&2
  exit 1
fi

if [ "$MODE" = "setup" ]; then
  if [ ! -d "$ROOT/.git" ]; then
    git_url="${AGENTLODGE_GIT_URL:-https://github.com/midotronn/MAESTRO.git}"
    git_ref="${AGENTLODGE_GIT_REF:-main}"
    if ! command -v git >/dev/null 2>&1; then
      apt-get update -qq
      apt-get install -y -qq ca-certificates git
    fi
    mkdir -p "$(dirname "$ROOT")"
    if git ls-remote --exit-code --heads "$git_url" \
        "refs/heads/$git_ref" >/dev/null 2>&1; then
      git clone --branch "$git_ref" --single-branch "$git_url" "$ROOT"
    else
      git clone "$git_url" "$ROOT"
      git -C "$ROOT" fetch --depth=1 origin "$git_ref"
      git -C "$ROOT" checkout --detach FETCH_HEAD
    fi
    exec env \
      WORKSPACE="$WORKSPACE" \
      AGENTLODGE_ROOT="$ROOT" \
      AGENTLODGE_GIT_URL="$git_url" \
      AGENTLODGE_GIT_REF="$git_ref" \
      bash "$ROOT/scripts/setup_four_gpu_pod.sh"
  fi
  if [ -z "${AGENTLODGE_GIT_REF:-}" ]; then
    AGENTLODGE_GIT_REF="$(
      git -C "$ROOT" symbolic-ref --quiet --short HEAD ||
        git -C "$ROOT" rev-parse HEAD
    )"
    export AGENTLODGE_GIT_REF
  fi

  WORKSPACE="$WORKSPACE" \
    AGENTLODGE_GIT_REF="$AGENTLODGE_GIT_REF" \
    bash "$ROOT/scripts/setup_gen_pod.sh"
  WORKSPACE="$WORKSPACE" bash "$ROOT/scripts/setup_filament_pod.sh"
  WORKSPACE="$WORKSPACE" \
    AGENTLODGE_FORCE_RESTART_WORKERS="${AGENTLODGE_FORCE_RESTART_WORKERS:-1}" \
    bash "$ROOT/scripts/start_four_gpu_workers.sh"
  WORKSPACE="$WORKSPACE" \
    AGENTLODGE_SLA_FORCE_RESTART="${AGENTLODGE_SLA_FORCE_RESTART:-1}" \
    bash "$ROOT/scripts/start_four_gpu_server.sh"
fi

WORKSPACE="$WORKSPACE" \
AGENTLODGE_ROOT="$ROOT" \
AGENTLODGE_SLA_PORT="$PORT" \
"$ROOT/.venv/bin/python" - <<'PY'
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

workspace = Path(os.environ["WORKSPACE"])
root = Path(os.environ["AGENTLODGE_ROOT"])
port = int(os.environ["AGENTLODGE_SLA_PORT"])

required_files = (
    workspace / ".maestro_gen_pod_ready",
    workspace / ".maestro_filament_ready",
    workspace / ".cache/jukemirlib/prior_level_2.pth.tar",
    workspace / "blender/blender",
    workspace / "ybot_scene.blend",
    workspace / "EDGE/SMPL-to-FBX/ybot.fbx",
    workspace / "LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt",
    workspace / "EDGE/checkpoint.pt",
    workspace / ".agentlodge/lib/libagentlodge_egl_cuda_device.so",
    workspace / "maestro-filament-poc/filament_bench",
    workspace / "maestro-filament-poc/ybot_production_animated.glb",
    root / ".venv/bin/python",
)
missing = [str(path) for path in required_files if not path.is_file()]
if missing:
    raise SystemExit(f"missing required four-GPU artifacts: {missing}")

prior = workspace / ".cache/jukemirlib/prior_level_2.pth.tar"
if prior.stat().st_size != 10_288_727_721:
    raise SystemExit(f"Jukebox prior has unexpected size: {prior.stat().st_size}")

registry_path = workspace / "maestro-workers/registry.json"
registry = json.loads(registry_path.read_text(encoding="utf-8"))
expected_workers = {
    *(f"jukebox-gpu{gpu}" for gpu in range(4)),
    "lodge-gpu0",
    "edge-gpu1",
    "audio-lodge-cpu",
    "audio-edge-cpu",
    "audio-beats-cpu",
    "dance-generate-cpu",
}
registered = {worker["id"] for worker in registry["workers"]}
if registered != expected_workers:
    raise SystemExit(
        f"worker registry mismatch: expected {sorted(expected_workers)}, got {sorted(registered)}"
    )

now = time.time()
for worker in registry["workers"]:
    heartbeat_path = Path(worker["task_dir"]) / "heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    worker_id = worker["id"]
    if heartbeat.get("worker_id") != worker_id or heartbeat.get("status") != "ready":
        raise SystemExit(f"worker is not ready: {worker_id}: {heartbeat}")
    age = now - float(heartbeat.get("updated_at", 0))
    if age > 60:
        raise SystemExit(f"worker heartbeat is stale: {worker_id}: {age:.1f}s")
    pid = int(heartbeat["pid"])
    if not Path(f"/proc/{pid}").exists():
        raise SystemExit(f"worker process is absent: {worker_id}: pid={pid}")

attestation_path = Path("/tmp/maestro-blend-daemon/d0/daemon.attestation.json")
attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
quality = attestation.get("quality", {})
expected_quality = {
    "width": 1080,
    "height": 1080,
    "samples": 96,
    "engine": "eevee",
    "denoise": 1,
    "frame_format": "tga",
}
if quality != expected_quality:
    raise SystemExit(f"warm Blender quality mismatch: {quality}")
if attestation.get("render_contract_version") != "render.frames-ffv1-v3":
    raise SystemExit(f"unexpected render contract: {attestation}")
if attestation.get("selector", {}).get("selected_cuda_index") != 0:
    raise SystemExit(f"warm Blender daemon is bound to the wrong GPU: {attestation}")

def get_json(path: str):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=10
    ) as response:
        return json.load(response)

songs = get_json("/api/songs")
if not isinstance(songs, (dict, list)):
    raise SystemExit("/api/songs did not return a JSON object or array")

planner = get_json("/api/planner/status")
configured = planner.get("configured") is True
verified = planner.get("verified") is True
if configured != verified:
    raise SystemExit(f"planner is configured but not verified: {planner}")
if os.environ.get("AGENTLODGE_REQUIRE_LLM_PLANNER") == "1" and not verified:
    raise SystemExit(f"planner verification is required but unavailable: {planner}")

ffmpeg_path_file = workspace / "maestro-filament-poc/ffmpeg.path"
ffmpeg = Path(ffmpeg_path_file.read_text(encoding="utf-8").strip())
if not ffmpeg.is_file():
    raise SystemExit(f"selected FFmpeg is absent: {ffmpeg}")
encoders = subprocess.run(
    [str(ffmpeg), "-hide_banner", "-encoders"],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
).stdout
if "h264_nvenc" not in encoders:
    raise SystemExit("selected FFmpeg is missing h264_nvenc")
subprocess.run(
    [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:r=1",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ],
    check=True,
    stdin=subprocess.DEVNULL,
)

status = {
    "gpu_count": 4,
    "workers_ready": len(expected_workers),
    "server_port": port,
    "planner_configured": configured,
    "planner_verified": verified,
    "ffmpeg": str(ffmpeg),
    "quality": expected_quality,
}
(workspace / ".maestro_four_gpu_ready.json").write_text(
    json.dumps(status, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": "FOUR_GPU_POD_READY", **status}, sort_keys=True))
PY
