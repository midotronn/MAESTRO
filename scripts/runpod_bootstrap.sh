#!/usr/bin/env bash
# Deprecated compatibility entry point. New automation lives in setup_gen_pod.sh and
# setup_four_gpu_pod.sh so all RunPod paths share the same fail-closed provisioning logic.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "runpod_bootstrap.sh is deprecated; dispatching to the maintained setup script" >&2
if [ "${1:-}" = "--four-gpu" ]; then
  shift
  exec bash "$SCRIPT_DIR/setup_four_gpu_pod.sh" "$@"
fi
exec bash "$SCRIPT_DIR/setup_gen_pod.sh" "$@"
