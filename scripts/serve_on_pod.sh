#!/usr/bin/env bash
# Host the MAESTRO editor ON the RunPod pod, served on port 8888 (RunPod's public HTTPS proxy port),
# rendering on the pod itself via localhost SSH. Idempotent: safe to re-run after a pod restart.
#
# Env in (set by scripts/host_on_pod.ps1, or export manually):
#   OPENAI_API_KEY        enables the LLM edit agent
#   MAESTRO_AUTH_USER     basic-auth username (public URL is protected when both are set)
#   MAESTRO_AUTH_PASS     basic-auth password
#   WORKSPACE             persistent volume root (default /workspace)
# Usage: bash serve_on_pod.sh
set -uo pipefail
WS="${WORKSPACE:-/workspace}"
A="$WS/AgentLODGE"
PY="$A/.venv/bin/python"
PORT=8888
LOG="$WS/maestro_editor.log"

echo "== MAESTRO pod host =="

# 1) web deps in the generation venv (torch is already there; add the server stack)
"$PY" -c "import fastapi, uvicorn, httpx, multipart" 2>/dev/null || {
  echo "installing web deps into the pod venv..."
  "$PY" -m pip install -q "fastapi>=0.110" "uvicorn[standard]>=0.29" "httpx>=0.27" "python-multipart>=0.0.9" || {
    echo "FATAL: could not install web deps"; exit 1; }
}

# 2) passwordless localhost SSH so the editor can render on THIS pod (host=127.0.0.1) with no change
#    to server/rendering.py (it drives Blender over ssh/scp).
mkdir -p /root/.ssh && chmod 700 /root/.ssh
[ -f /root/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f /root/.ssh/id_ed25519 -q
grep -qF "$(cat /root/.ssh/id_ed25519.pub)" /root/.ssh/authorized_keys 2>/dev/null \
  || cat /root/.ssh/id_ed25519.pub >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -R "[127.0.0.1]:22" -f /root/.ssh/known_hosts >/dev/null 2>&1 || true
if ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 22 -i /root/.ssh/id_ed25519 \
     root@127.0.0.1 "echo localhost_ssh_ok" 2>/dev/null | grep -q localhost_ssh_ok; then
  echo "localhost self-render SSH: OK"
else
  echo "WARN: localhost SSH not working; rendering may fail"
fi

# 3) the server-side FK template (licence-gated; already on the pod under LODGE/data)
mkdir -p "$A/server/data"
[ -f "$A/server/data/smplx_neu_J_1.npy" ] || cp "$WS/LODGE/data/smplx_neu_J_1.npy" \
  "$A/server/data/smplx_neu_J_1.npy" 2>/dev/null || echo "note: FK template not pre-copied (prewarm will fetch)"

# 4) free port 8888 (stop Jupyter / any prior editor) and relaunch
for pid in $(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
  echo "stopping process on :$PORT (pid $pid)"; kill "$pid" 2>/dev/null || true
done
sleep 2

cd "$A" || { echo "FATAL: $A missing"; exit 1; }
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export MAESTRO_AUTH_USER="${MAESTRO_AUTH_USER:-}"
export MAESTRO_AUTH_PASS="${MAESTRO_AUTH_PASS:-}"
export AGENTLODGE_LIVE=1
export AGENTLODGE_POD_HOST=127.0.0.1
export AGENTLODGE_POD_PORT=22
export AGENTLODGE_POD_KEY=/root/.ssh/id_ed25519
export AGENTLODGE_POD_WS="$WS"
export AGENTLODGE_POD_USER=root
export AGENTLODGE_POD_PYTHON="$PY"
export AGENTLODGE_LIVE_K="${AGENTLODGE_LIVE_K:-2}"
export AGENTLODGE_LIVE_CYCLES="${AGENTLODGE_LIVE_CYCLES:-1}"

rm -f "$LOG"
setsid "$PY" -m uvicorn server.app:app --host 0.0.0.0 --port "$PORT" >"$LOG" 2>&1 &
sleep 6
if ss -ltn | grep -q ":$PORT "; then
  echo "MAESTRO_EDITOR_UP on :$PORT"
  echo "auth: $([ -n "$MAESTRO_AUTH_USER" ] && echo enabled || echo OPEN)"
  tail -n 3 "$LOG" 2>/dev/null || true
else
  echo "FATAL: editor failed to start; log:"; tail -n 25 "$LOG" 2>/dev/null; exit 1
fi
