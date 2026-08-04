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
# the LLM edit agent lazily imports the openai client in agent_edit._llm_plan; without it the agent
# silently falls back to the offline keyword planner. Checked separately because the guard above
# short-circuits once fastapi/uvicorn exist (e.g. on a gen-provisioned venv).
"$PY" -c "import openai" 2>/dev/null || "$PY" -m pip install -q openai || echo "WARN: openai missing (agent will use keyword fallback)"

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
# Warm Blender daemons (server/warm_render.py) are started by the editor's prewarm and persist across
# restarts as detached processes. Kill any from a previous editor + clear the pool dir so the new
# editor's prewarm brings the pool back up with the CURRENT blender_daemon.py (a surviving pool would
# keep running stale code and double-process requests).
pkill -9 -f "scripts/blender_daemon.py" 2>/dev/null || true
rm -rf "$WS/blend_daemon" 2>/dev/null || true
# Same for the warm LODGE generation daemon (server/warm_gen.py) so a redeploy restarts it with the
# current gen_daemon.py + lodge.py (a surviving daemon would keep serving stale code).
pkill -9 -f "scripts/gen_daemon.py" 2>/dev/null || true
rm -rf "$WS/gen_daemon" 2>/dev/null || true
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
# A cold first start imports torch + the server modules, which can take well over 6s; poll the port
# for up to 45s instead of a single short sleep (else a slow-but-healthy start looks like a failure).
for _i in $(seq 1 45); do ss -ltn | grep -q ":$PORT " && break; sleep 1; done
if ss -ltn | grep -q ":$PORT "; then
  echo "MAESTRO_EDITOR_UP on :$PORT"
  echo "auth: $([ -n "$MAESTRO_AUTH_USER" ] && echo enabled || echo OPEN)"
  # Warm Blender render pool (fast before/after compare): the editor's prewarm builds the cached
  # scene + starts the daemons; report whether its prerequisites are present so cold-only renders
  # are not a surprise.
  if [ -x "$WS/blender/blender" ] && [ -f "$WS/EDGE/SMPL-to-FBX/ybot.fbx" ]; then
    echo "warm render: enabled (blender + ybot.fbx present; pool warms on startup)"
  else
    echo "warm render: DISABLED (need $WS/blender/blender + $WS/EDGE/SMPL-to-FBX/ybot.fbx) -> compare uses cold render"
  fi
  tail -n 3 "$LOG" 2>/dev/null || true
else
  echo "FATAL: editor failed to start; log:"; tail -n 25 "$LOG" 2>/dev/null; exit 1
fi
