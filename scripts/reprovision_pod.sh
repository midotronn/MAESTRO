#!/usr/bin/env bash
# Re-provision a migrated/fresh pod for the MAESTRO editor + rendering (torch-free path).
set -uo pipefail
WS=/workspace
A=$WS/AgentLODGE
export DEBIAN_FRONTEND=noninteractive

echo "### apt: blender libs + ffmpeg + tools"
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq wget xz-utils ffmpeg \
  libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libgl1 libglu1-mesa \
  libsm6 libice6 libxext6 libx11-6 libegl1 libglvnd0 libgles2 libopengl0 libgl1-mesa-dri \
  2>&1 | tail -n 1 || true
which ffmpeg && echo "ffmpeg OK" || echo "ffmpeg MISSING"

echo "### blender 4.2.3"
if [ ! -x "$WS/blender/blender" ]; then
  rm -rf "$WS/blender"; mkdir -p "$WS/blender"
  wget -qO /tmp/blender.tar.xz https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz
  tar -xf /tmp/blender.tar.xz -C "$WS/blender" --strip-components=1
fi
"$WS/blender/blender" --version 2>&1 | head -1 || echo "BLENDER FAIL"

echo "### fresh torch-free editor+render venv"
[ -d "$A/.venv" ] && mv "$A/.venv" "$A/.venv.broken.$$" 2>/dev/null || true
python3 -m venv "$A/.venv"
"$A/.venv/bin/pip" install -q --upgrade pip
"$A/.venv/bin/pip" install -q "fastapi>=0.110" "uvicorn[standard]>=0.29" "httpx>=0.27" \
  "python-multipart>=0.0.9" numpy scipy
"$A/.venv/bin/python" -c "import fastapi,uvicorn,numpy,scipy.signal;print('VENV_OK numpy',numpy.__version__)"

echo "### drop possibly-corrupt cached scene (prewarm rebuilds it from ybot.fbx)"
rm -f "$WS/ybot_scene.blend"

echo "PROVISION_DONE"
