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

echo "### cached Y-Bot scene for the warm render pool"
# A migrated scene may be corrupt, so always start from a clean rebuild. When the rig, build script
# and ybot.fbx are already present, build it here so the warm daemon pool (server/warm_render.py) is
# ready on the first render; otherwise drop it and let the editor's prewarm build it after the code +
# ybot.fbx are in place.
rm -f "$WS/ybot_scene.blend"
BS="$A/scripts/blender_render_ybot.py"; YB="$WS/EDGE/SMPL-to-FBX/ybot.fbx"
if [ -x "$WS/blender/blender" ] && [ -f "$BS" ] && [ -f "$YB" ]; then
  EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  __EGL_VENDOR_LIBRARY_FILENAMES=$EGL "$WS/blender/blender" -b -noaudio -P "$BS" -- \
    --build-scene "$WS/ybot_scene.blend" --ybot "$YB" --width 448 --height 448 --samples 8 \
    >/dev/null 2>&1 && echo "built ybot_scene.blend" || echo "scene build failed (prewarm will retry)"
else
  echo "rig/build-script/ybot.fbx not all present yet; prewarm builds the scene on editor start"
fi

echo "PROVISION_DONE"
