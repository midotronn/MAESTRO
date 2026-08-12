#!/usr/bin/env bash
# One-command AgentLODGE pod provisioning. Idempotent: safe to re-run after every RunPod restart
# (RunPod wipes apt-installed system libs AND anything on /root, keeping only /workspace).
#
# It installs, in order:
#   1. system libraries Blender/ffmpeg need (X11 / GL / EGL) + LODGE's OSMesa + ffmpeg
#                                                                      -> wiped on every restart
#   2. a Python venv with torch + pytorch3d + the LODGE/EDGE deps        -> gone if it lived on /root
#   3. LODGE/EDGE venv links so the subprocess backends resolve
# and verifies the result. Checkpoints/data live on the /workspace network volume and survive
# restarts; on a brand-new volume run scripts/setup_gen_pod.sh first to clone the repositories,
# fetch checkpoints, install Blender, and build the exact Y-Bot render scene.
#
# Configurable via env (sensible defaults):
#   WORKSPACE   (default /workspace)   root of the persistent volume with AgentLODGE/LODGE/EDGE
#   VENV        (default /root/al_venv) venv location (local disk = fast; re-created each restart)
#   TORCH_INDEX (default cu128)        'cu128' for GPU generation, 'cpu' for render-only boxes
#
# Usage (on the pod):   WORKSPACE=/workspace TORCH_INDEX=cu128 bash scripts/setup_pod.sh
set -uo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${VENV:-/root/al_venv}"
TORCH_INDEX="${TORCH_INDEX:-cu128}"
AL="$WORKSPACE/AgentLODGE"

echo "=== AgentLODGE pod setup (workspace=$WORKSPACE venv=$VENV torch=$TORCH_INDEX) ==="

echo "--- [1/4] system libraries (Blender X11/GL/EGL + LODGE OSMesa + ffmpeg) ---"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>/dev/null || true
# libosmesa6 is for LODGE, not Blender: dld/data/utils imports PyOpenGL, which resolves an OSMesa
# backend at import time, so without it generation dies on `ImportError: Unable to load OpenGL
# library` before a single diffusion step runs.
apt-get install -y -qq \
  libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libgl1 libglu1-mesa \
  libsm6 libice6 libxext6 libx11-6 libegl1 libglvnd0 libgles2 libopengl0 libgl1-mesa-dri \
  libosmesa6 \
  ffmpeg git build-essential 2>&1 | tail -n 2 || true

echo "--- [2/4] python venv ($VENV) ---"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip wheel "setuptools<82"
if [ "$TORCH_INDEX" = "cpu" ]; then
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi
# runtime + generation deps (LODGE uses pytorch-lightning; EDGE uses accelerate)
pip install -q numpy scipy librosa soundfile matplotlib tqdm smplx trimesh einops omegaconf \
  imageio imageio-ffmpeg pytorch-lightning torchmetrics accelerate wandb fire p_tqdm h5py \
  opencv-python-headless psutil gdown
[ -f "$AL/requirements.txt" ] && pip install -q -r "$AL/requirements.txt" || true

echo "--- [3/4] pytorch3d (CPU build; no nvcc needed) ---"
pip install -q ninja fvcore iopath
pip show pytorch3d >/dev/null 2>&1 || \
  CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install -q --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo "--- [4/4] backend venv links + verify ---"
ln -sfn "$VENV" "$AL/.venv" 2>/dev/null || true
ln -sfn "$VENV" "$WORKSPACE/LODGE/.venv" 2>/dev/null || true
# EDGE gets its own venv that shares the heavy packages via a .pth (keeps EDGE's old pins isolated)
if [ -d "$WORKSPACE/EDGE" ] && [ ! -x "$WORKSPACE/EDGE/.venv/bin/python" ]; then
  PYVER="$(python -c 'import sys;print("python%d.%d"%sys.version_info[:2])')"
  python3 -m venv "$WORKSPACE/EDGE/.venv"
  echo "$VENV/lib/$PYVER/site-packages" > "$WORKSPACE/EDGE/.venv/lib/$PYVER/site-packages/_shared.pth"
  "$WORKSPACE/EDGE/.venv/bin/pip" install -q -U pip fire unidecode wget accelerate 2>/dev/null || true
fi

python - <<'PY'
import importlib
mods = ["torch", "pytorch3d", "numpy", "smplx", "librosa"]
ok = []
for m in mods:
    try:
        importlib.import_module(m); ok.append(m)
    except Exception as e:  # noqa
        print(f"  MISSING {m}: {e}")
import torch
print("  torch", torch.__version__, "cuda", torch.cuda.is_available())
print("  ready:", ", ".join(ok))
PY
BB="$WORKSPACE/blender/blender"
[ -x "$BB" ] && "$BB" --version 2>/dev/null | head -1 || echo "  (Blender not found at $BB; run runpod_bootstrap.sh for first-time data)"
echo "SETUP_POD_DONE"
