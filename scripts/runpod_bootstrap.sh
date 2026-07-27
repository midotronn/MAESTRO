#!/usr/bin/env bash
# Bootstrap AgentLODGE on a RunPod GPU pod (CUDA + PyTorch template).
set -euo pipefail

WORK="${WORKSPACE:-/workspace}"
cd "$WORK"

echo "=== System packages (ffmpeg, headless OpenGL for pyrender, build tools) ==="
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq ffmpeg libsndfile1 build-essential \
    libosmesa6 libosmesa6-dev libgl1-mesa-glx libglu1-mesa freeglut3-dev libglib2.0-0 >/dev/null || true
fi
ffmpeg -version | head -1 || echo "WARNING: ffmpeg not found; stick-figure mp4 export will fail"

echo "=== System check ==="
nvidia-smi || { echo "ERROR: no GPU"; exit 1; }
free -h
df -h "$WORK"

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt 15000 ]; then
  echo "WARNING: GPU has ${VRAM_MB}MB VRAM; 16GB+ recommended for Jukebox 5B"
fi

DISK_FREE=$(df -BG "$WORK" | awk 'NR==2 {print $4}' | tr -d G)
if [ "$DISK_FREE" -lt 40 ]; then
  echo "WARNING: less than 40GB free on $WORK"
fi

echo "=== Clone / update repos ==="
[ -d AgentLODGE ] || git clone https://github.com/midotronn/AgentLODGE.git
[ -d LODGE ] || git clone https://github.com/li-ronghui/LODGE.git
[ -d EDGE ] || git clone https://github.com/Stanford-TML/EDGE.git
( cd AgentLODGE && git pull --ff-only )
( cd LODGE && git pull --ff-only || true )
( cd EDGE && git pull --ff-only || true )

echo "=== Python venv (shared) ==="
cd "$WORK/AgentLODGE"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip wheel setuptools

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# LODGE deps (opencv/pyrender/trimesh needed at render.py import; rest for model + FK)
pip install gdown omegaconf pytorch-lightning einops tqdm soundfile librosa \
  opencv-python-headless pyrender trimesh smplx p_tqdm h5py imageio psutil torchmetrics

# pytorch3d: no prebuilt wheel for recent torch, and PEP 517 build isolation hides the
# already-installed torch (ModuleNotFoundError: torch). Build CPU-only (no nvcc required)
# with build isolation disabled so the build backend can import torch.
pip install ninja fvcore iopath
pip show pytorch3d >/dev/null 2>&1 || CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
  || FORCE_CUDA=0 pip install --no-build-isolation pytorch3d

# EDGE app deps into the shared venv (EDGE model needs these; jukebox/jukemirlib are
# installed separately into a dedicated EDGE venv below so their old pins do NOT
# downgrade the shared LODGE stack: tqdm/soundfile/librosa).
pip install accelerate wandb einops

# LODGE shares the main venv; EDGE gets a DEDICATED venv. It inherits the shared heavy
# packages (torch, pytorch3d, numpy, librosa, ...) via a .pth file, then installs
# jukebox + jukemirlib with --no-deps so no versions are downgraded.
ln -sfn "$WORK/AgentLODGE/.venv" "$WORK/LODGE/.venv"

PYVER=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
rm -rf "$WORK/EDGE/.venv"
python3 -m venv "$WORK/EDGE/.venv"
echo "$WORK/AgentLODGE/.venv/lib/$PYVER/site-packages" \
  > "$WORK/EDGE/.venv/lib/$PYVER/site-packages/_shared_venv.pth"
"$WORK/EDGE/.venv/bin/pip" install -q -U pip
"$WORK/EDGE/.venv/bin/pip" install -q --no-build-isolation --no-deps \
  "git+https://github.com/rodrigo-castellon/jukebox.git"
"$WORK/EDGE/.venv/bin/pip" install -q --no-deps \
  "git+https://github.com/rodrigo-castellon/jukemirlib.git"
"$WORK/EDGE/.venv/bin/pip" install -q fire unidecode wget

echo "=== LODGE + EDGE config + patches ==="
mkdir -p "$WORK/LODGE/configs" "$WORK/LODGE/data"
cp -f "$WORK/AgentLODGE/scripts/lodge_infer_local.yaml" "$WORK/LODGE/configs/infer_local.yaml"
LODGE_CODE_PATH="$WORK/LODGE" python3 "$WORK/AgentLODGE/scripts/patch_lodge_pod.py"
EDGE_CODE_PATH="$WORK/EDGE" python3 "$WORK/AgentLODGE/scripts/patch_edge_pod.py"

SMPL_J="$WORK/LODGE/data/smplx_neu_J_1.npy"
if [ ! -f "$SMPL_J" ]; then
  echo "WARNING: missing $SMPL_J (needed for stick-figure video FK)"
  echo "  Copy from your local Runs/LODGE/data/ or create a symlink after LODGE data download."
fi

echo "=== Download checkpoints ==="
cd "$WORK/LODGE"
if [ ! -f "exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt" ]; then
  pip install gdown
  bash download_checkpoints.sh || {
    gdown 13Yp__EPAw0EjrSS898X5FtSQGmveBykA -O pretrained_models.tar.gz
    # --no-same-owner: the RunPod network FS rejects chown, which otherwise makes tar
    # exit non-zero and (under set -e/pipefail) aborts the whole bootstrap.
    gunzip -c pretrained_models.tar.gz | tar --no-same-owner -xf -
  }
fi

cd "$WORK/EDGE"
if [ ! -f checkpoint.pt ]; then
  # Google Drive's confirm-token flow returns an HTML page to wget/download_model.sh for
  # large files; gdown handles the token and fetches the real 1.2GB checkpoint.
  pip install gdown
  gdown 1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ -O checkpoint.pt
fi

echo "=== AgentLODGE .env ==="
cd "$WORK/AgentLODGE"
if [ -f .env ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  EXISTING_KEY=$(grep -E '^OPENAI_API_KEY=' .env | cut -d= -f2- || true)
  OPENAI_API_KEY="${EXISTING_KEY:-}"
fi
cat > .env <<EOF
OUTPUT_DIR=$WORK/AgentLODGE/outputs
LODGE_CODE_PATH=$WORK/LODGE
EDGE_CODE_PATH=$WORK/EDGE
LODGE_WEIGHTS_PATH=$WORK/LODGE/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt
LODGE_GLOBAL_WEIGHTS_PATH=$WORK/LODGE/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt
EDGE_WEIGHTS_PATH=$WORK/EDGE/checkpoint.pt
LODGE_GENRE=Hiphop
AGENTLODGE_MIN_AUDIO_SECONDS=20
AGENTLODGE_MAX_EDGE_SLICES=15
AGENTLODGE_PARALLEL=0
IMAGE_BACKEND=openai
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_IMAGE_MODEL=gpt-image-1
GEMINI_API_KEY=${GEMINI_API_KEY:-}
EOF

echo "=== Test audio ==="
python3 - <<'PY'
import soundfile as sf, numpy as np, os
path = "/workspace/AgentLODGE/test_audio_30s.wav"
if not os.path.exists(path):
    sr = 15360
    t = np.linspace(0, 30, 30 * sr)
    y = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, y.astype(np.float32), sr)
    print("wrote synthetic 30s test wav")
else:
    print("test audio exists")
PY

echo "=== Bootstrap complete ==="
echo "Set OPENAI_API_KEY in .env if not already set."
echo "Run: cd $WORK/AgentLODGE && source .venv/bin/activate && bash scripts/runpod_e2e.sh test_audio_30s.wav outputs/runpod_e2e12"
