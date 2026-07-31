#!/usr/bin/env bash
# ============================================================================================
# setup_gen_pod.sh -- provision a fresh RunPod GPU pod for AgentLODGE *generation* (live pod mode
# and candidate-bank builds), verified on an NVIDIA RTX PRO 4500 Blackwell (sm_120), CUDA 13 driver.
#
# This is the "full stack" superset of scripts/setup_pod.sh (render-only): it additionally installs
# CUDA PyTorch that works on Blackwell, the LODGE + EDGE diffusion weights, and the Jukebox venv used
# to extract EDGE features. Idempotent: safe to re-run after a pod restart (re-installs apt libs and
# re-links venvs; skips downloads that already exist).
#
# Usage (on the pod):   WORKSPACE=/workspace bash scripts/setup_gen_pod.sh
# After it finishes:     WORKSPACE=/workspace bash scripts/setup_gen_pod.sh --song <sid>   # preprocess
#                        WORKSPACE=/workspace /workspace/AgentLODGE/.venv/bin/python \
#                            scripts/gen_take.py <sid> lodge 1                              # smoke test
#
# HARD-WON LESSONS baked in (see comments):
#   * a pre-existing torch==*+cpu shadows the cu128 wheel -> must pip-uninstall torch* FIRST.
#   * gdown 6.1.0 on the pods is broken for large files -> use scripts/download_gdrive.py.
#   * LODGE render.py imports pyrender at module load -> pyrender + libosmesa6 are REQUIRED.
#   * EDGE inference + Jukebox run in /workspace/EDGE/.venv, which shares the CUDA venv via a .pth.
#   * /workspace/LODGE/.venv must point at the CUDA venv (not a CPU venv) for GPU LODGE.
# ============================================================================================
set -uo pipefail

WORK="${WORKSPACE:-/workspace}"
VENV="$WORK/AgentLODGE/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
TORCH_INDEX="${AGENTLODGE_TORCH_INDEX:-cu128}"        # cu128 verified on Blackwell; use 'cpu' for render-only
AL="$WORK/AgentLODGE"
step() { echo ""; echo "=== $* ==="; }
die()  { echo "SETUP_FAILED: $*" >&2; exit 1; }

# ---- optional per-song preprocessing mode -------------------------------------------------
if [ "${1:-}" = "--song" ]; then
  SID="${2:?--song needs a <sid>}"
  step "preprocess $SID (LODGE feats + EDGE Jukebox slices)"
  cd "$AL" && WORKSPACE="$WORK" "$PY" scripts/preprocess_song.py "$SID" "${3:-}" || die "preprocess failed"
  echo "PREPROCESS_OK $SID"; exit 0
fi

# ---- 1. system libraries (wiped on every pod restart) -------------------------------------
step "system libraries (ffmpeg + headless OpenGL/OSMesa for pyrender)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y -qq ffmpeg libsndfile1 build-essential git \
    libosmesa6 libosmesa6-dev libgl1-mesa-glx libglu1-mesa freeglut3-dev libglib2.0-0 \
    libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 >/dev/null 2>&1 || true
fi

# ---- 2. GPU present? ----------------------------------------------------------------------
step "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || die "no GPU"

# ---- 3. repos -----------------------------------------------------------------------------
step "repos"
cd "$WORK"
[ -d AgentLODGE ] || git clone -q https://github.com/midotronn/AgentLODGE.git
[ -d LODGE ]      || git clone -q https://github.com/li-ronghui/LODGE.git
# EDGE: a partial clone (only .venv/SMPL-to-FBX) has no model code -> re-clone if EDGE.py is missing.
if [ ! -f EDGE/EDGE.py ]; then rm -rf EDGE && git clone -q https://github.com/Stanford-TML/EDGE.git; fi
( cd AgentLODGE && git pull --ff-only -q || true )

# ---- 4. CUDA venv + PyTorch (Blackwell gate) ----------------------------------------------
step "CUDA venv + torch ($TORCH_INDEX)"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$PIP" install -q -U pip wheel setuptools
# CRITICAL: uninstall any pre-existing torch first. A leftover '2.13.0+cpu' has a higher version
# string than every cu128 wheel, so 'pip install torch --index-url .../cu128' becomes a no-op.
"$PIP" uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
if [ "$TORCH_INDEX" = "cpu" ]; then
  "$PIP" install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  "$PIP" install -q torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi

# ---- 5. python deps (pyrender is REQUIRED: LODGE render.py imports it at load) -------------
step "python deps"
"$PIP" install -q -r "$AL/requirements.txt"
"$PIP" install -q gdown omegaconf pytorch-lightning einops tqdm soundfile librosa \
  opencv-python-headless pyrender PyOpenGL trimesh smplx p_tqdm h5py imageio psutil \
  torchmetrics accelerate wandb fire
# pytorch3d (transforms only -> CPU build is fine; no nvcc). Build isolation OFF so it sees torch.
"$PIP" show pytorch3d >/dev/null 2>&1 || \
  CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 "$PIP" install -q --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable" || \
  echo "  (pytorch3d build skipped/failed -- LODGE/EDGE rotation ops may need it)"

# ---- 6. LODGE + EDGE weights (via robust gdrive helper; gdown fails on the virus-scan page) -
step "LODGE weights"
cd "$WORK/LODGE"
mkdir -p configs && cp -f "$AL/scripts/lodge_infer_local.yaml" configs/infer_local.yaml
LODGE_CODE_PATH="$WORK/LODGE" "$PY" "$AL/scripts/patch_lodge_pod.py" || true
if [ -f "exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt" ]; then
  echo "  LODGE weights present"
else
  "$PY" "$AL/scripts/download_gdrive.py" 13Yp__EPAw0EjrSS898X5FtSQGmveBykA pretrained_models.tar.gz || die "LODGE weights download"
  gunzip -c pretrained_models.tar.gz | tar --no-same-owner -xf - || die "LODGE weights extract"
fi
[ -f "$WORK/LODGE/data/smplx_neu_J_1.npy" ] || echo "  WARNING: missing LODGE/data/smplx_neu_J_1.npy (FK/render)"

step "EDGE weights"
cd "$WORK/EDGE"
EDGE_CODE_PATH="$WORK/EDGE" "$PY" "$AL/scripts/patch_edge_pod.py" || true
[ -f checkpoint.pt ] || "$PY" "$AL/scripts/download_gdrive.py" 1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ checkpoint.pt || die "EDGE checkpoint download"

# ---- 7. EDGE venv (shares CUDA venv via .pth) + Jukebox --------------------------------------
step "EDGE venv + Jukebox"
PYVER="$("$PY" -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
if [ ! -x "$WORK/EDGE/.venv/bin/python" ]; then
  python3 -m venv "$WORK/EDGE/.venv"
  echo "$VENV/lib/$PYVER/site-packages" > "$WORK/EDGE/.venv/lib/$PYVER/site-packages/_shared_venv.pth"
  "$WORK/EDGE/.venv/bin/pip" install -q -U pip
fi
"$WORK/EDGE/.venv/bin/python" -c "import jukemirlib" 2>/dev/null || {
  "$WORK/EDGE/.venv/bin/pip" install -q --no-build-isolation --no-deps "git+https://github.com/rodrigo-castellon/jukebox.git"
  "$WORK/EDGE/.venv/bin/pip" install -q --no-deps "git+https://github.com/rodrigo-castellon/jukemirlib.git"
  "$WORK/EDGE/.venv/bin/pip" install -q fire unidecode wget
}

# ---- 8. LODGE runs in the CUDA venv (GPU), not a CPU venv ----------------------------------
step "point LODGE at the CUDA venv (GPU)"
ln -sfn "$VENV" "$WORK/LODGE/.venv"

# ---- 9. Blackwell GPU gate ----------------------------------------------------------------
step "GPU gate (torch matmul on this card)"
"$PY" - <<'PY' || die "CUDA torch not usable on this GPU"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
c = (torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda"))
torch.cuda.synchronize()
print(f"  OK torch {torch.__version__} on {torch.cuda.get_device_name(0)} sm_{torch.cuda.get_device_capability(0)}")
PY

echo ""
echo "GEN_POD_READY"
echo "Next: WORKSPACE=$WORK bash scripts/setup_gen_pod.sh --song <sid>   # then gen_take.py / live mode"
