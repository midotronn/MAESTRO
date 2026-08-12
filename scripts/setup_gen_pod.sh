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
BLENDER="$WORK/blender/blender"
YBOT="$WORK/EDGE/SMPL-to-FBX/ybot.fbx"
YBOT_SCENE="$WORK/ybot_scene.blend"
step() { echo ""; echo "=== $* ==="; }
die()  { echo "SETUP_FAILED: $*" >&2; exit 1; }

# ---- optional per-song preprocessing mode -------------------------------------------------
if [ "${1:-}" = "--song" ]; then
  SID="${2:?--song needs a <sid>}"
  MODE="${3:-}"
  # EDGE features need Jukebox's ~10GB 5B prior. jukemirlib fetches it with a NON-resumable wget that
  # routinely dies mid-download; pre-fetch it resumably (curl -C -) to the cache path jukemirlib checks,
  # so setup_models() finds it and skips its own fragile download.
  if [ "$MODE" != "--lodge-only" ]; then
    PRIOR="$HOME/.cache/jukemirlib/prior_level_2.pth.tar"; PRIOR_SIZE=10288727721
    mkdir -p "$(dirname "$PRIOR")"
    if [ ! -f "$PRIOR" ] || [ "$(stat -c %s "$PRIOR" 2>/dev/null || echo 0)" -lt "$PRIOR_SIZE" ]; then
      TMP=$(ls "$(dirname "$PRIOR")"/prior_level_2.pth.tar*.tmp 2>/dev/null | head -1)
      [ -n "$TMP" ] && mv -f "$TMP" "$PRIOR"        # reuse any partial jukemirlib download
      step "fetching Jukebox 5B prior (~10GB, resumable, one-time)"
      curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors -o "$PRIOR" \
        https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar || die "prior download"
    fi
  fi
  step "preprocess $SID (LODGE feats + EDGE Jukebox slices)"
  cd "$AL" && WORKSPACE="$WORK" "$PY" scripts/preprocess_song.py "$SID" "$MODE" || die "preprocess failed"
  echo "PREPROCESS_OK $SID"; exit 0
fi

# ---- 1. system libraries (wiped on every pod restart) -------------------------------------
step "system libraries (ffmpeg + headless OpenGL/OSMesa for pyrender)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y -qq ffmpeg libsndfile1 build-essential git curl xz-utils \
    libosmesa6 libosmesa6-dev libgl1-mesa-glx libglu1-mesa freeglut3-dev libglib2.0-0 \
    libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 >/dev/null 2>&1 || true
fi

# ---- 2. GPU present? ----------------------------------------------------------------------
step "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || die "no GPU"

# ---- 3. Blender ---------------------------------------------------------------------------
step "Blender 4.2.3"
if [ ! -x "$BLENDER" ]; then
  archive="/tmp/blender-4.2.3-linux-x64.tar.xz"
  rm -rf "$WORK/blender"
  mkdir -p "$WORK/blender"
  curl -fL --retry 10 --retry-delay 5 --retry-all-errors \
    -o "$archive" \
    https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz \
    || die "Blender download"
  tar -xf "$archive" -C "$WORK/blender" --strip-components=1 || die "Blender extract"
  rm -f "$archive"
fi
"$BLENDER" --version 2>/dev/null | head -1 || die "Blender is not runnable"

# ---- 4. repos -----------------------------------------------------------------------------
step "repos"
cd "$WORK"
[ -d AgentLODGE ] || git clone -q https://github.com/midotronn/MAESTRO.git AgentLODGE
# LODGE: a bare/empty dir (fresh, or a volume migration that dropped file contents) has no code.
# Guard on the code package (dld/), and preserve downloaded weights (exp/) + the licence-gated
# SMPL-X template across a re-clone via mv (instant on the same fs, no multi-GB copy).
if [ ! -d LODGE/dld ]; then
  _lbak=$(mktemp -d)
  [ -d LODGE/exp ] && mv LODGE/exp "$_lbak/exp" 2>/dev/null || true
  [ -f LODGE/data/smplx_neu_J_1.npy ] && mkdir -p "$_lbak/data" && mv LODGE/data/smplx_neu_J_1.npy "$_lbak/data/" 2>/dev/null || true
  rm -rf LODGE && git clone -q https://github.com/li-ronghui/LODGE.git
  [ -d "$_lbak/exp" ] && mv "$_lbak/exp" LODGE/exp 2>/dev/null || true
  mkdir -p LODGE/data; [ -f "$_lbak/data/smplx_neu_J_1.npy" ] && mv "$_lbak/data/smplx_neu_J_1.npy" LODGE/data/ 2>/dev/null || true
  rm -rf "$_lbak"
fi
# EDGE: a partial clone (only .venv/SMPL-to-FBX) has no model code -> re-clone if EDGE.py is missing,
# preserving the caller-pushed ybot.fbx (lives under EDGE/SMPL-to-FBX, which rm -rf would delete).
if [ ! -f EDGE/EDGE.py ]; then
  _ebak=$(mktemp -d)
  [ -f EDGE/SMPL-to-FBX/ybot.fbx ] && cp -f EDGE/SMPL-to-FBX/ybot.fbx "$_ebak/" 2>/dev/null || true
  rm -rf EDGE && git clone -q https://github.com/Stanford-TML/EDGE.git
  mkdir -p EDGE/SMPL-to-FBX
  [ -f "$_ebak/ybot.fbx" ] && mv -f "$_ebak/ybot.fbx" EDGE/SMPL-to-FBX/ybot.fbx 2>/dev/null || true
  rm -rf "$_ebak"
fi
( cd AgentLODGE && git pull --ff-only -q || true )

# ---- 5. CUDA venv + PyTorch (Blackwell gate) ----------------------------------------------
step "CUDA venv + torch ($TORCH_INDEX)"
# The venv MUST live physically on /workspace to survive a pod restart. An earlier setup symlinked
# it to /root/al_venv (fast local disk, but /root is wiped on restart) -- replace such a symlink with
# a real venv so the whole gen+render stack persists.
if [ -L "$VENV" ]; then echo "  replacing $VENV symlink -> real venv (persist across restarts)"; rm -f "$VENV"; fi
[ -x "$VENV/bin/python" ] || { rm -rf "$VENV"; python3 -m venv "$VENV"; }
"$PIP" install -q -U pip wheel "setuptools<82"
# CRITICAL: uninstall any pre-existing torch first. A leftover '2.13.0+cpu' has a higher version
# string than every cu128 wheel, so 'pip install torch --index-url .../cu128' becomes a no-op.
torch_ready=0
if [ "$TORCH_INDEX" != "cpu" ] && "$PY" - <<'PY' >/dev/null 2>&1; then
import torch
assert torch.cuda.is_available()
value = torch.ones(1, device="cuda") + 1
torch.cuda.synchronize()
assert value.item() == 2
PY
  torch_ready=1
  echo "  existing CUDA torch is usable; keeping it"
fi
if [ "$torch_ready" -ne 1 ]; then
  "$PIP" uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  if [ "$TORCH_INDEX" = "cpu" ]; then
    "$PIP" install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  else
    "$PIP" install -q torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
  fi
fi

# ---- 6. python deps (pyrender is REQUIRED: LODGE render.py imports it at load) -------------
step "python deps"
"$PIP" install -q -r "$AL/requirements.txt"
"$PIP" install -q gdown omegaconf pytorch-lightning einops tqdm soundfile librosa \
  opencv-python-headless pyrender PyOpenGL trimesh smplx p_tqdm h5py imageio imageio-ffmpeg psutil \
  torchmetrics accelerate wandb fire
# pytorch3d (transforms only -> CPU build is fine; no nvcc). Build isolation OFF so it sees torch.
if ! "$PIP" show pytorch3d >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 MAX_JOBS="$(nproc)" \
    "$PIP" install -q --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    || die "pytorch3d build"
fi
"$PY" - <<'PY' || die "Python dependency import gate"
import importlib

for module in (
    "torch",
    "pytorch3d",
    "pyrender",
    "OpenGL",
    "smplx",
    "librosa",
    "omegaconf",
):
    importlib.import_module(module)
print("  Python dependency imports: OK")
PY

# ---- 7. LODGE + EDGE weights (via robust gdrive helper; gdown fails on the virus-scan page) -
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
mkdir -p "$AL/server/data"
if [ ! -f "$AL/server/data/smplx_neu_J_1.npy" ]; then
  cp "$WORK/LODGE/data/smplx_neu_J_1.npy" "$AL/server/data/smplx_neu_J_1.npy" \
    || die "FK template install"
fi

step "EDGE weights"
cd "$WORK/EDGE"
EDGE_CODE_PATH="$WORK/EDGE" "$PY" "$AL/scripts/patch_edge_pod.py" || true
[ -f checkpoint.pt ] || "$PY" "$AL/scripts/download_gdrive.py" 1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ checkpoint.pt || die "EDGE checkpoint download"

# ---- 8. EDGE venv (shares CUDA venv via .pth) + Jukebox -----------------------------------
step "EDGE venv (shares CUDA venv via .pth)"
PYVER="$("$PY" -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
if [ ! -x "$WORK/EDGE/.venv/bin/python" ]; then
  python3 -m venv "$WORK/EDGE/.venv"
  "$WORK/EDGE/.venv/bin/pip" install -q -U pip
fi
# Always (re)write the shared-venv path: after a restart the CUDA venv is rebuilt, and an old .pth may
# still point at the wiped /root path.
echo "$VENV/lib/$PYVER/site-packages" > "$WORK/EDGE/.venv/lib/$PYVER/site-packages/_shared_venv.pth"
if [ "${AGENTLODGE_SKIP_JUKEBOX:-0}" = "1" ]; then
  echo "  skipping Jukebox install (only NEW-song EDGE feature extraction needs it; cached slices don't)"
else
  "$WORK/EDGE/.venv/bin/python" -c "import jukemirlib" 2>/dev/null || {
    "$WORK/EDGE/.venv/bin/pip" install -q --no-build-isolation --no-deps "git+https://github.com/rodrigo-castellon/jukebox.git"
    "$WORK/EDGE/.venv/bin/pip" install -q --no-deps "git+https://github.com/rodrigo-castellon/jukemirlib.git"
    "$WORK/EDGE/.venv/bin/pip" install -q fire unidecode wget
  }
  "$WORK/EDGE/.venv/bin/python" -c "import jukemirlib" \
    || die "EDGE Jukebox import"
fi

# ---- 9. LODGE runs in the CUDA venv (GPU), not a CPU venv ----------------------------------
step "point LODGE at the CUDA venv (GPU)"
ln -sfn "$VENV" "$WORK/LODGE/.venv"

# ---- 10. Blackwell GPU gate ---------------------------------------------------------------
step "GPU gate (torch matmul on this card)"
"$PY" - <<'PY' || die "CUDA torch not usable on this GPU"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
c = (torch.randn(2048, 2048, device="cuda") @ torch.randn(2048, 2048, device="cuda"))
torch.cuda.synchronize()
print(f"  OK torch {torch.__version__} on {torch.cuda.get_device_name(0)} sm_{torch.cuda.get_device_capability(0)}")
PY

# ---- 11. exact Y-Bot render scene ---------------------------------------------------------
step "exact Y-Bot render scene"
[ -f "$YBOT" ] || die "missing $YBOT after cloning EDGE"
if [ ! -f "$YBOT_SCENE" ] \
  || [ "$AL/scripts/blender_render_ybot.py" -nt "$YBOT_SCENE" ] \
  || [ "$YBOT" -nt "$YBOT_SCENE" ]; then
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    "$BLENDER" -b -noaudio -P "$AL/scripts/blender_render_ybot.py" -- \
      --build-scene "$YBOT_SCENE" --ybot "$YBOT" \
      --width 448 --height 448 --samples 8 \
      >/tmp/maestro_ybot_scene.log 2>&1 \
    || { tail -n 80 /tmp/maestro_ybot_scene.log; die "Y-Bot scene build"; }
fi
echo "  ready: $YBOT_SCENE"

touch "$WORK/.maestro_gen_pod_ready"
echo ""
echo "GEN_POD_READY"
echo "Next: WORKSPACE=$WORK bash scripts/setup_gen_pod.sh --song <sid>   # then gen_take.py / live mode"
