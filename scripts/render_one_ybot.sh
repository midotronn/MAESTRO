#!/usr/bin/env bash
# Render ONE motion npy as the canonical gray Y-Bot (same look as the demos). Audio optional.
# Usage: ./render_one_ybot.sh <motion_npy> <out_mp4> <sid_for_audio>
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
A="${AGENTLODGE_ROOT:-$WORKSPACE/AgentLODGE}"
if [ -n "${AL_PY:-}" ]; then
  PY="$AL_PY"
elif [ -x "$A/.venv/bin/python" ]; then
  PY="$A/.venv/bin/python"
else
  PY="/root/al_venv/bin/python"
fi
[ -x "$PY" ] || { echo "Python environment not found: $PY" >&2; exit 1; }
EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
BB="$WORKSPACE/blender/blender"
BS="$A/scripts/blender_render_ybot.py"
YB="$WORKSPACE/EDGE/SMPL-to-FBX/ybot.fbx"
MOTION="$1"; OUT="$2"; SID="${3:-}"
# High-quality defaults (override via env). 1080^2 EEVEE-Next (raytraced GI/reflections + AgX):
# reliable + fast on the GPU. Set RENDER_ENGINE=cycles for path-traced photoreal (needs a Blender
# build with kernels for your GPU arch; Cycles OptiX stalls on Blackwell + Blender 4.2.3).
RW="${RENDER_W:-1080}"; RH="${RENDER_H:-1080}"; RS="${RENDER_SAMPLES:-96}"
RENGINE="${RENDER_ENGINE:-eevee}"; RDENOISE="${RENDER_DENOISE:-1}"
AUDIO_ARG=()
[ -n "$SID" ] && [ -f "$WORKSPACE/LODGE/data/finedance/music_wav/${SID}.wav" ] && \
  AUDIO_ARG=(--audio "$WORKSPACE/LODGE/data/finedance/music_wav/${SID}.wav")
CAMERA_ARG=()
[ "${RENDER_FIXED_CAMERA:-0}" = "1" ] && CAMERA_ARG=(--fixed-camera)
__EGL_VENDOR_LIBRARY_FILENAMES=$EGL "$PY" "$A/scripts/render_blender_dance.py" \
  --agentlodge-root "$A" --motion-npy "$MOTION" --output-mp4 "$OUT" \
  --lodge-code-path "$WORKSPACE/LODGE" --blender-bin "$BB" --blender-script "$BS" \
  --character ybot --ybot-fbx "$YB" --color 0.5,0.5,0.52 \
  "${AUDIO_ARG[@]}" --width "$RW" --height "$RH" --samples "$RS" \
  --engine "$RENGINE" --denoise "$RDENOISE" "${CAMERA_ARG[@]}"
echo "RENDER_ONE_DONE $OUT"
