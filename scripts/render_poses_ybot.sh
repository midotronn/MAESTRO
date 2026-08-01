#!/usr/bin/env bash
# Render a PRECOMPUTED poses.npz (server-side numpy FK) as the gray Y-Bot -- no torch on the pod, so
# the ~12-24s FK import is skipped entirely. poses.npz holds poses (L,24,3) / trans (L,3) / fk_joints
# (L,22,3), exactly what scripts/render_blender_dance.py used to compute with torch.
# Usage: ./render_poses_ybot.sh <poses_npz> <out_mp4> <sid_for_audio>
set -uo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
A="$WORKSPACE/AgentLODGE"
EGL=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
BB="$WORKSPACE/blender/blender"
BS="$A/scripts/blender_render_ybot.py"
YB="$WORKSPACE/EDGE/SMPL-to-FBX/ybot.fbx"
POSES="$1"; OUT="$2"; SID="${3:-}"
RW="${RENDER_W:-448}"; RH="${RENDER_H:-448}"; RS="${RENDER_SAMPLES:-8}"
RENGINE="${RENDER_ENGINE:-eevee}"; RDENOISE="${RENDER_DENOISE:-1}"
SCENE="${SCENE_BLEND:-$WORKSPACE/ybot_scene.blend}"   # pre-built scene skips the FBX import when present
FRAMES="$(mktemp -d)"

BLEND_OPEN=()
[ -f "$SCENE" ] && BLEND_OPEN=("$SCENE")               # open the cached scene (rig preloaded) if built
__EGL_VENDOR_LIBRARY_FILENAMES=$EGL "$BB" -b -noaudio "${BLEND_OPEN[@]}" -P "$BS" -- \
  --poses "$POSES" --ybot "$YB" --frames-dir "$FRAMES" \
  --width "$RW" --height "$RH" --samples "$RS" --engine "$RENGINE" --denoise "$RDENOISE" --color 0.5,0.5,0.52

AUDIO_ARG=()
[ -n "$SID" ] && [ -f "$WORKSPACE/LODGE/data/finedance/music_wav/${SID}.wav" ] && \
  AUDIO_ARG=(-i "$WORKSPACE/LODGE/data/finedance/music_wav/${SID}.wav")
SILENT="$FRAMES/silent.mp4"
ffmpeg -loglevel error -y -framerate 30 -i "$FRAMES/frame_%05d.png" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p "$SILENT"
if [ ${#AUDIO_ARG[@]} -gt 0 ]; then
  ffmpeg -loglevel error -y -i "$SILENT" "${AUDIO_ARG[@]}" -shortest -c:v copy -c:a aac -b:a 192k "$OUT"
else
  cp "$SILENT" "$OUT"
fi
rm -rf "$FRAMES"
echo "RENDER_POSES_DONE $OUT"
