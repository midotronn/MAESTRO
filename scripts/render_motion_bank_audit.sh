#!/usr/bin/env bash
# Render the blind real-host audit produced by build_motion_bank_audit.py.
set -euo pipefail

AUDIT_DIR="${1:?usage: render_motion_bank_audit.sh <audit-directory>}"
WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${AGENTLODGE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${AL_PY:-$APP/.venv/bin/python}"
BLENDER="${BLENDER_BIN:-$WORKSPACE/blender/blender}"
SCENE="${YBOT_SCENE:-$WORKSPACE/ybot_scene.blend}"
YBOT="${YBOT_FBX:-$WORKSPACE/EDGE/SMPL-to-FBX/ybot.fbx}"
SCRIPT="$APP/scripts/blender_render_ybot.py"
WIDTH="${AUDIT_WIDTH:-448}"
HEIGHT="${AUDIT_HEIGHT:-448}"
SAMPLES="${AUDIT_SAMPLES:-8}"
VIDEOS="$AUDIT_DIR/videos"
FIXED_CAMERA="${AUDIT_FIXED_CAMERA:-1}"
CAMERA_ARG=()
[ "$FIXED_CAMERA" = "1" ] && CAMERA_ARG=(--fixed-camera)

rm -f "$AUDIT_DIR/render_receipt.json" "$AUDIT_DIR/render_receipt.json.tmp"
rm -rf "$VIDEOS" "$AUDIT_DIR/phase_sheets"
find "$AUDIT_DIR" -maxdepth 1 -type d \
  \( -name '*_front_frames' -o -name '*_side_frames' \) -exec rm -rf -- {} +
find "$AUDIT_DIR" -maxdepth 1 -type f \
  \( -name '*_front.log' -o -name '*_side.log' \) -delete
mkdir -p "$VIDEOS"
"$PY" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); p["fixed_camera"]=sys.argv[2]=="1"; open(sys.argv[1],"w").write(json.dumps(p,indent=2))' \
  "$AUDIT_DIR/review.json" "$FIXED_CAMERA"
mapfile -t TAKES < <(
  "$PY" -c 'import json,sys; [print(x["take"]) for x in json.load(open(sys.argv[1]))["takes"]]' \
    "$AUDIT_DIR/review.json"
)
mapfile -t CONTROLS < <(
  "$PY" -c 'import json,sys; [print(x["control"]) for x in json.load(open(sys.argv[1])).get("controls", [])]' \
    "$AUDIT_DIR/review.json"
)

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
render_pair() {
  local take="$1"
  for view in front side; do
    frames="$AUDIT_DIR/${take}_${view}_frames"
    log="$AUDIT_DIR/${take}_${view}.log"
    mkdir -p "$frames"
    if ! "$BLENDER" -b -noaudio "$SCENE" -P "$SCRIPT" -- \
        --poses "$AUDIT_DIR/${take}_${view}.npz" \
        --ybot "$YBOT" \
        --frames-dir "$frames" \
        --width "$WIDTH" --height "$HEIGHT" --samples "$SAMPLES" \
        --engine eevee --denoise 1 --color 0.5,0.5,0.52 --fast "${CAMERA_ARG[@]}" \
        >"$log" 2>&1; then
      tail -n 80 "$log"
      exit 1
    fi
    ffmpeg -loglevel error -y -framerate 30 -i "$frames/frame_%05d.png" \
      -c:v libx264 -preset veryfast -pix_fmt yuv420p "$VIDEOS/${take}_${view}.mp4"
  done
  ffmpeg -loglevel error -y \
    -i "$VIDEOS/${take}_front.mp4" -i "$VIDEOS/${take}_side.mp4" \
    -filter_complex hstack=inputs=2 -c:v libx264 -preset veryfast \
    -pix_fmt yuv420p "$VIDEOS/${take}.mp4"
}

for control in "${CONTROLS[@]}"; do
  render_pair "$control"
  echo "AUDIT_CONTROL_RENDERED $control"
done
for take in "${TAKES[@]}"; do
  render_pair "$take"
  echo "AUDIT_RENDERED $take"
done

"$PY" "$APP/scripts/build_motion_audit_sheets.py" "$AUDIT_DIR"
"$PY" "$APP/scripts/record_motion_audit_render.py" "$AUDIT_DIR"
echo "AUDIT_PHASE_SHEETS_READY $AUDIT_DIR/phase_sheets"
echo "AUDIT_READY $AUDIT_DIR/review.html"
