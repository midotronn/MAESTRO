#!/usr/bin/env bash
# Render the named-motion review reel as the canonical gray Y-Bot, with each action's
# name burned in. Run on the GPU pod after setup_gen_pod.sh has finished.
#
#   ./scripts/render_motion_bank_reel.sh [out_mp4]
set -euo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
A="$WORKSPACE/AgentLODGE"
PY="${AL_PY:-$A/.venv/bin/python}"
REEL="$A/assets/motion_bank/review_reel.npy"
LABELS="$A/assets/motion_bank/review_reel.labels.json"
OUT="${1:-$WORKSPACE/motion_bank_review.mp4}"
RAW="${OUT%.mp4}.raw.mp4"

# The reel is derived, not committed, so build it here rather than relying on a stale copy.
"$PY" "$A/scripts/build_motion_bank_reel.py" --out "$REEL"

RENDER_W="${RENDER_W:-1080}" RENDER_H="${RENDER_H:-1080}" \
  bash "$A/scripts/render_one_ybot.sh" "$REEL" "$RAW"

"$PY" "$A/scripts/label_reel_video.py" \
  --video "$RAW" --labels "$LABELS" --out "$OUT" --height "${RENDER_H:-1080}"

rm -f "$RAW"
echo "REEL_DONE $OUT"
