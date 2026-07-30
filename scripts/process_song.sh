#!/usr/bin/env bash
# Full pipeline for an UPLOADED song: audio -> AgentLODGE dance + candidate bank + preview render.
# Stages results into $WORKSPACE/upload_<sid>/ for the server to pull:
#     base_motion.npy · beats.npy · preview.mp4 · bank_<sid>_*.npy
#
# Usage (on the pod):  WORKSPACE=/workspace AGENTLODGE_BANK_K=4 bash scripts/process_song.sh <sid>
# Prereqs: a provisioned pod (scripts/setup_pod.sh) with the demo pipeline scripts present on
# $WORKSPACE (preprocess_song.py / make_song_bestofk.py / make_energetic_recap_aligned.py). The
# song wav must already be at $WORKSPACE/LODGE/data/finedance/music_wav/<sid>.wav.
set -uo pipefail
SID="$1"
WORKSPACE="${WORKSPACE:-/workspace}"
K="${AGENTLODGE_BANK_K:-4}"
PY="${AL_PY:-/root/al_venv/bin/python}"
cd "$WORKSPACE"
export PYTHONUNBUFFERED=1 WORKSPACE
fail() { echo "PROCESS_${SID}_FAILED: $1" >&2; exit 1; }

find_script() {  # look in $WORKSPACE then the repo scripts dir
  for p in "$WORKSPACE/$1" "$WORKSPACE/AgentLODGE/scripts/$1"; do
    [ -f "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

# Jukebox prior is needed to extract EDGE features for a NEW song (wiped on /root each restart).
PRIOR=/root/.cache/jukemirlib/prior_level_2.pth.tar
if [ ! -s "$PRIOR" ]; then
  echo "### fetching jukebox prior (~10GB, first time only)"
  mkdir -p "$(dirname "$PRIOR")"
  wget -c -q -O "$PRIOR" https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar || \
    fail "jukebox prior download failed"
fi

PRE="$(find_script preprocess_song.py)" || fail "preprocess_song.py not found on the pod"
GEN="$(find_script make_song_bestofk.py)" || fail "make_song_bestofk.py not found on the pod"
RECAP="$(find_script make_energetic_recap_aligned.py || true)"
BANK="$(find_script build_window_bank.py)" || fail "build_window_bank.py not found"
REND="$(find_script render_one_ybot.sh)" || fail "render_one_ybot.sh not found"

echo "### [1/6] preprocess (LODGE feats + EDGE jukebox slices)"
"$PY" "$PRE" "$SID" || fail "preprocess failed"
echo "### [2/6] best-of-K generation + storyboard"
"$PY" "$GEN" "$SID" || fail "generation failed"
if [ -n "${RECAP:-}" ]; then echo "### [2b] recap alignment"; "$PY" "$RECAP" "$SID" || true; fi
echo "### [2c] resolve hand-through-body self-penetration"
PEN="$(find_script resolve_penetration.py || true)"
if [ -n "${PEN:-}" ]; then
  "$PY" "$PEN" "fd_${SID}_STORY_bestofk.npy" "fd_${SID}_STORY_bestofk.npy" \
    --radius 0.12 --margin 0.03 --max-deg 30 || echo "  (penetration cleanup skipped)"
fi
echo "### [3/6] candidate bank (K=$K real seeded takes)"
AGENTLODGE_BANK_K="$K" "$PY" "$BANK" "$SID" || fail "bank build failed"

echo "### [4/6] beats"
"$PY" - "$SID" <<'PY' || fail "beat tracking failed"
import sys, numpy as np, librosa
sid = sys.argv[1]
y, sr = librosa.load(f"/workspace/LODGE/data/finedance/music_wav/{sid}.wav", sr=22050, mono=True)
_, bt = librosa.beat.beat_track(y=y, sr=sr, units="time")
np.save(f"/workspace/beats_{sid}.npy", (np.asarray(bt, dtype=np.float32) * 30.0))
print("beats", len(bt))
PY

echo "### [5/6] render gray Y-Bot preview"
bash "$REND" "fd_${SID}_STORY_bestofk.npy" "v_${SID}_preview.mp4" "$SID" || fail "render failed"

echo "### [6/6] stage outputs"
OUT="$WORKSPACE/upload_${SID}"; mkdir -p "$OUT"
cp "fd_${SID}_STORY_bestofk.npy" "$OUT/base_motion.npy"
cp "beats_${SID}.npy"            "$OUT/beats.npy"
cp "v_${SID}_preview.mp4"        "$OUT/preview.mp4"
cp bank_${SID}_*.npy             "$OUT/" 2>/dev/null || true
echo "PROCESS_${SID}_DONE"
