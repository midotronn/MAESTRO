#!/usr/bin/env bash
# Critical-path pipeline for an uploaded song: audio -> initial dance -> optional preview render.
# Stages results into $WORKSPACE/upload_<sid>/ for the server to pull:
#     base_motion.npy · beats.npy · beat_strengths.npy · seed-0 bank · optional preview.mp4
#
# Usage (on the pod):  WORKSPACE=/workspace AGENTLODGE_BANK_K=4 bash scripts/process_song.sh <sid>
# Prereqs: a provisioned pod (scripts/setup_pod.sh) with the demo pipeline scripts present on
# $WORKSPACE (preprocess_song.py / make_song_bestofk.py / make_energetic_recap_aligned.py). The
# song wav must already be at $WORKSPACE/LODGE/data/finedance/music_wav/<sid>.wav.
set -uo pipefail
SID="$1"
WORKSPACE="${WORKSPACE:-/workspace}"
K="${AGENTLODGE_BANK_K:-4}"
AL="${AGENTLODGE_ROOT:-$WORKSPACE/AgentLODGE}"
PY="${AL_PY:-$AL/.venv/bin/python}"
SKIP_RENDER="${AGENTLODGE_SKIP_RENDER:-0}"
cd "$WORKSPACE"
export PYTHONUNBUFFERED=1 WORKSPACE AGENTLODGE_ROOT="$AL"
fail() { echo "PROCESS_${SID}_FAILED: $1" >&2; exit 1; }
progress() { printf 'MAESTRO_PROGRESS %s %s %s\n' "$1" "$2" "$3"; }
[ -x "$PY" ] || fail "Python environment not found: $PY"

find_script() {  # look in $WORKSPACE then the repo scripts dir
  for p in "$WORKSPACE/$1" "$AL/scripts/$1"; do
    [ -f "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

# Jukebox's 10GB prior must live on /workspace; /root is wiped on every pod restart.
progress assets 22 "Checking generation model assets"
PRIOR_SIZE=10288727721
PRIOR_STORE="${AGENTLODGE_JUKEBOX_PRIOR:-$WORKSPACE/.cache/jukemirlib/prior_level_2.pth.tar}"
PRIOR_LINK="$HOME/.cache/jukemirlib/prior_level_2.pth.tar"
mkdir -p "$(dirname "$PRIOR_STORE")" "$(dirname "$PRIOR_LINK")"
if [ "$(stat -c %s "$PRIOR_STORE" 2>/dev/null || echo 0)" -ne "$PRIOR_SIZE" ]; then
  progress assets 23 "Downloading the persistent Jukebox model"
  echo "### fetching persistent Jukebox prior (~10GB, first time only)"
  PRIOR_URL=https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --continue=true --max-connection-per-server=16 --split=16 \
      --min-split-size=16M --file-allocation=none --auto-file-renaming=false \
      --dir="$(dirname "$PRIOR_STORE")" --out="$(basename "$PRIOR_STORE")" "$PRIOR_URL" \
      || fail "Jukebox prior download failed"
  else
    curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors \
      -o "$PRIOR_STORE" "$PRIOR_URL" || fail "Jukebox prior download failed"
  fi
  [ "$(stat -c %s "$PRIOR_STORE" 2>/dev/null || echo 0)" -eq "$PRIOR_SIZE" ] \
    || fail "Jukebox prior download is incomplete"
fi
ln -sfn "$PRIOR_STORE" "$PRIOR_LINK"

PRE="$(find_script preprocess_song.py)" || fail "preprocess_song.py not found on the pod"
GEN="$(find_script make_song_bestofk.py)" || fail "make_song_bestofk.py not found on the pod"
RECAP="$(find_script make_energetic_recap_aligned.py || true)"
BANK="$(find_script build_window_bank.py)" || fail "build_window_bank.py not found"
REND="$(find_script render_one_ybot.sh)" || fail "render_one_ybot.sh not found"

progress preprocess 25 "Extracting LODGE and EDGE music features"
echo "### [1/6] preprocess (LODGE feats + EDGE jukebox slices)"
PREP_LOG_DIR="$WORKSPACE/gen${SID}_work"; mkdir -p "$PREP_LOG_DIR"
"$PY" "$PRE" "$SID" --lodge-only > "$PREP_LOG_DIR/preprocess_lodge.log" 2>&1 &
LODGE_PREP_PID=$!
"$PY" "$PRE" "$SID" --edge-only > "$PREP_LOG_DIR/preprocess_edge.log" 2>&1 &
EDGE_PREP_PID=$!
LODGE_PREP_RC=0; wait "$LODGE_PREP_PID" || LODGE_PREP_RC=$?
EDGE_PREP_RC=0; wait "$EDGE_PREP_PID" || EDGE_PREP_RC=$?
cat "$PREP_LOG_DIR/preprocess_lodge.log" "$PREP_LOG_DIR/preprocess_edge.log"
if [ "$LODGE_PREP_RC" -ne 0 ]; then
  echo "### parallel LODGE preprocessing failed; retrying alone"
  "$PY" "$PRE" "$SID" --lodge-only || fail "LODGE preprocess failed"
fi
if [ "$EDGE_PREP_RC" -ne 0 ]; then
  echo "### parallel EDGE preprocessing failed; retrying alone"
  "$PY" "$PRE" "$SID" --edge-only || fail "EDGE preprocess failed"
fi
progress generation 40 "Generating LODGE and EDGE motion"
echo "### [2/6] best-of-K generation + storyboard"
"$PY" "$GEN" "$SID" || fail "generation failed"
progress polish 68 "Assembling and polishing the choreography"
if [ -n "${RECAP:-}" ]; then echo "### [2b] recap alignment"; "$PY" "$RECAP" "$SID" || true; fi
echo "### [2c] resolve hand-through-body self-penetration"
PEN="$(find_script resolve_penetration.py || true)"
if [ -n "${PEN:-}" ]; then
  "$PY" "$PEN" "fd_${SID}_STORY_bestofk.npy" "fd_${SID}_STORY_bestofk.npy" \
    --radius 0.12 --margin 0.03 --max-deg 30 || echo "  (penetration cleanup skipped)"
fi
progress seed_bank 73 "Building the initial editing bank"
echo "### [3/6] seed-0 editing bank"
AGENTLODGE_BANK_K=1 "$PY" "$BANK" "$SID" || fail "seed-0 bank build failed"

progress beats 76 "Tracking beats and musical accents"
echo "### [4/6] beats"
"$PY" - "$SID" <<'PY' || fail "beat tracking failed"
import sys, numpy as np, librosa
sid = sys.argv[1]
y, sr = librosa.load(f"/workspace/LODGE/data/finedance/music_wav/{sid}.wav", sr=22050, mono=True)
_, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, units="frames")
beat_frames = np.asarray(beat_frames, dtype=np.int64).reshape(-1)
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512) * 30.0
onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
strengths = np.asarray([
    np.max(onset[max(0, frame - 1):min(len(onset), frame + 2)])
    if len(onset) else 0.0
    for frame in beat_frames
], dtype=np.float32)
strengths = np.nan_to_num(strengths, nan=0.0, posinf=0.0, neginf=0.0)
if strengths.size and float(strengths.max()) > 0.0:
    strengths /= float(strengths.max())
np.save(f"/workspace/beats_{sid}.npy", np.asarray(beat_times, dtype=np.float32))
np.save(f"/workspace/beat_strengths_{sid}.npy", strengths)
print("beats", len(beat_frames), "strongest", float(strengths.max()) if strengths.size else 0.0)
PY

progress staging 79 "Staging generated dance assets"
echo "### [5/6] stage initial outputs"
OUT="$WORKSPACE/upload_${SID}"; mkdir -p "$OUT"
cp "fd_${SID}_STORY_bestofk.npy" "$OUT/base_motion.npy"
cp "beats_${SID}.npy"            "$OUT/beats.npy"
cp "beat_strengths_${SID}.npy"   "$OUT/beat_strengths.npy"
cp bank_${SID}_*_seed0.npy       "$OUT/" 2>/dev/null || true

progress preview 80 "Preparing the initial preview"
if [ "$SKIP_RENDER" = "1" ]; then
  echo "### [6/6] preview render delegated to the hosted warm renderer"
else
  echo "### [6/6] render gray Y-Bot preview"
  bash "$REND" "fd_${SID}_STORY_bestofk.npy" "v_${SID}_preview.mp4" "$SID" || fail "render failed"
  cp "v_${SID}_preview.mp4" "$OUT/preview.mp4"
fi

progress preview 82 "Generation pipeline complete"
echo "BANK_DEFERRED $K"
echo "PROCESS_${SID}_DONE"
