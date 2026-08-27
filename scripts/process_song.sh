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
OUT="$WORKSPACE/upload_${SID}"
PENETRATION_MARKER="$WORKSPACE/penetration_cleanup_${SID}.done"
AUDIO_TIMING="$WORKSPACE/audio_timing_${SID}.json"
TIMING_FILE="${MAESTRO_TIMING_FILE:-$OUT/timings.tsv}"
cd "$WORKSPACE"
export PYTHONUNBUFFERED=1 WORKSPACE AGENTLODGE_ROOT="$AL"
mkdir -p "$OUT" "$(dirname "$TIMING_FILE")"
: > "$TIMING_FILE"
now_ms() { date +%s%3N; }
timing() {
  local line="MAESTRO_TIMING $1 $2 $(now_ms) ${3:-}"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$TIMING_FILE"
}
fail() {
  timing pipeline end "failed: $1"
  echo "PROCESS_${SID}_FAILED: $1" >&2
  exit 1
}
progress() { printf 'MAESTRO_PROGRESS %s %s %s\n' "$1" "$2" "$3"; }
[ -x "$PY" ] || fail "Python environment not found: $PY"
timing pipeline start "song=$SID"

find_script() {  # look in $WORKSPACE then the repo scripts dir
  for p in "$WORKSPACE/$1" "$AL/scripts/$1"; do
    [ -f "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

# Jukebox's 10GB prior must live on /workspace; /root is wiped on every pod restart.
timing assets start
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
timing assets end

PRE="$(find_script preprocess_song.py)" || fail "preprocess_song.py not found on the pod"
GEN="$(find_script make_song_bestofk.py)" || fail "make_song_bestofk.py not found on the pod"
if [ "${AGENTLODGE_DISTRIBUTED:-0}" = "1" ]; then
  case ",${AGENTLODGE_DISTRIBUTED_CAPABILITIES:-}," in
    *,dance.generate,*)
      DISPATCH_GEN="$(find_script dispatch_song_generation.py || true)"
      [ -n "$DISPATCH_GEN" ] && GEN="$DISPATCH_GEN"
      ;;
  esac
fi
DISPATCH_BEATS=""
DISPATCH_EARLY_LODGE=""
if [ "${AGENTLODGE_DISTRIBUTED:-0}" = "1" ]; then
  case ",${AGENTLODGE_DISTRIBUTED_CAPABILITIES:-}," in
    *,audio.beats,*)
      DISPATCH_BEATS="$(find_script dispatch_beat_tracking.py || true)"
      ;;
  esac
  case ",${AGENTLODGE_DISTRIBUTED_CAPABILITIES:-}," in
    *,lodge.generate,*)
      if [ "${AGENTLODGE_EARLY_LODGE_GENERATION:-0}" = "1" ] &&
         [ "${AGENTLODGE_BEST_OF_K:-1}" = "1" ]; then
        DISPATCH_EARLY_LODGE="$(find_script dispatch_backbone_generation.py || true)"
      fi
      ;;
  esac
fi
RECAP="$(find_script make_energetic_recap_aligned.py || true)"
BANK="$(find_script build_window_bank.py)" || fail "build_window_bank.py not found"
REND="$(find_script render_one_ybot.sh)" || fail "render_one_ybot.sh not found"

# A retry may reuse the same SID. Remove every derived fast-path artifact before
# launching concurrent preprocessing so no worker can consume stale results.
rm -f \
  "bank_${SID}_lodge_seed0.npy" \
  "bank_${SID}_edge_seed0.npy" \
  "beats_${SID}.npy" \
  "beat_strengths_${SID}.npy" \
  "lodge_early_${SID}.json" \
  "lodge_early_${SID}.pending" \
  "$AUDIO_TIMING" \
  "$OUT/bank_${SID}_lodge_seed0.npy" \
  "$OUT/bank_${SID}_edge_seed0.npy" \
  "$OUT/beats.npy" \
  "$OUT/beat_strengths.npy" \
  "$PENETRATION_MARKER"

progress preprocess 25 "Extracting music features and beat timing"
echo "### [1/6] preprocess (LODGE + EDGE + resident beat analysis)"
PREP_LOG_DIR="$WORKSPACE/gen${SID}_work"; mkdir -p "$PREP_LOG_DIR"
(
  timing preprocess_lodge start
  "$PY" "$PRE" "$SID" --lodge-only
  rc=$?
  timing preprocess_lodge end "rc=$rc"
  exit "$rc"
) > "$PREP_LOG_DIR/preprocess_lodge.log" 2>&1 &
LODGE_PREP_PID=$!
(
  timing preprocess_edge start
  "$PY" "$PRE" "$SID" --edge-only
  rc=$?
  timing preprocess_edge end "rc=$rc"
  exit "$rc"
) > "$PREP_LOG_DIR/preprocess_edge.log" 2>&1 &
EDGE_PREP_PID=$!
BEAT_PREP_PID=""
if [ -n "$DISPATCH_BEATS" ]; then
  (
    timing preprocess_beats start
    "$PY" "$DISPATCH_BEATS" "$SID"
    rc=$?
    timing preprocess_beats end "rc=$rc"
    exit "$rc"
  ) > "$PREP_LOG_DIR/preprocess_beats.log" 2>&1 &
  BEAT_PREP_PID=$!
fi
LODGE_PREP_RC=0; wait "$LODGE_PREP_PID" || LODGE_PREP_RC=$?
EARLY_LODGE_PID=""
EARLY_LODGE_LOG="$PREP_LOG_DIR/generation_lodge_early.log"
start_early_lodge() {
  if [ -z "$DISPATCH_EARLY_LODGE" ]; then
    return
  fi
  (
    timing generation_lodge_early start
    "$PY" "$DISPATCH_EARLY_LODGE" "$SID" lodge
    rc=$?
    timing generation_lodge_early end "rc=$rc"
    exit "$rc"
  ) > "$EARLY_LODGE_LOG" 2>&1 &
  EARLY_LODGE_PID=$!
}
if [ "$LODGE_PREP_RC" -eq 0 ]; then
  start_early_lodge
fi
EDGE_PREP_RC=0; wait "$EDGE_PREP_PID" || EDGE_PREP_RC=$?
BEAT_PREP_RC=0
if [ -n "$BEAT_PREP_PID" ]; then
  wait "$BEAT_PREP_PID" || BEAT_PREP_RC=$?
fi
cat "$PREP_LOG_DIR/preprocess_lodge.log" "$PREP_LOG_DIR/preprocess_edge.log"
if [ -f "$PREP_LOG_DIR/preprocess_beats.log" ]; then
  cat "$PREP_LOG_DIR/preprocess_beats.log"
fi
if [ "$LODGE_PREP_RC" -ne 0 ]; then
  echo "### parallel LODGE preprocessing failed; retrying alone"
  timing preprocess_lodge_retry start
  "$PY" "$PRE" "$SID" --lodge-only
  retry_rc=$?
  timing preprocess_lodge_retry end "rc=$retry_rc"
  [ "$retry_rc" -eq 0 ] || fail "LODGE preprocess failed"
  start_early_lodge
fi
if [ "$EDGE_PREP_RC" -ne 0 ]; then
  echo "### parallel EDGE preprocessing failed; retrying alone"
  timing preprocess_edge_retry start
  "$PY" "$PRE" "$SID" --edge-only
  retry_rc=$?
  timing preprocess_edge_retry end "rc=$retry_rc"
  [ "$retry_rc" -eq 0 ] || fail "EDGE preprocess failed"
fi
if [ "$BEAT_PREP_RC" -ne 0 ]; then
  echo "### resident beat analysis failed; generation will use the exact local fallback"
  rm -f "beats_${SID}.npy" "beat_strengths_${SID}.npy" "$AUDIO_TIMING"
fi
if [ -n "$EARLY_LODGE_PID" ] && ! kill -0 "$EARLY_LODGE_PID" 2>/dev/null; then
  EARLY_LODGE_RC=0
  wait "$EARLY_LODGE_PID" || EARLY_LODGE_RC=$?
  cat "$EARLY_LODGE_LOG"
  EARLY_LODGE_PID=""
  if [ "$EARLY_LODGE_RC" -ne 0 ]; then
    echo "### early LODGE generation failed; using normal generation"
    rm -f "lodge_early_${SID}.json" "lodge_early_${SID}.pending"
  fi
fi
progress generation 40 "Generating LODGE and EDGE motion"
echo "### [2/6] best-of-K generation + storyboard"
timing generation_total start
"$PY" "$GEN" "$SID"
GEN_RC=$?
timing generation_total end "rc=$GEN_RC"
if [ -n "$EARLY_LODGE_PID" ]; then
  EARLY_LODGE_RC=0
  wait "$EARLY_LODGE_PID" || EARLY_LODGE_RC=$?
  cat "$EARLY_LODGE_LOG"
  if [ "$EARLY_LODGE_RC" -ne 0 ]; then
    echo "### early LODGE generation failed; normal generation completed"
    rm -f "lodge_early_${SID}.json" "lodge_early_${SID}.pending"
  fi
fi
[ "$GEN_RC" -eq 0 ] || fail "generation failed"
progress polish 68 "Assembling and polishing the choreography"
timing polish start
if [ -n "${RECAP:-}" ]; then echo "### [2b] recap alignment"; "$PY" "$RECAP" "$SID" || true; fi
echo "### [2c] resolve hand-through-body self-penetration"
if [ -s "$PENETRATION_MARKER" ]; then
  echo "  resident penetration cleanup already complete; skipping standalone cleanup"
else
  rm -f "$PENETRATION_MARKER"
  PEN="$(find_script resolve_penetration.py)" || fail "resolve_penetration.py not found"
  "$PY" "$PEN" "fd_${SID}_STORY_bestofk.npy" "fd_${SID}_STORY_bestofk.npy" \
    --radius 0.12 --margin 0.03 --max-deg 30 || fail "penetration cleanup failed"
fi
timing polish end
progress seed_bank 73 "Building the initial editing bank"
echo "### [3/6] seed-0 editing bank"
timing seed_bank start
if [ -f "bank_${SID}_lodge_seed0.npy" ] && [ -f "bank_${SID}_edge_seed0.npy" ]; then
  echo "  seed-0 bank already exists; skipping standalone build"
  BANK_RC=0
else
  AGENTLODGE_BANK_K=1 "$PY" "$BANK" "$SID"
  BANK_RC=$?
fi
timing seed_bank end "rc=$BANK_RC"
[ "$BANK_RC" -eq 0 ] || fail "seed-0 bank build failed"
if [ "${MAESTRO_FRONT_FACING:-0}" = "1" ]; then
  FRONT="$(find_script normalize_front_facing.py)" || fail "normalize_front_facing.py not found"
  "$PY" "$FRONT" "fd_${SID}_STORY_bestofk.npy" bank_${SID}_*_seed0.npy \
    || fail "front-facing normalization failed"
fi

progress beats 76 "Tracking beats and musical accents"
echo "### [4/6] beats"
timing beats start
if [ -f "beats_${SID}.npy" ] && [ -f "beat_strengths_${SID}.npy" ]; then
  echo "  beat artifacts already exist; skipping standalone tracking"
  BEAT_RC=0
else
  "$PY" - "$SID" <<'PY'
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
  BEAT_RC=$?
fi
timing beats end "rc=$BEAT_RC"
[ "$BEAT_RC" -eq 0 ] || fail "beat tracking failed"

progress staging 79 "Staging generated dance assets"
echo "### [5/6] stage initial outputs"
timing staging start
mkdir -p "$OUT"
cp "fd_${SID}_STORY_bestofk.npy" "$OUT/base_motion.npy"
cp "beats_${SID}.npy"            "$OUT/beats.npy"
cp "beat_strengths_${SID}.npy"   "$OUT/beat_strengths.npy"
cp bank_${SID}_*_seed0.npy       "$OUT/" 2>/dev/null || true
timing staging end

progress preview 80 "Preparing the initial preview"
timing preview start
if [ "$SKIP_RENDER" = "1" ]; then
  echo "### [6/6] preview render delegated to the hosted warm renderer"
else
  echo "### [6/6] render gray Y-Bot preview"
  bash "$REND" "fd_${SID}_STORY_bestofk.npy" "v_${SID}_preview.mp4" "$SID" || fail "render failed"
  cp "v_${SID}_preview.mp4" "$OUT/preview.mp4"
fi
timing preview end "delegated=$SKIP_RENDER"

progress preview 82 "Generation pipeline complete"
echo "BANK_DEFERRED $K"
timing pipeline end "success"
echo "PROCESS_${SID}_DONE"
