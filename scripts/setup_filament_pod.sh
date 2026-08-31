#!/usr/bin/env bash
# Prepare the locally retained Filament renderer on a fresh multi-GPU RunPod.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT="${AGENTLODGE_FILAMENT_ROOT:-$WORKSPACE/maestro-filament-poc}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILAMENT_TAG="${AGENTLODGE_FILAMENT_TAG:-v1.75.0}"
FILAMENT_RELEASE_SHA256="${AGENTLODGE_FILAMENT_RELEASE_SHA256:-c5d2e0f692e5fb98ed029a5a3a52c8174660d02844d5db5a804dc5264bbab6d1}"
FFMPEG_ROOT="${AGENTLODGE_FFMPEG_ROOT:-$WORKSPACE/ffmpeg-nvenc}"
FFMPEG_ARCHIVE_NAME="ffmpeg-master-latest-linux64-gpl.tar.xz"
FFMPEG_URL="${AGENTLODGE_FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$FFMPEG_ARCHIVE_NAME}"
FFMPEG_SHA256="${AGENTLODGE_FFMPEG_SHA256:-5eed1ef9625abbcfeaeb8f7af137b9d0212a5c554624d4780f2fc2e344b64a26}"
SELECTOR_DIR="$ROOT/vulkan-selector"
IBL="$ROOT/filament/bin/assets/ibl/lightroom_14b/lightroom_14b_ibl.ktx"
FILAMENT_HEADER="$ROOT/filament/include/filament/Engine.h"
FILAMENT_LIBRARY="$ROOT/filament/lib/x86_64/libfilament.a"
SMOKE_GLB="${AGENTLODGE_FILAMENT_SMOKE_GLB:-$ROOT/ybot_production_animated.glb}"
STATIC_GLB="${AGENTLODGE_FILAMENT_STATIC_GLB:-$ROOT/ybot_visible_static.glb}"
NVIDIA_VK_ICD="${AGENTLODGE_NVIDIA_VK_ICD:-/etc/vulkan/icd.d/nvidia_icd.json}"
BLENDER="${AGENTLODGE_BLENDER:-$WORKSPACE/blender/blender}"
YBOT_SCENE="${AGENTLODGE_YBOT_SCENE:-$WORKSPACE/ybot_scene.blend}"
YBOT="${AGENTLODGE_YBOT_FBX:-$WORKSPACE/EDGE/SMPL-to-FBX/ybot.fbx}"
PYTHON="${AGENTLODGE_PYTHON:-$WORKSPACE/AgentLODGE/.venv/bin/python}"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  ca-certificates clang curl ffmpeg gcc git libc++-dev libc++abi-dev \
  libvulkan1 vulkan-tools xz-utils

mkdir -p "$ROOT"
for source in filament_bench.cpp vulkan_device_selector.c; do
  test -f "$SCRIPT_DIR/$source" || {
    echo "missing retained Filament source: $SCRIPT_DIR/$source" >&2
    exit 1
  }
  if [ ! -f "$ROOT/$source" ] || ! cmp -s "$SCRIPT_DIR/$source" "$ROOT/$source"; then
    cp -f "$SCRIPT_DIR/$source" "$ROOT/$source"
  fi
done

nvenc_works() {
  local candidate="$1"
  [ -x "$candidate" ] \
    && "$candidate" -hide_banner -encoders 2>/dev/null |
      grep -F h264_nvenc >/dev/null \
    && "$candidate" -hide_banner -loglevel error \
      -f lavfi -i color=c=black:s=256x256:r=1 \
      -frames:v 1 -c:v h264_nvenc -f null - </dev/null
}

FFMPEG_BIN="${AGENTLODGE_FFMPEG_BIN:-/usr/bin/ffmpeg}"
if ! nvenc_works "$FFMPEG_BIN"; then
  archive="$WORKSPACE/.cache/agentlodge/$FFMPEG_ARCHIVE_NAME"
  mkdir -p "$(dirname "$archive")"
  if [ ! -f "$archive" ] \
    || ! echo "$FFMPEG_SHA256  $archive" | sha256sum -c - >/dev/null 2>&1; then
    rm -f "$archive"
    curl -fL --retry 10 --retry-delay 3 --retry-all-errors \
      -o "$archive" "$FFMPEG_URL"
  fi
  echo "$FFMPEG_SHA256  $archive" | sha256sum -c -
  if [ ! -x "$FFMPEG_ROOT/bin/ffmpeg" ] \
    || [ "$(cat "$FFMPEG_ROOT/.archive-sha256" 2>/dev/null || true)" != "$FFMPEG_SHA256" ]; then
    rm -rf "$FFMPEG_ROOT"
    mkdir -p "$FFMPEG_ROOT"
    tar --no-same-owner -xJf "$archive" -C "$FFMPEG_ROOT" --strip-components=1
    printf '%s\n' "$FFMPEG_SHA256" >"$FFMPEG_ROOT/.archive-sha256"
  fi
  FFMPEG_BIN="$FFMPEG_ROOT/bin/ffmpeg"
fi
nvenc_works "$FFMPEG_BIN" || {
  echo "no installed or checksum-pinned FFmpeg can encode h264_nvenc on this Pod" >&2
  exit 1
}
FFPROBE_BIN="$(dirname "$FFMPEG_BIN")/ffprobe"
[ -x "$FFPROBE_BIN" ] || {
  echo "selected FFmpeg has no matching ffprobe: $FFPROBE_BIN" >&2
  exit 1
}
FFMPEG_BIN="$(readlink -f "$FFMPEG_BIN")"
FFPROBE_BIN="$(readlink -f "$FFPROBE_BIN")"
ln -sfn "$FFMPEG_BIN" /usr/local/bin/ffmpeg
ln -sfn "$FFPROBE_BIN" /usr/local/bin/ffprobe
printf '%s\n' "$FFMPEG_BIN" >"$ROOT/ffmpeg.path"
hash -r

test -f "$SCRIPT_DIR/filament_glibc_compat.cpp" || {
  echo "missing Filament glibc compatibility source: $SCRIPT_DIR/filament_glibc_compat.cpp" >&2
  exit 1
}

if [ ! -f "$SMOKE_GLB" ]; then
  for required in "$BLENDER" "$YBOT_SCENE" "$YBOT" "$PYTHON"; do
    test -e "$required" || {
      echo "cannot generate the Filament smoke asset; missing $required" >&2
      exit 1
    }
  done
  smoke_assets="/tmp/maestro-filament-smoke-assets"
  rm -rf "$smoke_assets"
  mkdir -p "$smoke_assets/frames"
  "$PYTHON" - "$smoke_assets/poses.npz" <<'PY'
import sys

import numpy as np

from server import fk

frames = 300
motion = np.zeros((frames, 139), dtype=np.float32)
identity_6d = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
motion[:, 3:135] = np.tile(identity_6d, 22)
motion[:, 0] = np.linspace(0.0, 0.25, frames, dtype=np.float32)
motion[:, 2] = 0.02 * np.sin(
    np.linspace(0.0, 4.0 * np.pi, frames, dtype=np.float32)
)
motion[:, 135:139] = 1.0
fk.save_poses_npz(motion, sys.argv[1])
PY
  "$BLENDER" -b "$YBOT_SCENE" -noaudio \
    -P "$WORKSPACE/AgentLODGE/scripts/blender_render_ybot.py" -- \
    --poses "$smoke_assets/poses.npz" \
    --ybot "$YBOT" \
    --frames-dir "$smoke_assets/frames" \
    --width 1080 --height 1080 --samples 96 \
    --engine eevee --denoise 1 --frame-format tga \
    --frame-start 0 --frame-end 300 --batch-render --fast \
    --export-glb "$SMOKE_GLB" \
    >"$smoke_assets/blender_export.log" 2>&1 || {
      tail -n 120 "$smoke_assets/blender_export.log" >&2
      echo "failed to generate the Filament animated smoke GLB" >&2
      exit 1
    }
fi
test -s "$SMOKE_GLB" || {
  echo "animated Filament smoke-test GLB is missing: $SMOKE_GLB" >&2
  exit 1
}
if [ ! -f "$STATIC_GLB" ]; then
  cp -f "$SMOKE_GLB" "$STATIC_GLB"
fi
test -s "$STATIC_GLB" || {
  echo "static Filament validation GLB is missing: $STATIC_GLB" >&2
  exit 1
}

if [ ! -f "$IBL" ] || [ ! -f "$FILAMENT_HEADER" ] || [ ! -f "$FILAMENT_LIBRARY" ]; then
  archive="/tmp/filament-${FILAMENT_TAG#v}-linux.tgz"
  curl -fL --retry 10 --retry-delay 3 --retry-all-errors \
    -o "$archive" \
    "https://github.com/google/filament/releases/download/$FILAMENT_TAG/filament-$FILAMENT_TAG-linux.tgz"
  echo "$FILAMENT_RELEASE_SHA256  $archive" | sha256sum -c -
  rm -rf "$ROOT/filament"
  tar --no-same-owner -xzf "$archive" -C "$ROOT" filament
  rm -f "$archive"
fi
for required in "$IBL" "$FILAMENT_HEADER" "$FILAMENT_LIBRARY"; do
  test -f "$required" || {
    echo "Filament SDK asset is missing after extraction: $required" >&2
    exit 1
  }
done

binary="$ROOT/filament_bench"
needs_build=0
if [ ! -x "$binary" ]; then
  needs_build=1
else
  usage="$("$binary" 2>&1 || true)"
  grep -F "Usage: filament_bench" <<<"$usage" >/dev/null || needs_build=1
  strings "$binary" | grep -F "MAESTRO_FILAMENT_ASYNC_FRAMES" >/dev/null \
    || needs_build=1
  [ "$ROOT/filament_bench.cpp" -nt "$binary" ] && needs_build=1
  [ "$SCRIPT_DIR/filament_glibc_compat.cpp" -nt "$binary" ] && needs_build=1
fi
if [ "$needs_build" -eq 1 ]; then
  filament_libs=("$ROOT/filament/lib/x86_64/"*.a)
  [ -e "${filament_libs[0]}" ] || {
    echo "Filament SDK has no x86_64 static libraries" >&2
    exit 1
  }
  candidate="$ROOT/.filament_bench.$$.build"
  clang++ -std=c++20 -O3 -DNDEBUG -stdlib=libc++ \
    -I"$ROOT/filament/include" \
    "$ROOT/filament_bench.cpp" \
    "$SCRIPT_DIR/filament_glibc_compat.cpp" \
    -o "$candidate" \
    -Wl,--start-group "${filament_libs[@]}" -Wl,--end-group \
    -lc++ -lc++abi -ldl -pthread
  chmod 0755 "$candidate"
  mv -f "$candidate" "$binary"
fi
usage="$("$binary" 2>&1 || true)"
grep -F "Usage: filament_bench" <<<"$usage" >/dev/null || {
  echo "Filament binary is not runnable on this Pod: $usage" >&2
  exit 1
}

REAL_VULKAN="$(
  ldconfig -p |
    awk '/libvulkan[.]so[.]1 / && !found {print $NF; found=1}'
)"
[ -n "$REAL_VULKAN" ] && [ -f "$REAL_VULKAN" ] || {
  echo "unable to locate the real libvulkan.so.1" >&2
  exit 1
}
mkdir -p "$SELECTOR_DIR"
cp -Lf "$REAL_VULKAN" "$SELECTOR_DIR/libvulkan.real.so.1"
gcc -shared -fPIC -O2 "$ROOT/vulkan_device_selector.c" -ldl \
  -o "$SELECTOR_DIR/libvulkan.so.1"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
gpu_count="$(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d' |
    wc -l |
    tr -d ' '
)"
[ "$gpu_count" -gt 0 ] || {
  echo "nvidia-smi reported no GPUs" >&2
  exit 1
}
test -f "$NVIDIA_VK_ICD" || {
  echo "NVIDIA Vulkan ICD manifest is missing: $NVIDIA_VK_ICD" >&2
  exit 1
}
vulkan_gpu_count="$(
  VK_ICD_FILENAMES="$NVIDIA_VK_ICD" vulkaninfo --summary 2>/dev/null |
    grep -F "deviceName" |
    grep -F "NVIDIA" |
    wc -l |
    tr -d ' '
)"
[ "$vulkan_gpu_count" -eq "$gpu_count" ] || {
  echo "Vulkan enumerated $vulkan_gpu_count NVIDIA GPUs; expected $gpu_count" >&2
  exit 1
}
ffmpeg -hide_banner -encoders 2>/dev/null | grep -F h264_nvenc >/dev/null || {
  echo "ffmpeg does not expose h264_nvenc" >&2
  exit 1
}
ffmpeg -hide_banner -loglevel error \
  -f lavfi -i color=c=black:s=256x256:r=1 \
  -frames:v 1 -c:v h264_nvenc -f null - </dev/null || {
  echo "ffmpeg exposes h264_nvenc but cannot encode on this Pod" >&2
  exit 1
}

smoke_root="/tmp/maestro-filament-setup-smoke"
rm -rf "$smoke_root"
mkdir -p "$smoke_root"
smoke_video="$smoke_root/smoke.mp4"
smoke_log="$smoke_root/filament.log"
CUDA_VISIBLE_DEVICES=0 \
VK_ICD_FILENAMES="$NVIDIA_VK_ICD" \
LD_LIBRARY_PATH="$SELECTOR_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MAESTRO_VK_REAL_LIBRARY="$SELECTOR_DIR/libvulkan.real.so.1" \
MAESTRO_VK_DEVICE_INDEX=0 \
MAESTRO_FILAMENT_IBL="$IBL" \
MAESTRO_FILAMENT_JOB_ONLY=1 \
MAESTRO_FILAMENT_FRAME_OFFSET=0 \
MAESTRO_FILAMENT_ASYNC_FRAMES=1 \
MAESTRO_FILAMENT_ASYNC_VIDEO_PATH="$smoke_video" \
MAESTRO_FILAMENT_ASYNC_ENCODER=h264_nvenc \
MAESTRO_FILAMENT_WRITE_ASYNC_SAMPLES=0 \
  "$binary" "$STATIC_GLB" "$SMOKE_GLB" "$smoke_root" \
  >"$smoke_log" 2>&1
grep -F \
  "MAESTRO_VK_SELECTOR selected Vulkan physical device index 0 of $gpu_count" \
  "$smoke_log" >/dev/null || {
  tail -n 80 "$smoke_log" >&2
  echo "Filament selector did not attest physical GPU 0" >&2
  exit 1
}
[ "$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name \
  -of default=nw=1:nk=1 "$smoke_video")" = "h264" ]
[ "$(ffprobe -v error -select_streams v:0 -show_entries stream=width \
  -of default=nw=1:nk=1 "$smoke_video")" = "1080" ]
[ "$(ffprobe -v error -select_streams v:0 -show_entries stream=height \
  -of default=nw=1:nk=1 "$smoke_video")" = "1080" ]
[ "$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate \
  -of default=nw=1:nk=1 "$smoke_video")" = "30/1" ]
[ "$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of default=nw=1:nk=1 \
  "$smoke_video")" = "1" ]
rm -rf "$smoke_root"

touch "$ROOT/.ready" "$WORKSPACE/.maestro_filament_ready"
echo "FILAMENT_POD_READY root=$ROOT ibl=$IBL"
