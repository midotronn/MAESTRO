#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
AGENTLODGE_ROOT="${AGENTLODGE_ROOT:-$WORKSPACE/AgentLODGE}"
SOURCE="$AGENTLODGE_ROOT/scripts/egl_cuda_device_selector.c"
OUTPUT="${1:-$WORKSPACE/.agentlodge/lib/libagentlodge_egl_cuda_device.so}"

fail() {
  echo "EGL_SELECTOR_BUILD_FAILED: $*" >&2
  exit 1
}

[ -f "$SOURCE" ] || fail "missing source: $SOURCE"
command -v gcc >/dev/null 2>&1 || fail "gcc is required (install build-essential)"
command -v nm >/dev/null 2>&1 || fail "nm is required (install binutils)"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required (install coreutils)"

mkdir -p "$(dirname "$OUTPUT")"
TEMPORARY="$OUTPUT.$$.tmp"
METADATA="$OUTPUT.build.json"
TEMP_METADATA="$METADATA.$$.tmp"
trap 'rm -f "$TEMPORARY" "$TEMP_METADATA"' EXIT
SOURCE_SHA256="$(sha256sum "$SOURCE" | awk '{print $1}')"
BUILD_ID="sha256:$SOURCE_SHA256"

if ! gcc -shared -fPIC -O2 -Wall -Wextra -Werror -Wl,-z,defs \
  "-DAGENTLODGE_SELECTOR_BUILD_ID=\"$BUILD_ID\"" \
  -o "$TEMPORARY" "$SOURCE" -ldl; then
  fail "gcc could not link the selector against libdl"
fi
for symbol in \
  agentlodge_egl_selector_build_id \
  agentlodge_egl_selector_version \
  dlsym \
  eglGetDisplay \
  eglGetPlatformDisplayEXT \
  eglQueryDevicesEXT; do
  nm -D "$TEMPORARY" | grep -Eq "[[:space:]]${symbol}$" \
    || fail "built selector is missing exported symbol $symbol"
done
BINARY_SHA256="$(sha256sum "$TEMPORARY" | awk '{print $1}')"
printf '%s\n' \
  "{\"schema_version\":1,\"selector_version\":2,\"build_id\":\"$BUILD_ID\",\"source_sha256\":\"$SOURCE_SHA256\",\"binary_sha256\":\"$BINARY_SHA256\"}" \
  > "$TEMP_METADATA"
mv -f "$TEMPORARY" "$OUTPUT"
mv -f "$TEMP_METADATA" "$METADATA"
trap - EXIT
echo "EGL_SELECTOR_READY $OUTPUT $METADATA"
