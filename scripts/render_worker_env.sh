#!/usr/bin/env bash

agentlodge_validate_selector_shim() {
  local shim="$1"
  [ -s "$shim" ] || {
    echo "render.frames multi-GPU selector shim is missing: $shim" >&2
    return 1
  }
  command -v nm >/dev/null 2>&1 || {
    echo "nm is required to validate the EGL selector shim" >&2
    return 1
  }
  command -v ldd >/dev/null 2>&1 || {
    echo "ldd is required to validate the EGL selector shim" >&2
    return 1
  }
  local dependencies
  dependencies="$(ldd "$shim" 2>&1)" || {
    echo "EGL selector shim is not loadable: $shim" >&2
    printf '%s\n' "$dependencies" >&2
    return 1
  }
  printf '%s\n' "$dependencies" | grep -q "not found" && {
    echo "EGL selector shim has missing dependencies: $shim" >&2
    printf '%s\n' "$dependencies" >&2
    return 1
  }
  for symbol in \
    agentlodge_egl_selector_build_id \
    agentlodge_egl_selector_version \
    dlsym \
    eglGetDisplay \
    eglGetPlatformDisplayEXT \
    eglQueryDevicesEXT; do
    nm -D "$shim" 2>/dev/null | grep -Eq "[[:space:]]${symbol}$" || {
      echo "invalid EGL selector shim (missing $symbol): $shim" >&2
      return 1
    }
  done
}

agentlodge_configure_gpu() {
  local capability="$1"
  local workspace="$2"
  local gpu_lines gpu_count requested only_index selector

  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is required" >&2
    return 1
  }
  gpu_lines="$(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null |
      sed 's/[[:space:]]//g' |
      sed '/^$/d'
  )"
  gpu_count="$(printf '%s\n' "$gpu_lines" | sed '/^$/d' | wc -l | tr -d ' ')"
  [ "$gpu_count" -gt 0 ] || {
    echo "nvidia-smi reported no visible GPUs" >&2
    return 1
  }

  requested="${AGENTLODGE_GPU_INDEX:-}"
  if [ "$gpu_count" -eq 1 ]; then
    only_index="$(printf '%s\n' "$gpu_lines" | head -1)"
    if [ -n "$requested" ] && [ "$requested" != "$only_index" ]; then
      echo "single-GPU container exposes index $only_index, not $requested" >&2
      return 1
    fi
    export AGENTLODGE_RESOLVED_GPU_INDEX="${requested:-$only_index}"
    unset AGENTLODGE_RENDER_MULTI_GPU
  else
    [ -n "$requested" ] || {
      echo "multi-GPU container requires AGENTLODGE_GPU_INDEX; detected $gpu_count GPUs" >&2
      return 1
    }
    printf '%s\n' "$gpu_lines" | grep -qx "$requested" || {
      echo "AGENTLODGE_GPU_INDEX=$requested is not exposed by this container" >&2
      return 1
    }
    export AGENTLODGE_RESOLVED_GPU_INDEX="$requested"
    if [ "$capability" = "render.frames" ]; then
      selector="${AGENTLODGE_EGL_SELECTOR_SHIM:-$workspace/.agentlodge/lib/libagentlodge_egl_cuda_device.so}"
      agentlodge_validate_selector_shim "$selector" || return 1
      export AGENTLODGE_EGL_SELECTOR_SHIM="$selector"
      export AGENTLODGE_RENDER_MULTI_GPU=1
    fi
  fi

  # CVD hides/reindexes the physical ordinals that EGL_CUDA_DEVICE_NV must match.
  if [ "$capability" = "render.frames" ] && [ "$gpu_count" -gt 1 ]; then
    unset CUDA_VISIBLE_DEVICES
    unset NVIDIA_VISIBLE_DEVICES
  elif [ -n "$requested" ]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$requested"
  fi
}

agentlodge_configure_render_paths() {
  local worker_id="$1"
  local safe_id base use_shm shm_root reservation_bytes max_percent headroom
  local reservation_file

  safe_id="$(printf '%s' "$worker_id" | tr -c 'A-Za-z0-9_.-' '_')"
  use_shm="${AGENTLODGE_RENDER_USE_SHM:-0}"
  shm_root="${AGENTLODGE_SHM_ROOT:-/dev/shm}"
  reservation_bytes="${AGENTLODGE_SHM_RESERVATION_BYTES:-2147483648}"
  max_percent="${AGENTLODGE_SHM_MAX_PERCENT:-50}"
  headroom="${AGENTLODGE_SHM_HEADROOM_BYTES:-4294967296}"
  case "$use_shm" in
    1|true|TRUE|yes|YES|on|ON) use_shm=1 ;;
    0|false|FALSE|no|NO|off|OFF|'') use_shm=0 ;;
    *)
      echo "AGENTLODGE_RENDER_USE_SHM must be a boolean" >&2
      return 1
      ;;
  esac
  case "$reservation_bytes" in
    ''|*[!0-9]*)
      echo "AGENTLODGE_SHM_RESERVATION_BYTES must be a positive integer" >&2
      return 1
      ;;
  esac
  [ "$reservation_bytes" -gt 0 ] || {
    echo "AGENTLODGE_SHM_RESERVATION_BYTES must be greater than zero" >&2
    return 1
  }
  case "$headroom" in
    ''|*[!0-9]*)
      echo "AGENTLODGE_SHM_HEADROOM_BYTES must be a non-negative integer" >&2
      return 1
      ;;
  esac
  case "$max_percent" in
    ''|*[!0-9]*)
      echo "AGENTLODGE_SHM_MAX_PERCENT must be an integer from 1 to 90" >&2
      return 1
      ;;
  esac
  [ "$max_percent" -ge 1 ] && [ "$max_percent" -le 90 ] || {
    echo "AGENTLODGE_SHM_MAX_PERCENT must be an integer from 1 to 90" >&2
    return 1
  }

  if [ -n "${AGENTLODGE_RENDER_LOCAL_ROOT:-}" ]; then
    base="$AGENTLODGE_RENDER_LOCAL_ROOT"
  elif [ -n "${AGENTLODGE_RENDER_LOCAL_BASE:-}" ]; then
    base="$AGENTLODGE_RENDER_LOCAL_BASE/agentlodge-render-$safe_id"
  elif [ "$use_shm" -eq 1 ]; then
    base="$shm_root/agentlodge-render-$safe_id"
  else
    base="/tmp/agentlodge-render-$safe_id"
  fi

  case "$base/" in
    "$shm_root/"*)
      [ "$use_shm" -eq 1 ] || {
        echo "paths under $shm_root require AGENTLODGE_RENDER_USE_SHM=1" >&2
        return 1
      }
      ;;
  esac
  if [ "$use_shm" -eq 1 ]; then
    [ -d "$shm_root" ] && [ -w "$shm_root" ] || {
      echo "shared-memory scratch is unavailable or not writable: $shm_root" >&2
      return 1
    }
    command -v flock >/dev/null 2>&1 || {
      echo "flock is required for shared-memory worker reservations" >&2
      return 1
    }
    reservation_file="$(
      (
        set -e
        reservation_root="$shm_root/.agentlodge-reservations"
        mkdir -p "$reservation_root"
        exec 9>"$reservation_root/.lock"
        flock -x 9
        current="$reservation_root/$safe_id.reservation"
        reserved=0
        for record in "$reservation_root"/*.reservation; do
          [ -f "$record" ] || continue
          read -r record_pid record_bytes _record_worker < "$record" || true
          case "${record_pid:-}:${record_bytes:-}" in
            *[!0-9:]*|:|*:)
              echo "invalid shared-memory reservation ledger entry: $record" >&2
              exit 1
              ;;
          esac
          if [ "$record" = "$current" ] && [ "$record_pid" = "$$" ]; then
            if [ "$record_bytes" != "$reservation_bytes" ]; then
              echo "worker $worker_id already has a different shm reservation" >&2
              exit 1
            fi
            printf '%s\n' "$current"
            exit 0
          fi
          if ! kill -0 "$record_pid" 2>/dev/null; then
            rm -f "$record"
            continue
          fi
          if [ "$record" = "$current" ]; then
            echo "worker $worker_id already has a live shm reservation" >&2
            exit 1
          fi
          reserved=$((reserved + record_bytes))
        done
        total_bytes="$(
          df -Pk "$shm_root" |
            awk 'NR==2 {printf "%.0f", $2 * 1024}'
        )"
        free_bytes="$(
          df -Pk "$shm_root" |
            awk 'NR==2 {printf "%.0f", $4 * 1024}'
        )"
        capacity=$((total_bytes * max_percent / 100))
        if [ $((reserved + reservation_bytes)) -gt "$capacity" ]; then
          echo "shared-memory reservations would exceed ${max_percent}% of $shm_root" >&2
          exit 1
        fi
        if [ "$free_bytes" -lt $((reserved + reservation_bytes + headroom)) ]; then
          echo "shared-memory scratch lacks aggregate reservations plus headroom" >&2
          exit 1
        fi
        temporary="$current.$$.tmp"
        printf '%s %s %s\n' "$$" "$reservation_bytes" "$safe_id" > "$temporary"
        mv -f "$temporary" "$current"
        printf '%s\n' "$current"
      )
    )" || return 1
    export AGENTLODGE_SHM_RESERVATION_FILE="$reservation_file"
  else
    unset AGENTLODGE_SHM_RESERVATION_FILE
  fi

  export AGENTLODGE_RENDER_LOCAL_ROOT="$base"
  export AGENTLODGE_RENDER_DAEMON_ROOT="${AGENTLODGE_RENDER_DAEMON_ROOT:-$base/daemon}"
  export AGENTLODGE_WORKER_TMP="${AGENTLODGE_WORKER_TMP:-$base/tmp}"
  export AGENTLODGE_HTTP_WORKER_SCRATCH="${AGENTLODGE_HTTP_WORKER_SCRATCH:-$base/http}"
  export AGENTLODGE_RENDER_FALLBACK_ROOT="${AGENTLODGE_RENDER_FALLBACK_ROOT:-/tmp}"
  mkdir -p \
    "$AGENTLODGE_RENDER_LOCAL_ROOT" \
    "$AGENTLODGE_RENDER_DAEMON_ROOT" \
    "$AGENTLODGE_WORKER_TMP"
}
