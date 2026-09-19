#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${SDKB_IMAGE:-sdkb-spark:0.4}"
cache="${SDKB_CACHE_DIR:-$HOME/.cache/sdkb}"
configured_runs="${SDKB_RUNS_DIR:-}"
if [[ -z "$configured_runs" && -f "$root/.sdkb/runs-dir" ]]; then
  IFS= read -r configured_runs < "$root/.sdkb/runs-dir" || true
  [[ -n "$configured_runs" ]] || { echo 'Empty .sdkb/runs-dir storage setting.' >&2; exit 1; }
fi
runs="${configured_runs:-$root/runs}"
archive="${SDKB_ARCHIVE_DIR:-}"
if [[ -z "$archive" && -f "$root/.sdkb/archive-dir" ]]; then
  IFS= read -r archive < "$root/.sdkb/archive-dir" || true
  [[ -n "$archive" ]] || { echo 'Empty .sdkb/archive-dir storage setting.' >&2; exit 1; }
fi
container="${SDKB_CONTAINER:-sdkb-training}"
command="${1:-shell}"
shift || true
case "$command" in
  build)
    [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]] || {
      echo 'Build natively on Spark/ARM64, not under x86 emulation.' >&2; exit 1;
    }
    base="${SDKB_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
    docker pull --platform linux/arm64 "$base"
    [[ "$(docker image inspect --format '{{.Architecture}}' "$base")" == arm64 ]] || {
      echo 'Base image is not ARM64.' >&2; exit 1;
    }
    pinned="$(docker image inspect --format '{{index .RepoDigests 0}}' "$base")"
    mkdir -p "$root/.sdkb"
    printf '%s\n' "$pinned" >"$root/.sdkb/base-image.txt"
    exec docker build --platform linux/arm64 -f "$root/docker/Dockerfile.spark" \
      --build-arg "BASE_IMAGE=$pinned" -t "$image" "$root"
    ;;
  shell|run|start)
    # An explicit storage location must already exist; a missing mount must not
    # quietly turn into a directory on the internal filesystem.
    if [[ -n "$configured_runs" ]]; then
      [[ -d "$runs" ]] || { echo 'Configured run storage is unavailable; check the mounted disk.' >&2; exit 1; }
    else
      mkdir -p "$runs"
    fi
    mkdir -p "$cache"
    if [[ "$command" == start || ( "${1:-}" == sdkb && ( "${2:-}" == launch || "${2:-}" == train || ( "${2:-}" == runs && "${3:-}" == start ) ) ) ]]; then
      previous=""
      for argument in "$@"; do
        if [[ "$previous" == --output || "$argument" == --output=* ]]; then
          output="${argument#--output=}"
          if [[ "$output" != /* ]]; then
            echo 'Use an absolute container output path, normally /runs/NAME for configured storage.' >&2
            exit 2
          fi
        fi
        previous="$argument"
      done
    fi
    flags=(--init --gpus all --shm-size=8g --stop-timeout=600 \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --user "$(id -u):$(id -g)" --env HOME=/tmp \
      --env HF_HOME=/cache/huggingface --env NVIDIA_IMEX_CHANNELS=0 \
      --mount "type=bind,src=$root,dst=/workspace/sdkb" \
      --mount "type=bind,src=$cache,dst=/cache" \
      --mount "type=bind,src=$runs,dst=/runs" --workdir /workspace/sdkb)
    if [[ -n "$archive" ]]; then
      [[ -d "$archive" ]] || { echo 'Archive directory must exist on the mounted disk.' >&2; exit 1; }
      flags+=(--mount "type=bind,src=$archive,dst=/archive")
      # Relocated checkpoint links use a host-absolute path. Expose that same
      # path so existing run directories remain readable inside/outside Docker.
      archive_absolute="$(cd "$archive" && pwd -P)"
      [[ "$archive_absolute" == /archive ]] || flags+=(--mount "type=bind,src=$archive_absolute,dst=$archive_absolute")
    fi
    [[ -z "${HF_TOKEN:-}" ]] || flags+=(--env HF_TOKEN)
    [[ -z "${WANDB_API_KEY:-}" ]] || flags+=(--env WANDB_API_KEY)
    [[ -z "${WANDB_BASE_URL:-}" ]] || flags+=(--env WANDB_BASE_URL)
    if [[ "$command" == start ]]; then
      [[ $# -gt 0 ]] || { echo 'Supply sdkb launch arguments after start.' >&2; exit 2; }
      exec docker run --detach --name "$container" "${flags[@]}" "$image" sdkb launch "$@"
    fi
    flags+=(--rm)
    if [[ "$command" == shell ]]; then
      exec docker run -it "${flags[@]}" "$image" bash
    fi
    [[ $# -gt 0 ]] || { echo 'Supply a command after run.' >&2; exit 1; }
    exec docker run "${flags[@]}" "$image" "$@"
    ;;
  status) exec docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}}' "$container" ;;
  logs) exec docker logs --follow "$container" ;;
  stop) exec docker stop --time "${SDKB_STOP_TIMEOUT:-600}" "$container" ;;
  *) echo 'Usage: scripts/spark.sh build|shell|run COMMAND...|start LAUNCH_ARGS...|status|logs|stop' >&2; exit 2 ;;
esac
