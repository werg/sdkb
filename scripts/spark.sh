#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${SDKB_IMAGE:-sdkb-spark:0.4}"
cache="${SDKB_CACHE_DIR:-$HOME/.cache/sdkb}"
runs="${SDKB_RUNS_DIR:-$root/runs}"
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
    mkdir -p "$cache" "$runs"
    flags=(--init --gpus all --shm-size=8g --stop-timeout=600 \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --user "$(id -u):$(id -g)" --env HOME=/tmp \
      --env HF_HOME=/cache/huggingface --env NVIDIA_IMEX_CHANNELS=0 \
      --mount "type=bind,src=$root,dst=/workspace/sdkb" \
      --mount "type=bind,src=$cache,dst=/cache" \
      --mount "type=bind,src=$runs,dst=/runs" --workdir /workspace/sdkb)
    if [[ -n "${SDKB_ARCHIVE_DIR:-}" ]]; then
      [[ -d "$SDKB_ARCHIVE_DIR" ]] || { echo 'Archive directory must exist on the mounted disk.' >&2; exit 1; }
      flags+=(--mount "type=bind,src=$SDKB_ARCHIVE_DIR,dst=/archive")
      # Relocated checkpoint links use a host-absolute path. Expose that same
      # path so existing run directories remain readable inside/outside Docker.
      archive_absolute="$(cd "$SDKB_ARCHIVE_DIR" && pwd -P)"
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
