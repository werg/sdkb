#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${SDKB_IMAGE:-sdkb-spark:0.4}"
cache="${SDKB_CACHE_DIR:-$HOME/.cache/sdkb}"
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
  shell|run)
    mkdir -p "$cache"
    flags=(--rm --init --gpus all --shm-size=8g --stop-timeout=120 \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --user "$(id -u):$(id -g)" --env HOME=/tmp \
      --env HF_HOME=/cache/huggingface --env NVIDIA_IMEX_CHANNELS=0 \
      --mount "type=bind,src=$root,dst=/workspace/sdkb" \
      --mount "type=bind,src=$cache,dst=/cache" --workdir /workspace/sdkb)
    [[ -z "${HF_TOKEN:-}" ]] || flags+=(--env HF_TOKEN)
    if [[ "$command" == shell ]]; then
      exec docker run -it "${flags[@]}" "$image" bash
    fi
    [[ $# -gt 0 ]] || { echo 'Supply a command after run.' >&2; exit 1; }
    exec docker run "${flags[@]}" "$image" "$@"
    ;;
  *) echo 'Usage: scripts/spark.sh build|shell|run COMMAND...' >&2; exit 2 ;;
esac
