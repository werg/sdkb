#!/usr/bin/env bash
# Usage: scripts/spark.sh build | shell | run elm doctor --require-spark
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ELM_IMAGE:-elm-spark:dev}"
cache="${ELM_CACHE_DIR:-$HOME/.cache/elm}"
command="${1:-shell}"
shift || true
case "$command" in
  build)
    [[ "$(uname -m)" == 'aarch64' || "$(uname -m)" == 'arm64' ]] || {
      echo 'Build natively on Spark/ARM64, not with CPU emulation.' >&2; exit 1;
    }
    exec docker build -f "$root/docker/Dockerfile.spark" \
      --build-arg "BASE_IMAGE=${ELM_BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.08-py3}" \
      -t "$image" "$root"
    ;;
  shell|run)
    mkdir -p "$cache"
    flags=(--rm --gpus all --shm-size=8g --user "$(id -u):$(id -g)" \
      --env HOME=/tmp --env HF_HOME=/cache/huggingface \
      --mount "type=bind,src=$root,dst=/workspace/elm" \
      --mount "type=bind,src=$cache,dst=/cache" --workdir /workspace/elm)
    if [[ "$command" == shell ]]; then
      exec docker run -it "${flags[@]}" "$image" bash
    fi
    [[ $# -gt 0 ]] || { echo 'Supply a command after run.' >&2; exit 1; }
    exec docker run "${flags[@]}" "$image" "$@"
    ;;
  *) echo 'Usage: scripts/spark.sh build|shell|run COMMAND...' >&2; exit 2 ;;
esac
