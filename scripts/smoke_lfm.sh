#!/usr/bin/env bash
# First real-model validation, intentionally only two optimizer steps.
set -euo pipefail
config="${1:-configs/lfm25_230m_spark.yaml}"
output="${2:-runs/lfm-smoke-$(date +%Y%m%d-%H%M%S)}"
elm doctor --require-spark
python -m pytest -q -m integration
elm train --config "$config" --output "$output" --steps 2
elm evaluate --run "$output" --count 2
