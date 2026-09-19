#!/usr/bin/env bash
# First real-model validation, intentionally only two optimizer steps.
set -euo pipefail
config="${1:-configs/lfm25_230m_spark.yaml}"
output="${2:-runs/lfm-smoke-$(date +%Y%m%d-%H%M%S)}"
sdkb doctor --require-spark
python -m pytest -q -m integration
sdkb model-probe --config "$config" --output "${output}-model-probe.json"
sdkb train --config "$config" --output "$output" --steps 2
sdkb evaluate --run "$output" --count 2
