#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

bash "${PROJECT_ROOT}/scripts/run_data_generation.sh"
bash "${PROJECT_ROOT}/scripts/run_training.sh"

if [[ "${RUN_PRODUCTION_SCORING:-0}" == "1" ]]; then
  bash "${PROJECT_ROOT}/scripts/run_production_scoring.sh"
fi

if [[ "${START_API:-0}" == "1" ]]; then
  bash "${PROJECT_ROOT}/scripts/run_api.sh"
fi
