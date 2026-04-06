#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/Scripts/python.exe}"
export PYTHONPATH="${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m src.data_generation.generate_synthetic_data \
  --num-customers "${NUM_CUSTOMERS:-9000}" \
  --target-claims "${TARGET_CLAIMS:-60000}" \
  --seed "${PIPELINE_SEED:-42}"

"${PYTHON_BIN}" -m src.data_generation.generate_pdfs_and_images \
  --train-document-sample-size "${TRAIN_DOCUMENT_SAMPLE_SIZE:-20}" \
  --prod-document-sample-size "${PROD_DOCUMENT_SAMPLE_SIZE:-10}" \
  --seed "${PIPELINE_SEED:-42}"

EXTRACT_ARGS=(--split all)
if [[ "${RESET_DB:-0}" == "1" ]]; then
  EXTRACT_ARGS+=(--reset-db)
fi

"${PYTHON_BIN}" -m src.data_generation.extract_claim_packages "${EXTRACT_ARGS[@]}"
