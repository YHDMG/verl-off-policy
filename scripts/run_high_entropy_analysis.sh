#!/usr/bin/env bash
set -euo pipefail

# Standalone high-entropy analysis for existing artifacts.
# Edit the variables below, or override them with environment variables.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Required inputs.
GENERATIONS_PATH="${GENERATIONS_PATH:-/path/to/generations.parquet}"
TOKEN_METRICS_PATH="${TOKEN_METRICS_PATH:-/path/to/token_metrics.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/high_entropy_analysis}"

# Optional settings.
RUN_NAME="${RUN_NAME:-}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-}"
ENTROPY_QUANTILE="${ENTROPY_QUANTILE:-0.8}"
POSITION_BINS="${POSITION_BINS:-10}"
MIN_TOKEN_FREQUENCY="${MIN_TOKEN_FREQUENCY:-2}"
TOP_K_TOKEN_TEXTS="${TOP_K_TOKEN_TEXTS:-50}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"

if [[ ! -f "${GENERATIONS_PATH}" ]]; then
  echo "[error] generations file not found: ${GENERATIONS_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TOKEN_METRICS_PATH}" ]]; then
  echo "[error] token_metrics file not found: ${TOKEN_METRICS_PATH}" >&2
  exit 1
fi

echo "========== High Entropy Analysis =========="
echo "REPO_ROOT           = ${REPO_ROOT}"
echo "PYTHON_BIN          = ${PYTHON_BIN}"
echo "GENERATIONS_PATH    = ${GENERATIONS_PATH}"
echo "TOKEN_METRICS_PATH  = ${TOKEN_METRICS_PATH}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "RUN_NAME            = ${RUN_NAME}"
echo "ENTROPY_THRESHOLD   = ${ENTROPY_THRESHOLD}"
echo "ENTROPY_QUANTILE    = ${ENTROPY_QUANTILE}"
echo "POSITION_BINS       = ${POSITION_BINS}"
echo "MIN_TOKEN_FREQUENCY = ${MIN_TOKEN_FREQUENCY}"
echo "TOP_K_TOKEN_TEXTS   = ${TOP_K_TOKEN_TEXTS}"
echo "SKIP_PLOTS          = ${SKIP_PLOTS}"
echo "==========================================="

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  -m math_entropy_analysis.high_entropy
  --generations "${GENERATIONS_PATH}"
  --token-metrics "${TOKEN_METRICS_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --entropy-quantile "${ENTROPY_QUANTILE}"
  --position-bins "${POSITION_BINS}"
  --min-token-frequency "${MIN_TOKEN_FREQUENCY}"
  --top-k-token-texts "${TOP_K_TOKEN_TEXTS}"
)

if [[ -n "${RUN_NAME}" ]]; then
  ARGS+=(--run-name "${RUN_NAME}")
fi

if [[ -n "${ENTROPY_THRESHOLD}" ]]; then
  ARGS+=(--entropy-threshold "${ENTROPY_THRESHOLD}")
fi

if [[ "${SKIP_PLOTS}" == "1" ]]; then
  ARGS+=(--skip-plots)
fi

echo "[1/1] Running standalone high-entropy analysis ..."
"${PYTHON_BIN}" "${ARGS[@]}"

echo
echo "Done. Outputs written under: ${OUTPUT_DIR}${RUN_NAME:+/${RUN_NAME}}"