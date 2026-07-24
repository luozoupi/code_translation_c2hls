#!/usr/bin/env bash
# Fir HLS baseline smoke wrapper — loads Apptainer Vitis container then runs Python.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${C2HLS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export C2HLS_ROOT="${ROOT}"
export C2HLS_SITE=fir

module purge 2>/dev/null || true
unset LIBRARY_PATH LD_LIBRARY_PATH

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
fir_setup_vitis_env

if ! command -v vitis-run >/dev/null 2>&1; then
  echo "ERROR: vitis-run not on PATH after fir_setup_vitis_env" >&2
  exit 1
fi

echo "vitis-run=$(command -v vitis-run)"
echo "XILINX_SIF=${XILINX_SIF:-<unset>} C2HLS_USE_CONTAINER=${C2HLS_USE_CONTAINER:-<unset>}"

exec "${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/run_hls_baseline_smoke.py" --fir "$@"
