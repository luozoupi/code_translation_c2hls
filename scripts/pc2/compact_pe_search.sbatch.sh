#!/usr/bin/env bash
#SBATCH --job-name=mmsearch
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/pc2/compact_pe_search/%x-%j.out
#SBATCH --error=artifacts/pc2/compact_pe_search/%x-%j.err

set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${C2HLS_PE_SEARCH_STAMP:?set C2HLS_PE_SEARCH_STAMP}"
OUT="${C2HLS_PE_SEARCH_OUT:?set C2HLS_PE_SEARCH_OUT}"
HEADER="${C2HLS_PE_SEARCH_HEADER:-${C2HLS_ROOT}/related_work/benchmarks/autosa_ready/autosa_mm/kernel.h}"
TESTBENCH="${C2HLS_PE_SEARCH_TESTBENCH:-${C2HLS_ROOT}/related_work/benchmarks/autosa_ready/autosa_mm/testbench.cpp}"

mkdir -p "${OUT}" artifacts/pc2/compact_pe_search c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

if [[ -n "${C2HLS_PYTHON:-}" && -x "${C2HLS_PYTHON}" ]]; then
  PY="${C2HLS_PYTHON}"
elif [[ -x "${C2HLS_ROOT}/.venv/bin/python" ]]; then
  PY="${C2HLS_ROOT}/.venv/bin/python"
else
  PY=python3
fi
export C2HLS_PYTHON="${PY}"

pc2_log "compact PE search stamp=${STAMP} out=${OUT} part=${C2HLS_PART} clock=${C2HLS_CLOCK_NS}"

"${PY}" "${C2HLS_ROOT}/compact_pe_search_main.py" \
  --stamp "${STAMP}" \
  --out "${OUT}" \
  --header "${HEADER}" \
  --testbench "${TESTBENCH}"

pc2_log "compact PE search done stamp=${STAMP} out=${OUT}"
