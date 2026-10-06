#!/usr/bin/env bash
#SBATCH --job-name=w1retry
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/pc2/post_flash_dse/%x-%j.out
#SBATCH --error=artifacts/pc2/post_flash_dse/%x-%j.err

# Serial DSE then stream on wave-1 cells. One LLM at a time (login5 timeouts
# killed the 6-way campaign chain). Always run stream even if some DSE miss.
set -euo pipefail

_REPO_ROOT="${C2HLS_ROOT:-${SLURM_SUBMIT_DIR:?missing SLURM_SUBMIT_DIR}}"
_SCRIPT_DIR="${_REPO_ROOT}/scripts/pc2"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"
mkdir -p artifacts/pc2/post_flash_dse artifacts/pc2/post_flash_stream c2hls_tmp

export C2HLS_SITE=pc2
export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
export C2HLS_POST_FLASH_DSE=1
export C2HLS_POST_FLASH_STREAM=1
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_MODEL="${C2HLS_DSE_MODEL:-${C2HLS_MODEL:-deepseek-v4-flash}}"
export C2HLS_DSE_MAX_TOKENS="${C2HLS_DSE_MAX_TOKENS:-65536}"
export C2HLS_STREAM_MAX_TOKENS="${C2HLS_STREAM_MAX_TOKENS:-65536}"
export C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES:-3}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/setup_emu_env.sh"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

MATRIX_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:?set C2HLS_POST_FLASH_MATRIX_ROOT}"
BENCHES_FILE="${C2HLS_POST_FLASH_BENCHES_FILE:-${MATRIX_ROOT}/wave1_retry_benches.txt}"
if [[ -f "${BENCHES_FILE}" ]]; then
  BENCHES="$(tr '\n' ',' < "${BENCHES_FILE}" | sed 's/,$//')"
else
  BENCHES="${C2HLS_POST_FLASH_BENCHES:-autosa_mm_hcl,autosa_mm_hcl_intel,autosa_mm_intel,autosa_mm_int16,autosa_mm_catapult,autosa_mm_getting_started}"
fi
pc2_log "benches_file=${BENCHES_FILE} benches=${BENCHES}"

STREAM_ONLY="${C2HLS_WAVE1_STREAM_ONLY:-0}"
dse_rc=0
if [[ "${STREAM_ONLY}" == "1" ]]; then
  pc2_log "stream-only: skip DSE"
else
  pc2_log "seed phase-B selected for flash-failed cells"
  "${PY}" "${C2HLS_ROOT}/scripts/pc2/seed_wave1_phase_b_selected.py" \
    --matrix-root "${MATRIX_ROOT}" \
    --benches autosa_mm_hcl,autosa_mm_catapult || true

  pc2_log "wave1 DSE serial matrix=${MATRIX_ROOT} benches=${BENCHES} model=${C2HLS_MODEL:-} tokens=${C2HLS_DSE_MAX_TOKENS}"
  set +e
  "${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_dse.py" --pc2 \
    --matrix-root "${MATRIX_ROOT}" --benches "${BENCHES}" --force \
    ${C2HLS_MODEL:+--model "${C2HLS_MODEL}"}
  dse_rc=$?
  set -e
  pc2_log "wave1 DSE exit=${dse_rc}"
fi

pc2_log "wave1 stream serial (runs even if DSE partial)"
set +e
"${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_stream.py" --pc2 \
  --matrix-root "${MATRIX_ROOT}" --benches "${BENCHES}" --force \
  ${C2HLS_MODEL:+--model "${C2HLS_MODEL}"}
stream_rc=$?
set -e
pc2_log "wave1 stream exit=${stream_rc}"

if [[ "${dse_rc}" -ne 0 || "${stream_rc}" -ne 0 ]]; then
  exit 1
fi
exit 0
