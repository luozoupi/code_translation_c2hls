#!/usr/bin/env bash
# PE-array + fuse + ping-pong DATAFLOW ablation on existing multi-PE (dse) cells.
#
# Usage:
#   ./scripts/pc2/start_autosa_pe_overlap.sh --dry-run
#   ./scripts/pc2/start_autosa_pe_overlap.sh --submit --endpoint-url http://login5:18092/v1
#   ./scripts/pc2/start_autosa_pe_overlap.sh --submit --mm-only --mm-root artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

# Default MM_ROOT is the Aug 23 gap campaign. Frozen mmflow (do not overwrite
# selected.cpp/stream.cpp) is:
#   artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow
MM_ROOT="${C2HLS_MM_MATRIX_ROOT:-artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_aav_n_gf_20260823_mm_gap_flash_ds_v4f_aav_n_gf}"
WAVE1_ROOT="${C2HLS_POST_FLASH_MATRIX_ROOT:-artifacts/pc2/batch_parallel_autosa_wave1_aav_n_gf_20260825_wave1_aav_n_gf}"
DRY_RUN=0
MM_ONLY=0
ENDPOINT_URL_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mm-root) shift; MM_ROOT="$1"; shift ;;
    --wave1-root) shift; WAVE1_ROOT="$1"; shift ;;
    --mm-only) MM_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --submit) shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"
export C2HLS_MODEL="${C2HLS_OVERLAP_MODEL:-deepseek-v4-flash}"
export C2HLS_STREAM_MAX_TOKENS="${C2HLS_STREAM_MAX_TOKENS:-65536}"
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${ENDPOINT_URL_ARG}" ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "WARNING: --endpoint-url not given; dry-run only" >&2
  else
    echo "ERROR: --endpoint-url is required for a real start (e.g. http://login5:18092/v1)." >&2
    exit 2
  fi
else
  export OPENAI_BASE_URL="${ENDPOINT_URL_ARG}"
  export C2HLS_OPENAI_HOSTED_URL="${ENDPOINT_URL_ARG}"
  export CHATHLS_API_BASE="${ENDPOINT_URL_ARG}"
fi

CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
if [[ -f "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
fi
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY:-}}"

JOBS_FILE="${C2HLS_ROOT}/artifacts/pc2/post_flash_overlap/overlap_jobs.txt"
mkdir -p artifacts/pc2/post_flash_overlap
{
  printf '%s\t%s\n' "${MM_ROOT}" "autosa_mm"
  if [[ "${MM_ONLY}" -eq 0 ]]; then
    printf '%s\t%s\n' "${WAVE1_ROOT}" "autosa_mm_hcl,autosa_mm_hcl_intel,autosa_mm_intel,autosa_mm_int16,autosa_mm_catapult,autosa_mm_getting_started"
  fi
} > "${JOBS_FILE}"
export C2HLS_OVERLAP_JOBS_FILE="${JOBS_FILE}"

echo "=== PE overlap ablation (fuse + ping-pong DATAFLOW on multi-PE) ==="
echo "jobs_file=${JOBS_FILE}"
if [[ "${MM_ONLY}" -eq 1 ]]; then
  echo "mm_only=1 (wave1 skipped)"
fi
cat "${JOBS_FILE}"
echo "model=${C2HLS_MODEL} endpoint=${OPENAI_BASE_URL:-}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_overlap.py" --pc2 \
    --matrix-root "${MM_ROOT}" --benches autosa_mm --dry-run
  if [[ "${MM_ONLY}" -eq 0 ]]; then
    "${PY}" "${C2HLS_ROOT}/scripts/pc2/run_post_flash_overlap.py" --pc2 \
      --matrix-root "${WAVE1_ROOT}" \
      --benches autosa_mm_hcl,autosa_mm_hcl_intel,autosa_mm_intel,autosa_mm_int16,autosa_mm_catapult,autosa_mm_getting_started \
      --dry-run
  fi
  exit 0
fi

OUT_STAMP="$(date -u +%Y%m%d_%H%M%S)"
JOB="$(sbatch --parsable \
  --chdir="${C2HLS_ROOT}" \
  --export=ALL \
  "${SCRIPT_DIR}/post_flash_overlap.sbatch.sh")"
echo "job=${JOB}"
echo "${JOB}" > "artifacts/pc2/post_flash_overlap/slurm_job_${OUT_STAMP}"
