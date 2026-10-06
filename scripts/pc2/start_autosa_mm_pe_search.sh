#!/usr/bin/env bash
# autosa_mm compact PE×SIMD HLS search (csim+csynth, no LLM, no GPU).
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_pe_search.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_pe_search.sh --stamp 20260830_pesearch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "${C2HLS_PYTHON:-}" && -x "${C2HLS_PYTHON}" ]]; then
  PY="${C2HLS_PYTHON}"
elif [[ -x "${C2HLS_ROOT}/.venv/bin/python" ]]; then
  PY="${C2HLS_ROOT}/.venv/bin/python"
else
  PY=python3
fi
export C2HLS_PYTHON="${PY}"

OUT="${C2HLS_ROOT}/artifacts/pc2/compact_pe_search_${STAMP}"
HEADER="${C2HLS_PE_SEARCH_HEADER:-${C2HLS_ROOT}/related_work/benchmarks/autosa_ready/autosa_mm/kernel.h}"
TESTBENCH="${C2HLS_PE_SEARCH_TESTBENCH:-${C2HLS_ROOT}/related_work/benchmarks/autosa_ready/autosa_mm/testbench.cpp}"

export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_REFERENCE_COSIM=0
export PC2_BATCH_JOB_PREFIX=mmsearch
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset OPENAI_BASE_URL OPENAI_API_KEY CHATHLS_API_KEY CHATHLS_API_BASE \
  C2HLS_OPENAI_HOSTED_URL C2HLS_MODEL C2HLS_DSE_MODEL \
  BATCH_PARALLEL_EXTERNAL_LLM BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL \
  BATCH_PARALLEL_EXTERNAL_MODEL || true

echo "=== autosa_mm compact PE search ==="
echo "stamp=${STAMP}"
echo "campaign=${OUT}"
echo "header=${HEADER}"
echo "testbench=${TESTBENCH}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "job_prefix=${PC2_BATCH_JOB_PREFIX} (no LLM, no GPU)"
echo "candidates:"
"${PY}" - <<'PY'
from compact_pe_search import enumerate_mm_mesh_recipes, enumerate_mm_recipes, search_candidate_id
for rec in list(enumerate_mm_recipes()) + list(enumerate_mm_mesh_recipes()):
    print(search_candidate_id(rec))
PY

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

mkdir -p "${OUT}/slurm" artifacts/pc2/compact_pe_search
JOB="$(sbatch --parsable \
  --chdir="${C2HLS_ROOT}" \
  --job-name=mmsearch \
  --partition="${PC2_COMPUTE_PARTITION:-normal}" \
  --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
  --cpus-per-task=16 \
  --mem=64G \
  --time=12:00:00 \
  --output="${OUT}/slurm/%x-%j.out" \
  --error="${OUT}/slurm/%x-%j.err" \
  --export=ALL,C2HLS_ROOT="${C2HLS_ROOT}",C2HLS_PYTHON="${PY}",C2HLS_PE_SEARCH_STAMP="${STAMP}",C2HLS_PE_SEARCH_OUT="${OUT}",C2HLS_PE_SEARCH_HEADER="${HEADER}",C2HLS_PE_SEARCH_TESTBENCH="${TESTBENCH}",C2HLS_PART="${C2HLS_PART}",C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS}",C2HLS_RUN_COSIM=0,C2HLS_COSIM_REQUIRED=0,C2HLS_REFERENCE_COSIM=0,PC2_BATCH_JOB_PREFIX=mmsearch,PYTHONPATH="${PYTHONPATH}" \
  "${SCRIPT_DIR}/compact_pe_search.sbatch.sh")"
echo "job=${JOB}"
echo "campaign=${OUT}"
echo "${JOB}" > "${OUT}/slurm_job_id"
