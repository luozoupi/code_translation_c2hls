#!/usr/bin/env bash
# Submit pragma-only DSE on the frozen autosa_mm flash kernel (no LLM, no GPU).
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_agent_pragma_dse.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_agent_pragma_dse.sh --stamp 20260818_mm_pragma_dse
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

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

OUT="${C2HLS_ROOT}/artifacts/pc2/autosa_mm_agent_pragma_dse_${STAMP}"
export AUTOSA_MM_PRAGMA_DSE_OUT="${OUT}"
export AUTOSA_MM_PRAGMA_DSE_SEED="${AUTOSA_MM_PRAGMA_DSE_SEED:-}"
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"

echo "=== autosa_mm agent pragma-only DSE ==="
echo "stamp=${STAMP}"
echo "out=${OUT}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "no LLM, no GPU, frozen flash kernel"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${PY}" "${C2HLS_ROOT}/scripts/autosa_mm_agent_pragma_dse.py" --dry-run
  exit 0
fi

mkdir -p "${OUT}" artifacts/pc2/autosa_mm_agent_pragma_dse
JOB="$(sbatch --parsable \
  --chdir="${C2HLS_ROOT}" \
  --export=ALL,C2HLS_ROOT="${C2HLS_ROOT}",AUTOSA_MM_PRAGMA_DSE_OUT="${OUT}",AUTOSA_MM_PRAGMA_DSE_SEED="${AUTOSA_MM_PRAGMA_DSE_SEED}",C2HLS_PYTHON="${PY}",C2HLS_PART="${C2HLS_PART}",C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS}" \
  "${SCRIPT_DIR}/autosa_mm_agent_pragma_dse.sbatch.sh")"
echo "job=${JOB}"
echo "out=${OUT}"
echo "${JOB}" > "${OUT}/slurm_job_id"
