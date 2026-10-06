#!/usr/bin/env bash
# autosa_mm variant sweep (43 base + 43 mem siblings × N). DAG: flash → DSE →
# stream/enf; every family also has a mem_* child. Unique PC2_BATCH_JOB_PREFIX
# per cell. Never writes frozen trees (includes 20260918).
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_variant_sweep.sh --dry-run --reps 1 --stamp 20260919
#   ./scripts/pc2/start_autosa_mm_variant_sweep.sh --wave flash,oneshot --submit \
#       --endpoint-url http://login5:18092/v1
#   ./scripts/pc2/start_autosa_mm_variant_sweep.sh --wave dse --submit
#   sbatch scripts/pc2/start_autosa_mm_mem_follow.sbatch.sh
#     1-rep +mem: flash/oneshot/DSE/stream/enf/mem (MAX_INFLIGHT=3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
if [[ -f "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
fi
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY:-}}"

exec "${PY}" "${SCRIPT_DIR}/autosa_mm_variant_sweep.py" "$@"
