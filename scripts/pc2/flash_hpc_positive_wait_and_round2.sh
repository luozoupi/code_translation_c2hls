#!/usr/bin/env bash
# Wait 5 hours, analyze round-1 results, patch skills v2→v3, submit round-2 (noskills + all_skills).
#
# Usage:
#   nohup ./scripts/pc2/flash_hpc_positive_wait_and_round2.sh 20260623_120000 > artifacts/pc2/flash_hpc_positive_round2.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

ROUND1_STAMP="${1:?usage: $0 <round1_stamp>}"
WAIT_SEC="${C2HLS_HPC_POSITIVE_ROUND2_WAIT_SEC:-18000}"  # 5 hours

if [[ -x "${C2HLS_ROOT}/.venv/bin/python3" ]]; then
  PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python3}"
else
  PY="${C2HLS_PYTHON:-python3}"
fi

LOG_DIR="${C2HLS_ROOT}/artifacts/pc2"
LOG="${LOG_DIR}/flash_hpc_positive_round2_${ROUND1_STAMP}.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

log "round2 scheduler: waiting ${WAIT_SEC}s before analyzing stamp=${ROUND1_STAMP}"
sleep "${WAIT_SEC}"

log "analyzing round-1 matrices..."
"${PY}" scripts/pc2/flash_hpc_positive_analyze_and_patch.py \
  --stamp "${ROUND1_STAMP}" \
  --from-version v2 \
  --to-version v3 \
  --artifact-prefix flash_hpc_positive_v2 \
  --report "${LOG_DIR}/flash_hpc_positive_patch_v2_to_v3_${ROUND1_STAMP}.md" \
  2>&1 | tee -a "${LOG}"

ROUND2_STAMP="$(date +%Y%m%d_%H%M%S)"
log "submitting round-2 with skills v3 stamp=${ROUND2_STAMP}"

export C2HLS_HPC_POSITIVE_SKILLS_VERSION=v3
export C2HLS_HPC_POSITIVE_VARIANTS=noskills,all_skills

"${SCRIPT_DIR}/start_flash_hpc_positive_v2.sh" \
  --auto-stop-on-complete \
  --stamp "${ROUND2_STAMP}" \
  --variants noskills,all_skills \
  2>&1 | tee -a "${LOG}"

log "round-2 submitted stamp=${ROUND2_STAMP}"
