#!/usr/bin/env bash
# Parallel 3-way flash test: skills_flash_hpc_positive_v1.json
#
#   flash_hpc_positive_v1_noskills
#   flash_hpc_positive_v1_all_skills
#   flash_hpc_positive_v1_bn_4_2
#
# Usage:
#   ./scripts/pc2/start_flash_hpc_positive_v1.sh --dry-run
#   ./scripts/pc2/start_flash_hpc_positive_v1.sh --auto-stop-on-complete
#   ./scripts/pc2/start_flash_hpc_positive_v1.sh --stamp 20260623_120000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
STAMP="${C2HLS_FLASH_HPC_POSITIVE_V1_STAMP:-$(date +%Y%m%d_%H%M%S)}"

VARIANT_KEYS=(
  noskills
  all_skills
  bn_4_2
)

declare -A SESSION_ID=(
  [noskills]=flash_hpc_positive_v1_noskills
  [all_skills]=flash_hpc_positive_v1_all_skills
  [bn_4_2]=flash_hpc_positive_v1_bn_4_2
)

declare -A ARTIFACT_PREFIX=(
  [noskills]=flash_hpc_positive_v1_noskills
  [all_skills]=flash_hpc_positive_v1_all_skills
  [bn_4_2]=flash_hpc_positive_v1_bn_4_2
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

export PC2_FORCE_WALLTIME="${PC2_HPC_POSITIVE_V1_WALLTIME:-12:00:00}"
if [[ -x "${C2HLS_ROOT}/.venv/bin/python3" ]]; then
  PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python3}"
else
  PY="${C2HLS_PYTHON:-python3}"
fi

echo "flash_hpc_positive_v1 stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME}"
echo "mode: parallel (3 independent sessions, each with own GPU + Vitis job)"
echo "skills: hls_full_optimization_skills_schema_1_1_package/skills_flash_hpc_positive_v1.json"
echo "validation: csim + csynth only (C2HLS_RUN_COSIM=0)"
echo "variants: ${VARIANT_KEYS[*]}"

for key in "${VARIANT_KEYS[@]}"; do
  echo "  dry-run ${key}..."
  C2HLS_FLASH_HPC_POSITIVE_V1_STAMP="${STAMP}" \
    "${PY}" scripts/pc2/run_flash_hpc_positive_batch.py --pc2 --variant "${key}" --dry-run --stamp "${STAMP}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok — no sessions started"
  exit 0
fi

start_one() {
  local session_id="$1"
  local worker_cmd="$2"
  local extra=()
  if [[ "${AUTO_STOP}" -eq 1 ]]; then
    extra+=(--auto-stop-on-complete)
  fi
  "${SCRIPT_DIR}/start_session.sh" --session-id "${session_id}" --worker-cmd "${worker_cmd}" "${extra[@]}"
}

for key in "${VARIANT_KEYS[@]}"; do
  sid="${SESSION_ID[$key]}"
  if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
    echo "stopping existing session ${sid}..."
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" || true
  else
    echo "resetting session ${sid} (no active watch)..."
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" 2>/dev/null || true
  fi
done

for key in "${VARIANT_KEYS[@]}"; do
  sid="${SESSION_ID[$key]}"
  worker_cmd="C2HLS_FLASH_HPC_POSITIVE_V1_STAMP=${STAMP} ${PY} scripts/pc2/run_flash_hpc_positive_batch.py --pc2 --variant ${key}"
  echo "starting session ${sid} (parallel)..."
  start_one "${sid}" "${worker_cmd}"
done

echo ""
echo "All 3 flash_hpc_positive_v1 sessions submitted (stamp=${STAMP}). Monitor:"
for key in "${VARIANT_KEYS[@]}"; do
  sid="${SESSION_ID[$key]}"
  echo "  tail -f artifacts/pc2/sessions/${sid}/watch.log"
done
echo ""
echo "Artifacts:"
for key in "${VARIANT_KEYS[@]}"; do
  echo "  artifacts/pc2/${ARTIFACT_PREFIX[$key]}_${STAMP}/"
done
