#!/usr/bin/env bash
# Start one PC2 flash wave (5 variants) for the LLM-curated skills matrix.
#
# Usage:
#   ./scripts/pc2/start_curated_skills_flash_wave.sh --focus bottleneck --dry-run
#   ./scripts/pc2/start_curated_skills_flash_wave.sh --focus warnings --stamp 20260622_120000 --auto-stop-on-complete
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
STAMP="${C2HLS_FLASH_CURATED_STAMP:-$(date +%Y%m%d_%H%M%S)}"
FOCUS="${C2HLS_FLASH_CURATED_FOCUS:-bottleneck}"
VARIANTS_CSV="${C2HLS_FLASH_CURATED_VARIANTS:-noskills,all_avoids_json,all_avoids_llm,no_avoids_json,no_avoids_llm}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --focus) shift; FOCUS="$1"; shift ;;
    --variants) shift; VARIANTS_CSV="$1"; shift ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "${FOCUS}" in
  bottleneck|warnings|combined) ;;
  *)
    echo "invalid --focus ${FOCUS}; use bottleneck, warnings, or combined" >&2
    exit 2
    ;;
esac

export PC2_FORCE_WALLTIME="${PC2_CURATED_WALLTIME:-12:00:00}"
PY="${C2HLS_PYTHON:-python3}"

IFS=',' read -r -a VARIANT_KEYS <<< "${VARIANTS_CSV}"

echo "curated-skills flash wave focus=${FOCUS} stamp=${STAMP} walltime=${PC2_FORCE_WALLTIME} watch=${PC2_WATCH_INTERVAL_SEC:-60}s"
echo "skills: hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(73skills).json"
echo "variants: ${VARIANTS_CSV}"

for key in "${VARIANT_KEYS[@]}"; do
  key="$(echo "${key}" | xargs)"
  [[ -z "${key}" ]] && continue
  echo "  dry-run ${key}..."
  C2HLS_FLASH_CURATED_STAMP="${STAMP}" \
    "${PY}" scripts/pc2/run_flash_curated_skills_batch.py \
      --pc2 --variant "${key}" --focus "${FOCUS}" --dry-run --stamp "${STAMP}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run: wave ok, no Slurm jobs submitted"
  exit 0
fi

declare -A SESSION_ID=(
  [noskills]=flash_curated_noskills
  [all_avoids_json]=flash_curated_all_avoids_json
  [all_avoids_llm]=flash_curated_all_avoids_llm
  [no_avoids_json]=flash_curated_no_avoids_json
  [no_avoids_llm]=flash_curated_no_avoids_llm
)

declare -A STAMP_ENV=(
  [noskills]=C2HLS_FLASH_CURATED_NOSKILLS_STAMP
  [all_avoids_json]=C2HLS_FLASH_CURATED_ALL_AVOIDS_JSON_STAMP
  [all_avoids_llm]=C2HLS_FLASH_CURATED_ALL_AVOIDS_LLM_STAMP
  [no_avoids_json]=C2HLS_FLASH_CURATED_NO_AVOIDS_JSON_STAMP
  [no_avoids_llm]=C2HLS_FLASH_CURATED_NO_AVOIDS_LLM_STAMP
)

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
  key="$(echo "${key}" | xargs)"
  [[ -z "${key}" ]] && continue
  sid="${SESSION_ID[$key]:-}"
  if [[ -z "${sid}" ]]; then
    echo "unknown variant: ${key}" >&2
    exit 2
  fi
  export PC2_SESSION_ID="${sid}"
  _pc2_configure_session_paths
  if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
    "${SCRIPT_DIR}/stop_session.sh" --session-id "${sid}" || true
  fi
done
unset PC2_SESSION_ID

for key in "${VARIANT_KEYS[@]}"; do
  key="$(echo "${key}" | xargs)"
  [[ -z "${key}" ]] && continue
  sid="${SESSION_ID[$key]}"
  stamp_var="${STAMP_ENV[$key]}"
  worker_cmd="${stamp_var}=${STAMP} python3 scripts/pc2/run_flash_curated_skills_batch.py --pc2 --variant ${key} --focus ${FOCUS}"
  echo "starting session ${sid} (focus=${FOCUS})..."
  start_one "${sid}" "${worker_cmd}"
done

echo ""
echo "Curated-skills wave submitted (focus=${FOCUS} stamp=${STAMP}). Monitor:"
for key in "${VARIANT_KEYS[@]}"; do
  key="$(echo "${key}" | xargs)"
  [[ -z "${key}" ]] && continue
  sid="${SESSION_ID[$key]}"
  echo "  tail -f artifacts/pc2/sessions/${sid}/watch.log"
done
