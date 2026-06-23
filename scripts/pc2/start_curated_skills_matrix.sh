#!/usr/bin/env bash
# Sequential LLM-curated flash matrix: 3 waves × 5 variants = 15 runs.
#
# Wave order: bottleneck → warnings → combined
# Each wave starts only when squeue is empty for $USER.
# All sessions: 12h walltime, 60s watch (via common.sh / wave launcher).
#
# Usage:
#   ./scripts/pc2/start_curated_skills_matrix.sh --dry-run
#   ./scripts/pc2/start_curated_skills_matrix.sh --auto-stop-on-complete
#   ./scripts/pc2/start_curated_skills_matrix.sh --from-wave warnings --stamp 20260622_150000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
AUTO_STOP=0
FORCE=0
FROM_WAVE="bottleneck"
STAMP_BASE="${C2HLS_FLASH_CURATED_MATRIX_STAMP:-$(date +%Y%m%d_%H%M%S)}"
POLL_SQUEUE_SEC="${C2HLS_CURATED_SQUEUE_POLL_SEC:-300}"
POLL_WAVE_SEC="${C2HLS_CURATED_WAVE_POLL_SEC:-120}"

WAVES=(bottleneck warnings combined)

declare -A ARTIFACT_PREFIX=(
  [noskills]=flash_curated_noskills
  [all_avoids_json]=flash_curated_all_avoids_json
  [all_avoids_llm]=flash_curated_all_avoids_llm
  [no_avoids_json]=flash_curated_no_avoids_json
  [no_avoids_llm]=flash_curated_no_avoids_llm
)
VARIANT_KEYS=(noskills all_avoids_json all_avoids_llm no_avoids_json no_avoids_llm)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --auto-stop-on-complete) AUTO_STOP=1; shift ;;
    --stamp) shift; STAMP_BASE="$1"; shift ;;
    --from-wave) shift; FROM_WAVE="$1"; shift ;;
    --force) FORCE=1; shift ;;
    --focus)
      shift
      WAVES=("$1")
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

wave_index() {
  local target="$1"
  local i
  for i in "${!WAVES[@]}"; do
    if [[ "${WAVES[$i]}" == "${target}" ]]; then
      echo "${i}"
      return 0
    fi
  done
  return 1
}

if ! wave_index "${FROM_WAVE}" >/dev/null; then
  echo "invalid --from-wave ${FROM_WAVE}" >&2
  exit 2
fi
START_IDX="$(wave_index "${FROM_WAVE}")"

if [[ "${FORCE}" -eq 0 ]]; then
  for sid in flash_curated_noskills flash_curated_all_avoids_json flash_curated_all_avoids_llm \
             flash_curated_no_avoids_json flash_curated_no_avoids_llm; do
    if pgrep -u "$(whoami)" -f "watch_session.sh ${sid}" >/dev/null 2>&1; then
      echo "refusing to start: watch_session still running for ${sid} (use --force)" >&2
      exit 1
    fi
  done
fi

wait_for_empty_squeue() {
  echo "waiting for empty squeue (poll ${POLL_SQUEUE_SEC}s)..."
  while squeue -u "$(whoami)" -h 2>/dev/null | grep -q .; do
    squeue -u "$(whoami)" -oh "%.18i %.9P %.20j %.8T %.10M %.6D %R" 2>/dev/null | head -5 || true
    sleep "${POLL_SQUEUE_SEC}"
  done
  echo "squeue empty"
}

wave_matrix_paths() {
  local focus="$1"
  local stamp="$2"
  local key
  for key in "${VARIANT_KEYS[@]}"; do
    echo "${C2HLS_ROOT}/artifacts/pc2/${ARTIFACT_PREFIX[$key]}_${focus}_${stamp}/matrix.json"
  done
}

wait_for_wave_complete() {
  local focus="$1"
  local stamp="$2"
  echo "waiting for wave focus=${focus} stamp=${stamp} (poll ${POLL_WAVE_SEC}s)..."
  while true; do
    local all_ok=1
    local path
    for path in $(wave_matrix_paths "${focus}" "${stamp}"); do
      if [[ ! -f "${path}" ]]; then
        all_ok=0
        break
      fi
    done
    if [[ "${all_ok}" -eq 1 ]]; then
      echo "wave complete: all matrix.json present"
      return 0
    fi
    sleep "${POLL_WAVE_SEC}"
  done
}

run_wave() {
  local focus="$1"
  local stamp="$2"
  local extra=()
  if [[ "${AUTO_STOP}" -eq 1 ]]; then
    extra+=(--auto-stop-on-complete)
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    extra+=(--dry-run)
  fi
  "${SCRIPT_DIR}/start_curated_skills_flash_wave.sh" \
    --focus "${focus}" --stamp "${stamp}" "${extra[@]}"
}

echo "curated skills matrix stamp_base=${STAMP_BASE} from_wave=${FROM_WAVE} dry_run=${DRY_RUN}"
echo "waves: ${WAVES[*]}"

for ((i=START_IDX; i<${#WAVES[@]}; i++)); do
  focus="${WAVES[$i]}"
  stamp="${STAMP_BASE}"

  if [[ "${DRY_RUN}" -eq 0 ]]; then
    wait_for_empty_squeue
  else
    echo "[dry-run] would wait for empty squeue before wave ${focus}"
  fi

  echo "=== wave $((i+1))/${#WAVES[@]} focus=${focus} stamp=${stamp} ==="
  run_wave "${focus}" "${stamp}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] would wait for wave ${focus} matrix.json files"
    continue
  fi

  wait_for_wave_complete "${focus}" "${stamp}"

  if [[ $((i+1)) -lt ${#WAVES[@]} ]]; then
    wait_for_empty_squeue
  fi
done

echo ""
echo "Curated skills matrix finished (stamp_base=${STAMP_BASE})."
echo "Artifacts: artifacts/pc2/flash_curated_*_{bottleneck,warnings,combined}_${STAMP_BASE}/"
