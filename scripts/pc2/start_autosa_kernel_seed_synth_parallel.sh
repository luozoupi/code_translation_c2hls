#!/usr/bin/env bash
# Parallel seed csynth for AutoSA-ready kernels (inventory order, skip frozen
# autosa_mm). No DeepSeek proxies: gold-gate only.
#
# Usage:
#   ./scripts/pc2/start_autosa_kernel_seed_synth_parallel.sh --wave all
#   ./scripts/pc2/start_autosa_kernel_seed_synth_parallel.sh --kernels autosa_mm_hcl,autosa_lu
#   ./scripts/pc2/start_autosa_kernel_seed_synth_parallel.sh --wave all --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

WAVE="all"
KERNELS_CSV=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wave) shift; WAVE="$1"; shift ;;
    --kernels) shift; KERNELS_CSV="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

if [[ -n "${KERNELS_CSV}" ]]; then
  IFS=',' read -r -a KERNELS <<< "${KERNELS_CSV}"
else
  mapfile -t KERNELS < <("${PY}" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from autosa_skill_bins import kernels_for_wave
for k in kernels_for_wave('${WAVE}'):
    print(k)
")
fi

_trimmed=()
for k in "${KERNELS[@]}"; do
  k="${k#"${k%%[![:space:]]*}"}"
  k="${k%"${k##*[![:space:]]}"}"
  [[ -n "${k}" ]] && _trimmed+=("${k}")
done
KERNELS=("${_trimmed[@]:-}")

if [[ ${#KERNELS[@]} -eq 0 ]]; then
  echo "ERROR: no kernels (wave=${WAVE})" >&2
  exit 2
fi

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/kernel_seed_synth_parallel_${STAMP}"
mkdir -p "${SEQ_ROOT}"

echo "=== parallel autosa kernel seed synth (no LLM) ==="
echo "stamp=${STAMP} wave=${WAVE} n=${#KERNELS[@]}"
echo "kernels=${KERNELS[*]}"
echo "seq_root=${SEQ_ROOT}"

{
  echo "stamp=${STAMP}"
  echo "llm=off"
  echo "flash=off"
} > "${SEQ_ROOT}/launch_summary.txt"

for bench in "${KERNELS[@]}"; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submit ${bench} (no proxy)"
  (
    unset BATCH_PARALLEL_ARTIFACT_PREFIX PC2_BATCH_JOB_PREFIX BATCH_PARALLEL_VARIANT || true
    unset C2HLS_FLASH_MIN_DSP C2HLS_FLASH_MAX_DSP C2HLS_FLASH_PE_BLK C2HLS_FLASH_ONCHIP || true
    unset C2HLS_FLASH_ROW_UF C2HLS_FLASH_K_TILE C2HLS_FLASH_ONCHIP_TILE || true
    unset C2HLS_FLASH_SKILL_ENTRIES_JSON C2HLS_PACKAGED_SKILLS_JSON || true
    args=(--kernel "${bench}" --stamp "${STAMP}_${bench}")
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      args+=(--dry-run)
    fi
    "${SCRIPT_DIR}/start_autosa_kernel_seed_synth.sh" "${args[@]}"
  )
  echo "${bench} llm=off" >> "${SEQ_ROOT}/launch_summary.txt"
done

echo "=== launched ${#KERNELS[@]} seed-synth campaigns ==="
cat "${SEQ_ROOT}/launch_summary.txt"
echo "seq_root=${SEQ_ROOT}"
