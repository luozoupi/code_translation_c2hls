#!/usr/bin/env bash
# A/B flash+cosim campaigns on Fir (28 parallel compute nodes each):
#   A) skills_ii_target_miss_solutions_added(90skills).json only
#   B) 90skills + flash_no_RMW_m_axi_skill_entries.json overlay
#
# Usage:
#   ./scripts/fir/start_flash_cosim_ab_campaign.sh --dry-run
#   ./scripts/fir/start_flash_cosim_ab_campaign.sh --submit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)_ab}"
DRY_RUN=0
SUBMIT=0

usage() {
  cat <<EOF
Usage: $0 [--dry-run | --submit] [options]

  --dry-run       Init both campaigns; no Slurm jobs
  --submit        Launch both campaigns (dedicated GPU each; no borrow, no park)
  --stamp STAMP   Shared stamp suffix for artifact dirs
  -h, --help

Policies (fixed): borrow=off, park=off (gpu_policy=always_on), 28 compute nodes each.
Skills A: skills_ii_target_miss_solutions_added(90skills).json only
Skills B: 90skills + flash_no_RMW_m_axi_skill_entries.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --submit) SUBMIT=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 && "${SUBMIT}" -eq 1 ]]; then
  echo "ERROR: use --dry-run or --submit, not both" >&2
  exit 2
fi
if [[ "${DRY_RUN}" -eq 0 && "${SUBMIT}" -eq 0 ]]; then
  echo "ERROR: specify --dry-run or --submit" >&2
  usage >&2
  exit 2
fi

export C2HLS_FIR_FLASH_COSIM=1
export C2HLS_LLM_TIMEOUT="${C2HLS_LLM_TIMEOUT:-7200}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-7200}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-57600}"
export C2HLS_TURNS="${C2HLS_TURNS:-4}"
export C2HLS_QUALITY_REPAIR_TURNS="${C2HLS_QUALITY_REPAIR_TURNS:-2}"
# GPU partition gpubase_bynode_b1 MaxTime=03:00:00 — use presubmit handoff for long runs.
export FIR_FORCE_WALLTIME="${FIR_FORCE_WALLTIME:-3:00:00}"
export FIR_COMPUTE_WALLTIME="${FIR_COMPUTE_WALLTIME:-24:00:00}"
export FIR_BATCH_PARALLEL_WALLTIME="${FIR_BATCH_PARALLEL_WALLTIME:-3:00:00}"
export FIR_GPU_PRESUBMIT_SEC="${FIR_GPU_PRESUBMIT_SEC:-600}"
# Slurm (match working zero_shot cosim campaign): auto-route compute, explicit GPU partition.
export FIR_GPU_PARTITION="${FIR_GPU_PARTITION:-gpubase_bynode_b1}"
export FIR_COMPUTE_PARTITION="${FIR_COMPUTE_PARTITION:-}"
export FIR_SLURM_ACCOUNT="${FIR_SLURM_ACCOUNT:-def-zhenman_gpu}"
export FIR_COMPUTE_SLURM_ACCOUNT="${FIR_COMPUTE_SLURM_ACCOUNT:-def-zhenman}"

PY="${C2HLS_PYTHON:-python3}"

CONFIG_OVERLAY="${SCRIPT_DIR}/batch_parallel_flash_cosim_90_overlay.json"
CONFIG_BASE="${SCRIPT_DIR}/batch_parallel_flash_cosim_90_base.json"
STAMP_OVERLAY="${STAMP}_overlay"
STAMP_BASE="${STAMP}_base"

run_one() {
  local config="$1"
  local stamp="$2"
  shift 2
  local artifact_prefix
  artifact_prefix="$("${PY}" - "${config}" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1]).read())["artifact_prefix"])
PY
)"
  BATCH_PARALLEL_CONFIG="${config}" \
    BATCH_PARALLEL_STAMP="${stamp}" \
    BATCH_PARALLEL_ARTIFACT_PREFIX="${artifact_prefix}" \
    "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
    --config "${config}" \
    --stamp "${stamp}" \
    "$@"
}

echo "=== Fir flash cosim A/B (stamp=${STAMP}) ==="
echo "gpu_walltime=${FIR_FORCE_WALLTIME} compute_walltime=${FIR_COMPUTE_WALLTIME} presubmit=${FIR_GPU_PRESUBMIT_SEC}s"
echo "cosim_timeout=${C2HLS_COSIM_TIMEOUT}s repair turns=${C2HLS_TURNS}"
echo "borrow=off park=off gpu_policy=always_on (dedicated GPU per campaign)"
echo ""

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "--- overlay (90skills + no_RMW overlay) ---"
  run_one "${CONFIG_OVERLAY}" "${STAMP_OVERLAY}" --dry-run
  echo ""
  echo "--- base (90skills only) ---"
  run_one "${CONFIG_BASE}" "${STAMP_BASE}" --dry-run
  echo ""
  echo "dry-run ok"
  exit 0
fi

echo "Submitting overlay campaign (90skills + overlay)..."
run_one "${CONFIG_OVERLAY}" "${STAMP_OVERLAY}" --no-borrow-gpu
echo ""
echo "Submitting base campaign (90skills only)..."
run_one "${CONFIG_BASE}" "${STAMP_BASE}" --no-borrow-gpu

echo ""
echo "Submitted both campaigns (56 parallel compute slots when fully allocated)."
echo "Overlay artifacts: artifacts/fir/flash_cosim_90_overlay_${STAMP_OVERLAY}/"
echo "Base artifacts:    artifacts/fir/flash_cosim_90_base_${STAMP_BASE}/"
echo "Monitor overlay: tail -f artifacts/fir/flash_cosim_90_overlay_${STAMP_OVERLAY}/flow/watch.log"
echo "Monitor base:    tail -f artifacts/fir/flash_cosim_90_base_${STAMP_BASE}/flow/watch.log"
