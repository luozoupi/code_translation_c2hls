#!/usr/bin/env bash
# One-bench c2hls flash on autosa_ready/autosa_mm (original 64^3 nest).
# Goal: L_agent vs AutoSA rank1 (4228 cycles @ 3.33 ns U280).
#
# Policy: Devstral-2, flash then DSE then stream, csim+csynth, cosim/lat-opt/RAG/RAG2/dataflow/pragma_opt off.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_gap_flash.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_gap_flash.sh --stamp 20260818_mm_gap_flash
#   ./scripts/pc2/start_autosa_mm_gap_flash.sh --no-borrow-gpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
TRY_BORROW=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --borrow-gpu) TRY_BORROW=1; shift ;;
    --no-borrow-gpu) TRY_BORROW=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Scrub parent-shell poison before sbatch --export=ALL.
unset C2HLS_RAG_MODE C2HLS_RAG2_OPT_CORPUS C2HLS_RAG2_REPAIR_CORPUS C2HLS_RAG_SCRAPE_CORPUS || true
unset C2HLS_LATENCY_OPT_CHAIN_FLASH C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
unset C2HLS_LATENCY_OPT_ROUNDS C2HLS_LATENCY_OPT_REPAIR_ROUNDS || true
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
export C2HLS_POST_FLASH_LATENCY_OPT=0
export C2HLS_POST_FLASH_PRAGMA_OPT=0
export C2HLS_POST_FLASH_DATAFLOW=0
export C2HLS_POST_FLASH_DSE=1
export C2HLS_DSE_CHAIN_FLASH=1
export C2HLS_POST_FLASH_STREAM=1
export C2HLS_STREAM_CHAIN_FLASH=1
export C2HLS_PHASEB_FROM_GOLD=0

export C2HLS_STRATEGY=flash
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-1200}"
export C2HLS_MODEL="${C2HLS_MODEL:-mistralai/Devstral-2-123B-Instruct-2512}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

echo "=== preparing autosa_ready/autosa_mm (C-linkage csim ABI) ==="
"${PY}" "${C2HLS_ROOT}/scripts/prepare_autosa_ready.py" --kernel mm

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_autosa_mm_gap.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-autosa_nav_n}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_mm_gap}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpautmm}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-24:00:00}"
export PC2_GPU_WALLTIME="${PC2_GPU_WALLTIME:-24:00:00}"

EXTRA_ARGS=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "=== autosa_mm gap flash ==="
echo "stamp=${STAMP}"
echo "bench=autosa_mm (original nest, not rank1 stripped)"
echo "model=${C2HLS_MODEL}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "cosim=off lat-opt=off rag=off rag2=off dataflow=off pragma_opt=off dse=on"
echo "reference=autosa_mm_rank1 4228 cycles"

if [[ "${TRY_BORROW}" -eq 1 ]]; then
  CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
  export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
  export PC2_SESSION_DIR="${CAMPAIGN_ROOT}"
  export PC2_ENDPOINT_FILE="${CAMPAIGN_ROOT}/llm_endpoint.json"
  export PC2_WATCH_LOG="${CAMPAIGN_ROOT}/flow/watch.log"
  mkdir -p "${CAMPAIGN_ROOT}/flow"
  if "${SCRIPT_DIR}/borrow_gpu.sh" 2>/dev/null; then
    echo "borrowed existing GPU endpoint; submitting campaign without new gpu_h100 job"
    exec env BATCH_PARALLEL_STAMP="${STAMP}" \
      "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
      --stamp "${STAMP}" \
      --borrow-gpu \
      "${EXTRA_ARGS[@]}"
  fi
  echo "no borrowable GPU found; submitting dedicated gpu_h100 job with batch_park policy"
fi

exec env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"
