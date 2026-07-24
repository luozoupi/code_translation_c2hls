#!/usr/bin/env bash
# Start ChatHLS flash+streaming-dataflow batch_parallel.
#
# Flash: pure 90-skills (no no_RMW overlay).
# Dataflow (per-bench as soon as flash selected): flash_no_RMW_m_axi_skill_entries.json
# Cosim: on if possible (C2HLS_COSIM_REQUIRED=0).
# GPU: 1 node, borrow OFF, batch_park ON, park_grace_s=5400 (+1h).
# Compute: synth_nodes=cosim_nodes=#benches (16), 1 worker/node.
#
# Usage:
#   ./scripts/pc2/start_chathls_flash_dataflow_batch_parallel.sh --dry-run
#   ./scripts/pc2/start_chathls_flash_dataflow_batch_parallel.sh --stamp 20260712_chathls_fd
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
    --borrow-gpu)
      echo "ERROR: this campaign requires --no-borrow-gpu (GPU borrow off)" >&2
      exit 2
      ;;
    --no-borrow-gpu) shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Ensure corpus exists
READY_ROOT="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/chathls_ready"
if [[ ! -d "${READY_ROOT}/chathls_gemm" ]]; then
  echo "preparing chathls_ready corpus..."
  "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/prepare_chathls_ready.py" \
    --output-root "${READY_ROOT}"
fi

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_chathls_flash_dataflow.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-chathls_aav_n}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_chathls_fd}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpchfd}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_COSIM_REQUIRED=0
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_DATAFLOW_MAX_PARALLEL="${C2HLS_DATAFLOW_MAX_PARALLEL:-16}"

# RAG scrape for this ChatHLS campaign only (flash workers inherit via --export=ALL).
export C2HLS_RAG=1
export C2HLS_RAG_ENABLE=1
export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
export C2HLS_RAG_SCRAPE=1
export C2HLS_RAG_SCRAPE_CORPUS="${C2HLS_RAG_SCRAPE_CORPUS:-${C2HLS_ROOT}/artifacts/rag/knowledge_repo}"
export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

EXTRA_ARGS=(--no-borrow-gpu)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "=== ChatHLS flash+streaming-dataflow batch_parallel ==="
echo "stamp=${STAMP}"
echo "config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT}"
echo "gpu_borrow=off park_policy=on park_grace_s=5400"
echo "flash_skills=90 (no overlay) dataflow_skills=no_RMW overlay cosim=on-if-possible"
echo "streaming=per-bench flash→dataflow max_parallel=${C2HLS_DATAFLOW_MAX_PARALLEL}"
echo "rag=on scrape=on mode=${C2HLS_RAG_MODE} corpus=${C2HLS_RAG_SCRAPE_CORPUS}"
echo "tmp_run=${C2HLS_TMP_RUN}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exec env BATCH_PARALLEL_STAMP="${STAMP}" \
    "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
    --stamp "${STAMP}" \
    "${EXTRA_ARGS[@]}"
fi

env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
WATCH_LOG="${CAMPAIGN_ROOT}/flow/stream_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="bpchfd-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=72:00:00 \
    --export=ALL,C2HLS_RAG=1,C2HLS_RAG_ENABLE=1,C2HLS_RAG_MODE=${C2HLS_RAG_MODE},C2HLS_RAG_SCRAPE=1,C2HLS_RAG_SCRAPE_CORPUS=${C2HLS_RAG_SCRAPE_CORPUS},C2HLS_TMP_RUN=${C2HLS_TMP_RUN} \
    --wrap="bash ${SCRIPT_DIR}/wait_chathls_flash_stream_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} --max-parallel ${C2HLS_DATAFLOW_MAX_PARALLEL} >> ${WATCH_LOG} 2>&1"
)"

echo "submitted streaming flash→dataflow watcher job ${POST_JOB}"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"
