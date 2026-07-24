#!/usr/bin/env bash
# Start ONE Devstral-2 ChatHLS flash+streaming-dataflow RAG2 campaign (GPU vLLM).
#
# Usage:
#   ./scripts/pc2/start_chathls_devstral_rag2_one.sh --flavor rag2_skills|rag2_ns \
#       [--stamp STAMP] [--dry-run]
#
# Flavors:
#   rag2_skills  RAG2 + 90-skills flash / no_RMW dataflow (U280 3.33ns)
#   rag2_ns      RAG2 + no skills (U280 3.33ns)
#
# GPU: 1x node, 4x H100, borrow OFF, batch_park ON (same as other ChatHLS fd campaigns).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

FLAVOR=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
LATENCY_OPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor) shift; FLAVOR="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --latency-opt) LATENCY_OPT=1; shift ;;
    --no-latency-opt) LATENCY_OPT=0; shift ;;
    --borrow-gpu)
      echo "ERROR: this campaign requires --no-borrow-gpu (GPU borrow off)" >&2
      exit 2
      ;;
    --no-borrow-gpu) shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${FLAVOR}" in
  rag2_skills|rag2_ns) ;;
  "")
    echo "ERROR: --flavor is required (rag2_skills|rag2_ns)" >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown --flavor '${FLAVOR}' (expected rag2_skills|rag2_ns)" >&2
    exit 2
    ;;
esac

READY_ROOT="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/chathls_ready"
if [[ ! -d "${READY_ROOT}/chathls_gemm" ]]; then
  echo "preparing chathls_ready corpus..."
  "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/prepare_chathls_ready.py" \
    --output-root "${READY_ROOT}"
fi

RAG2_OPT="${C2HLS_ROOT}/artifacts/rag/rag2_opt"
RAG2_REPAIR="${C2HLS_ROOT}/artifacts/rag/rag2_repair"
if [[ ! -f "${RAG2_OPT}/chunks.jsonl" ]] || [[ ! -f "${RAG2_REPAIR}/chunks.jsonl" ]]; then
  echo "building RAG2 indexes ..."
  "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/build_rag2_indexes.py" \
    --knowledge-repo "${C2HLS_ROOT}/artifacts/rag/knowledge_repo" \
    --opt-out "${RAG2_OPT}" \
    --repair-out "${RAG2_REPAIR}"
fi

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_chathls_flash_dataflow.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-chathls_aav_n}"
# Export both so start_batch_parallel_campaign.sh cannot fall back to 13h.
export PC2_BATCH_PARALLEL_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_COSIM_REQUIRED=0
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-7200}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_DATAFLOW_MAX_PARALLEL="${C2HLS_DATAFLOW_MAX_PARALLEL:-16}"

# Fair compare with DeepSeek U280 campaigns.
export C2HLS_PART="${C2HLS_PART:-xcu280-fsvh2892-2L-e}"
export C2HLS_CLOCK_NS="${C2HLS_CLOCK_NS:-3.33}"

# RAG2 on; scrape off.
export C2HLS_RAG2=1
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
unset C2HLS_RAG_SCRAPE_CORPUS || true
export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
export C2HLS_RAG2_OPT_CORPUS="${C2HLS_RAG2_OPT_CORPUS:-${RAG2_OPT}}"
export C2HLS_RAG2_REPAIR_CORPUS="${C2HLS_RAG2_REPAIR_CORPUS:-${RAG2_REPAIR}}"

case "${FLAVOR}" in
  rag2_skills)
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_chathls_fd_rag2"
    export PC2_BATCH_JOB_PREFIX="bpchr2"
    FLAVOR_DESC="Devstral-2 RAG2+skills U280"
    unset C2HLS_CHATHLS_NOSKILLS || true
    unset C2HLS_DATAFLOW_NO_SKILLS || true
    ;;
  rag2_ns)
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_chathls_fd_rag2_ns"
    export PC2_BATCH_JOB_PREFIX="bpchr2n"
    FLAVOR_DESC="Devstral-2 RAG2-noskills U280"
    export C2HLS_CHATHLS_NOSKILLS=1
    export C2HLS_DATAFLOW_NO_SKILLS=1
    unset C2HLS_DATAFLOW_SKILL_ENTRIES_JSON || true
    unset C2HLS_FLASH_SKILL_ENTRIES_JSON || true
    ;;
esac

if [[ "${LATENCY_OPT}" -eq 1 ]]; then
  export C2HLS_POST_FLASH_LATENCY_OPT=1
  export C2HLS_LATENCY_OPT_CHAIN_FLASH=1
  export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1
  export C2HLS_LATENCY_OPT_ROUNDS="${C2HLS_LATENCY_OPT_ROUNDS:-3}"
  export C2HLS_LATENCY_OPT_REPAIR_ROUNDS="${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3}"
  export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX}_lat"
  FLAVOR_DESC="${FLAVOR_DESC} +latency_opt"
else
  unset C2HLS_POST_FLASH_LATENCY_OPT || true
  unset C2HLS_LATENCY_OPT_CHAIN_FLASH || true
  unset C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
  export C2HLS_POST_FLASH_LATENCY_OPT=0
fi

export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

EXTRA_ARGS=(--no-borrow-gpu)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

echo "=== ChatHLS Devstral-2 RAG2 batch_parallel: flavor=${FLAVOR} ==="
echo "stamp=${STAMP}"
echo "config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT}"
echo "flavor_desc=${FLAVOR_DESC}"
echo "part=${C2HLS_PART} clock_ns=${C2HLS_CLOCK_NS}"
echo "rag2=1 mode=${C2HLS_RAG_MODE} opt=${C2HLS_RAG2_OPT_CORPUS} repair=${C2HLS_RAG2_REPAIR_CORPUS}"
echo "skills=$([[ "${C2HLS_CHATHLS_NOSKILLS:-0}" == "1" ]] && echo off || echo on)"
echo "latency_opt=${LATENCY_OPT} rounds=${C2HLS_LATENCY_OPT_ROUNDS:--} repair=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-}"
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
echo "${FLAVOR}" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "${LATENCY_OPT}" > "${CAMPAIGN_ROOT}/latency_opt.txt"

GPU_JOB_ID="$(
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
print((json.loads(p.read_text()).get("gpu_job_id") or "") if p.is_file() else "")
PY
)"
DEP_ARGS=()
if [[ -n "${GPU_JOB_ID}" ]]; then
  DEP_ARGS=(--dependency="after:${GPU_JOB_ID}")
fi

POST_EXPORT="ALL,C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=1,C2HLS_RAG_MODE=${C2HLS_RAG_MODE},C2HLS_RAG2_OPT_CORPUS=${C2HLS_RAG2_OPT_CORPUS},C2HLS_RAG2_REPAIR_CORPUS=${C2HLS_RAG2_REPAIR_CORPUS},C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_CHATHLS_NOSKILLS=${C2HLS_CHATHLS_NOSKILLS:-0},C2HLS_DATAFLOW_NO_SKILLS=${C2HLS_DATAFLOW_NO_SKILLS:-0},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_ENDPOINT_WAIT_SEC=${C2HLS_ENDPOINT_WAIT_SEC:-172800},C2HLS_POST_FLASH_LATENCY_OPT=${C2HLS_POST_FLASH_LATENCY_OPT:-0},C2HLS_LATENCY_OPT_CHAIN_FLASH=${C2HLS_LATENCY_OPT_CHAIN_FLASH:-0},C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=${C2HLS_LATENCY_OPT_CHAIN_DATAFLOW:-0},C2HLS_LATENCY_OPT_ROUNDS=${C2HLS_LATENCY_OPT_ROUNDS:-3},C2HLS_LATENCY_OPT_REPAIR_ROUNDS=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3}"

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="${PC2_BATCH_JOB_PREFIX}-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=72:00:00 \
    "${DEP_ARGS[@]}" \
    --export="${POST_EXPORT}" \
    --wrap="bash ${SCRIPT_DIR}/wait_chathls_flash_stream_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} --max-parallel ${C2HLS_DATAFLOW_MAX_PARALLEL} >> ${WATCH_LOG} 2>&1"
)"

"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_watcher_job_id"] = "${POST_JOB}"
doc["flavor"] = "${FLAVOR}"
doc["rag_method"] = "rag2"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted streaming flash→dataflow watcher job ${POST_JOB} (dependency after:${GPU_JOB_ID:-none})"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"
