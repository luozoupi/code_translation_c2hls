#!/usr/bin/env bash
# Start ONE GLM-4.7-FP8 ChatHLS flash+streaming-dataflow batch_parallel campaign
# against an already-running GLM vLLM endpoint (external_llm).
#
# Model: GLM-4.7-FP8 (2-node TP4+PP2 serve started by the sequence orchestrator).
# Compute: 16 combined-HLS nodes, gpu_policy=always_on.
#
# Usage:
#   ./scripts/pc2/start_chathls_glm_one.sh --flavor rag2_skills|rag2_ns|rag_skills|skills \
#       [--stamp STAMP] [--dry-run] [--endpoint-url URL]
#
# Flavors:
#   rag2_skills  RAG2 + 90-skills flash / no_RMW dataflow.
#   rag2_ns      RAG2 + no skills.
#   rag_skills   RAG scrape (full knowledge_repo) + 90-skills.
#   skills       noRAG / no scrape + 90-skills.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

FLAVOR=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
LATENCY_OPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor) shift; FLAVOR="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --latency-opt) LATENCY_OPT=1; shift ;;
    --no-latency-opt) LATENCY_OPT=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${FLAVOR}" in
  rag2_skills|rag2_ns|rag_skills|skills) ;;
  "")
    echo "ERROR: --flavor is required (rag2_skills|rag2_ns|rag_skills|skills)" >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown --flavor '${FLAVOR}' (expected rag2_skills|rag2_ns|rag_skills|skills)" >&2
    exit 2
    ;;
esac

# vLLM accepts any key; campaign starter requires OPENAI_API_KEY to be set.
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" || "${OPENAI_API_KEY}" == "empty" ]]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:-local-glm}"
  if [[ "${OPENAI_API_KEY}" == "EMPTY" || "${OPENAI_API_KEY}" == "empty" ]]; then
    export OPENAI_API_KEY="local-glm"
  fi
fi

READY_ROOT="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/chathls_ready"
if [[ ! -d "${READY_ROOT}/chathls_gemm" ]]; then
  echo "preparing chathls_ready corpus..."
  "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/prepare_chathls_ready.py" \
    --output-root "${READY_ROOT}"
fi

KR="${C2HLS_ROOT}/artifacts/rag/knowledge_repo"
RAG2_OPT="${C2HLS_ROOT}/artifacts/rag/rag2_opt"
RAG2_REPAIR="${C2HLS_ROOT}/artifacts/rag/rag2_repair"

_rag2_ensure_indexes() {
  if [[ ! -f "${RAG2_OPT}/chunks.jsonl" ]] || [[ ! -f "${RAG2_REPAIR}/chunks.jsonl" ]]; then
    echo "building RAG2 indexes under artifacts/rag/rag2_{opt,repair} ..."
    "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/build_rag2_indexes.py" \
      --knowledge-repo "${KR}" \
      --opt-out "${RAG2_OPT}" \
      --repair-out "${RAG2_REPAIR}"
  fi
}

case "${FLAVOR}" in
  rag2_skills)
    _rag2_ensure_indexes
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_chathls_fd_glm_rag2"
    FLAVOR_DESC="GLM-4.7 RAG2(opt/repair indexes)+skills"
    unset C2HLS_CHATHLS_NOSKILLS || true
    unset C2HLS_DATAFLOW_NO_SKILLS || true
    export C2HLS_RAG2=1
    export C2HLS_RAG=0
    export C2HLS_RAG_ENABLE=0
    export C2HLS_RAG_SCRAPE=0
    unset C2HLS_RAG_SCRAPE_CORPUS || true
    export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
    export C2HLS_RAG2_OPT_CORPUS="${C2HLS_RAG2_OPT_CORPUS:-${RAG2_OPT}}"
    export C2HLS_RAG2_REPAIR_CORPUS="${C2HLS_RAG2_REPAIR_CORPUS:-${RAG2_REPAIR}}"
    ;;
  rag2_ns)
    _rag2_ensure_indexes
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_chathls_fd_glm_rag2_ns"
    FLAVOR_DESC="GLM-4.7 RAG2(opt/repair indexes)-noskills"
    export C2HLS_CHATHLS_NOSKILLS=1
    export C2HLS_DATAFLOW_NO_SKILLS=1
    unset C2HLS_DATAFLOW_SKILL_ENTRIES_JSON || true
    unset C2HLS_FLASH_SKILL_ENTRIES_JSON || true
    export C2HLS_RAG2=1
    export C2HLS_RAG=0
    export C2HLS_RAG_ENABLE=0
    export C2HLS_RAG_SCRAPE=0
    unset C2HLS_RAG_SCRAPE_CORPUS || true
    export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
    export C2HLS_RAG2_OPT_CORPUS="${C2HLS_RAG2_OPT_CORPUS:-${RAG2_OPT}}"
    export C2HLS_RAG2_REPAIR_CORPUS="${C2HLS_RAG2_REPAIR_CORPUS:-${RAG2_REPAIR}}"
    ;;
  rag_skills)
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_chathls_fd_glm_rag"
    FLAVOR_DESC="GLM-4.7 RAG(full knowledge_repo)+skills"
    unset C2HLS_CHATHLS_NOSKILLS || true
    unset C2HLS_DATAFLOW_NO_SKILLS || true
    export C2HLS_RAG2=0
    export C2HLS_RAG=1
    export C2HLS_RAG_ENABLE=1
    export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
    export C2HLS_RAG_SCRAPE=1
    export C2HLS_RAG_SCRAPE_CORPUS="${KR}"
    ;;
  skills)
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_chathls_fd_glm_skills"
    FLAVOR_DESC="GLM-4.7 noRAG+skills"
    unset C2HLS_CHATHLS_NOSKILLS || true
    unset C2HLS_DATAFLOW_NO_SKILLS || true
    export C2HLS_RAG=0
    export C2HLS_RAG_ENABLE=0
    export C2HLS_RAG_SCRAPE=0
    export C2HLS_RAG2=0
    unset C2HLS_RAG_SCRAPE_CORPUS || true
    ;;
esac

export BATCH_PARALLEL_CONFIG="${SCRIPT_DIR}/batch_parallel_chathls_glm_u280.json"
export BATCH_PARALLEL_VARIANT="chathls_aav_n"
export C2HLS_MODEL=GLM-4.7-FP8
export BATCH_PARALLEL_EXTERNAL_MODEL=GLM-4.7-FP8
export C2HLS_COMBINED_HLS=1
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-7200}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_DATAFLOW_MAX_PARALLEL="${C2HLS_DATAFLOW_MAX_PARALLEL:-16}"
export PC2_BATCH_JOB_PREFIX="bpchglm"
export PC2_BATCH_PARALLEL_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"

if [[ "${LATENCY_OPT}" -eq 1 ]]; then
  export C2HLS_POST_FLASH_LATENCY_OPT=1
  export C2HLS_LATENCY_OPT_CHAIN_FLASH=1
  export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1
  export C2HLS_LATENCY_OPT_ROUNDS="${C2HLS_LATENCY_OPT_ROUNDS:-3}"
  export C2HLS_LATENCY_OPT_REPAIR_ROUNDS="${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3}"
  export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX}_lat"
  FLAVOR_DESC="${FLAVOR_DESC} +latency_opt"
else
  unset C2HLS_LATENCY_OPT_CHAIN_FLASH || true
  unset C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
  export C2HLS_POST_FLASH_LATENCY_OPT=0
fi

export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:8000/v1"
  echo "WARNING: --endpoint-url not given; using placeholder ${ENDPOINT_URL} for --dry-run only" >&2
else
  echo "ERROR: --endpoint-url is required for a real (non-dry-run) start." >&2
  echo "       Run the sequence orchestrator:" >&2
  echo "         ./scripts/pc2/start_chathls_glm_u280_sequence.sh" >&2
  exit 2
fi

export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"

echo "=== ChatHLS GLM-4.7 U280 batch_parallel: flavor=${FLAVOR} ==="
echo "stamp=${STAMP}"
echo "config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT}"
echo "flavor_desc=${FLAVOR_DESC}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "part=${C2HLS_PART} clock_ns=${C2HLS_CLOCK_NS} combined_hls=1 gpu_policy=always_on"
echo "rag=${C2HLS_RAG:-0} rag2=${C2HLS_RAG2:-0} rag_scrape_corpus=${C2HLS_RAG_SCRAPE_CORPUS:-none}"
echo "skills=$([[ "${C2HLS_CHATHLS_NOSKILLS:-0}" == "1" ]] && echo off || echo on)"
echo "latency_opt=${LATENCY_OPT} rounds=${C2HLS_LATENCY_OPT_ROUNDS:--} repair=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-}"
echo "tmp_run=${C2HLS_TMP_RUN}"

EXTRA_ARGS=(--external-llm)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
mkdir -p "${CAMPAIGN_ROOT}"
echo "${FLAVOR}" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "${LATENCY_OPT}" > "${CAMPAIGN_ROOT}/latency_opt.txt"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok"
  echo "campaign=${CAMPAIGN_ROOT}"
  exit 0
fi

WATCH_LOG="${CAMPAIGN_ROOT}/flow/stream_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

POST_EXPORT="ALL,C2HLS_RAG=${C2HLS_RAG:-0},C2HLS_RAG_ENABLE=${C2HLS_RAG_ENABLE:-0},C2HLS_RAG_MODE=${C2HLS_RAG_MODE:-everywhere},C2HLS_RAG_SCRAPE=${C2HLS_RAG_SCRAPE:-0},C2HLS_RAG_SCRAPE_CORPUS=${C2HLS_RAG_SCRAPE_CORPUS:-},C2HLS_RAG2=${C2HLS_RAG2:-0},C2HLS_RAG2_OPT_CORPUS=${C2HLS_RAG2_OPT_CORPUS:-},C2HLS_RAG2_REPAIR_CORPUS=${C2HLS_RAG2_REPAIR_CORPUS:-},C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_CHATHLS_NOSKILLS=${C2HLS_CHATHLS_NOSKILLS:-0},C2HLS_DATAFLOW_NO_SKILLS=${C2HLS_DATAFLOW_NO_SKILLS:-0},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_ENDPOINT_WAIT_SEC=${C2HLS_ENDPOINT_WAIT_SEC:-172800},C2HLS_POST_FLASH_LATENCY_OPT=${C2HLS_POST_FLASH_LATENCY_OPT:-0},C2HLS_LATENCY_OPT_CHAIN_FLASH=${C2HLS_LATENCY_OPT_CHAIN_FLASH:-0},C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=${C2HLS_LATENCY_OPT_CHAIN_DATAFLOW:-0},C2HLS_LATENCY_OPT_ROUNDS=${C2HLS_LATENCY_OPT_ROUNDS:-3},C2HLS_LATENCY_OPT_REPAIR_ROUNDS=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3}"

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="bpchglm-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=72:00:00 \
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
doc["model"] = "GLM-4.7-FP8"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted streaming flash->dataflow watcher job ${POST_JOB} (external_llm: no GPU dependency)"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"
